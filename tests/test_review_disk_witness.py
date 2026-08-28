"""Tests for the disk-witness detection of background-review writes.

Local overlay tests (not upstream). They guard the bug measured on 2026-07-26:
two background skill reviews patched five skills between them and both reported
"no changes", because on the native ACP path the review fork's tool calls show
up neither as ``tool`` role messages nor in the ledger the summariser reads.

Coverage:
  A. snapshot/diff labels created / updated / archived skills and memory writes
  B. a review that only READS skills is not reported as a change
     (``.usage.json`` bumps ``last_viewed_at`` on reads — hence dot-skipping)
  C. end-to-end ``_run_review_in_thread``: a fork that writes a skill file while
     producing zero tool messages and an empty ledger still notifies the user
     (this is the exact regression)
  D. end-to-end: a fork that writes nothing stays SILENT in chat, but records
     the pass in ``state/background-review.json`` so silence is auditable
  F. end-to-end: a fork that RAISES is recorded too (source="error"), on its own
     ``error_streak`` — recording only the success path recreated the same
     "silence == health" ambiguity from the other side
  G. ``no_change_streak`` is tracked per review kind, so a memory review that
     still writes cannot mask a skill detector that went blind
  H. the memory witness globs ``*.md`` rather than a hardcoded pair of names

Standalone runner, plain asserts — matches tests/test_background_review_list_shapes.py.
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestRunner:
    def __init__(self):
        self.passed = []
        self.failed = []

    def run(self, name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — runner summary uses it
            import traceback
            self.failed.append((name, e, traceback.format_exc()))
        else:
            self.passed.append(name)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 70}\nResults: {len(self.passed)}/{total} passed")
        for n in self.passed:
            print(f"  [ok] {n}")
        if self.failed:
            print(f"\n--- {len(self.failed)} failure(s) ---")
            for n, _e, tb in self.failed:
                print(f"\n[FAIL] {n}\n{tb}")
        return 0 if not self.failed else 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Home:
    """A throwaway HERMES_HOME with a skills/ and memory/ tree.

    The tree is deleted at PROCESS EXIT, never at test teardown. Deleting it
    mid-run is what made this suite poison the rest of a pytest session: a
    module imported while HERMES_HOME pointed here can freeze a derived path
    in a module-level constant (``hermes_state.DEFAULT_DB_PATH``), and after
    the directory is gone that constant points at nothing for every later
    test — ``session_search`` gates on ``DEFAULT_DB_PATH.parent.exists()``, so
    the ACP tools-server suite failed while passing on its own.

    This is the same discipline pytest's own ``_hermetic_environment``
    fixture (tests/conftest.py) follows for its per-test HERMES_HOME: hand it
    out, never remove it while the process lives. Module eviction stays as a
    second layer so teardown still leaves a clean ``sys.modules``.
    """

    def __init__(self):
        self.path = Path(tempfile.mkdtemp(prefix="hermes-review-witness-"))
        atexit.register(shutil.rmtree, str(self.path), ignore_errors=True)
        (self.path / "skills" / "hermes" / "alpha").mkdir(parents=True)
        (self.path / "skills" / "hermes" / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\n---\nbody\n", encoding="utf-8"
        )
        (self.path / "skills" / ".archive").mkdir()
        (self.path / "skills" / ".usage.json").write_text("{}", encoding="utf-8")
        (self.path / "memories").mkdir()
        (self.path / "memories" / "MEMORY.md").write_text("- one\n", encoding="utf-8")
        self._mods_before = set(sys.modules)
        self._prev = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.path)

    def witness(self):
        """Fresh import of the witness bound to this HERMES_HOME."""
        for mod in ("agent.review_disk_witness",):
            sys.modules.pop(mod, None)
        return importlib.import_module("agent.review_disk_witness")

    def close(self):
        if self._prev is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._prev
        self._evict_modules_bound_to_home()
        # NOT removed here — see the class docstring. atexit owns the cleanup.

    def _evict_modules_bound_to_home(self):
        """Drop modules that froze a path inside this throwaway home.

        Restoring the env var is not enough: a module imported while
        HERMES_HOME pointed here can derive a path once, at import, and keep
        it in a module-level constant (`hermes_state.DEFAULT_DB_PATH`). The
        next test would then read this home's state instead of its own.

        Second layer only — the directory now outlives the process, so even a
        constant this pass fails to evict still points at a real tree.
        """
        root = str(self.path)
        for name in set(sys.modules) - self._mods_before:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            try:
                values = list(vars(mod).values())
            except TypeError:  # extension modules without a __dict__
                continue
            if any(isinstance(v, (str, Path)) and str(v).startswith(root) for v in values):
                sys.modules.pop(name, None)


def _bump(path: Path, text: str):
    """Write and force a distinct mtime so the diff cannot be a stat race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_mtime_ns + 2_000_000_000, st.st_mtime_ns + 2_000_000_000))


# ---------------------------------------------------------------------------
# A. diff labels
# ---------------------------------------------------------------------------


def test_a_diff_reports_created_updated_archived_and_memory():
    home = _Home()
    try:
        w = home.witness()
        before = w.snapshot_review_targets()
        assert "hermes/alpha/SKILL.md" in before["skills"], before["skills"]

        # created skill, updated skill, new supporting file, archived skill,
        # memory write — all in one pass.
        _bump(home.path / "skills" / "hermes" / "beta" / "SKILL.md", "new\n")
        _bump(home.path / "skills" / "hermes" / "alpha" / "SKILL.md", "changed\n")
        (home.path / "skills" / "hermes" / "gamma").mkdir(parents=True)
        _bump(home.path / "skills" / "hermes" / "gamma" / "SKILL.md", "g\n")
        _bump(home.path / "memories" / "MEMORY.md", "- one\n- two\n")
        after1 = w.snapshot_review_targets()
        actions = w.diff_review_targets(before, after1)
        assert "Skill 'beta' created" in actions, actions
        assert "Skill 'gamma' created" in actions, actions
        assert "Skill 'alpha' updated" in actions, actions
        assert "Memory updated" in actions, actions

        # supporting file only -> counts as an update of its skill
        base = w.snapshot_review_targets()
        _bump(home.path / "skills" / "hermes" / "beta" / "references" / "x.md", "ref\n")
        sup = w.diff_review_targets(base, w.snapshot_review_targets())
        assert sup == ["Skill 'beta' updated"], sup

        # deletion -> archived
        base2 = w.snapshot_review_targets()
        shutil.rmtree(home.path / "skills" / "hermes" / "gamma")
        gone = w.diff_review_targets(base2, w.snapshot_review_targets())
        assert gone == ["Skill 'gamma' archived"], gone
    finally:
        home.close()


def test_b_reads_and_dotfiles_are_not_changes():
    """A review that only viewed skills must diff empty.

    ``skills/.usage.json`` gets ``last_viewed_at`` rewritten on every
    skill_view, and ``.curator_backups/`` churns on curator passes. Including
    either would turn "the review looked around" into "the review changed
    things" — the false-positive mirror of the bug being fixed.
    """
    home = _Home()
    try:
        w = home.witness()
        before = w.snapshot_review_targets()
        _bump(home.path / "skills" / ".usage.json",
              json.dumps({"alpha": {"last_viewed_at": "2026-07-26T00:00:00Z"}}))
        _bump(home.path / "skills" / ".curator_backups" / "snap" / "alpha.md", "x")
        assert w.diff_review_targets(before, w.snapshot_review_targets()) == []
    finally:
        home.close()


# ---------------------------------------------------------------------------
# C/D. end-to-end through _run_review_in_thread
# ---------------------------------------------------------------------------


class _ParentStub:
    """Enough of AIAgent for _run_review_in_thread; unknown attrs read as None."""

    def __init__(self):
        self.model = "claude-opus-5"
        self.provider = "copilot-acp"
        self.platform = "telegram"
        self.session_id = "sess-1"
        self.memory_notifications = "on"
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self.request_overrides = {}
        self.max_tokens = 4096
        self.valid_tool_names = ["memory", "skill_manage"]
        self.printed = []
        self.notified = []
        self.aux_failures = []
        self.background_review_callback = self.notified.append
        self._memory_store = None
        self._memory_enabled = True
        self._user_profile_enabled = True
        self._cached_system_prompt = "cached"
        self._acp_native_tool_calls = []
        # Real containers, not the permissive __getattr__ default: v2026.8.13
        # registers the review fork on the parent's ``_active_children`` and
        # ``_background_review_agent``. A stub that answers None to everything
        # turns "attribute exists" into "attribute is None", and upstream's
        # ``hasattr`` guard then walks straight into ``None.append`` — five
        # tests failed on that, all reporting the same swallowed AttributeError
        # instead of the thing they were about.
        self._active_children = []
        self._background_review_agent = None

    def __getattr__(self, name):  # noqa: D105 — permissive stub
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def _current_main_runtime(self):
        return {"api_key": "k", "base_url": None, "api_mode": None}

    def _safe_print(self, text):
        self.printed.append(text)

    def _emit_auxiliary_failure(self, label, exc):
        self.aux_failures.append((label, exc))


def _install_fake_aiagent(on_run):
    """Inject a fake ``run_agent.AIAgent`` whose run_conversation calls on_run.

    The fake records NO tool messages and leaves the ledger empty — exactly the
    native-ACP shape in which the real reviews went undetected.
    """
    class _Fork:
        def __init__(self, **kwargs):
            self._session_messages = []
            self._acp_native_tool_calls = []

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return None

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

        def run_conversation(self, user_message=None, conversation_history=None):
            on_run()
            return "done"

        def shutdown_memory_provider(self):
            return None

        def close(self):
            return None

    mod = sys.modules.get("run_agent")
    created = False
    if mod is None:
        mod = types.ModuleType("run_agent")
        sys.modules["run_agent"] = mod
        created = True
    prev = getattr(mod, "AIAgent", None)
    mod.AIAgent = _Fork
    return mod, prev, created


_SKILL_PROMPT = "Review the conversation above and update the skill library."
_MEM_PROMPT = "Review the conversation above, consider saving to memory."


def _run_review(home, on_run, prompt=_SKILL_PROMPT):
    sys.modules.pop("agent.background_review", None)
    bg = importlib.import_module("agent.background_review")
    mod, prev, created = _install_fake_aiagent(on_run)
    parent = _ParentStub()
    try:
        bg._run_review_in_thread(parent, [], prompt)
    finally:
        if created:
            sys.modules.pop("run_agent", None)
        elif prev is not None:
            mod.AIAgent = prev
    return parent


def test_c_disk_write_is_reported_even_with_empty_ledger():
    """THE regression: fork patches a skill, no tool messages, empty ledger."""
    home = _Home()
    try:
        home.witness()
        target = home.path / "skills" / "hermes" / "alpha" / "SKILL.md"

        parent = _run_review(home, lambda: _bump(target, "patched by review\n"))

        assert not parent.aux_failures, parent.aux_failures
        joined = " ".join(parent.notified)
        assert "Skill 'alpha' updated" in joined, (
            f"review write went unreported: notified={parent.notified!r} "
            f"printed={parent.printed!r}"
        )
        state = json.loads(
            (home.path / "state" / "background-review.json").read_text(encoding="utf-8")
        )
        assert state["last_source"] == "disk", state
        assert state["last_actions"] == ["Skill 'alpha' updated"], state
        assert state["no_change_streak"] == 0, state
    finally:
        home.close()


def test_d_no_change_is_silent_but_recorded():
    """No write -> nothing in chat, but the pass is on record."""
    home = _Home()
    try:
        home.witness()
        parent = _run_review(home, lambda: None)

        assert parent.notified == [], parent.notified
        assert not any("Self-improvement" in p for p in parent.printed), parent.printed
        state = json.loads(
            (home.path / "state" / "background-review.json").read_text(encoding="utf-8")
        )
        assert state["last_actions"] == [], state
        assert state["no_change_streak"] == 1, state
        assert state["recent"][-1]["notified"] is False, state["recent"][-1]
        assert state["recent"][-1]["kind"] == "скиллы", state["recent"][-1]
    finally:
        home.close()


def test_e_memory_write_is_reported():
    home = _Home()
    try:
        home.witness()
        mem = home.path / "memories" / "MEMORY.md"
        parent = _run_review(home, lambda: _bump(mem, "- one\n- fresh fact\n"))
        assert any("Memory updated" in n for n in parent.notified), parent.notified
    finally:
        home.close()


def _state(home) -> dict:
    return json.loads(
        (home.path / "state" / "background-review.json").read_text(encoding="utf-8")
    )


def test_f_crashed_review_is_recorded_not_silent():
    """A review that RAISES must still leave a record.

    The regression this guards: ``record_review_pass`` used to sit on the
    success path only, so a crashing review wrote nothing — no record, no
    streak, ``last_run_at`` frozen at the last good pass. Chat silent, health
    guard silent: the original bug from the other side.
    """
    home = _Home()
    try:
        home.witness()

        def _boom():
            raise RuntimeError("fork exploded")

        parent = _run_review(home, _boom)
        assert parent.aux_failures, "the failure should still surface to the agent"

        st = _state(home)
        assert st["last_source"] == "error", st
        assert st["error_streak"] == 1, st
        assert st["errors_total"] == 1, st
        assert "fork exploded" in (st.get("last_error") or ""), st
        # a crash is NOT a no-change pass — different diagnosis, different counter
        assert st["no_change_streak"] == 0, st
        assert st["recent"][-1].get("error"), st["recent"][-1]

        # ...and a healthy pass afterwards clears the error streak
        _run_review(home, lambda: None)
        st2 = _state(home)
        assert st2["error_streak"] == 0, st2
        assert st2["no_change_streak"] == 1, st2
        assert st2["errors_total"] == 1, st2
    finally:
        home.close()


def test_g_no_change_streak_is_per_kind():
    """A working memory review must not mask a blind skill detector.

    With one shared counter, every memory pass that writes something resets the
    streak, so a skill detector that stopped seeing writes never reaches any
    alarm threshold.
    """
    home = _Home()
    try:
        home.witness()
        mem = home.path / "memories" / "MEMORY.md"
        for i in range(4):
            # skills review: sees nothing (the blindness being simulated)
            _run_review(home, lambda: None, prompt=_SKILL_PROMPT)
            # memory review: writes every time, so it keeps zeroing the global
            _run_review(home, lambda i=i: _bump(mem, f"- fact {i}\n"), prompt=_MEM_PROMPT)

        st = _state(home)
        assert st["no_change_streak"] == 0, st  # global: masked, as expected
        kinds = st["kinds"]
        assert kinds["скиллы"]["no_change_streak"] == 4, kinds
        assert kinds["память"]["no_change_streak"] == 0, kinds
        assert kinds["скиллы"]["runs_total"] == 4, kinds
        assert kinds["память"].get("last_change_at"), kinds
    finally:
        home.close()


def test_h_memory_witness_globs_new_files():
    """Any *.md in the memory dir is witnessed, not just MEMORY.md/USER.md."""
    home = _Home()
    try:
        w = home.witness()
        # non-".md" siblings are bookkeeping and must stay invisible
        (home.path / "memories" / "MEMORY.md.lock").write_text("", encoding="utf-8")
        before = w.snapshot_review_targets()
        assert "MEMORY.md.lock" not in before["memory"], before["memory"]

        _bump(home.path / "memories" / "PROJECTS.md", "- new memory file\n")
        actions = w.diff_review_targets(before, w.snapshot_review_targets())
        assert actions == ["Memory updated"], actions

        base = w.snapshot_review_targets()
        _bump(home.path / "memories" / "MEMORY.md.lock", "touched")
        assert w.diff_review_targets(base, w.snapshot_review_targets()) == []
    finally:
        home.close()


def test_i_teardown_leaves_no_module_bound_to_the_throwaway_home():
    """Teardown must not poison the rest of the pytest process.

    Regression: the suite passed alone and made
    tests/agent/transports/test_hermes_acp_tools_server.py fail when run in
    the same command, because a module kept a path into the deleted temp home.
    """
    home = _Home()
    try:
        home.witness()
    finally:
        home.close()

    # First layer: the tree survives teardown, so any path a module froze is
    # still a real directory (this is what actually broke the other suite).
    assert home.path.is_dir(), "teardown must not delete a live HERMES_HOME"

    root = str(home.path)
    leaked = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            values = list(vars(mod).values())
        except TypeError:
            continue
        if any(isinstance(v, (str, Path)) and str(v).startswith(root) for v in values):
            leaked.append(name)
    assert leaked == [], leaked

    from tools.session_search_tool import check_session_search_requirements

    assert check_session_search_requirements() is True


if __name__ == "__main__":
    r = TestRunner()
    r.run("A diff labels created/updated/archived/memory", test_a_diff_reports_created_updated_archived_and_memory)
    r.run("B reads and dotfiles are not changes", test_b_reads_and_dotfiles_are_not_changes)
    r.run("C disk write reported with empty ledger", test_c_disk_write_is_reported_even_with_empty_ledger)
    r.run("D no change is silent but recorded", test_d_no_change_is_silent_but_recorded)
    r.run("E memory write reported", test_e_memory_write_is_reported)
    r.run("F crashed review is recorded", test_f_crashed_review_is_recorded_not_silent)
    r.run("G no_change_streak is per kind", test_g_no_change_streak_is_per_kind)
    r.run("H memory witness globs new files", test_h_memory_witness_globs_new_files)
    r.run("I teardown leaks no module paths", test_i_teardown_leaves_no_module_bound_to_the_throwaway_home)
    sys.exit(r.summary())
