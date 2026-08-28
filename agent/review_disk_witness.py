"""Deciding what a background review changed by looking at the DISK.

Local overlay module (not upstream). Companion to ``agent.acp_native_calls``,
which reconstructs review actions from the engine's native-tool ledger.

Why this exists
---------------
Upstream detects review actions by walking the fork's session for ``tool`` role
messages. On the native ACP path there are none. The ledger fallback was added
next, and it too can come back empty: the ledger lives on whichever object got
the ACP client, and a fork that creates its own ``shared=True`` client records
where the summariser does not look.

Measured 2026-07-26: two consecutive skill reviews (11:22:47-11:25:38 and
11:35:28-11:37:50 MSK) patched five skills between them -- promise-arm-watcher,
linux-host-triage, memory-and-skill-hygiene, cron-silent-stall-guard,
fable-refusal-acp-native-tools, all timestamped inside those windows in
``skills/.usage.json`` -- while the user-facing summary said "no changes".

The filesystem is the one witness that does not depend on provider, transport,
or which object holds the ledger: if a review wrote a skill or a memory entry,
the bytes moved. So snapshot before, snapshot after, diff.

Deliberately NOT snapshotted: dot-entries under the skills dir. ``.usage.json``
records ``last_viewed_at`` on every read, so including it would report a review
that only LOOKED at skills as one that changed them.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# path -> (mtime_ns, size); ``kind`` distinguishes the two stores.
Snapshot = Dict[str, Dict[str, Tuple[int, int]]]

_MAX_FILES = 20_000  # runaway guard: a stat walk, but never an unbounded one


def _stat_pair(path: Path) -> Optional[Tuple[int, int]]:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _skills_root() -> Optional[Path]:
    """The live skills dir for the CURRENT profile, or None if unavailable.

    Resolved through skill_manager_tool._skills_dir() rather than composing it
    from HERMES_HOME, because the review fork runs in a thread that inherits a
    propagated profile context and that helper is what the write path itself
    honours (#54937).
    """
    try:
        from tools.skill_manager_tool import _skills_dir
        root = Path(_skills_dir())
    except Exception:
        home = os.environ.get("HERMES_HOME")
        if not home:
            return None
        root = Path(home) / "skills"
    return root if root.is_dir() else None


def _memory_files() -> List[Path]:
    """Every markdown file in the profile's memory dir.

    A glob rather than the hardcoded [MEMORY.md, USER.md] pair it replaced: the
    two happen to be all that exists today, so the old list was accidentally
    correct, and would have silently stopped witnessing the moment a review
    created any other memory file. Non-".md" siblings (``*.lock``,
    ``*.bak.premigrate``) stay out — they are bookkeeping, not memory.
    """
    try:
        from tools.memory_tool import get_memory_dir
        mem_dir = Path(get_memory_dir())
    except Exception:
        return []
    try:
        return sorted(p for p in mem_dir.glob("*.md") if p.is_file())
    except OSError:
        return []


def snapshot_review_targets() -> Snapshot:
    """Stat every skill file and memory file the review is allowed to write.

    Never raises: a snapshot that failed is an empty snapshot, and an empty
    ``before`` simply makes the diff report nothing (the caller still has the
    ledger and message-walk paths).
    """
    skills: Dict[str, Tuple[int, int]] = {}
    memory: Dict[str, Tuple[int, int]] = {}
    try:
        root = _skills_root()
        if root is not None:
            count = 0
            for path in root.rglob("*"):
                rel = path.relative_to(root)
                # Skip .archive/, .curator_backups/, .usage.json, locks, caches.
                if any(part.startswith(".") for part in rel.parts):
                    continue
                if not path.is_file():
                    continue
                pair = _stat_pair(path)
                if pair is not None:
                    skills[str(rel)] = pair
                count += 1
                if count >= _MAX_FILES:
                    logger.warning(
                        "review disk witness: skills walk hit the %d-file cap; "
                        "diff may be incomplete", _MAX_FILES,
                    )
                    break
    except Exception as e:
        logger.debug("review disk witness: skills snapshot failed: %s", e)
    try:
        for path in _memory_files():
            pair = _stat_pair(path)
            if pair is not None:
                memory[path.name] = pair
    except Exception as e:
        logger.debug("review disk witness: memory snapshot failed: %s", e)
    return {"skills": skills, "memory": memory}


def _skill_name(rel_path: str) -> str:
    """'hermes/cron-silent-stall-guard/SKILL.md' -> 'cron-silent-stall-guard'.

    Skills live at ``<root>/<category>/<name>/SKILL.md`` or ``<root>/<name>/
    SKILL.md``, and supporting files sit under ``<name>/references|scripts|...``.
    The skill is therefore the directory that directly contains SKILL.md; for a
    supporting file it is the ancestor two levels up at most. Walking from the
    right and taking the first part that is not a known subdir is close enough
    for a notification label and never raises.
    """
    parts = [p for p in Path(rel_path).parts if p]
    if not parts:
        return ""
    if parts[-1] == "SKILL.md":
        return parts[-2] if len(parts) >= 2 else ""
    known = {"references", "templates", "scripts", "assets"}
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] in known:
            return parts[idx - 1] if idx >= 1 else ""
    return parts[-2] if len(parts) >= 2 else parts[0]


def diff_review_targets(before: Snapshot, after: Snapshot) -> List[str]:
    """Action labels for what changed on disk between two snapshots.

    Labels match ``summarize_background_review_actions`` wording so the user
    sees the same phrasing no matter which detector produced the line.
    """
    actions: List[str] = []
    try:
        b_skills = (before or {}).get("skills") or {}
        a_skills = (after or {}).get("skills") or {}
        created: Dict[str, bool] = {}
        touched: Dict[str, bool] = {}
        removed: Dict[str, bool] = {}
        for rel, pair in a_skills.items():
            name = _skill_name(rel)
            if not name:
                continue
            if rel not in b_skills:
                # A brand-new SKILL.md is a new skill; a new supporting file
                # under an existing skill is an update to that skill.
                if Path(rel).name == "SKILL.md":
                    created[name] = True
                else:
                    touched[name] = True
            elif b_skills[rel] != pair:
                touched[name] = True
        for rel in b_skills:
            if rel in a_skills:
                continue
            name = _skill_name(rel)
            if name and Path(rel).name == "SKILL.md":
                removed[name] = True
            elif name:
                touched[name] = True
        for name in created:
            actions.append(f"Skill '{name}' created")
        for name in touched:
            if name in created or name in removed:
                continue
            actions.append(f"Skill '{name}' updated")
        for name in removed:
            if name in created:
                continue
            actions.append(f"Skill '{name}' archived")

        b_mem = (before or {}).get("memory") or {}
        a_mem = (after or {}).get("memory") or {}
        if any(a_mem.get(k) != b_mem.get(k) for k in set(a_mem) | set(b_mem)):
            actions.append("Memory updated")
    except Exception as e:
        logger.debug("review disk witness: diff failed: %s", e)
        return []
    return list(dict.fromkeys(actions))


def record_review_pass(
    kind: str,
    actions: List[str],
    source: str,
    elapsed_s: float,
    notified: bool,
    error: Optional[str] = None,
) -> None:
    """Append-light audit of every FINISHED review to state/background-review.json.

    The chat notification is intentionally silent on a no-change pass (the user
    asked for that: a message saying nothing happened is noise). Silence must
    still be distinguishable from "the review never ran", so each pass leaves a
    record here. ``no_change_streak`` is what a health check would alarm on --
    a review that suddenly stops writing anything for days is a broken review,
    and now that is visible without reading the chat.

    Three properties this has to hold, each learned the hard way:

    * **Crashes are recorded too.** The caller invokes this from ``finally``,
      with ``error`` set on the exception path. A review that raised used to
      leave no trace at all -- no record, no streak, a frozen ``last_run_at``
      -- which is the original "silence is indistinguishable from health" bug
      wearing a different hat.
    * **Errors are not no-change passes.** They bump ``error_streak``, never
      ``no_change_streak``. "Ran and wrote nothing" and "did not survive the
      run" are different diagnoses and want different alerts.
    * **Streaks are per kind.** ``kinds[<kind>].no_change_streak`` is tracked
      alongside the global one, because a global counter is reset by whichever
      review still works: if the skill detector goes blind while memory reviews
      keep writing, every memory pass zeroes the shared streak and the blindness
      never crosses any threshold. The global fields stay for back-compat with
      readers written before this.
    """
    try:
        home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        state_dir = Path(home) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "background-review.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        failed = bool(error)
        runs = int(data.get("runs_total") or 0) + 1
        streak = int(data.get("no_change_streak") or 0)
        err_streak = int(data.get("error_streak") or 0)
        if failed:
            err_streak += 1
        else:
            err_streak = 0
            streak = 0 if actions else streak + 1
        history = data.get("recent") or []
        if not isinstance(history, list):
            history = []
        entry = {
            "at": now,
            "kind": kind,
            "actions": actions,
            "source": source,
            "elapsed_s": round(float(elapsed_s), 1),
            "notified": bool(notified),
        }
        if failed:
            entry["error"] = str(error)[:400]
        history.append(entry)
        data.update({
            "last_run_at": now,
            "last_kind": kind,
            "last_actions": actions,
            "last_source": source,
            "last_elapsed_s": round(float(elapsed_s), 1),
            "runs_total": runs,
            "no_change_streak": streak,
            "error_streak": err_streak,
            "errors_total": int(data.get("errors_total") or 0) + (1 if failed else 0),
            "recent": history[-40:],
        })
        if actions:
            data["last_change_at"] = now
        if failed:
            data["last_error_at"] = now
            data["last_error"] = str(error)[:400]

        # Per-kind counters — see the docstring for why the global streak alone
        # cannot see a detector that went blind for only one kind of review.
        kinds = data.get("kinds")
        if not isinstance(kinds, dict):
            kinds = {}
        slot = kinds.get(kind)
        if not isinstance(slot, dict):
            slot = {}
        k_streak = int(slot.get("no_change_streak") or 0)
        k_err = int(slot.get("error_streak") or 0)
        if failed:
            k_err += 1
        else:
            k_err = 0
            k_streak = 0 if actions else k_streak + 1
        slot.update({
            "last_run_at": now,
            "last_source": source,
            "runs_total": int(slot.get("runs_total") or 0) + 1,
            "no_change_streak": k_streak,
            "error_streak": k_err,
        })
        if actions:
            slot["last_change_at"] = now
        if failed:
            slot["last_error_at"] = now
        kinds[kind] = slot
        data["kinds"] = kinds
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        logger.debug("review disk witness: state write failed: %s", e)


def merge_actions(*groups: Any) -> List[str]:
    """Union of action lists, order preserved, duplicates dropped."""
    out: List[str] = []
    for group in groups:
        for item in group or []:
            if isinstance(item, str) and item:
                out.append(item)
    return list(dict.fromkeys(out))
