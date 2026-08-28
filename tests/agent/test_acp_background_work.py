"""Background work the ACP bridge cannot keep alive must be reported, loudly.

Local overlay test (not upstream).

The bridge is one process per API call: ``_run_prompt`` spawns it, ``close()``
terminates it when the call returns. An ``Agent(run_in_background=true)`` lives
inside that process, so it is killed the moment the turn is answered — while
the tool call itself already returned a task id and looked successful. From
inside the turn "killed with the bridge" and "finished with no output" are the
same observation, so the agent reports success.

Measured on the ``founder`` profile 2026-08-01: eight subagents with
``run_in_background: false`` ran 47-264 lines each and finished fine; the two
with ``run_in_background: true`` were cut off after ~5 seconds, and the agent
reported both as started. It did this twice in one evening — the second time
minutes after diagnosing the first — which is why the notice goes into the
ANSWER (and therefore the transcript) rather than onto the progress rail.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path

from agent import acp_background_work as bgw
from agent.acp_session_update import handle_session_update

# The founder call that started this: the flag sits AFTER a multi-kilobyte
# prompt, i.e. outside the 2000-char slice the native-tool ledger keeps.
_LONG_PROMPT = "Реализуй веху «Импорт файлом» целиком. " * 200


def _agent_call(
    tool_id: str,
    *,
    background: bool | None,
    tool: str = "Agent",
    description: str = "Implement task 005",
    kind: str = "tool_call",
    prompt: str = "do the thing",
    meta: bool = True,
) -> dict:
    """``background=None`` omits the flag — the real founder call of 05:39."""
    raw: dict = {
        "description": description,
        "subagent_type": "claude",
        "prompt": prompt,
    }
    if background is not None:
        raw["run_in_background"] = background
    update: dict = {
        "sessionUpdate": kind,
        "toolCallId": tool_id,
        "title": description,
        "rawInput": raw,
    }
    if meta:
        update["_meta"] = {"claudeCode": {"toolName": tool}}
    return update


def _fetch_call(tool_id: str, *, meta: bool = True) -> dict:
    """A real ``WebFetch`` call — ``{url, prompt}``, and no flag anywhere.

    Recorded 2026-08-02 06:57 from the live transcript. It is synchronous:
    the page comes back inside the same turn and the answer is written from
    it. The shape is what makes it dangerous — a ``prompt`` with no
    ``command`` looked exactly like a subagent spawn to the first predicate.
    """
    update: dict = {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_id,
        "title": "https://nuxt.com/docs/4.x/guide/ai/mcp",
        "rawInput": {
            "url": "https://nuxt.com/docs/4.x/guide/ai/mcp",
            "prompt": "What is the Nuxt MCP feature? List the tools it exposes.",
        },
    }
    if meta:
        update["_meta"] = {"claudeCode": {"toolName": "WebFetch"}}
    return update


def _bash_call(tool_id: str, *, background: bool, command: str = "cargo build") -> dict:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_id,
        "_meta": {"claudeCode": {"toolName": "Bash"}},
        "title": command,
        "rawInput": {
            "command": command,
            "description": "Build api crate",
            "run_in_background": background,
        },
    }


class _FakeClient:
    """Just enough client for the overlay's tool-event bookkeeping."""

    def __init__(self) -> None:
        self.pulses: list[tuple] = []
        self.ledger: list[tuple[str, str]] = []
        self._activity_agent = None
        self._hermes_cancelled = None

    def _record_native_tool_call(self, update: dict, kind: str) -> None:
        self.ledger.append((kind, str(update.get("toolCallId") or "")))

    def _fire_engine_activity(self, event_type, tool_name=None, preview=None, args=None):
        self.pulses.append((event_type, tool_name, preview, args))


class _BackgroundWorkTestCase(unittest.TestCase):
    """Redirects the jsonl so a test run never writes the real log."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = Path(self._tmp.name) / "acp-background-work.jsonl"
        self._real_log = bgw._LOG_PATH
        bgw._LOG_PATH = self.log
        self.addCleanup(lambda: setattr(bgw, "_LOG_PATH", self._real_log))
        # And the cap hook's deny log: without this a test run reads the real
        # one out of the operator's home, so its verdict depends on what the
        # fleet happened to do this afternoon.
        self.deny_log = Path(self._tmp.name) / "subagent-cap-denials.log"
        self._real_deny = bgw._DENY_LOG
        bgw._DENY_LOG = self.deny_log
        self.addCleanup(lambda: setattr(bgw, "_DENY_LOG", self._real_deny))

    def write_denial(self, tool_id: str) -> None:
        with self.deny_log.open("a", encoding="utf-8") as fh:
            fh.write(
                "2026-08-15 20:49:38\tbg\tsess=72f2db53\tid=%s\ttool=Bash\t"
                "live=-\tcwd=/home/hun/.hermes\tdesc=sleep 300\n" % tool_id
            )

    def rows(self) -> list[dict]:
        if not self.log.is_file():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]

    def feed(self, client, *updates) -> None:
        text: list[str] = []
        rail: list[str] = []
        for update in updates:
            handle_session_update(client, update, text, rail)


class TestDetection(_BackgroundWorkTestCase):
    def test_background_subagent_is_recorded_and_named(self) -> None:
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=True))
        calls = client._acp_background_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "Agent")
        self.assertEqual(calls[0]["kind"], "subagent")
        self.assertEqual(calls[0]["label"], "Implement task 005")

    def test_foreground_subagent_is_not_touched(self) -> None:
        """The eight foreground fan-outs of 2026-08-01 must stay silent."""
        client = _FakeClient()
        self.feed(client, *[
            _agent_call(f"t{i}", background=False) for i in range(8)
        ])
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])
        self.assertEqual(bgw.warning_text(client), "")
        self.assertEqual(self.rows(), [])

    def test_flag_is_read_past_the_ledgers_truncation_point(self) -> None:
        """The ledger caps arguments at 2000 chars; the flag often sits after."""
        client = _FakeClient()
        update = _agent_call("t1", background=True, prompt=_LONG_PROMPT)
        # Reorder so run_in_background lands beyond the ledger's slice.
        raw = update["rawInput"]
        update["rawInput"] = {
            "description": raw["description"],
            "prompt": raw["prompt"],
            "run_in_background": True,
        }
        blob = json.dumps(update["rawInput"], ensure_ascii=False)
        self.assertGreater(
            blob.index("run_in_background"),
            2000,
            "the fixture no longer reproduces the truncation it exists for",
        )
        self.feed(client, update)
        self.assertEqual(len(client._acp_background_calls), 1)

    def test_background_shell_is_a_different_kind(self) -> None:
        client = _FakeClient()
        self.feed(client, _bash_call("t1", background=True))
        self.assertEqual(client._acp_background_calls[0]["kind"], "shell")
        note = bgw.warning_text(client)
        self.assertIn("BashOutput", note)
        # The note must not claim the engine's own background shell survives:
        # two live probes on 2026-08-02 showed the loop writing one iteration
        # and dying with the turn.
        self.assertNotIn("BashOutput handle survives", note)
        # It must name the escape hatch that actually works. Until 2026-08-13
        # that was `setsid nohup`, and it was wrong: a detached child keeps the
        # gateway's cgroup, so ExecStopPost=gateway.cgroup_cleanup SIGKILLs it
        # at the next restart. The runtime that survives is job_start.
        self.assertIn("job_start", note)
        self.assertIn("setsid", note)  # named only to warn against it
        setsid_clause = note[note.index("setsid"):]
        self.assertTrue(
            "cgroup" in setsid_clause,
            "the setsid mention must carry the reason it fails, not read as advice",
        )

    def test_kind_falls_back_to_argument_shape_without_meta(self) -> None:
        """A bridge that stops stamping _meta must not blind the detector."""
        client = _FakeClient()
        update = _agent_call("t1", background=True, meta=False)
        update["title"] = "Implement task 005"  # engine's display title
        self.feed(client, update)
        self.assertEqual(client._acp_background_calls[0]["kind"], "subagent")

    def test_a_subagent_with_no_flag_at_all_is_still_background(self) -> None:
        """The 2026-08-02 05:39 loss: no flag, and async is the default.

        ``founder`` called ``Agent`` without ``run_in_background`` — the engine
        answered "Async agent launched successfully", the reply went out five
        seconds later, and strata task 006 was never touched. The first version
        of this detector required a truthy flag, so it stayed silent through
        exactly the shape of call that caused the incident it was built for.
        """
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=None))
        calls = client._acp_background_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kind"], "subagent")
        self.assertEqual(calls[0]["asked"], "default")
        self.assertIn("by default", bgw.warning_text(client).lower())

    def test_a_synchronous_fetch_is_not_called_background_work(self) -> None:
        """The 2026-08-02 06:57 false alarm: ``WebFetch`` is not a spawn.

        ``WebFetch(url, prompt)`` returns its page inside the turn — the very
        answer that carried the warning was written from the fetched page. The
        predicate defaulted every ``prompt``-shaped call to background, so it
        told the agent to assume work that plainly happened had been lost.
        A detector that cries wolf on the synchronous case teaches the reader
        to skip the line that matters.
        """
        client = _FakeClient()
        self.feed(client, _fetch_call("t1"))
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])
        self.assertEqual(bgw.warning_text(client), "")
        self.assertEqual(self.rows(), [])

    def test_a_fetch_without_meta_is_still_not_a_spawn(self) -> None:
        """Degraded mode must not turn a URL into a subagent either."""
        client = _FakeClient()
        self.feed(client, _fetch_call("t1", meta=False))
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])

    def test_a_spawn_without_meta_is_still_caught(self) -> None:
        """The shape path stays useful: description + subagent_type + prompt."""
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=None, meta=False))
        calls = client._acp_background_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kind"], "subagent")

    def test_a_url_beats_a_spawn_looking_field(self) -> None:
        """A URL means the tool hands content back inside the turn.

        Today's ``WebFetch`` carries no ``description``, so the spawn
        signature alone would already reject it — this pins the guard that
        keeps a fetch-shaped call with one extra field from sliding back in.
        """
        client = _FakeClient()
        update = _fetch_call("t1", meta=False)
        update["rawInput"]["description"] = "Read the Nuxt MCP page"
        self.feed(client, update)
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])

    def test_a_bare_prompt_is_not_a_spawn_signature(self) -> None:
        """"Has a prompt" is what over-fired; it must stay insufficient."""
        client = _FakeClient()
        update = {
            "sessionUpdate": "tool_call",
            "toolCallId": "t1",
            "title": "Ask the oracle",
            "rawInput": {"prompt": "summarise this file"},
        }
        self.feed(client, update)
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])

    def test_a_task_list_tool_is_not_the_spawn_tool(self) -> None:
        """``Task`` is the spawn tool; ``TaskCreate``/``TaskList`` are not.

        Prefix matching on the name made every sibling tool async-by-default,
        which is the same false alarm as the fetch, arriving by another door.
        """
        client = _FakeClient()
        update = {
            "sessionUpdate": "tool_call",
            "toolCallId": "t1",
            "_meta": {"claudeCode": {"toolName": "TaskCreate"}},
            "title": "Add tasks",
            "rawInput": {"tasks": [{"title": "ship it"}]},
        }
        self.feed(client, update)
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])

    def test_an_unknown_tool_that_says_background_is_still_reported(self) -> None:
        """Tightening the default must not silence an explicit claim."""
        client = _FakeClient()
        update = _fetch_call("t1")
        update["rawInput"]["run_in_background"] = True
        self.feed(client, update)
        self.assertEqual(len(client._acp_background_calls), 1)

    def test_a_shell_with_no_flag_is_foreground(self) -> None:
        """Bash defaults the other way; the default must be per-tool."""
        client = _FakeClient()
        update = _bash_call("t1", background=False)
        update["rawInput"].pop("run_in_background")
        self.feed(client, update)
        self.assertEqual(getattr(client, "_acp_background_calls", []), [])
        self.assertEqual(bgw.warning_text(client), "")

    def test_the_engines_own_words_outrank_the_arguments(self) -> None:
        """Evidence beats inference: the result says it launched async."""
        client = _FakeClient()
        update = _agent_call("t1", background=False, kind="tool_call_update")
        update["content"] = [
            {
                "type": "content",
                "content": {
                    "type": "text",
                    "text": "Async agent launched successfully.\nagentId: a3ea8a8",
                },
            }
        ]
        self.feed(client, update)
        self.assertEqual(len(client._acp_background_calls), 1)

    def test_same_call_seen_twice_is_reported_once(self) -> None:
        """The engine sends tool_call and tool_call_update for one call."""
        client = _FakeClient()
        self.feed(
            client,
            _agent_call("t1", background=True),
            _agent_call("t1", background=True, kind="tool_call_update"),
        )
        self.assertEqual(len(client._acp_background_calls), 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertIn("×1", bgw.warning_text(client))


class TestWarning(_BackgroundWorkTestCase):
    def test_notice_names_the_loss_and_the_alternatives(self) -> None:
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=True))
        note = bgw.warning_text(client)
        self.assertIn("did not survive", note)
        self.assertIn("`Agent` (background)", note)
        self.assertIn("Implement task 005", note)
        self.assertIn("Kanban", note)

    def test_notice_is_appended_to_the_answer_not_replacing_it(self) -> None:
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=True))
        out = bgw.append_warning(client, "Запустил задачу 005 в фоне.")
        self.assertTrue(out.startswith("Запустил задачу 005 в фоне."))
        self.assertIn("did not survive", out)

    def test_clean_turn_is_left_byte_for_byte_alone(self) -> None:
        client = _FakeClient()
        self.feed(client, _bash_call("t1", background=False))
        self.assertEqual(bgw.append_warning(client, "Готово."), "Готово.")

    def test_both_kinds_in_one_turn_get_separate_lines(self) -> None:
        client = _FakeClient()
        self.feed(
            client,
            _agent_call("t1", background=True),
            _bash_call("t2", background=True),
        )
        note = bgw.warning_text(client)
        self.assertEqual(note.count("\n•"), 2)
        self.assertIn("Kanban", note)
        self.assertIn("BashOutput", note)
        self.assertIn("setsid", note)


class TestRefusedCallsAreNotLosses(_BackgroundWorkTestCase):
    """A call the cap hook refused never ran — warning about it is a lie.

    Since 15.08.2026 the PreToolUse hook denies `run_in_background` under the
    bridge, so without this the operator would read "the work did not happen"
    on every turn where the guard did its job. A warning that cries wolf is
    the warning people learn to skip, and this one must be read.
    """

    def test_denied_call_produces_no_damage_report(self) -> None:
        client = _FakeClient()
        self.feed(client, _bash_call("toolu_denied", background=True))
        self.write_denial("toolu_denied")
        self.assertEqual(bgw.warning_text(client), "")

    def test_a_call_that_actually_ran_still_warns(self) -> None:
        """The inverse: suppression must not become a mute button."""
        client = _FakeClient()
        self.feed(client, _bash_call("toolu_ran", background=True))
        self.write_denial("toolu_somethingelse")
        self.assertIn("did not survive", bgw.warning_text(client))

    def test_mixed_turn_keeps_only_the_real_loss(self) -> None:
        client = _FakeClient()
        self.feed(
            client,
            _bash_call("toolu_denied", background=True),
            _agent_call("toolu_ran", background=True),
        )
        self.write_denial("toolu_denied")
        note = bgw.warning_text(client)
        self.assertEqual(note.count("\n•"), 1)
        self.assertIn("`Agent` (background)", note)

    def test_absent_deny_log_changes_nothing(self) -> None:
        client = _FakeClient()
        self.feed(client, _bash_call("t1", background=True))
        self.assertFalse(self.deny_log.exists())
        self.assertIn("did not survive", bgw.warning_text(client))


class TestLog(_BackgroundWorkTestCase):
    def test_each_detection_writes_one_row(self) -> None:
        client = _FakeClient()
        self.feed(
            client,
            _agent_call("t1", background=True),
            _bash_call("t2", background=True),
        )
        rows = self.rows()
        self.assertEqual([r["kind"] for r in rows], ["subagent", "shell"])
        self.assertTrue(all(r["ts"] > 0 for r in rows))
        self.assertTrue(all(r.get("profile") for r in rows))

    def test_the_prompt_body_never_reaches_the_log(self) -> None:
        """Rows are shipped to bridge-doctor; prompts are not observability."""
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=True, prompt=_LONG_PROMPT))
        blob = self.log.read_text()
        self.assertNotIn("Реализуй веху", blob)

    def test_a_broken_log_path_does_not_break_the_turn(self) -> None:
        bgw._LOG_PATH = Path("/proc/definitely/not/writable/x.jsonl")
        client = _FakeClient()
        self.feed(client, _agent_call("t1", background=True))
        self.assertEqual(len(client._acp_background_calls), 1)


class _FailoverClient(_FakeClient):
    """Drives the REAL run_prompt_with_failover over scripted attempts."""

    def __init__(self, scripts) -> None:
        super().__init__()
        self._scripts = list(scripts)

    def _run_prompt(self, prompt_text, *, timeout_seconds, model_override=None):
        script = self._scripts.pop(0)
        text: list[str] = []
        rail: list[str] = []
        for update in script.get("updates", []):
            handle_session_update(self, update, text, rail)
        exc = script.get("raises")
        if exc is not None:
            raise exc
        return script.get("text", "".join(text)), "".join(rail)


class TestThroughTheRealTurn(_BackgroundWorkTestCase):
    """End to end: the notice must reach the string the caller returns."""

    def setUp(self) -> None:
        super().setUp()
        import agent.acp_failover as failover

        self.failover = failover
        self._real_record = failover.record_usage
        failover.record_usage = lambda **kw: None
        self._env = {
            k: os.environ.get(k)
            for k in ("HERMES_ACP_FAILOVER", "HERMES_ACP_FALLBACK_MODEL",
                      "HERMES_ACP_MODEL")
        }
        os.environ["HERMES_ACP_FAILOVER"] = "1"
        os.environ["HERMES_ACP_FALLBACK_MODEL"] = "claude-opus-4-8"
        os.environ["HERMES_ACP_MODEL"] = "claude-opus-5"
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        self.failover.record_usage = self._real_record
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_answer_carries_the_notice(self) -> None:
        client = _FailoverClient([
            {"updates": [_agent_call("t1", background=True)],
             "text": "Запустил 005 в фоне, дам знать."},
        ])
        answer, _reasoning, _used, fired = self.failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-opus-5",
        )
        self.assertFalse(fired)
        self.assertIn("Запустил 005 в фоне", answer)
        self.assertIn("did not survive", answer)

    def test_the_next_turn_does_not_repeat_a_stale_notice(self) -> None:
        """Shared clients outlive a turn; the record must not."""
        client = _FailoverClient([
            {"updates": [_agent_call("t1", background=True)], "text": "Запустил."},
            {"updates": [], "text": "Готово."},
        ])
        first, *_ = self.failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-opus-5")
        second, *_ = self.failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-opus-5")
        self.assertIn("did not survive", first)
        self.assertEqual(second, "Готово.")

    def test_work_spawned_by_a_dead_attempt_is_still_reported(self) -> None:
        """The primary parked a subagent, then failed over — still lost."""
        client = _FailoverClient([
            {"updates": [_agent_call("t1", background=True)],
             "raises": TimeoutError("primary wedged")},
            {"updates": [], "text": "Готово."},
        ])
        answer, _reasoning, used, fired = self.failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-opus-5",
        )
        self.assertTrue(fired)
        self.assertEqual(used, "claude-opus-4-8")
        self.assertIn("Готово.", answer)
        self.assertIn("did not survive", answer)


class TestDeliveredCallsAreNotLosses(_BackgroundWorkTestCase):
    """A subagent that came back before the reply lost nothing.

    Async-by-default makes the RECORD unavoidable — every flagless `Agent` is
    logged at spawn, because at spawn there is nothing else to go on. But the
    engine writes the answer down: a subagent that returned left a
    `tool_result` carrying its `tool_use_id` in the transcript. Measured
    16.08.2026 on `personal`: 15 flagless spawns, 15 results on disk, and the
    doctor called all 15 "the work did not happen". Nothing was lost except
    the detector's credibility — which is the expensive part, since this is
    the one warning that must be read.
    """

    def setUp(self) -> None:
        super().setUp()
        self.projects = Path(self._tmp.name) / "claude" / "projects"
        self.projects.mkdir(parents=True)
        self._real_cfg = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = str(self.projects.parent)
        self.addCleanup(self._restore_cfg)
        self.cwd = "/home/hun/.hermes"

    def _restore_cfg(self) -> None:
        if self._real_cfg is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = self._real_cfg

    def _session_dir(self) -> Path:
        d = self.projects / re.sub(r"[^A-Za-z0-9]", "-", self.cwd)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_transcript(self, sid: str, *delivered: str,
                         in_subagent_file: bool = False) -> None:
        """A transcript whose `tool_result` blocks name `delivered`."""
        rows = [{"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": i, "content": "done"}
        ]}} for i in delivered]
        base = self._session_dir() / f"{sid}.jsonl"
        if in_subagent_file:
            base.write_text('{"type": "summary"}\n', encoding="utf-8")
            sub = self._session_dir() / sid / "subagents"
            sub.mkdir(parents=True, exist_ok=True)
            target = sub / "agent-a1b2.jsonl"
        else:
            target = base
        target.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def client_on(self, sid: str) -> _FakeClient:
        client = _FakeClient()
        client._acp_engine_sessions = [
            {"id": sid, "cwd": self.cwd, "ts": time.time()}]
        return client

    def test_a_delivered_subagent_produces_no_damage_report(self) -> None:
        client = self.client_on("sess-1")
        self.feed(client, _agent_call("toolu_done", background=None))
        self.write_transcript("sess-1", "toolu_done")
        self.assertEqual(bgw.warning_text(client), "")

    def test_a_spawn_with_no_result_on_disk_still_warns(self) -> None:
        """The inverse: evidence must not become a mute button."""
        client = self.client_on("sess-1")
        self.feed(client, _agent_call("toolu_running", background=None))
        self.write_transcript("sess-1", "toolu_somethingelse")
        self.assertIn("did not survive", bgw.warning_text(client))

    def test_a_result_in_the_fan_out_file_counts(self) -> None:
        """Subagent transcripts sit BESIDE the session file, not inside it."""
        client = self.client_on("sess-1")
        self.feed(client, _agent_call("toolu_nested", background=None))
        self.write_transcript("sess-1", "toolu_nested", in_subagent_file=True)
        self.assertEqual(bgw.warning_text(client), "")

    def test_a_missing_transcript_leaves_the_warning_alone(self) -> None:
        """No evidence is not evidence of delivery — fail loud, not silent."""
        client = self.client_on("sess-gone")
        self.feed(client, _agent_call("toolu_x", background=True))
        self.assertIn("did not survive", bgw.warning_text(client))

    def test_mixed_turn_keeps_only_the_call_with_no_result(self) -> None:
        client = self.client_on("sess-1")
        self.feed(
            client,
            _agent_call("toolu_back", background=None),
            _agent_call("toolu_stuck", background=None),
        )
        self.write_transcript("sess-1", "toolu_back")
        note = bgw.warning_text(client)
        self.assertEqual(note.count("\n•"), 1)
        self.assertIn("×1", note)

    def test_every_call_gets_exactly_one_verdict_row(self) -> None:
        """The doctor reads the verdict; a second read must not double it."""
        client = self.client_on("sess-1")
        self.feed(
            client,
            _agent_call("toolu_back", background=None),
            _agent_call("toolu_stuck", background=None),
        )
        self.write_transcript("sess-1", "toolu_back")
        bgw.warning_text(client)
        bgw.warning_text(client)
        verdicts = {r["id"]: r["outcome"] for r in self.rows() if r.get("outcome")}
        self.assertEqual(verdicts,
                         {"toolu_back": "delivered", "toolu_stuck": "lost"})
        self.assertEqual(len([r for r in self.rows() if r.get("outcome")]), 2)

    def test_a_refused_call_is_journalled_as_denied_not_lost(self) -> None:
        client = self.client_on("sess-1")
        self.feed(client, _agent_call("toolu_denied", background=True))
        self.write_denial("toolu_denied")
        self.assertEqual(bgw.warning_text(client), "")
        verdicts = {r["id"]: r["outcome"] for r in self.rows() if r.get("outcome")}
        self.assertEqual(verdicts, {"toolu_denied": "denied"})


if __name__ == "__main__":
    unittest.main()
