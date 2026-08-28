"""A follow-up sent mid-turn must reach the model IN that turn.

Local overlay test (not upstream).

Under native ACP tool-calling Hermes owns no ``tool``-role message, so
upstream's steer drain has nothing to append to and the text used to wait for
the next turn — after the wrong answer had already been sent. The transport
now pushes it as a second ``session/prompt`` on the live engine session.

The scripted bridge below reproduces the hand-off measured on a real bridge
2026-08-07 (``/tmp/steer-probe/midturn_probe.py``, sonnet-5), which is the
part that makes this non-obvious:

    4.5s   second session/prompt accepted, no error
    24.3s  FIRST id settles — stopReason end_turn, usage all zeros
    28.9s  answer text streams
    29.6s  SECOND id settles, carrying the real usage

i.e. the answer belongs to the follow-up's id. A loop that returns on its own
id gets an empty string and tears the bridge down mid-sentence, so these tests
drive the REAL ``_run_prompt``/``_request`` rather than a fake.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent import acp_midturn_steer
from agent.copilot_acp_client import CopilotACPClient
from agent.prompt_builder import STEER_MARKER_OPEN


class _FakeStdout:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def emit(self, obj: dict) -> None:
        self._q.put(json.dumps(obj) + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item

    def close(self) -> None:
        pass


class _FakeStderr:
    def __iter__(self):
        return iter(())

    def close(self) -> None:
        pass


class _HandoffBridge:
    """Engine that hands the turn over when a second prompt arrives.

    ``prompt_queueing`` controls the ``initialize`` capability. ``reject_second``
    makes the follow-up prompt fail, which must cost the steer and nothing else.
    ``die_after_second`` kills the bridge once the follow-up is on the wire —
    the case where a delivered steer would otherwise exist nowhere.
    """

    def __init__(
        self,
        *,
        prompt_queueing: bool = True,
        reject_second: bool = False,
        die_after_second: bool = False,
        solo_timeout: float = 1.5,
    ) -> None:
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr()
        self.stdin = self
        self.returncode = None
        self._prompt_queueing = prompt_queueing
        self._reject_second = reject_second
        self._die_after_second = die_after_second
        self._solo_timeout = solo_timeout
        self.prompts: list[dict] = []
        self._second = threading.Event()
        self._second_id = None
        self._lock = threading.Lock()

    # -- stdin -----------------------------------------------------------
    def write(self, data: str) -> None:
        for line in data.splitlines():
            if line.strip():
                self._dispatch(json.loads(line))

    def flush(self) -> None:
        pass

    def _dispatch(self, msg: dict) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        if method == "initialize":
            caps: dict = {}
            if self._prompt_queueing:
                caps = {"_meta": {"claudeCode": {"promptQueueing": True}}}
            self.stdout.emit(
                {"jsonrpc": "2.0", "id": req_id, "result": {"agentCapabilities": caps}}
            )
        elif method == "session/new":
            self.stdout.emit(
                {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "s-1"}}
            )
        elif method == "session/prompt":
            with self._lock:
                self.prompts.append(msg)
                first = len(self.prompts) == 1
            if first:
                threading.Thread(
                    target=self._run_first_turn, args=(req_id,), daemon=True
                ).start()
            else:
                self._second_id = req_id
                self._second.set()

    # -- engine behaviour -------------------------------------------------
    def _run_first_turn(self, first_id) -> None:
        # Work out loud so the idle deadline never fires while we wait.
        deadline = time.monotonic() + self._solo_timeout
        while time.monotonic() < deadline and not self._second.is_set():
            self._emit_update("thinking. ", kind="agent_thought_chunk")
            time.sleep(0.02)

        if not self._second.is_set():
            # Nobody steered: ordinary turn.
            self._emit_update("solo answer.")
            self.stdout.emit(
                {"jsonrpc": "2.0", "id": first_id, "result": {"stopReason": "end_turn"}}
            )
            return

        if self._die_after_second:
            self.returncode = -15
            self.stdout.eof()
            return

        # The measured hand-off: the ORIGINAL id settles first, empty.
        self.stdout.emit(
            {
                "jsonrpc": "2.0",
                "id": first_id,
                "result": {
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                },
            }
        )
        time.sleep(0.05)

        if self._reject_second:
            self.stdout.emit(
                {
                    "jsonrpc": "2.0",
                    "id": self._second_id,
                    "error": {"code": -32000, "message": "prompt rejected"},
                }
            )
            return

        # ...and the real answer streams under the FOLLOW-UP's id.
        self._emit_update("answer after the steer.")
        self.stdout.emit(
            {
                "jsonrpc": "2.0",
                "id": self._second_id,
                "result": {"stopReason": "end_turn", "usage": {"outputTokens": 42}},
            }
        )

    def _emit_update(self, text: str, kind: str = "agent_message_chunk") -> None:
        self.stdout.emit(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": kind,
                        "messageId": "m-1",
                        "content": {"type": "text", "text": text},
                    }
                },
            }
        )

    # -- process ----------------------------------------------------------
    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0
        self.stdout.eof()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode

    # -- assertions helpers ------------------------------------------------
    def prompt_texts(self) -> list[str]:
        out = []
        for msg in self.prompts:
            blocks = (msg.get("params") or {}).get("prompt") or []
            out.append("".join(b.get("text", "") for b in blocks))
        return out


class _FakeAgent:
    """Just enough AIAgent for the steer plumbing."""

    def __init__(self, pending: str | None = None) -> None:
        self._pending_steer = pending
        self._pending_steer_lock = threading.Lock()
        self.session_id = None
        self.provider = "copilot-acp"

    def _drain_pending_steer(self):
        with self._pending_steer_lock:
            text = self._pending_steer
            self._pending_steer = None
        return text

    def steer(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        with self._pending_steer_lock:
            self._pending_steer = (
                self._pending_steer + "\n" + text.strip()
                if self._pending_steer
                else text.strip()
            )
        return True


def _client(tmpdir: str, agent) -> CopilotACPClient:
    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=tmpdir,
    )
    client._activity_agent = agent
    return client


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        # The capability cache is process-global and deliberately sticky;
        # a test that saw a bridge without the capability must not poison
        # the next one.
        acp_midturn_steer._capability_seen_false = False
        self.addCleanup(
            lambda: setattr(acp_midturn_steer, "_capability_seen_false", False)
        )

    def run_turn(self, bridge, agent, *, native: bool = True, timeout: float = 3.0):
        with tempfile.TemporaryDirectory() as tmp:
            client = _client(tmp, agent)
            with patch(
                "agent.copilot_acp_client.subprocess.Popen", return_value=bridge
            ), patch(
                "agent.acp_native_tools._acp_native_tools_enabled", return_value=native
            ):
                return client._run_prompt("original question", timeout_seconds=timeout)


class MidTurnDeliveryTests(_Base):
    def test_pending_steer_is_pushed_as_a_second_prompt(self) -> None:
        agent = _FakeAgent("stop and answer in Russian")
        bridge = _HandoffBridge()
        self.run_turn(bridge, agent)

        self.assertEqual(len(bridge.prompts), 2, bridge.prompt_texts())
        followup = bridge.prompt_texts()[1]
        self.assertIn("stop and answer in Russian", followup)
        self.assertIn(STEER_MARKER_OPEN, followup)

    def test_the_answer_after_the_handoff_is_returned(self) -> None:
        """The whole point: returning on our own id yields an empty answer."""
        agent = _FakeAgent("also mention pineapple")
        bridge = _HandoffBridge()
        text, _reasoning = self.run_turn(bridge, agent)
        self.assertIn("answer after the steer.", text)

    def test_the_second_prompt_reuses_the_live_session(self) -> None:
        agent = _FakeAgent("go on")
        bridge = _HandoffBridge()
        self.run_turn(bridge, agent)
        sessions = {
            (m.get("params") or {}).get("sessionId") for m in bridge.prompts
        }
        self.assertEqual(sessions, {"s-1"})

    def test_no_pending_steer_means_exactly_one_prompt(self) -> None:
        agent = _FakeAgent(None)
        bridge = _HandoffBridge(solo_timeout=0.2)
        text, _ = self.run_turn(bridge, agent)
        self.assertEqual(len(bridge.prompts), 1)
        self.assertIn("solo answer.", text)

    def test_delivered_text_is_recorded_for_the_transcript(self) -> None:
        agent = _FakeAgent("switch to bullet points")
        bridge = _HandoffBridge()
        self.run_turn(bridge, agent)
        self.assertEqual(
            getattr(agent, "_acp_delivered_steers", None), ["switch to bullet points"]
        )

    def test_the_steer_slot_is_emptied_so_it_is_not_delivered_twice(self) -> None:
        agent = _FakeAgent("once only")
        bridge = _HandoffBridge()
        self.run_turn(bridge, agent)
        self.assertIsNone(agent._pending_steer)


class RefusalToPushTests(_Base):
    def test_engine_without_prompt_queueing_is_not_pushed_to(self) -> None:
        agent = _FakeAgent("please also do X")
        bridge = _HandoffBridge(prompt_queueing=False, solo_timeout=0.2)
        self.run_turn(bridge, agent)
        self.assertEqual(len(bridge.prompts), 1)
        # Nothing was drained, so upstream's leftover path still owns it.
        self.assertEqual(agent._pending_steer, "please also do X")

    def test_capability_absence_is_remembered_for_the_busy_ack(self) -> None:
        agent = _FakeAgent(None)
        bridge = _HandoffBridge(prompt_queueing=False, solo_timeout=0.2)
        self.assertTrue(acp_midturn_steer.midturn_delivery_available())
        self.run_turn(bridge, agent)
        self.assertFalse(acp_midturn_steer.midturn_delivery_available())

    def test_hermes_own_tool_loop_keeps_its_steer(self) -> None:
        """Draining here would STEAL the text from the pre-API drain."""
        agent = _FakeAgent("do it differently")
        bridge = _HandoffBridge(solo_timeout=0.2)
        self.run_turn(bridge, agent, native=False)
        self.assertEqual(len(bridge.prompts), 1)
        self.assertEqual(agent._pending_steer, "do it differently")

    def test_a_cancelled_turn_is_not_steered(self) -> None:
        agent = _FakeAgent("too late")
        client_holder = {}

        class _Bridge(_HandoffBridge):
            def _dispatch(self, msg):
                if msg.get("method") == "session/new":
                    # Mark the turn cancelled the moment the session exists,
                    # i.e. before the prompt loop's first steer check.
                    client_holder["client"]._hermes_cancelled = "/stop"
                super()._dispatch(msg)

        bridge = _Bridge(solo_timeout=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            client = _client(tmp, agent)
            client_holder["client"] = client
            with patch(
                "agent.copilot_acp_client.subprocess.Popen", return_value=bridge
            ), patch(
                "agent.acp_native_tools._acp_native_tools_enabled", return_value=True
            ):
                client._run_prompt("q", timeout_seconds=3.0)
        self.assertEqual(len(bridge.prompts), 1)
        self.assertEqual(agent._pending_steer, "too late")


class LossProtectionTests(_Base):
    def test_a_rejected_followup_costs_the_steer_not_the_turn(self) -> None:
        agent = _FakeAgent("late thought")
        bridge = _HandoffBridge(reject_second=True)
        # No exception: the original turn already settled, so there is an
        # answer to hand back even though the follow-up bounced.
        text, _ = self.run_turn(bridge, agent)
        self.assertNotIn("answer after the steer.", text)

    def test_a_bridge_that_dies_hands_the_steer_back(self) -> None:
        agent = _FakeAgent("do not lose me")
        bridge = _HandoffBridge(die_after_second=True)
        with self.assertRaises(Exception):
            self.run_turn(bridge, agent)
        self.assertIsNotNone(agent._pending_steer)
        self.assertIn("do not lose me", agent._pending_steer)
        self.assertEqual(getattr(agent, "_acp_delivered_steers", []), [])


class TranscriptFoldTests(unittest.TestCase):
    def test_delivered_steer_is_folded_into_the_last_user_message(self) -> None:
        agent = _FakeAgent(None)
        agent._acp_delivered_steers = ["and use tables"]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        folded = acp_midturn_steer.fold_delivered_into_messages(agent, messages)
        self.assertEqual(folded, 1)
        self.assertIn("and use tables", messages[1]["content"])
        self.assertIn(STEER_MARKER_OPEN, messages[1]["content"])
        self.assertEqual(agent._acp_delivered_steers, [])

    def test_fold_is_a_noop_without_delivered_text(self) -> None:
        agent = _FakeAgent(None)
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(
            acp_midturn_steer.fold_delivered_into_messages(agent, messages), 0
        )
        self.assertEqual(messages[0]["content"], "hello")

    def test_no_user_message_keeps_the_record_for_later(self) -> None:
        agent = _FakeAgent(None)
        agent._acp_delivered_steers = ["x"]
        messages = [{"role": "system", "content": "sys"}]
        self.assertEqual(
            acp_midturn_steer.fold_delivered_into_messages(agent, messages), 0
        )
        self.assertEqual(agent._acp_delivered_steers, ["x"])

    def test_multimodal_user_content_gets_a_text_block(self) -> None:
        agent = _FakeAgent(None)
        agent._acp_delivered_steers = ["also this"]
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        acp_midturn_steer.fold_delivered_into_messages(agent, messages)
        self.assertEqual(len(messages[0]["content"]), 2)
        self.assertIn("also this", messages[0]["content"][1]["text"])


class BusyAckTests(_Base):
    def test_steer_is_advertised_as_a_real_carrier_on_native_acp(self) -> None:
        from agent.acp_busy_followup import steer_lands_this_turn

        agent = _FakeAgent(None)
        with patch(
            "agent.acp_native_tools._acp_native_tools_enabled", return_value=True
        ):
            self.assertTrue(steer_lands_this_turn(agent))

    def test_an_engine_without_the_capability_falls_back_to_the_queue(self) -> None:
        from agent.acp_busy_followup import steer_lands_this_turn

        agent = _FakeAgent(None)
        acp_midturn_steer._capability_seen_false = True
        with patch(
            "agent.acp_native_tools._acp_native_tools_enabled", return_value=True
        ):
            self.assertFalse(steer_lands_this_turn(agent))

    def test_non_acp_providers_are_untouched(self) -> None:
        from agent.acp_busy_followup import steer_lands_this_turn

        agent = _FakeAgent(None)
        agent.provider = "custom:local-llama"
        acp_midturn_steer._capability_seen_false = True
        self.assertTrue(steer_lands_this_turn(agent))


if __name__ == "__main__":
    unittest.main()
