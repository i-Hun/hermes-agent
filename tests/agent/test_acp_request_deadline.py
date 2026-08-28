"""The ACP transport deadline measures SILENCE, not total elapsed time.

Local overlay test (not upstream).

`session/prompt` is not one API call: the engine drives its own tool loop
inside it, so a healthy turn holds the request open for 30+ minutes while
streaming `session/update` the whole time. The request loop in
``copilot_acp_client._request`` used to arm one wall-clock deadline before the
loop and never touch it again, so such a turn was cut at exactly
``timeout_seconds`` (1800s by default) and reported as ``TimeoutError`` — which
``run_prompt_with_failover`` correctly reads as a provider failure and answers
on the backup model.

Measured on the ``founder`` profile 2026-08-01: bridge spawned 18:06:33, tool
calls in ``logs/acp-permission-audit.log`` at 18:26, 18:31 and 18:36, timeout
declared 18:36:33 — 1800.000s after the spawn. Both failovers that day had this
shape and both were answered by ``claude-opus-4-8``.

These tests drive the REAL ``_run_prompt``/``_request`` loop against a scripted
in-process bridge, because the defect lives in the loop's bookkeeping and a
faked ``_run_prompt`` (the harness the other ACP suites use) cannot see it.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import unittest
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient


class _FakeStdout:
    """Blocking line iterator, fed by the scripted bridge thread."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def emit(self, obj: dict) -> None:
        self._q.put(json.dumps(obj) + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self) -> "_FakeStdout":
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


class _ScriptedBridge:
    """A fake `claude-agent-acp` that answers on its own schedule.

    ``updates``/``gap`` describe the engine working out loud: N session updates
    spaced ``gap`` seconds apart. ``final_delay`` is the silence between the
    last update and the prompt result. Total prompt time is therefore
    ``updates * gap + final_delay`` — deliberately longer than the timeout the
    tests pass, while every individual gap stays shorter.
    """

    def __init__(self, *, updates: int, gap: float, final_delay: float) -> None:
        self._updates = updates
        self._gap = gap
        self._final_delay = final_delay
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr()
        self.stdin = self
        self.returncode = None
        self._threads: list[threading.Thread] = []

    # -- stdin side -------------------------------------------------------
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
            self.stdout.emit({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/new":
            self.stdout.emit(
                {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "s-1"}}
            )
        elif method == "session/prompt":
            t = threading.Thread(target=self._answer_prompt, args=(req_id,), daemon=True)
            t.start()
            self._threads.append(t)

    def _answer_prompt(self, req_id) -> None:
        for i in range(self._updates):
            time.sleep(self._gap)
            self.stdout.emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "m-1",
                            "content": {"type": "text", "text": f"chunk {i}. "},
                        }
                    },
                }
            )
        time.sleep(self._final_delay)
        self.stdout.emit(
            {"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}}
        )

    # -- process side -----------------------------------------------------
    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.eof()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


def _client(tmpdir: str) -> CopilotACPClient:
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=tmpdir,
    )


class TransportDeadlineTests(unittest.TestCase):
    def _run(self, bridge: _ScriptedBridge, timeout: float):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = _client(tmp)
            with patch(
                "agent.copilot_acp_client.subprocess.Popen", return_value=bridge
            ):
                return client._run_prompt("hi", timeout_seconds=timeout)

    def test_a_turn_longer_than_the_timeout_survives_while_the_engine_talks(self) -> None:
        """The founder 18:06→18:36 case: 6x the timeout, never silent."""
        bridge = _ScriptedBridge(updates=12, gap=0.05, final_delay=0.05)
        text, _reasoning = self._run(bridge, timeout=0.1)
        # 12 * 0.05 + 0.05 = 0.65s of work under a 0.1s timeout.
        self.assertIn("chunk 11.", text)
        self.assertIn("chunk 0.", text)

    def test_silence_longer_than_the_timeout_still_times_out(self) -> None:
        """The watchdog must keep catching a bridge that actually wedged."""
        bridge = _ScriptedBridge(updates=2, gap=0.05, final_delay=5.0)
        with self.assertRaises(TimeoutError) as caught:
            self._run(bridge, timeout=0.3)
        self.assertIn("session/prompt", str(caught.exception))

    def test_silence_from_the_very_first_byte_still_times_out(self) -> None:
        """No activity at all is the case the deadline was written for."""
        bridge = _ScriptedBridge(updates=0, gap=0.0, final_delay=5.0)
        with self.assertRaises(TimeoutError):
            self._run(bridge, timeout=0.3)

    def test_partial_text_is_not_returned_when_the_bridge_wedges(self) -> None:
        """A timeout must stay a timeout — not a truncated answer."""
        bridge = _ScriptedBridge(updates=2, gap=0.05, final_delay=5.0)
        with self.assertRaises(TimeoutError):
            self._run(bridge, timeout=0.3)


if __name__ == "__main__":
    unittest.main()
