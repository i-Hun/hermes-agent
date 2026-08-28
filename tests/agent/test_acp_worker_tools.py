"""The ACP tool server must inherit the dispatcher's Kanban context.

A dispatcher-spawned worker is told (by the stop guard) that it must end on
``kanban_complete``/``kanban_block``. Those tools live in a *separate* process
— the ``hermes-tools`` MCP server the engine spawns from the spec built in
``agent/acp_native_tools.py`` — and that process resolves its own tool list,
gated on ``HERMES_KANBAN_TASK``. Forgetting to forward it does not raise: the
tools are silently absent, the worker exits clean, and the dispatcher records
``protocol_violation``. Retrying reproduces it forever, so it reads like a
dispatcher bug (that is how it was misdiagnosed for four runs of
``t_aca5ab55`` on 2026-08-05/06).

The last test here is the one that matters: it runs the real tool resolution
in a subprocess under the exact env we hand the engine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from agent import acp_native_tools


def _spec_env() -> dict[str, str]:
    servers = acp_native_tools._hermes_tools_mcp_servers()
    assert len(servers) == 1, servers
    pairs = servers[0]["env"]
    names = [p["name"] for p in pairs]
    assert len(names) == len(set(names)), f"duplicate env names: {names}"
    return {p["name"]: p["value"] for p in pairs}


_WORKER_ENV = {
    "HERMES_KANBAN_TASK": "t_aca5ab55",
    "HERMES_KANBAN_RUN_ID": "80",
    "HERMES_KANBAN_WORKSPACE": "/home/hun/.hermes/kanban/workspaces/t_aca5ab55",
    "HERMES_KANBAN_CLAIM_LOCK": "hun-server:914297",
}


class KanbanWorkerEnvForwardingTests(unittest.TestCase):
    def test_plain_session_forwards_no_kanban_context(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_KANBAN_")}
        with patch.dict(os.environ, clean, clear=True):
            env = _spec_env()
        self.assertEqual([k for k in env if k.startswith("HERMES_KANBAN_")], [])

    def test_worker_session_forwards_every_dispatcher_variable(self):
        with patch.dict(os.environ, _WORKER_ENV):
            env = _spec_env()
        for key, value in _WORKER_ENV.items():
            self.assertEqual(env.get(key), value, key)

    def test_blank_variable_is_not_forwarded_as_empty(self):
        with patch.dict(os.environ, {**_WORKER_ENV, "HERMES_KANBAN_BOARD": "   "}):
            env = _spec_env()
        self.assertNotIn("HERMES_KANBAN_BOARD", env)
        self.assertEqual(env.get("HERMES_KANBAN_TASK"), "t_aca5ab55")

    def test_profile_context_still_forwarded(self):
        with patch.dict(os.environ, {**_WORKER_ENV, "HERMES_PROFILE": "founder"}):
            env = _spec_env()
        self.assertEqual(env.get("HERMES_PROFILE"), "founder")
        self.assertIn("PYTHONPATH", env)
        self.assertIn("HOME", env)

    def test_delegated_child_gets_no_kanban_context(self):
        """delegate_task children run inside a worker but are not the worker."""
        from agent.delegation_context import delegated_child_context

        with patch.dict(os.environ, _WORKER_ENV):
            with delegated_child_context():
                env = _spec_env()
        self.assertEqual([k for k in env if k.startswith("HERMES_KANBAN_")], [])

    def test_delegated_child_marker_in_env_also_fails_closed(self):
        """Lineage that crossed a process boundary carries a marker, not a var."""
        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

        with patch.dict(os.environ, {**_WORKER_ENV, DELEGATED_CHILD_ENV_MARKER: "1"}):
            env = _spec_env()
        self.assertEqual([k for k in env if k.startswith("HERMES_KANBAN_")], [])


class ToolServerActuallyRegistersKanbanTests(unittest.TestCase):
    """End-to-end over the seam: resolve tools in the env we hand the engine.

    Unit tests above only prove the spec carries a variable. This proves the
    variable buys the tools — the step the original bug hid behind.
    """

    _PROBE = (
        "import json;"
        "from model_tools import get_tool_definitions;"
        "print(json.dumps(sorted("
        "t['function']['name'] for t in (get_tool_definitions(quiet_mode=True) or [])"
        " if t.get('type') == 'function'"
        ")))"
    )

    def _tools(self, env: dict[str, str]) -> list[str]:
        proc = subprocess.run(
            [sys.executable, "-c", self._PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_worker_env_yields_terminal_handoff_tools(self):
        with patch.dict(os.environ, _WORKER_ENV):
            env = _spec_env()
        names = self._tools(env)
        self.assertIn("kanban_complete", names)
        self.assertIn("kanban_block", names)

    def test_plain_session_env_yields_none_of_them(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_KANBAN_")}
        with patch.dict(os.environ, clean, clear=True):
            env = _spec_env()
        names = self._tools(env)
        self.assertEqual([n for n in names if n.startswith("kanban_")], [])


if __name__ == "__main__":
    unittest.main()
