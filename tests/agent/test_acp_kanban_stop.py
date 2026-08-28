"""A finished kanban worker must not be told it is still running.

Local overlay test (not upstream).

Two halves of one incident, kept in one file because neither is legible alone:

1. Upstream's stop-guard decides "this worker never closed its card" by
   walking Hermes' message list for a ``kanban_complete``/``kanban_block``
   tool call. Under native ACP tool-calling the engine's calls never appear
   there, and the fleet's documented way to close a card is
   ``hermes kanban complete <id>`` in a shell — a command, not a tool call.
   So the guard fires on healthy completions.

2. Closing a card ``rmtree``s the scratch workspace, which is the worker
   process's own cwd. The nudge's next API call therefore builds a client,
   calls ``os.getcwd()``, and dies with a path-less
   ``FileNotFoundError: [Errno 2] No such file or directory`` — which Hermes'
   retry layer reports as "Provider unreachable".

Measured 2026-08-04: twelve worker logs across ``founder``, ``secretary`` and
``default``, every task terminal on the board, every one nudged anyway, each
costing three failed API calls. The transcript of ``t_e3324d0a`` carries the
whole chain in the open: ``hermes kanban complete t_e3324d0a --summary ...``,
then the worker prefixing its next shell call with ``cd /home/hun &&`` because
its own directory had vanished.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import acp_cwd, acp_kanban_stop
from agent.kanban_stop import build_kanban_stop_nudge


class _Agent:
    """Just enough agent for the ledger read."""

    def __init__(self, calls=None, client_calls=None):
        if calls is not None:
            self._acp_native_tool_calls = calls
        if client_calls is not None:
            self.client = mock.Mock(_native_tool_calls=client_calls)


def _board(tmp: Path, rows: dict[str, str]) -> Path:
    """A minimal board file with just the column the guard reads."""
    db = tmp / "kanban.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT)")
    conn.executemany("INSERT INTO tasks VALUES (?, ?)", list(rows.items()))
    conn.commit()
    conn.close()
    return db


class BoardIsTheAuthorityTests(unittest.TestCase):
    """``board_says_terminal`` answers from the board, not from a transcript."""

    def setUp(self):
        self._env = dict(os.environ)
        self.tmp = Path(tempfile.mkdtemp(prefix="kanban-stop-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _with_board(self, rows):
        db = _board(self.tmp, rows)
        real_task = __import__(
            "hermes_cli.kanban_db", fromlist=["get_task"]
        )

        class _Row:
            def __init__(self, status):
                self.status = status

        def fake_get_task(conn, tid):
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (tid,)
            ).fetchone()
            return _Row(row[0]) if row else None

        import contextlib

        @contextlib.contextmanager
        def fake_connect(*a, **kw):
            conn = sqlite3.connect(db)
            try:
                yield conn
            finally:
                conn.close()

        return mock.patch.multiple(
            real_task, get_task=fake_get_task, connect_closing=fake_connect
        )

    def test_terminal_statuses_silence_the_guard(self):
        rows = {
            "t_done": "done",
            "t_arch": "archived",
            "t_block": "blocked",
            "t_fail": "failed",
            "t_cancel": "cancelled",
        }
        with self._with_board(rows):
            for tid in rows:
                os.environ["HERMES_KANBAN_TASK"] = tid
                with self.subTest(status=rows[tid]):
                    self.assertTrue(acp_kanban_stop.board_says_terminal())

    def test_live_statuses_leave_the_guard_armed(self):
        rows = {
            "t_run": "running",
            "t_ready": "ready",
            "t_todo": "todo",
            "t_sched": "scheduled",
            "t_review": "review",
            "t_triage": "triage",
        }
        with self._with_board(rows):
            for tid in rows:
                os.environ["HERMES_KANBAN_TASK"] = tid
                with self.subTest(status=rows[tid]):
                    self.assertFalse(acp_kanban_stop.board_says_terminal())

    def test_unknown_card_is_not_terminal(self):
        with self._with_board({"t_done": "done"}):
            os.environ["HERMES_KANBAN_TASK"] = "t_missing"
            self.assertFalse(acp_kanban_stop.board_says_terminal())

    def test_a_broken_board_fails_open(self):
        """A database error must leave upstream's nudge intact, not cancel it."""
        os.environ["HERMES_KANBAN_TASK"] = "t_done"
        with mock.patch(
            "hermes_cli.kanban_db.connect_closing", side_effect=OSError("locked")
        ):
            self.assertFalse(acp_kanban_stop.board_says_terminal())

    def test_not_a_worker_at_all(self):
        os.environ.pop("HERMES_KANBAN_TASK", None)
        self.assertFalse(acp_kanban_stop.worker_already_finished(_Agent()))


class EngineLedgerTests(unittest.TestCase):
    """The engine's own MCP call counts, under either naming mode."""

    def test_plain_and_namespaced_tool_names(self):
        for name in (
            "kanban_complete",
            "kanban_block",
            "mcp__hermes-tools__kanban_complete",
            "mcp__hermes-tools__kanban_block",
        ):
            with self.subTest(name=name):
                agent = _Agent(calls=[{"name": "Read"}, {"name": name}])
                self.assertTrue(
                    acp_kanban_stop.engine_called_terminal_tool(agent)
                )

    def test_non_terminal_board_tools_do_not_count(self):
        agent = _Agent(
            calls=[
                {"name": "kanban_show"},
                {"name": "kanban_comment"},
                {"name": "kanban_heartbeat"},
            ]
        )
        self.assertFalse(acp_kanban_stop.engine_called_terminal_tool(agent))

    def test_ledger_read_falls_back_to_the_client(self):
        agent = _Agent(client_calls=[{"name": "kanban_complete"}])
        self.assertTrue(acp_kanban_stop.engine_called_terminal_tool(agent))

    def test_empty_ledger_is_not_a_completion(self):
        self.assertFalse(acp_kanban_stop.engine_called_terminal_tool(_Agent()))


class UpstreamPredicateIsBlindTests(unittest.TestCase):
    """The regression itself: what upstream sees on the paths we actually use."""

    def setUp(self):
        self._env = dict(os.environ)
        os.environ["HERMES_KANBAN_TASK"] = "t_e3324d0a"
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_cli_completion_still_produces_an_upstream_nudge(self):
        """`hermes kanban complete` is a Bash command — no tool call to find."""
        messages = [
            {"role": "user", "content": "work kanban task t_e3324d0a"},
            {
                "role": "assistant",
                "content": "Готово, карточка закрыта.",
                "tool_calls": [],
            },
        ]
        self.assertIsNotNone(
            build_kanban_stop_nudge(messages=messages, attempts=0)
        )

    def test_native_engine_completion_still_produces_an_upstream_nudge(self):
        """Native ACP: the engine's tool calls never enter Hermes' messages."""
        messages = [
            {"role": "user", "content": "work kanban task t_e3324d0a"},
            {"role": "assistant", "content": "Done."},
        ]
        self.assertIsNotNone(
            build_kanban_stop_nudge(messages=messages, attempts=0)
        )
        # ...and the overlay is what rescues both cases.
        agent = _Agent(calls=[{"name": "mcp__hermes-tools__kanban_complete"}])
        self.assertTrue(acp_kanban_stop.worker_already_finished(agent))


class SessionCwdRepairTests(unittest.TestCase):
    """Half two: the client must survive standing in a deleted directory."""

    def setUp(self):
        self._origin = os.getcwd()
        self.addCleanup(self._restore_cwd)

    def _restore_cwd(self):
        try:
            os.chdir(self._origin)
        except OSError:
            os.chdir(tempfile.gettempdir())

    def _delete_own_cwd(self) -> str:
        d = tempfile.mkdtemp(prefix="acp-cwd-")
        os.chdir(d)
        shutil.rmtree(d)
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def test_bare_getcwd_is_the_production_signature(self):
        """Guards the diagnosis: no filename is what made it unclassifiable."""
        self._delete_own_cwd()
        with self.assertRaises(FileNotFoundError) as ctx:
            os.getcwd()
        self.assertIsNone(ctx.exception.filename)
        self.assertEqual(str(ctx.exception), "[Errno 2] No such file or directory")

    def test_deleted_cwd_is_recreated_in_place(self):
        d = self._delete_own_cwd()
        resolved = acp_cwd.session_cwd(None)
        self.assertEqual(resolved, os.path.realpath(d))
        self.assertTrue(os.path.isdir(d))

    def test_repair_keeps_the_containment_root_it_had(self):
        """Never relocate to $HOME: that widens _ensure_path_within_cwd."""
        d = self._delete_own_cwd()
        resolved = acp_cwd.session_cwd(None)
        self.assertNotEqual(resolved, os.path.realpath(os.path.expanduser("~")))
        self.assertTrue(resolved.startswith(os.path.realpath(tempfile.gettempdir())))

    def test_unrepairable_cwd_names_the_cause(self):
        self._delete_own_cwd()
        with mock.patch("agent.acp_cwd._deleted_cwd_path", return_value=None):
            with self.assertRaises(acp_cwd.SessionCwdDeleted) as ctx:
                acp_cwd.session_cwd(None)
        self.assertIn("not a provider failure", str(ctx.exception))

    def test_explicit_cwd_wins_and_never_touches_getcwd(self):
        self._delete_own_cwd()
        target = tempfile.mkdtemp(prefix="acp-explicit-")
        self.addCleanup(shutil.rmtree, target, ignore_errors=True)
        self.assertEqual(acp_cwd.session_cwd(target), os.path.realpath(target))

    def test_a_real_directory_named_deleted_is_not_stripped(self):
        base = tempfile.mkdtemp(prefix="acp-odd-")
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        odd = Path(base) / "workspace (deleted)"
        odd.mkdir()
        os.chdir(odd)
        self.assertEqual(acp_cwd._deleted_cwd_path(), str(odd))

    def test_client_construction_survives_the_real_failure(self):
        """End to end: the exact call that raised in production."""
        from agent.copilot_acp_client import CopilotACPClient

        d = self._delete_own_cwd()
        client = CopilotACPClient()
        self.assertEqual(client._acp_cwd, os.path.realpath(d))


if __name__ == "__main__":
    unittest.main()
