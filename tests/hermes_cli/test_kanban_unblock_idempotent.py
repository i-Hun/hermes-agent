"""``hermes kanban unblock`` is idempotent: "nothing parked here" is success.

A one-shot cron alarm's whole job is ``hermes kanban unblock <id>``. If the card was
closed, archived or purged before the alarm fired, the command used to exit non-zero
("not blocked/scheduled?"), the run was recorded as failed and the owner was paged on
an empty board. Every alarm therefore had to be wrapped in a status-checking shell
script — a rule that only works for alarms someone remembered to wrap.

The contract exercised here: parked (``blocked``/``scheduled``) is unblocked, and any
other state — including a missing card — still exits 0 with a line saying so.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

ROOT = Path(__file__).parents[2]


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated board; ``HERMES_KANBAN_TASK`` cleared so the CLI is not taken for a worker."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    return home


def _unblock(*task_ids: str, reason: str | None = None) -> int:
    return kc._cmd_unblock(argparse.Namespace(task_ids=list(task_ids), reason=reason))


def _status(task_id: str) -> str:
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        return task.status


def test_unblock_scheduled_card_returns_it_to_ready(board, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="wake target")
        assert kb.schedule_task(conn, tid)

    assert _unblock(tid) == 0

    assert f"Unblocked {tid}" in capsys.readouterr().out
    assert _status(tid) == "ready"


def test_unblock_blocked_card_resumes_it(board, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs input")
        assert kb.block_task(conn, tid, reason="waiting")

    assert _unblock(tid, reason="unparked by hand") == 0

    out = capsys.readouterr().out
    assert f"Unblocked {tid}" in out
    assert "unparked by hand" in out
    assert _status(tid) == "ready"
    with kbc.connect_closing() as conn:
        notes = [c.body for c in kb.list_comments(conn, tid) if "UNBLOCK" in (c.body or "")]
    assert notes == ["UNBLOCK: unparked by hand"]


def test_unblock_done_card_exits_zero_with_a_message(board, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="closed early")
        assert kb.complete_task(conn, tid, result="ok", force=True)

    assert _unblock(tid) == 0

    out = capsys.readouterr().out
    assert f"{tid} is already done — nothing to unblock" in out
    assert _status(tid) == "done"


def test_unblock_archived_card_exits_zero_with_a_message(board, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="already gone")
        assert kb.archive_task(conn, tid)

    assert _unblock(tid) == 0

    out = capsys.readouterr().out
    assert f"{tid} is already archived — nothing to unblock" in out
    assert _status(tid) == "archived"


def test_unblock_missing_card_exits_zero_with_a_message(board, capsys):
    assert _unblock("t_doesnotexist") == 0

    assert "no such task — nothing to unblock" in capsys.readouterr().out


def test_unblock_closed_card_records_no_reason_comment(board):
    """A no-op must not append an ``UNBLOCK: …`` note to a closed card's history."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="closed early")
        assert kb.complete_task(conn, tid, result="ok", force=True)
        before = kb.list_events(conn, tid)

    assert _unblock(tid, reason="alarm fired") == 0

    with kbc.connect_closing() as conn:
        after = kb.list_events(conn, tid)
        comments = [c for c in kb.list_comments(conn, tid) if "UNBLOCK" in (c.body or "")]
    assert len(after) == len(before)
    assert comments == []


def test_unblock_losing_the_race_still_exits_zero(board, capsys, monkeypatch):
    """Another caller unparked the card between our read and the write: not an error."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="raced")
        assert kb.schedule_task(conn, tid)

    monkeypatch.setattr(kb, "unblock_task", lambda conn, task_id: False)

    assert _unblock(tid, reason="alarm fired") == 0

    out = capsys.readouterr().out
    assert f"{tid} left scheduled while unblocking" in out
    assert "nothing to do" in out
    with kbc.connect_closing() as conn:
        notes = [c.body for c in kb.list_comments(conn, tid) if "UNBLOCK" in (c.body or "")]
    assert notes == [], "a lost race must not leave an UNBLOCK note behind"


def test_unblock_survives_a_failed_reason_comment(board, capsys, monkeypatch):
    """The unpark already landed: a comment failure is a warning, not an error exit."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs input")
        assert kb.block_task(conn, tid, reason="waiting")

    def boom(*args, **kwargs):
        raise ValueError("task vanished")

    monkeypatch.setattr(kb, "add_comment", boom)

    assert _unblock(tid, reason="alarm fired") == 0

    out, err = capsys.readouterr()
    assert f"Unblocked {tid}" in out
    assert "reason was not recorded" in err
    assert _status(tid) == "ready"


def test_cli_unblock_done_card_exits_zero(tmp_path):
    """The acceptance shape: the alarm's literal command line, on a card closed early."""
    home = tmp_path / "hermes"
    home.mkdir()
    db = home / "kanban.db"
    with kbc.connect_closing(db) as conn:
        tid = kb.create_task(conn, title="alarm target", assignee="default")
        assert kb.complete_task(conn, tid, result="done before the alarm", force=True)

    env = os.environ.copy()
    env.pop("HERMES_KANBAN_TASK", None)
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_DB"] = str(db)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "unblock", tid],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert f"{tid} is already done — nothing to unblock" in proc.stdout
    with kbc.connect_closing(db) as conn:
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "done"
