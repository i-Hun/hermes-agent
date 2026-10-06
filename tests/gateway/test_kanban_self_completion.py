"""Completion provenance gates only the actor's own wake, not worker results."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session_context import set_session_vars, reset_session_vars
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tests.gateway.test_kanban_notifier import RecordingAdapter, _make_runner, _run_one_notifier_tick


@pytest.mark.parametrize("actor,target,kind,wakes", [
    ("origin", "origin", "completed", 0),
    ("worker", "origin", "completed", 1),
    ("", "origin", "completed", 1),
    ("origin", "reset-session", "completed", 1),
    ("origin", "origin", "blocked", 1),
    ("origin", "origin", "review_requested", 1),
])
def test_completion_cli_to_notifier(tmp_path, monkeypatch, actor, target, kind, wakes):
    from hermes_cli.kanban import _cmd_complete
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    conn = kbc.connect()
    tid = kb.create_task(conn, title="self echo", assignee="default", session_id="origin")
    kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                       notifier_profile="default", delivery_mode="notify+wake", chat_type="dm")
    key = "agent:main:telegram:dm:chat-1"
    set_session_vars(platform="telegram", chat_id="chat-1", chat_type="dm",
                     session_id=actor, session_key=key, profile="default")
    try:
        if kind == "completed":
            args = SimpleNamespace(task_ids=[tid], result="closed from CLI", summary=None, metadata=None,
                                   created_cards=None, force=False, board=None)
            assert _cmd_complete(args) == 0
        elif kind == "blocked":
            kb.block_task(conn, tid, reason="attention")
        else:
            kb.request_review(conn, tid, summary="review me")
    finally:
        reset_session_vars()
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._kanban_notifier_profile = "default"
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda key: SimpleNamespace(session_id=target))
    wake = AsyncMock()
    monkeypatch.setattr("gateway.wake.deliver_wake", wake)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert wake.await_count == wakes
    assert len(adapter.sent) == 1  # passive delivery is unchanged
    assert any(e.kind == kind for e in kb.list_events(conn, tid))  # audit retained
    _, unseen = kbn.unseen_events_for_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1", kinds=[kind])
    assert unseen == []
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert wake.await_count == wakes
    assert len(adapter.sent) == 1
    conn.close()


@pytest.mark.parametrize("actor,profile,wakes", [("origin", "default", 0), ("worker", "default", 1), ("origin", "other", 1), ("", "default", 1)])
def test_subprocess_cli_api_delivery(tmp_path, monkeypatch, actor, profile, wakes):
    import os
    import subprocess
    import sys
    from tests.gateway.test_kanban_notifier_apiserver_wake import ApiServerLikeAdapter
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "api.db"))
    conn = kbc.connect()
    tid = kb.create_task(conn, title="api completion", assignee="default", session_id="unrelated-creator")
    kbn.add_notify_sub(conn, task_id=tid, platform="api_server", chat_id="origin",
                       notifier_profile="default", delivery_mode="wake")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_SESSION_") and not k.startswith("HERMES_KANBAN_TASK")}
    env.update(HERMES_SESSION_ID=actor, HERMES_SESSION_PROFILE=profile, HERMES_SESSION_PLATFORM="api_server")
    code = "from argparse import Namespace; from hermes_cli.kanban import _cmd_complete; raise SystemExit(_cmd_complete(Namespace(task_ids=[%r],result='cli child finished')))" % tid
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    event = [e for e in kb.list_events(conn, tid) if e.kind == "completed"][-1]
    if actor:
        assert event.payload["completion_origin"]["session_id"] == actor
    adapter = ApiServerLikeAdapter()
    runner = _make_runner(adapter)
    runner.adapters = {Platform.API_SERVER: adapter}
    runner._kanban_notifier_profile = "default"
    wake = AsyncMock()
    monkeypatch.setattr("gateway.wake.deliver_wake", wake)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert wake.await_count == wakes
    _, unseen = kbn.unseen_events_for_sub(conn, task_id=tid, platform="api_server", chat_id="origin", kinds=["completed"])
    assert not unseen
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert wake.await_count == wakes
    conn.close()


def test_unbound_actor_does_not_borrow_gateway_environment(monkeypatch):
    from hermes_cli.kanban_completion_origin import completion_origin
    set_session_vars(session_id="real-turn", profile="default")
    reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-other-turn")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "default")
    assert completion_origin() == {}
