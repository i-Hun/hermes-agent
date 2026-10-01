"""Owner routing and receipts across block/unblock/complete on a real board."""
import asyncio
import pytest
from evals.heartbeat_idle_wire import WireAdapter
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.kanban_watchers_notifier import _notifier_collect, _KanbanNotification
from gateway.session_context import set_session_vars, clear_session_vars
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
from tools import kanban_tools as kt

@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['wake','notify+wake','notify'])
async def test_owner_route_survives_terminal_cycle(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    runner = object.__new__(GatewayRunner)
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner.adapters = {Platform.TELEGRAM:adapter}
    runner._profile_adapters = {'secretary': {Platform.TELEGRAM: adapter}}
    runner._primary_profile_name = 'secretary'
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda source: adapter
    runner._kanban_dispatcher_lock_handle = object()
    received = []
    async def handler(event):
        received.append(event)
    adapter.set_message_handler(handler)
    await adapter.connect()
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='owned product', assignee='default')
        kn.add_notify_sub(conn, task_id=tid, platform='telegram', chat_id='42',
                          thread_id='22', chat_type='group', notifier_profile='secretary', delivery_mode=mode)
        kb.block_task(conn, tid, reason='controlled block', kind='needs_input')
    async def tick():
        deliveries = await asyncio.to_thread(_notifier_collect, runner, kb,
                           notifier_profile='secretary', gc_due=False, gc_retention_days=30)
        for delivery in deliveries:
            await _KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts={}).deliver()
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
    try:
        await tick()
        first = (len(received),len(adapter.wire))
        with kbc.connect_closing() as conn:
            before = kn.list_notify_subs(conn, tid)
            tokens = set_session_vars(platform='telegram', chat_id='42', thread_id='1', profile='default')
            try:
                for _ in range(3):
                    assert kt._maybe_auto_subscribe(conn, tid)
            finally:
                clear_session_vars(tokens)
            assert kn.list_notify_subs(conn, tid) == before
        for _ in range(3):
            await tick()
        assert (len(received),len(adapter.wire)) == first
        with kbc.connect_closing() as conn:
            kb.unblock_task(conn, tid)
            kb.complete_task(conn, tid, summary='controlled complete')
        await tick()
        last = (len(received),len(adapter.wire))
        for _ in range(3):
            await tick()
        assert (len(received),len(adapter.wire)) == last
        assert len(received) == (0 if mode=='notify' else 2)
        assert len(adapter.wire) == (0 if mode=='wake' else 2)
        assert all(str(e.source.thread_id)=='22' for e in received)
        with kbc.connect_closing() as conn:
            row = kn.list_notify_subs(conn, tid)[0]
            assert row['notifier_profile']=='secretary' and row['thread_id']=='22' and row['delivery_mode']==mode
            assert not kn.unseen_events_for_sub(conn, task_id=tid, platform='telegram',chat_id='42', thread_id='22', kinds=['blocked','completed'])[1]
    finally:
        await adapter.disconnect()
