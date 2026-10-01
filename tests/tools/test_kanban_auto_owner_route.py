"""Automatic subscription must not add a competing origin to an owned task."""
from gateway.session_context import set_session_vars, clear_session_vars
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
from tools import kanban_tools as kt


def test_auto_subscription_preserves_entire_owner_policy(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='owned', assignee='default')
        kn.add_notify_sub(conn, task_id=tid, platform='telegram', chat_id='group',
                          thread_id='22', notifier_profile='secretary', delivery_mode='wake')
        expected = kn.list_notify_subs(conn, tid)
        tokens = set_session_vars(platform='telegram', chat_id='group', thread_id='1', profile='default')
        try:
            for _ in range(3):
                assert kt._maybe_auto_subscribe(conn, tid)
        finally:
            clear_session_vars(tokens)
        assert kn.list_notify_subs(conn, tid) == expected


def test_explicit_resubscribe_can_change_owner_without_replaying_history(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='owned')
        kn.add_notify_sub(conn, task_id=tid, platform='telegram', chat_id='group',
                          notifier_profile='default', delivery_mode='notify+wake')
        before = kn.list_notify_subs(conn, tid)[0]
        from argparse import Namespace
        from hermes_cli.kanban import _cmd_notify_subscribe
        assert _cmd_notify_subscribe(Namespace(
            task_id=tid, platform='telegram', chat_id='group', thread_id=None,
            user_id=None, chat_type='group', notifier_profile='secretary',
            delivery_mode='wake', only_if_missing=False)) == 0
        after = kn.list_notify_subs(conn, tid)[0]
        assert after['notifier_profile'] == 'secretary'
        assert after['delivery_mode'] == 'wake'
        assert after['last_event_id'] == before['last_event_id']
        assert after['last_ping_event_id'] == before['last_ping_event_id']
