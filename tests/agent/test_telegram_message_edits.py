"""A Telegram edit must not become a second turn — overlay tests.

Measured on the researcher profile 2026-08-06: one message sent at 20:19:46
and edited at 20:20:19 produced two ``Flushing text batch`` lines of identical
size and two queued follow-ups (``Handing 2 queued follow-up(s) back`` on the
following ``/stop``).

The tests below drive the REAL ``GatewayRunner._queue_or_replace_pending_event``
rather than the helper alone — a module-only test would stay green even if the
thin hook were never wired into the gateway.
"""

import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub telegram like the other gateway tests do, so gateway.run imports.
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from agent.telegram_message_edits import (  # noqa: E402
    EDIT_METADATA_KEY,
    is_edit,
    replace_pending_copy,
)
from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
)

SESSION = "agent:main:telegram:group:-100:3"


def _event(text, message_id="7001", edited=False):
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="-100",
        chat_type="group",
        user_id="u1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
        metadata={EDIT_METADATA_KEY: True} if edited else {},
    )


def _adapter():
    a = MagicMock()
    a._pending_messages = {}
    a.interrupt_session_activity = AsyncMock()
    return a


def _runner(adapter):
    """Real GatewayRunner with only the routing plumbing stubbed."""
    from gateway.run import GatewayRunner

    r = object.__new__(GatewayRunner)
    r._sessions = {}
    r.adapters = {}
    r.config = MagicMock()
    r._adapter_for_source = lambda source: adapter

    state = MagicMock()
    state.conversation.queued_events = []
    r._peek_session_state = lambda key: state
    r._session_state = lambda key: state
    r._state = state
    return r


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------

class TestIsEdit:
    def test_plain_message_is_not_an_edit(self):
        assert is_edit(_event("hi")) is False

    def test_flagged_message_is_an_edit(self):
        assert is_edit(_event("hi", edited=True)) is True

    def test_missing_metadata_is_not_an_edit(self):
        assert is_edit(object()) is False


# ---------------------------------------------------------------------------
# reconciliation against the pending queue
# ---------------------------------------------------------------------------

class TestReplacePendingCopy:
    def test_replaces_the_head_slot(self):
        adapter = _adapter()
        runner = _runner(adapter)
        adapter._pending_messages[SESSION] = _event("original")
        edit = _event("corrected", edited=True)

        assert replace_pending_copy(runner, SESSION, edit, adapter) is True
        assert adapter._pending_messages[SESSION].text == "corrected"

    def test_replaces_an_overflow_item(self):
        adapter = _adapter()
        runner = _runner(adapter)
        adapter._pending_messages[SESSION] = _event("first", message_id="7000")
        runner._state.conversation.queued_events = [
            _event("second", message_id="7001"),
            _event("third", message_id="7002"),
        ]
        edit = _event("second, fixed", message_id="7001", edited=True)

        assert replace_pending_copy(runner, SESSION, edit, adapter) is True
        texts = [e.text for e in runner._state.conversation.queued_events]
        assert texts == ["second, fixed", "third"]

    def test_non_edit_is_never_absorbed(self):
        adapter = _adapter()
        runner = _runner(adapter)
        adapter._pending_messages[SESSION] = _event("original")

        assert replace_pending_copy(runner, SESSION, _event("new"), adapter) is False

    def test_edit_of_an_unqueued_message_falls_through(self):
        """Already answered ⇒ nothing to replace; caller must still queue it.

        Silently swallowing here would delete instruction text.
        """
        adapter = _adapter()
        runner = _runner(adapter)
        edit = _event("late correction", edited=True)

        assert replace_pending_copy(runner, SESSION, edit, adapter) is False

    def test_different_message_is_not_overwritten(self):
        adapter = _adapter()
        runner = _runner(adapter)
        adapter._pending_messages[SESSION] = _event("unrelated", message_id="6000")
        edit = _event("corrected", message_id="7001", edited=True)

        assert replace_pending_copy(runner, SESSION, edit, adapter) is False
        assert adapter._pending_messages[SESSION].text == "unrelated"

    def test_unknown_ids_never_match(self):
        """``None`` id means "unknown" — matching it would clobber a stranger."""
        adapter = _adapter()
        runner = _runner(adapter)
        adapter._pending_messages[SESSION] = _event("unrelated", message_id=None)
        edit = _event("corrected", message_id=None, edited=True)

        assert replace_pending_copy(runner, SESSION, edit, adapter) is False

    def test_hostile_adapter_fails_open(self):
        """Any error ⇒ False ⇒ the caller queues, exactly like upstream."""
        runner = _runner(MagicMock())

        class Hostile:
            @property
            def _pending_messages(self):
                raise RuntimeError("boom")

        assert replace_pending_copy(runner, SESSION, _event("x", edited=True), Hostile()) is False


# ---------------------------------------------------------------------------
# the gateway itself — is the hook wired in?
# ---------------------------------------------------------------------------

class TestGatewayIntegration:
    def test_edit_does_not_create_a_second_turn(self):
        """The measured incident, replayed through the real gateway method."""
        adapter = _adapter()
        runner = _runner(adapter)

        runner._queue_or_replace_pending_event(SESSION, _event("original", message_id="7001"))
        runner._queue_or_replace_pending_event(
            SESSION, _event("original, fixed", message_id="7001", edited=True)
        )

        assert adapter._pending_messages[SESSION].text == "original, fixed"
        assert runner._state.conversation.queued_events == []

    def test_two_distinct_messages_still_get_two_turns(self):
        """The dedupe must not swallow genuine consecutive follow-ups."""
        adapter = _adapter()
        runner = _runner(adapter)
        runner._BUSY_QUEUE_MAX_PENDING = 10
        runner._queue_depth = lambda key, adapter=None: 0

        runner._queue_or_replace_pending_event(SESSION, _event("one", message_id="7001"))
        runner._queue_or_replace_pending_event(SESSION, _event("two", message_id="7002"))

        assert adapter._pending_messages[SESSION].text == "one"
        assert [e.text for e in runner._state.conversation.queued_events] == ["two"]


# ---------------------------------------------------------------------------
# root cause + producer seam
# ---------------------------------------------------------------------------

class TestRootCauseAndSeam:
    def test_ptb_really_routes_edits_through_the_text_handler(self):
        """The premise of this whole module, checked against real PTB.

        If upstream python-telegram-bot ever stops matching ``edited_message``
        here, the reconciler becomes dead code — and this test says so.
        """
        # This module stubs ``telegram`` at import time so ``gateway.run`` can
        # load, so the real library must be exercised in a clean interpreter —
        # otherwise the check passes against a MagicMock and proves nothing.
        import subprocess

        probe = (
            "import datetime\n"
            "from telegram import Chat, Message, Update, User\n"
            "from telegram.ext import filters\n"
            "now = datetime.datetime.now()\n"
            "m = Message(message_id=5, date=now, chat=Chat(id=-100, type='supergroup'),\n"
            "            from_user=User(id=1, first_name='O', is_bot=False), text='hi',\n"
            "            edit_date=now)\n"
            "print(bool((filters.TEXT & ~filters.COMMAND).check_update(\n"
            "    Update(update_id=1, edited_message=m))))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0 and "No module named 'telegram'" in result.stderr:
            pytest.skip("python-telegram-bot not installed")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True", (
            "PTB no longer routes edited_message through the text handler — "
            "the edit reconciler may be dead code now"
        )

    def test_adapter_flags_edited_messages(self):
        """The gateway side is useless unless the adapter sets the flag."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        src = inspect.getsource(TelegramAdapter._build_message_event)
        assert "edit_date" in src, "adapter no longer inspects edit_date"
        assert EDIT_METADATA_KEY in src, "adapter no longer stamps the edit flag"
        assert "metadata=" in src, "edit flag is not passed into MessageEvent"
