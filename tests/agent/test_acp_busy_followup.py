"""Where a follow-up to a BUSY session goes — overlay tests.

Two defects, measured on the secretary profile 2026-08-06 (see
``agent/acp_busy_followup``):

* ``steer`` has no landing site under native ACP tool-calling, so the text was
  stashed in ``_pending_steer`` and never reached a model (zero steer markers
  fleet-wide);
* ``/stop`` popped the queued follow-up with ``# consume and discard``,
  deleting the exact message the operator was waiting on.

The integration tests below drive the REAL ``GatewayRunner``
``_interrupt_and_clear_session`` and the real busy-handler steer branch — a
unit test of the helper module alone would pass even if the hooks were never
wired in.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub telegram like the existing gateway tests do, so gateway.run imports.
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

from agent.acp_busy_followup import (  # noqa: E402
    capture_pending_events,
    describe_events,
    restore_pending_events,
    should_redeliver,
    steer_lands_this_turn,
)
from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _agent(provider="copilot-acp"):
    a = MagicMock()
    a.provider = provider
    return a


def _event(text="follow-up"):
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
        message_id="m1",
    )


def _runner():
    """Real GatewayRunner instance with only the plumbing stubbed."""
    from gateway.run import GatewayRunner

    r = object.__new__(GatewayRunner)
    r._sessions = {}
    r._draining = False
    r._busy_input_mode = "steer"
    r._busy_text_mode = "interrupt"
    r.adapters = {}
    r.config = MagicMock()
    r._invalidate_session_run_generation = lambda key, reason=None: 1
    r._release_running_agent_state = lambda key, **kw: r._sessions.pop(key, None) is not None
    r._evict_cached_agent = lambda key: None
    r._thread_metadata_for_source = lambda *a, **kw: None
    r._is_session_run_current = lambda *a, **kw: True
    return r


def _adapter():
    a = MagicMock()
    a._pending_messages = {}
    a.interrupt_session_activity = AsyncMock()
    # Real semantics (BasePlatformAdapter.get_pending_message): pop the slot.
    # A MagicMock no-op here would make the discard tests pass vacuously.
    a.get_pending_message = lambda key: a._pending_messages.pop(key, None)
    return a


# ---------------------------------------------------------------------------
# steer_lands_this_turn — the landing-site predicate
# ---------------------------------------------------------------------------

class TestSteerLandingSite:
    def test_non_acp_provider_can_steer(self):
        """Hermes owns the tool loop there — the drain points have a target."""
        assert steer_lands_this_turn(_agent(provider="openai")) is True

    def test_acp_with_native_tools_steers_through_the_transport(self, monkeypatch):
        """Changed 2026-08-07: the bridge now delivers the steer mid-turn.

        Hermes still owns no ``tool`` message here — the landing site is a
        second ``session/prompt`` on the live engine session, see
        ``agent/acp_midturn_steer``.
        """
        from agent import acp_midturn_steer

        monkeypatch.setenv("HERMES_ACP_NATIVE_TOOLS", "1")
        monkeypatch.setattr(acp_midturn_steer, "_capability_seen_false", False)
        assert steer_lands_this_turn(_agent()) is True

    def test_acp_queues_again_if_the_engine_refuses_queued_prompts(self, monkeypatch):
        """The one thing that reopens the hole: an engine without the capability."""
        from agent import acp_midturn_steer

        monkeypatch.setenv("HERMES_ACP_NATIVE_TOOLS", "1")
        monkeypatch.setattr(acp_midturn_steer, "_capability_seen_false", True)
        assert steer_lands_this_turn(_agent()) is False

    def test_acp_without_native_tools_can_steer(self, monkeypatch):
        """Legacy ACP: Hermes drives the tools, so tool messages exist."""
        monkeypatch.setenv("HERMES_ACP_NATIVE_TOOLS", "0")
        assert steer_lands_this_turn(_agent()) is True

    def test_no_agent_is_not_steerable(self):
        assert steer_lands_this_turn(None) is False

    def test_broken_agent_fails_closed(self, monkeypatch):
        """Any error ⇒ queue. Queue delivers everywhere; steer does not."""
        monkeypatch.setenv("HERMES_ACP_NATIVE_TOOLS", "1")

        class Hostile:
            @property
            def provider(self):
                raise RuntimeError("boom")

        assert steer_lands_this_turn(Hostile()) is False

    def test_gateway_hook_is_wired(self, monkeypatch):
        """The gateway's own thin hook must agree — not just the module."""
        from agent import acp_midturn_steer
        from gateway.overlay_hooks import acp_steer_lands_this_turn

        monkeypatch.setenv("HERMES_ACP_NATIVE_TOOLS", "1")
        monkeypatch.setattr(acp_midturn_steer, "_capability_seen_false", True)
        assert acp_steer_lands_this_turn(_agent()) is False
        monkeypatch.setattr(acp_midturn_steer, "_capability_seen_false", False)
        assert acp_steer_lands_this_turn(_agent()) is True
        assert acp_steer_lands_this_turn(_agent(provider="anthropic")) is True


# ---------------------------------------------------------------------------
# capture / restore
# ---------------------------------------------------------------------------

class TestCaptureAndRestore:
    def test_captures_head_slot(self):
        r, a = _runner(), _adapter()
        ev = _event()
        a._pending_messages["k"] = ev
        assert capture_pending_events(r, "k", a) == [ev]
        assert "k" not in a._pending_messages

    def test_captures_overflow_tail_in_order(self):
        r, a = _runner(), _adapter()
        head, t1, t2 = _event("head"), _event("t1"), _event("t2")
        a._pending_messages["k"] = head
        r._session_state("k").conversation.queued_events.extend([t1, t2])
        assert capture_pending_events(r, "k", a) == [head, t1, t2]
        assert r._session_state("k").conversation.queued_events == []

    def test_nothing_queued_is_empty(self):
        r, a = _runner(), _adapter()
        assert capture_pending_events(r, "k", a) == []

    def test_capture_fails_open(self):
        """A broken adapter behaves exactly like upstream: nothing captured."""
        r = _runner()
        broken = MagicMock()
        broken._pending_messages = "not-a-dict"
        assert capture_pending_events(r, "k", broken) == []

    def test_restore_puts_head_and_tail_back(self):
        r, a = _runner(), _adapter()
        head, tail = _event("head"), _event("tail")
        assert restore_pending_events(r, "k", a, [head, tail]) is True
        assert a._pending_messages["k"] is head
        assert r._session_state("k").conversation.queued_events == [tail]

    def test_restore_never_clobbers_a_newer_follow_up(self):
        """A message that landed while /stop ran is newer — keep its place."""
        r, a = _runner(), _adapter()
        newer, older = _event("newer"), _event("older")
        a._pending_messages["k"] = newer
        assert restore_pending_events(r, "k", a, [older]) is True
        assert a._pending_messages["k"] is newer
        assert r._session_state("k").conversation.queued_events == [older]

    def test_restore_of_nothing_is_a_noop(self):
        r, a = _runner(), _adapter()
        assert restore_pending_events(r, "k", a, []) is False
        assert a._pending_messages == {}


class TestRedeliverScope:
    def test_stop_hands_work_back(self):
        assert should_redeliver("stop_command") is True

    @pytest.mark.parametrize("reason", ["new_command", "session_reset", "", None])
    def test_forget_it_reasons_still_discard(self, reason):
        assert should_redeliver(reason) is False

    def test_describe_events_previews_text(self):
        assert describe_events([_event("hello"), _event("world")]) == "hello; world"

    def test_describe_events_truncates(self):
        long = "x" * 200
        assert describe_events([_event(long)], limit=10) == "x" * 10 + "…"


# ---------------------------------------------------------------------------
# integration — the real _interrupt_and_clear_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBusyHandlerDoesNotSteerIntoTheVoid:
    """The real busy handler must queue, not steer, on an ACP-native turn."""

    async def _handle(self, monkeypatch, provider, native="1"):
        import gateway.run as _gr
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_ACP_NATIVE_TOOLS", native)
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})
        r, a = _runner(), _adapter()
        r._sessions_map = lambda: r._sessions
        r._adapter_for_source = lambda _s: a
        r._agent_has_active_subagents = lambda _agent: False
        r._session_has_compression_in_flight = AsyncMock(return_value=False)
        r._reply_anchor_for_event = lambda _e: None
        r._queue_depth = lambda key, adapter=None: 0
        r._is_user_authorized = lambda _source: True
        agent = _agent(provider=provider)
        agent.steer.return_value = True
        ev = _event("уточнение во время работы")
        r._session_state("k").turn.agent = agent
        handled = await GatewayRunner._handle_active_session_busy_message(r, ev, "k")
        return handled, agent, a, ev

    async def test_acp_native_turn_steers_when_the_engine_takes_queued_prompts(
        self, monkeypatch
    ):
        """Since 2026-08-07 this is the normal path: the transport delivers it."""
        from agent import acp_midturn_steer

        monkeypatch.setattr(acp_midturn_steer, "_capability_seen_false", False)
        handled, agent, a, ev = await self._handle(monkeypatch, "copilot-acp")
        agent.steer.assert_called_once()
        assert "k" not in a._pending_messages
        assert handled is True

    async def test_acp_native_turn_queues_when_the_engine_cannot(self, monkeypatch):
        """The original 2026-08-06 loss: no landing site ⇒ queue, never steer."""
        from agent import acp_midturn_steer

        monkeypatch.setattr(acp_midturn_steer, "_capability_seen_false", True)
        handled, agent, a, ev = await self._handle(monkeypatch, "copilot-acp")
        agent.steer.assert_not_called()
        assert a._pending_messages.get("k") is ev
        assert handled is True

    async def test_non_acp_turn_still_steers(self, monkeypatch):
        """Local qwen / native APIs keep real mid-turn steering."""
        handled, agent, a, ev = await self._handle(monkeypatch, "openai")
        agent.steer.assert_called_once()
        assert "k" not in a._pending_messages


class TestBothSteerSitesAreGuarded:
    """There are TWO steer call sites; the behavioural test above only drives
    one. Guard the other (``_handle_message``'s PRIORITY branch) at the seam —
    an unguarded site would silently reopen the same hole."""

    def test_priority_steer_branch_asks_about_the_landing_site(self):
        import inspect

        from gateway.run import GatewayRunner

        src = inspect.getsource(GatewayRunner._handle_message)
        # Anchor on the COMPARISON, not on the receiver: v2026.8.13 made the
        # busy-input mode per-profile and the branch became
        # ``effective_busy_input_mode == "steer"``. The seam was untouched, but
        # a receiver-shaped anchor still reddened — the same rot the seam test
        # avoids by keying on AST function names instead of text.
        marker = '_busy_input_mode == "steer"'
        assert marker in src, "PRIORITY steer branch moved"
        branch = src.split(marker, 1)[1]
        head = branch.split("running_agent.steer(", 1)[0]
        assert "acp_steer_lands_this_turn" in head, (
            "PRIORITY steer site calls agent.steer() without checking whether a "
            "steer can land — on native ACP that silently drops the message"
        )


@pytest.mark.asyncio
class TestStopKeepsTheFollowUp:
    async def _stop(self, r, a, reason="stop_command"):
        source = SessionSource(
            platform=MagicMock(value="telegram"),
            chat_id="-100",
            chat_type="group",
            user_id="u1",
        )
        r._adapter_for_source = lambda _s: a
        await r._interrupt_and_clear_session(
            "k",
            source,
            interrupt_reason="stop",
            invalidation_reason=reason,
        )

    async def test_stop_leaves_the_follow_up_for_the_normal_drain(self):
        """Upstream's _drain_pending_after_session_command runs right after
        this and pops the head slot — so surviving here IS delivery."""
        r, a = _runner(), _adapter()
        ev = _event("вот ссылка на просьбу коллеги")
        a._pending_messages["k"] = ev
        await self._stop(r, a)
        assert a._pending_messages.get("k") is ev

    async def test_stop_preserves_order_of_several_follow_ups(self):
        r, a = _runner(), _adapter()
        head, tail = _event("first"), _event("second")
        a._pending_messages["k"] = head
        r._session_state("k").conversation.queued_events.append(tail)
        await self._stop(r, a)
        assert a._pending_messages.get("k") is head
        assert r._session_state("k").conversation.queued_events == [tail]

    async def test_new_command_still_discards(self):
        """/new means "forget it" — upstream behaviour must not change."""
        r, a = _runner(), _adapter()
        a._pending_messages["k"] = _event("stale /reset text")
        await self._stop(r, a, reason="new_command")
        assert "k" not in a._pending_messages

    async def test_stop_with_empty_queue_is_harmless(self):
        r, a = _runner(), _adapter()
        await self._stop(r, a)
        assert a._pending_messages == {}
