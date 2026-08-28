"""Where does a follow-up sent to a BUSY session actually go?

Local overlay module (not upstream).

Two independent losses, both measured on the secretary profile 2026-08-06.

**1. ``steer`` has no landing site under native ACP.**
``AIAgent.steer()`` stashes the text in ``_pending_steer`` and every drain
point appends it to *the last ``tool``-role message*
(``conversation_loop`` pre-API drain, ``tool_executor`` post-batch drain).
With native ACP tool-calling the engine runs its whole tool loop inside one
``session/prompt``, so Hermes' message list carries no ``tool`` message at
all — there is nothing to append to, on any iteration.  Evidence: the steer
marker (``prompt_builder.STEER_MARKER_OPEN``) appears **zero** times in every
transcript of every profile since the fleet switched to ``steer``.  The
gateway still reports success, because ``steer()`` returns True for any
non-empty string, and the ack even promises "your message arrives after the
next tool call" — a tool call that never comes.

**2. ``/stop`` destroys the queued follow-up.**
``_interrupt_and_clear_session`` pops the adapter's pending slot with a
literal ``# consume and discard``, and ``request_hard_interrupt`` →
``AIAgent.interrupt()`` deliberately drops ``_pending_steer`` as well.  So
both carriers die on ``/stop``, silently.  That is what the operator hits in
practice: a follow-up to a 25-minute turn looks ignored, he sends ``/stop``,
and the text he was waiting on is deleted at that exact moment.  Twice on
2026-08-06 he re-typed the identical message by hand afterwards (115-char at
19:06, 56-char at 19:41 — both provable by byte count in the adapter's
"Flushing text batch" line).

So: pick a carrier that can actually deliver, and stop deleting it.

**Update 2026-08-07.**  Loss 1 is now fixed at the transport instead of
routed around: ``agent/acp_midturn_steer`` pushes the pending steer as a
second ``session/prompt`` on the LIVE engine session, which the model absorbs
at its next model call.  So ``steer`` is a real carrier on native ACP again
and ``steer_lands_this_turn`` says True — unless the engine does not
advertise ``promptQueueing``, in which case this module's original verdict
(queue it) still stands and is still correct.

``steer_lands_this_turn`` answers the first question — the gateway's steer
sites already fall back to queue semantics when it says no, which is the
documented contract of ``steer()`` returning False.  ``capture_pending_events``
and ``redeliver_events`` answer the second: what ``/stop`` used to discard is
handed back as the next user turn.

Fail-safe direction differs per helper and is deliberate:

* ``steer_lands_this_turn`` fails **closed** (returns False on any error) —
  queue is always a correct carrier, steer only sometimes is.
* ``capture_pending_events`` fails **open** (returns [] on any error) — a
  caller that gets nothing behaves exactly like upstream did.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ``agent_runtime_helpers`` routes a session onto the ACP transport on exactly
# this provider label (see its ``client_kwargs`` branch); keep the string in
# one place so a rename shows up as one grep hit.
_ACP_PROVIDER = "copilot-acp"

# Only ``/stop`` hands work back. ``/new`` and ``/reset`` mean "forget it",
# so for those the old discard is the correct behaviour.
_REDELIVER_REASONS = frozenset({"stop_command"})


def _native_tools_enabled() -> bool:
    """Is the engine driving its own tool loop this session?"""
    from agent.acp_native_tools import _acp_native_tools_enabled

    return bool(_acp_native_tools_enabled())


def steer_lands_this_turn(agent: Any) -> bool:
    """Can a mid-turn steer actually reach the model on THIS agent?

    True for the ordinary Hermes tool loop: the next tool result is a real
    message in the list and the drain points can append to it.

    True for native ACP tool-calling **since 2026-08-07**: Hermes still owns
    no ``tool`` message there, but the transport now delivers the steer as a
    second ``session/prompt`` on the live engine session, which the model
    absorbs at its next model call (``agent/acp_midturn_steer``).  The one
    thing that would make it false again is an engine that does not accept a
    prompt while a turn runs — and that is exactly what
    ``midturn_delivery_available()`` reports, learned from ``initialize``.

    The caller is expected to treat False the same way it treats ``steer()``
    rejecting the payload — i.e. queue it.
    """
    if agent is None:
        return False
    try:
        provider = str(getattr(agent, "provider", "") or "")
        if provider != _ACP_PROVIDER:
            # Non-ACP providers (local qwen, native APIs) run Hermes' own tool
            # loop — steering works there and is worth keeping.
            return True
        if not _native_tools_enabled():
            # Engine-side tools off ⇒ Hermes runs the loop ⇒ upstream's
            # tool-result drain is the carrier, exactly as it always was.
            return True
        from agent.acp_midturn_steer import midturn_delivery_available

        return bool(midturn_delivery_available())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("steer_lands_this_turn check failed: %s", exc)
        return False


def capture_pending_events(runner: Any, session_key: str, adapter: Any) -> list:
    """Pop every queued follow-up for a session and return them in order.

    Covers both storage tiers the gateway uses: the adapter's single
    ``_pending_messages`` head slot and the session-state overflow tail
    (``conversation.queued_events``, written by ``_enqueue_fifo``).  A caller
    that only popped the head slot would still lose everything past the
    first message.
    """
    events: list = []
    if not session_key:
        return events
    try:
        slot = getattr(adapter, "_pending_messages", None)
        if isinstance(slot, dict):
            head = slot.pop(session_key, None)
            if head is not None:
                events.append(head)
        state = runner._peek_session_state(session_key)
        tail = getattr(getattr(state, "conversation", None), "queued_events", None)
        if isinstance(tail, list) and tail:
            events.extend(tail)
            tail.clear()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("capture_pending_events failed for %s: %s", session_key, exc)
        return []
    return events


def restore_pending_events(runner: Any, session_key: str, adapter: Any, events: list) -> bool:
    """Put captured follow-ups back so the normal drain picks them up.

    Upstream already owns the redelivery step: ``BasePlatformAdapter.
    _drain_pending_after_session_command`` runs at the tail of ``/stop``
    dispatch, pops the head slot and spawns a fresh processing task for it.
    It found nothing only because the discard above ran first.  So the fix is
    to hand the events back into the same two tiers they came from, after the
    session state has been released (which resets the overflow tail), and let
    upstream deliver.

    Returns True when at least the head slot was restored.
    """
    if not events:
        return False
    try:
        slot = getattr(adapter, "_pending_messages", None)
        if not isinstance(slot, dict):
            return False
        # Never clobber a follow-up that arrived while the command itself ran —
        # that one is newer, and upstream's drain is already about to take it.
        if session_key in slot:
            tail_target = runner._session_state(session_key).conversation.queued_events
            tail_target.extend(events)
            return True
        slot[session_key] = events[0]
        if len(events) > 1:
            runner._session_state(session_key).conversation.queued_events.extend(events[1:])
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("restore_pending_events failed for %s: %s", session_key, exc)
        return False


def should_redeliver(invalidation_reason: str) -> bool:
    """Does this session teardown owe the user their queued text back?"""
    return str(invalidation_reason or "") in _REDELIVER_REASONS


def describe_events(events: list, *, limit: int = 60) -> str:
    """Short human preview of what is being handed back, for the /stop reply."""
    previews = []
    for event in events:
        text = (getattr(event, "text", "") or "").strip().replace("\n", " ")
        if text:
            previews.append(text[:limit] + ("…" if len(text) > limit else ""))
    return "; ".join(previews)
