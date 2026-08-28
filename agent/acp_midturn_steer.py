"""Deliver a follow-up INTO the live ACP turn, instead of after it.

Local overlay module (not upstream).

Background.  ``AIAgent.steer()`` stashes the operator's mid-turn message in
``_pending_steer``; every upstream drain point appends it to the last
``tool``-role message.  Under native ACP tool-calling Hermes owns no tool
messages — the engine runs its whole tool loop inside one ``session/prompt``
— so the text had no landing site at all (see ``acp_busy_followup``).  The
2026-08-06 fix picked the queue as carrier: correct, but the follow-up only
reached the model on the NEXT turn, after the wrong answer had already been
sent.

Measured 2026-08-07 on a raw bridge (``/tmp/steer-probe/midturn_probe.py``),
sonnet-5, engine ``claude-agent-acp``:

* ``initialize`` advertises ``agentCapabilities._meta.claudeCode
  .promptQueueing = true``;
* a SECOND ``session/prompt`` sent on the same session while the first is
  still in flight is accepted (no error) — the engine pushes it onto the
  SDK's streaming input (``session.input.push``) and queues a Turn;
* the model absorbs it at its next model call, i.e. genuinely mid-turn: the
  probe's turn answered the first prompt's question *and* obeyed the
  follow-up that arrived 21s into it;
* **the first request id settles early and empty** — ``stopReason:end_turn``
  with all-zero usage at the moment the hand-off happens (24.28s), while the
  answer text streams at 28.85s and the *second* id carries the real usage.

That last point is the trap.  A caller that returns as soon as its own
request id settles gets an EMPTY answer and tears the bridge down while the
engine is still writing.  So the transport must wait for every outstanding
prompt id, not just its own.  This module holds the policy; the loop lives
in ``copilot_acp_client._run_prompt`` (a file upstream does not touch).

Fail direction is deliberate: every helper here fails **closed** — on any
doubt we do not push, the text stays in ``_pending_steer``, and upstream's
``turn_finalizer`` hands it back as ``result["pending_steer"]`` for the
gateway to deliver as the next turn.  Losing mid-turn delivery costs a turn;
losing the text costs the operator his message twice (2026-08-06).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Last observation of the engine's prompt-queueing capability, learned from
# `initialize`.  Optimistic default: the pinned engine advertises it, and
# being wrong only makes the busy-ack read "steered" for a message that in
# fact lands next turn — a cosmetic error on a lossless path.  A single
# bridge reporting False flips it and it stays flipped for the process.
_capability_lock = threading.Lock()
_capability_seen_false = False


def note_initialize_result(result: Any) -> bool:
    """Record whether this engine accepts a prompt while a turn is live.

    Returns the capability for the bridge that just initialized.
    """
    global _capability_seen_false
    supported = _prompt_queueing_flag(result)
    if not supported:
        with _capability_lock:
            _capability_seen_false = True
    return supported


def _prompt_queueing_flag(result: Any) -> bool:
    try:
        caps = (result or {}).get("agentCapabilities") or {}
        meta = caps.get("_meta") or {}
        claude = meta.get("claudeCode") or {}
        return bool(claude.get("promptQueueing"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("prompt-queueing capability probe failed: %s", exc)
        return False


def midturn_delivery_available() -> bool:
    """Best current guess, for surfaces that must answer before a bridge runs."""
    with _capability_lock:
        return not _capability_seen_false


def pull_followup(client: Any) -> str | None:
    """Drain a pending steer and return the wire text to push, or None.

    Only ever returns text when the engine is driving its own tool loop.  On
    the plain Hermes loop the pre-API drain in ``conversation_loop`` is the
    correct consumer and draining here would STEAL the message from it.
    """
    try:
        from agent.acp_native_tools import _acp_native_tools_enabled

        if not _acp_native_tools_enabled():
            return None
        agent = getattr(client, "_activity_agent", None)
        if agent is None:
            return None
        # A turn we ourselves cancelled is about to be torn down; anything
        # pushed now would be delivered to a bridge nobody is reading.
        if getattr(client, "_hermes_cancelled", None):
            return None
        drain = getattr(agent, "_drain_pending_steer", None)
        if not callable(drain):
            return None
        text = drain()
        if not text or not str(text).strip():
            return None
        text = str(text).strip()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("mid-turn steer drain failed: %s", exc)
        return None

    record_delivered(client, text)
    from agent.prompt_builder import format_steer_marker

    # Same marker the system prompt teaches the model to trust, so an
    # out-of-band instruction reads identically on both transports.
    return format_steer_marker(text).strip()


def record_delivered(client: Any, text: str) -> None:
    """Remember what went onto the wire, for the transcript and for rollback."""
    agent = getattr(client, "_activity_agent", None)
    if agent is None:
        return
    try:
        delivered = getattr(agent, "_acp_delivered_steers", None)
        if not isinstance(delivered, list):
            delivered = []
            agent._acp_delivered_steers = delivered
        delivered.append(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not record delivered steer: %s", exc)


def restore_undelivered(client: Any) -> None:
    """Put mid-turn text back into ``_pending_steer`` after a failed attempt.

    A bridge that dies (eviction, failover, transport error) takes the
    pushed prompt with it.  The text was drained out of ``_pending_steer``,
    so without this it would exist nowhere: not in the answer, not in the
    queue, not in the transcript.  Re-stashing hands it to the retry — or,
    if the turn is over, to ``turn_finalizer``'s leftover path.
    """
    agent = getattr(client, "_activity_agent", None)
    if agent is None:
        return
    try:
        delivered = getattr(agent, "_acp_delivered_steers", None)
        if not delivered:
            return
        pending = "\n".join(str(t) for t in delivered if str(t).strip())
        agent._acp_delivered_steers = []
        if not pending:
            return
        steer = getattr(agent, "steer", None)
        if callable(steer):
            steer(pending)
            logger.info(
                "mid-turn follow-up re-queued after a failed ACP attempt (%d chars)",
                len(pending),
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not restore undelivered steer: %s", exc)


def fold_delivered_into_messages(agent: Any, messages: Any) -> int:
    """Write mid-turn follow-ups into the transcript, after the answer lands.

    The engine saw them; Hermes' own message list did not, and Hermes is
    what gets replayed on every subsequent turn.  Without this the next turn
    reads as an assistant that obeyed an instruction nobody gave.

    Appended to the LAST ``user`` message rather than added as a new one:
    a second consecutive user message would be rewritten by the
    alternation repairer, and the marker already says the text arrived
    out of band.  Returns how many follow-ups were folded.
    """
    if agent is None:
        return 0
    try:
        delivered = getattr(agent, "_acp_delivered_steers", None)
        if not delivered:
            return 0
        if not isinstance(messages, list):
            return 0
        from agent.prompt_builder import format_steer_marker

        target = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                target = msg
                break
        if target is None:
            # Nothing to attach to: leave the record in place so a later
            # iteration (or the leftover path) can still see it.
            return 0
        marker = "".join(format_steer_marker(str(t)) for t in delivered)
        existing = target.get("content", "")
        if isinstance(existing, str):
            target["content"] = existing + marker
        else:
            try:
                blocks = list(existing) if existing else []
                blocks.append({"type": "text", "text": marker})
                target["content"] = blocks
            except Exception:
                return 0
        count = len(delivered)
        agent._acp_delivered_steers = []
        logger.debug("folded %d mid-turn follow-up(s) into the transcript", count)
        return count
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not fold delivered steers into messages: %s", exc)
        return 0


__all__ = [
    "note_initialize_result",
    "midturn_delivery_available",
    "pull_followup",
    "record_delivered",
    "restore_undelivered",
    "fold_delivered_into_messages",
]
