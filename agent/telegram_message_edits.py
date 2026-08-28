"""A Telegram *edit* is a correction, not a second message.

Local overlay module (not upstream).

Measured on the researcher profile 2026-08-06 20:19:46 / 20:20:19: two
``Flushing text batch`` lines of identical size (104 chars) 33 s apart, and
``Handing 2 queued follow-up(s) back`` on the following ``/stop``. The
operator had sent one message and then edited it — the gateway queued the
edit as a *separate* turn next to the original.

Why it happens: PTB's ``MessageHandler`` matches ``update.effective_message``,
which resolves ``edited_message`` exactly like ``message`` (verified against
python-telegram-bot 22.6 — ``filters.TEXT`` returns True for an update whose
only field is ``edited_message``). The registered handlers carry no
``~filters.UpdateType.EDITED_MESSAGE``, so an edit walks the whole inbound
path a second time and lands as a fresh ``MessageEvent``.

The fix is deliberately narrow: an edit whose original is *still waiting* in
the pending queue replaces that copy in place. It does not touch the two
other cases, and that is a decision, not an omission:

* original is the turn running right now → the edit is a live correction and
  ``busy_input_mode`` already routes it (redirect / queue). Nothing to dedupe.
* original was already answered → the edit is new instruction text. Dropping
  it would be a silent loss, which is worse than an extra turn.

Fails **open**: any error returns False and the caller queues exactly as
upstream did.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Set by the Telegram adapter on every event built from a message that
# carries ``edit_date``. Kept in one place so a rename is one grep hit.
EDIT_METADATA_KEY = "telegram_edited"


def is_edit(event: Any) -> bool:
    """Was this event built from an edited platform message?"""
    try:
        metadata = getattr(event, "metadata", None) or {}
        return bool(metadata.get(EDIT_METADATA_KEY))
    except Exception:  # pragma: no cover - defensive
        return False


def _same_message(a: Any, b: Any) -> bool:
    """Do two events describe the same platform message?

    Both ids must be present and non-empty: a ``None`` id on either side is
    "unknown", and treating unknown as a match would let an edit overwrite an
    unrelated queued follow-up.
    """
    left = getattr(a, "message_id", None)
    right = getattr(b, "message_id", None)
    if left is None or right is None:
        return False
    left, right = str(left), str(right)
    return bool(left) and left == right


def replace_pending_copy(runner: Any, session_key: str, event: Any, adapter: Any) -> bool:
    """Replace an already-queued copy of the same message with ``event``.

    Returns True when the edit was absorbed (caller must NOT queue it) and
    False when there is nothing to replace (caller proceeds as upstream).
    """
    if not is_edit(event):
        return False
    try:
        pending_slot = getattr(adapter, "_pending_messages", None)
        if isinstance(pending_slot, dict):
            existing = pending_slot.get(session_key)
            if existing is not None and _same_message(existing, event):
                pending_slot[session_key] = event
                logger.info(
                    "Telegram edit of message %s replaced the queued copy for "
                    "session %s (no extra turn).",
                    getattr(event, "message_id", None),
                    session_key,
                )
                return True

        # Overflow tail: same FIFO the gateway uses for /queue.
        peek = getattr(runner, "_peek_session_state", None)
        state = peek(session_key) if callable(peek) else None
        overflow = getattr(getattr(state, "conversation", None), "queued_events", None)
        if isinstance(overflow, list):
            for index, queued in enumerate(overflow):
                if _same_message(queued, event):
                    overflow[index] = event
                    logger.info(
                        "Telegram edit of message %s replaced queued follow-up "
                        "#%d for session %s (no extra turn).",
                        getattr(event, "message_id", None),
                        index,
                        session_key,
                    )
                    return True
    except Exception as exc:
        logger.debug("Telegram edit reconciliation failed for %s: %s", session_key, exc)
        return False
    return False
