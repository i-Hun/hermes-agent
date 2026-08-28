"""Safe wrappers for overlay hooks called from ``gateway/run.py``.

Why this module exists — collision surface, not code size.

Until 2026-08-09 these three wrappers were *defined* inside ``gateway/run.py``.
That file is the highest-churn file in the engine (+20096/-17680 over the 872
commits between our pinned base and origin/main), and a definition occupies a
fixed anchor in it: upstream inserted a new module-level function at exactly the
anchor our two steer/edit wrappers used, and ``git apply --3way`` reported a
conflict that had nothing to do with behaviour — two unrelated functions wanting
the same blank lines.

Keeping only *call sites* in ``run.py`` (each a local import on a line we
already own) removes that class of conflict entirely: a call site sits inside a
hunk we author, so upstream churn around it merges instead of colliding.

The ``try/except`` guards stay here rather than at the call site so behaviour is
unchanged: each wrapper still degrades on its own if the deeper overlay module
is missing, with the failure direction documented per hook.
"""

from typing import Any, Optional

__all__ = [
    "usage_limit_reply_safe",
    "acp_steer_lands_this_turn",
    "telegram_edit_absorbed",
]


def usage_limit_reply_safe(text: str) -> Optional[str]:
    """Thin hook → ``gateway/usage_limit_reply`` (session/weekly/usage windows).

    Returns None when the text is not a usage-limit envelope.
    """
    try:
        from gateway.usage_limit_reply import usage_limit_reply

        return usage_limit_reply(text)
    except Exception:  # never let the overlay leak raw text or crash delivery
        return None


def acp_steer_lands_this_turn(running_agent: Any) -> bool:
    """Thin hook → ``agent/acp_busy_followup``.

    Fails closed (False ⇒ queue instead of steer): queue delivers on every
    provider, steer only where the payload has a landing site.
    """
    try:
        from agent.acp_busy_followup import steer_lands_this_turn
    except Exception:
        return False
    return steer_lands_this_turn(running_agent)


def telegram_edit_absorbed(
    runner: Any, session_key: str, event: Any, adapter: Any
) -> bool:
    """Thin hook → ``agent/telegram_message_edits``.

    Fails open (False ⇒ queue exactly as upstream): an edit that cannot be
    reconciled must still reach the user's turn rather than vanish.
    """
    try:
        from agent.telegram_message_edits import replace_pending_copy
    except Exception:
        return False
    return replace_pending_copy(runner, session_key, event, adapter)
