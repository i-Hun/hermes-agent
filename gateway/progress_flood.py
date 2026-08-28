"""Flood-control budget for the per-run tool-progress bubble.

Local overlay module (ACP-native fleet). Telegram rate-limits *edits* the same
way it rate-limits sends, and the progress rail edits one bubble every
``_PROGRESS_EDIT_INTERVAL`` seconds for the whole length of a tool-heavy run.
When the budget runs out the server answers 429 ``RetryAfter``; this module
owns what happens next.

Two defects it exists to close, both observed live on 2026-07-29 (secretary,
omnirecs run 16:25–16:32):

* a 429 on the edit was answered by *sending a new message*, so every flood
  cycle spawned a fresh bubble — a long run arrived as one ⚙️ line per minute
  instead of one collapsing bubble (the "fragmentation" report). The line is
  already in ``progress_lines``, so holding the edit loses nothing: the next
  successful edit shows it. Sending it separately is pure duplication that
  also spends the very budget that is exhausted.
* the backoff ignored the server's ``retry_after`` and re-armed the same 1.5s
  cadence, so the run re-entered flood immediately and stayed there. A group
  chat tolerates ~20 messages/minute; 1.5s is ~40/min, i.e. permanently over
  budget on a tool-heavy run. The steady-state interval therefore *grows* once
  the server says the rail is too fast, and stays grown for the rest of the
  run (a fresh run starts from the base again).

Logic lives here on purpose: ``gateway/run.py`` and the Telegram adapter are
hot upstream files, so they keep only thin hook lines — the conflict surface
of a ``hermes update`` stays small. Same rule as ``gateway/progress_collapse``.
"""

from __future__ import annotations

import re

# "Retry in 27 seconds" (Telegram prose), "flood_control:27.0" (our own
# SendResult.error), "retry after 27" (python-telegram-bot RetryAfter str).
_RETRY_AFTER_RE = re.compile(
    r"(?:retry(?:\s+after|\s+in)?|flood_control)\D{0,12}?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_FLOOD_MARKERS = ("flood control", "flood_control", "retry after", "retry in", "too many requests")


def _text_of(obj) -> str:
    return (obj if isinstance(obj, str) else str(obj or "")).lower()


def is_flood_error(obj) -> bool:
    """True when an exception/error string is a rate limit, not a real fault.

    Deliberately structural first: python-telegram-bot's ``RetryAfter`` carries
    a numeric ``retry_after`` attribute, and that is worth more than any
    substring match on a server message that is free to be reworded.
    """
    if getattr(obj, "retry_after", None) is not None:
        return True
    text = _text_of(obj)
    return any(m in text for m in _FLOOD_MARKERS)


def flood_retry_after(obj) -> float | None:
    """Seconds the server asked us to wait, or None when it did not say."""
    value = getattr(obj, "retry_after", None)
    if value is not None:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None and seconds > 0:
            return seconds
    match = _RETRY_AFTER_RE.search(_text_of(obj))
    if not match:
        return None
    try:
        seconds = float(match.group(1))
    except ValueError:
        return None
    return seconds if seconds > 0 else None


class ProgressFloodBudget:
    """Adaptive edit cadence + failure classification for one run's bubble.

    Not thread-safe: like ``ProgressCollapser`` it is only touched from the
    progress task of a single run.
    """

    BASE_INTERVAL = 1.5     # upstream cadence, kept as the optimistic start
    MAX_INTERVAL = 12.0     # ceiling — beyond this the bubble stops feeling live
    GROWTH = 2.0            # multiplier per flood hit
    FALLBACK_HOLD = 5.0     # hold when the server refused to name a delay
    MAX_HOLD = 120.0        # never park the rail longer than this

    def __init__(self, base_interval: float = BASE_INTERVAL) -> None:
        try:
            base = float(base_interval)
        except (TypeError, ValueError):
            base = self.BASE_INTERVAL
        self._base = base if base > 0 else self.BASE_INTERVAL
        self._interval = self._base
        self.floods = 0

    @property
    def interval(self) -> float:
        """Current minimum seconds between progress edits."""
        return self._interval

    def classify(self, result) -> str:
        """``"transient"`` | ``"flood"`` | ``"permanent"`` for a failed edit.

        Order matters. ``retryable`` is the adapter's own verdict on network
        faults and wins; a rate limit is next; anything left is a real, sticky
        failure (message deleted, bot kicked, no edit permission) and only that
        may retire the bubble.
        """
        if getattr(result, "retryable", False):
            return "transient"
        if getattr(result, "retry_after", None) is not None:
            return "flood"
        if is_flood_error(getattr(result, "error", "") or ""):
            return "flood"
        return "permanent"

    def on_flood(self, retry_after=None) -> float:
        """Record a flood hit; return how long to hold the rail, in seconds.

        Widening the steady-state interval is the half that actually ends the
        flood: honouring ``retry_after`` alone would clear the penalty and then
        walk straight back into it at the same over-budget cadence.
        """
        self.floods += 1
        self._interval = min(self.MAX_INTERVAL, self._interval * self.GROWTH)
        try:
            hold = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            hold = None
        if hold is None or hold <= 0:
            hold = max(self._interval, self.FALLBACK_HOLD)
        return min(hold, self.MAX_HOLD)
