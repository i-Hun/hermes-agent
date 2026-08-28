"""Boundary-aware excerpts for kanban notification pings (overlay).

Upstream builds the "task done" ping as ``first_line[:200]`` — a hard character
cut with no ellipsis. For a fleet whose workers hand off a paragraph (the
operator reads this ping AS the report, not as a pointer to the board) that
truncates mid-word:

    ... баннер «Повторить» вместо тихого «Списк

Two separate defects hide in that one line:

1. **The cut lands mid-word and mid-sentence** and is not marked, so the reader
   cannot tell a finished summary from a severed one.
2. **Only the first line survives.** A worker whose summary is one long
   paragraph loses everything past 200 chars; a worker whose summary is a
   headline plus detail lines loses the detail.

This module answers both with one function. It is deliberately *not* a general
"smart truncate": the contract is narrow — produce the text of a chat ping that
either says everything or visibly says "there is more".

Design notes:

* The limit is generous (900 chars by default) because the transport is a chat
  message (Telegram allows 4096) and the ping is the deliverable. It is still a
  ceiling: a worker that dumps its whole transcript into ``--summary`` must not
  flood the topic.
* The cut prefers a sentence end, then a word boundary, and only then falls back
  to a hard cut. Every truncated result ends with ``…`` — an unmarked cut is the
  bug being fixed, so a "prettier" cut without the marker would be a regression.
* ``HERMES_KANBAN_HANDOFF_CHARS`` overrides the limit at runtime. Parsing is
  fail-safe: garbage in the environment falls back to the default rather than
  raising inside a gateway background loop.
"""

from __future__ import annotations

import os

# Chat transports allow far more (Telegram: 4096), but the ping shares the
# message with the title and the id, and it is a ping — not a transcript.
DEFAULT_HANDOFF_CHARS = 900

# Write-side cap for the copy of the handoff carried in the ``completed`` event
# payload. Upstream stored ``first_line[:400]`` there, so the notifier could
# never deliver more than 400 characters no matter what budget it applied — the
# ping was starved before it was ever built, and the "first line only" rule
# discarded every detail line. MUST stay >= the ping budget; the full,
# untouched summary still lives on the ``task_runs`` row.
EVENT_PAYLOAD_CHARS = 1200

# Minimum room a limit must leave for the boundary search to be meaningful.
_MIN_LIMIT = 16

_SENTENCE_END = ".!?…"

# Trailing punctuation that reads as broken when an ellipsis follows it.
_DANGLING = " \t,;:—–-("


def handoff_limit(default: int = DEFAULT_HANDOFF_CHARS) -> int:
    """Resolve the ping character budget, honouring the env override.

    Fail-safe by construction: any unparseable or non-positive value yields the
    default. This runs inside the notifier's background loop, where raising
    would cost a delivery, not just a wrong number.
    """
    raw = os.environ.get("HERMES_KANBAN_HANDOFF_CHARS")
    if not raw:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def handoff_excerpt(text: object, limit: int | None = None) -> str:
    """Excerpt a worker handoff for a one-message notification.

    Keeps the whole summary (all of its lines) when it fits; otherwise cuts on
    the nearest sentence end, else the nearest word boundary, and marks the cut
    with ``…``.

    Returns ``""`` for empty/whitespace/``None`` input so the caller's
    ``if summary:`` guards keep working unchanged.
    """
    if text is None:
        return ""
    body = str(text).strip()
    if not body:
        return ""

    cap = handoff_limit() if limit is None else limit
    if cap < _MIN_LIMIT:
        cap = _MIN_LIMIT
    if len(body) <= cap:
        return body

    window = body[:cap]
    # Only accept a boundary in the tail of the window; a sentence end at 5% of
    # the budget would throw away almost everything the worker wrote.
    floor = max(1, int(cap * 0.6))

    cut = -1
    for i in range(len(window) - 1, floor - 1, -1):
        if window[i] in _SENTENCE_END:
            cut = i + 1
            break
    if cut < 0:
        space = max(window.rfind(" ", floor), window.rfind("\n", floor))
        cut = space if space > floor else cap

    excerpt = window[:cut].rstrip(_DANGLING)
    if not excerpt:  # pathological: the whole window was punctuation
        excerpt = window.rstrip()
    return excerpt + "…"


def handoff_payload_excerpt(text: object) -> str:
    """Excerpt for the copy carried in a task event payload.

    Wider than the ping budget on purpose: the payload is what every consumer
    (gateway notifier, dashboard WS, TUI) renders from, so a cap below the ping
    budget silently caps the ping too — which is exactly how a 900-char budget
    still delivered 400 characters.
    """
    return handoff_excerpt(text, limit=max(EVENT_PAYLOAD_CHARS, handoff_limit()))
