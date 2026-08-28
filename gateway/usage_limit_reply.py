"""ACP-bridge overlay: surface plan/quota (usage-limit) reset times to chat.

Rationale (Координатор, 2026-07-28 — researcher stall at 17:03 MSK):
The gateway sanitizer collapses every provider/API failure envelope into the
catch-all "The model provider failed after retries … check gateway logs for
diagnostics." That is right for genuinely opaque errors, but it also swallowed
the one class where the raw text carried the whole answer — a hit **usage
limit** and its reset time:

    API call failed after 3 retries: Copilot ACP session/prompt failed:
    Internal error: You've hit your session limit · resets 5:50pm (Europe/Moscow)

The user got "go read the logs" when the honest answer was "wait 47 minutes".
This module recognises session / weekly / generic usage-limit envelopes and
surfaces the reset time — but only when it is genuinely timestamp-shaped, so a
prose "resets after we finish migrating account …" fragment is dropped rather
than leaked. The caller has already run ``_redact_gateway_user_facing_secrets``
before we see the text, and the reset clause carries only a time/zone, so the
output is safe for chat.

Kept out of ``gateway/run.py`` (upstream file) so upstream churn there cannot
conflict with this logic — same convention as ``gateway/progress_collapse.py``.
The thin ``_gateway_usage_limit_reply`` hook in run.py delegates here.
"""

from __future__ import annotations

import re
from typing import Optional

# Window classifiers, most specific first (session/weekly are named windows;
# "usage limit" / "hit your … limit" / "quota" are the generic fallback).
_SESSION_RE = re.compile(r"\bsession\s+limit\b", re.IGNORECASE)
_WEEKLY_RE = re.compile(r"\b(?:weekly|week)\s+limit\b", re.IGNORECASE)
_GENERIC_QUOTA_RE = re.compile(
    r"\busage\s+limit\b"
    r"|\bplan\s+limit\b"
    r"|\bquota\b"
    r"|\b(?:hit|reach(?:ed|ing)?|exceed(?:ed|ing)?)\s+your\s+(?:\w+\s+){0,2}limit\b",
    re.IGNORECASE,
)

# The "resets …" clause, bounded to one sentence/segment.
_RESET_CLAUSE_RE = re.compile(r"reset[s]?\b[:\s]*([^\n.;·|]*)", re.IGNORECASE)
# A reset value is only surfaced when it *starts* with a real timestamp/date —
# never arbitrary prose (which could smuggle account ids, notes, secrets).
_TIMESTAMP_HEAD_RE = re.compile(
    r"^("
    r"(?:mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d"
    r"|\d{1,2}\s*:\s*\d{2}"
    r"|\d{1,2}\s*(?:am|pm)\b"
    r"|tomorrow\b|today\b"
    r")",
    re.IGNORECASE,
)

_TAIL = (
    "The rolling usage window frees up on its own — retry once it resets; "
    "no action needed."
)


def _reset_suffix(body: str) -> str:
    """Return ' — <timestamp>' for a timestamp-shaped reset clause, else ''."""
    m = _RESET_CLAUSE_RE.search(body)
    if not m:
        return ""
    clause = " ".join(m.group(1).split()).rstrip(" .;,·-")
    # Drop a leading "at " so the suffix reads cleanly ("at 3pm" -> "3pm").
    clause = re.sub(r"^at\s+", "", clause, flags=re.IGNORECASE)
    if not clause or not _TIMESTAMP_HEAD_RE.match(clause):
        return ""
    return f" — {clause}"


def usage_limit_reply(text: str) -> Optional[str]:
    """Friendly, actionable reply for a plan/quota (usage-limit) envelope.

    Returns ``None`` when the text is not a usage-limit error, so the caller
    falls through to its normal provider-error handling. Self-sufficient: it
    matches only genuine "<window> limit" quota phrases, so ordinary prose that
    merely mentions a "context limit" or "hit a wall" is never rewritten.
    """
    if not text:
        return None
    body = str(text)

    if _SESSION_RE.search(body):
        head = "⏱️ Model session limit reached"
    elif _WEEKLY_RE.search(body):
        head = "⏱️ Model weekly limit reached"
    elif _GENERIC_QUOTA_RE.search(body):
        head = "⏱️ Provider usage limit reached"
    else:
        return None

    return f"{head}{_reset_suffix(body)}. {_TAIL}"
