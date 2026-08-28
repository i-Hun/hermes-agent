"""Tests for the progress-bubble flood budget (gateway/progress_flood.py).

Regression: 2026-07-29, secretary/omnirecs run 16:25–16:32. A tool-heavy run
arrived in Telegram as one ⚙️ line per minute instead of a single collapsing
bubble. Root cause was two-layered:

* ``gateway/run.py`` answered a 429 on the progress edit by SENDING the line
  as its own message — every flood cycle spawned a fresh bubble;
* the backoff ignored the server's ``retry_after`` and re-armed the same 1.5s
  cadence, so the run walked straight back into flood and stayed there.

The end-to-end proof that no replacement bubble is sent lives next to the
harness that can drive the real progress loop:
``tests/gateway/test_run_progress_topics.py::
test_flood_controlled_progress_edit_does_not_fragment_into_new_messages``.
"""

from __future__ import annotations

import pytest

from gateway.platforms.base import SendResult
from gateway.progress_flood import (
    ProgressFloodBudget,
    flood_retry_after,
    is_flood_error,
)


class _RetryAfterError(Exception):
    """Stands in for python-telegram-bot's RetryAfter."""

    def __init__(self, seconds):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


# ---------------------------------------------------------------------------
# 1. Flood detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obj", [
    "Flood control exceeded. Retry in 27 seconds",
    "flood_control:27.0",
    "Too Many Requests: retry after 30",
    "RetryAfter: retry after 5",
])
def test_flood_strings_are_detected(obj):
    assert is_flood_error(obj) is True


def test_retry_after_attribute_wins_over_wording():
    """Structure beats substrings: the server is free to reword its message."""
    exc = _RetryAfterError(27)
    exc.args = ("some future wording nobody greps for",)
    assert is_flood_error(exc) is True
    assert flood_retry_after(exc) == 27.0


@pytest.mark.parametrize("obj", [
    "Bad Request: message to edit not found",
    "Bad Request: message can't be edited",
    "Bad Request: not enough rights to edit the message",
    "Bad Request: can't parse entities: unexpected end of input",
    "Forbidden: bot was blocked by the user",
    "httpx.ConnectError: Connection refused",
    "",
    None,
])
def test_non_flood_errors_are_not_flood(obj):
    assert is_flood_error(obj) is False


@pytest.mark.parametrize("text,expected", [
    ("Flood control exceeded. Retry in 27 seconds", 27.0),
    ("flood_control:27.0", 27.0),
    ("Too Many Requests: retry after 30", 30.0),
    ("message to edit not found", None),
    ("", None),
])
def test_retry_after_parsing(text, expected):
    assert flood_retry_after(text) == expected


def test_zero_retry_after_is_treated_as_unstated():
    """A 0 from the server is not a delay — the caller must pick a floor."""
    assert flood_retry_after("retry after 0") is None


# ---------------------------------------------------------------------------
# 2. Failure classification
# ---------------------------------------------------------------------------

def test_transient_beats_everything():
    r = SendResult(success=False, error="httpx.ConnectError", retryable=True)
    assert ProgressFloodBudget().classify(r) == "transient"


def test_flood_from_retry_after_field():
    r = SendResult(success=False, error="flood_control:27.0", retry_after=27.0)
    assert ProgressFloodBudget().classify(r) == "flood"


def test_flood_from_error_string_only():
    """The adapter does not always attach retry_after; the string must do."""
    r = SendResult(success=False, error="Flood control exceeded. Retry in 27 seconds")
    assert ProgressFloodBudget().classify(r) == "flood"


@pytest.mark.parametrize("err", [
    "Bad Request: message to edit not found",
    "Bad Request: not enough rights to edit the message",
    "Forbidden: bot was blocked by the user",
])
def test_permanent_failures_stay_permanent(err):
    """Only these may retire the bubble — the regression was flood landing here."""
    assert ProgressFloodBudget().classify(SendResult(success=False, error=err)) == "permanent"


# ---------------------------------------------------------------------------
# 3. Backoff + adaptive cadence
# ---------------------------------------------------------------------------

def test_starts_at_the_upstream_cadence():
    assert ProgressFloodBudget().interval == 1.5


def test_server_retry_after_is_honoured():
    b = ProgressFloodBudget()
    assert b.on_flood(27.0) == 27.0


def test_cadence_widens_on_every_flood_and_caps():
    """Honouring retry_after alone would clear the penalty, then re-enter it."""
    b = ProgressFloodBudget()
    seen = []
    for _ in range(6):
        b.on_flood(1.0)
        seen.append(b.interval)
    assert seen[:3] == [3.0, 6.0, 12.0]
    assert all(v == b.MAX_INTERVAL for v in seen[2:])
    assert b.floods == 6


def test_unstated_delay_falls_back_to_a_floor():
    b = ProgressFloodBudget()
    assert b.on_flood(None) == max(b.interval, b.FALLBACK_HOLD)
    assert b.on_flood("") == max(b.interval, b.FALLBACK_HOLD)


def test_absurd_retry_after_is_clamped():
    """A stuck rail is worse than a late one — never park longer than MAX_HOLD."""
    b = ProgressFloodBudget()
    assert b.on_flood(9999.0) == b.MAX_HOLD


def test_garbage_retry_after_does_not_crash_the_rail():
    b = ProgressFloodBudget()
    assert b.on_flood("not-a-number") == max(b.interval, b.FALLBACK_HOLD)


def test_a_fresh_run_starts_from_the_base_again():
    """The widened cadence is per-run state, not a global penalty."""
    b = ProgressFloodBudget()
    b.on_flood(1.0)
    assert b.interval == 3.0
    assert ProgressFloodBudget().interval == 1.5


# ===========================================================================
# 5. The rail, end to end.
#
# Moved here on 2026-08-04 from two upstream test files. They lived there
# because that is where the harness is — and that is exactly why v2026.8.3
# conflicted on both. Nothing is lost: the harness is imported, not copied.
# ===========================================================================

from gateway.platforms.base import Platform  # noqa: E402


class FloodControlledEditProgressAdapter:
    """Answer the first progress edit with Telegram flood control (429).

    Mirrors what the live adapter returns once the per-chat edit budget is
    spent: ``success=False`` with a server-stated ``retry_after``. The delay is
    small so the test stays fast; the code path is identical.

    Built as a mixin over the upstream capture adapter at import time (see
    ``_flood_adapter_cls``) so this file does not re-implement the harness.
    """


def _flood_adapter_cls():
    from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter

    class _Adapter(ProgressCaptureAdapter):
        def __init__(self, platform=Platform.TELEGRAM):
            super().__init__(platform=platform)
            self.flood_edits = 0

        async def edit_message(self, chat_id, message_id, content):
            self.edits.append(
                {"chat_id": chat_id, "message_id": message_id, "content": content}
            )
            if self.flood_edits == 0:
                self.flood_edits += 1
                return SendResult(
                    success=False, error="flood_control:0.2", retry_after=0.2
                )
            return SendResult(success=True, message_id=message_id)

    return _Adapter


@pytest.mark.asyncio
async def test_flood_controlled_progress_edit_does_not_fragment_into_new_messages(
    monkeypatch, tmp_path
):
    """A 429 on the progress edit must hold the bubble, not spawn a new one.

    Regression 2026-07-29 (secretary/omnirecs): flood control was answered by
    sending the tool line as its own message, so a long run arrived as one ⚙️
    line per minute instead of a single collapsing bubble. The line is already
    accumulated in the bubble, so holding loses nothing — and sending spends
    the very budget that is exhausted.
    """
    from tests.gateway.test_run_progress_topics import (
        RetryableEditProgressAgent,
        _run_with_agent,
    )

    adapter_cls = _flood_adapter_cls()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        RetryableEditProgressAgent,
        session_id="sess-progress-flood-no-fragment",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
            }
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=adapter_cls,
    )

    assert result["final_response"] == "done"
    assert adapter.flood_edits == 1, "the fake adapter never got to flood"
    # The regression, stated as an artifact: exactly one progress bubble was
    # ever sent. Before the fix this was 2+ — one per flood cycle.
    assert len(adapter.sent) == 1
    # And every edit kept aiming at that same bubble.
    assert {call["message_id"] for call in adapter.edits} == {"progress-1"}
    # The line whose edit was flood-blocked is not lost: it stays accumulated
    # and rides along on a later edit of the same bubble.
    assert any("fourth command" in call["content"] for call in adapter.edits)


# ---------------------------------------------------------------------------
# 6. can_edit, decided by the REAL classifier.
#
# Upstream's tests/gateway/test_telegram_progress_edit_transient.py runs the
# same scenarios against a hand-copied if/else, which is why its
# ``test_flood_control_sets_can_edit_false`` still passes while describing a
# gateway we no longer run (the root conftest deselects it and says so).
# ---------------------------------------------------------------------------

def _decide_can_edit(edit_results):
    """Decide can_edit for a sequence of edit_message results.

    Uses the real classifier gateway/run.py calls, so this cannot drift from
    run.py the way a hand-copied branch would.
    """
    budget = ProgressFloodBudget()
    can_edit = True
    for result in edit_results:
        if not result.success:
            kind = budget.classify(result)
            if kind == "transient":
                continue
            if kind == "flood":
                budget.on_flood(getattr(result, "retry_after", None))
                continue
            can_edit = False
            break
    return can_edit


def test_permanent_failure_retires_the_bubble():
    results = [SendResult(success=False, error="message to edit not found")]
    assert _decide_can_edit(results) is False


def test_transient_failure_keeps_the_bubble():
    results = [
        SendResult(success=False, error="ConnectError", retryable=True),
        SendResult(success=True, message_id="42"),
    ]
    assert _decide_can_edit(results) is True


def test_flood_control_keeps_the_bubble_editable():
    """Flood control is a delay, not a broken bubble.

    Retiring the bubble on 429 is what made a tool-heavy run arrive as one
    message per minute (regression 2026-07-29). The no-fragmentation proof is
    ``test_flood_controlled_progress_edit_does_not_fragment_into_new_messages``
    above.
    """
    results = [
        SendResult(success=False, error="flood_control:30.0", retryable=False),
        SendResult(success=True, message_id="42"),
    ]
    assert _decide_can_edit(results) is True


# ---------------------------------------------------------------------------
# 7. Adapter: a rate limit is not a formatting failure.
#
# The finalize path used to catch EVERY exception from the MarkdownV2 edit and
# retry the same edit as plain text. On a 429 that costs a second request out
# of the exhausted budget (deepening the penalty) and strips the formatting of
# a perfectly good bubble. Flood must reach the outer handler untouched.
# ---------------------------------------------------------------------------

def _telegram_adapter():
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_flood_on_markdown_edit_is_not_retried_as_plain_text():
    """One request per flood, and the bubble keeps its formatting."""
    from unittest.mock import AsyncMock

    adapter = _telegram_adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=_RetryAfterError(27))

    result = await adapter.edit_message("-1001", "42", "hello", finalize=True)

    assert result.success is False
    assert result.retry_after == 27
    # Exactly one attempt: no plain-text retry burning the same budget.
    assert adapter._bot.edit_message_text.await_count == 1


@pytest.mark.asyncio
async def test_real_markdown_failure_still_falls_back_to_plain_text():
    """The fallback must survive for the case it was actually written for."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    adapter = _telegram_adapter()
    calls = []

    async def _edit(**kwargs):
        calls.append(kwargs)
        if kwargs.get("parse_mode") is not None:
            raise Exception("Bad Request: can't parse entities: unexpected end of input")
        return SimpleNamespace(message_id=42)

    adapter._bot.edit_message_text = AsyncMock(side_effect=_edit)

    result = await adapter.edit_message("-1001", "42", "*broken", finalize=True)

    assert result.success is True
    assert len(calls) == 2
    assert calls[1].get("parse_mode") is None
