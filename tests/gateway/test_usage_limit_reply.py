"""Plan/quota provider failures keep their reset time on the way to chat.

Regression for the 2026-07-28 researcher stall: the engine reported
"You've hit your session limit · resets 5:50pm (Europe/Moscow)" and chat
received the catch-all "check gateway logs for diagnostics" instead. The
reset time was the whole answer, and the sanitizer threw it away.
"""

import pytest

from gateway.config import Platform
from gateway.overlay_hooks import usage_limit_reply_safe as _gateway_usage_limit_reply
from gateway.run import (
    _gateway_provider_error_reply,
    _sanitize_gateway_final_response,
)

# The exact strings the ACP bridge produced, straight from the gateway logs.
SESSION_LIMIT_ERROR = (
    "API call failed after 3 retries: Copilot ACP session/prompt failed: "
    "Internal error: You've hit your session limit · resets 5:50pm (Europe/Moscow)"
)
WEEKLY_LIMIT_ERROR = (
    "API call failed after 3 retries: Copilot ACP session/prompt failed: "
    "Internal error: You've hit your weekly limit · resets Jul 25, 3pm (Europe/Moscow)"
)


@pytest.mark.parametrize(
    "raw, window, reset",
    [
        (SESSION_LIMIT_ERROR, "session", "5:50pm (Europe/Moscow)"),
        (WEEKLY_LIMIT_ERROR, "weekly", "Jul 25, 3pm (Europe/Moscow)"),
    ],
)
def test_quota_reply_carries_window_and_reset(raw, window, reset):
    reply = _gateway_provider_error_reply(raw)
    assert f"{window} limit reached" in reply
    assert reset in reply
    assert "gateway logs" not in reply


def test_quota_reply_survives_the_full_chat_sanitizer():
    """End-to-end through the path a real Telegram reply takes."""
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, SESSION_LIMIT_ERROR)
    assert "5:50pm (Europe/Moscow)" in sanitized
    # The raw provider envelope still must not reach chat.
    assert "Copilot ACP session/prompt failed" not in sanitized
    assert "Internal error" not in sanitized


def test_unknown_window_still_reports_the_limit():
    reply = _gateway_provider_error_reply(
        "API call failed after 3 retries: You've hit your usage limit"
    )
    assert "Provider usage limit reached" in reply
    # No reset time was stated, so none may be invented.
    assert "— resets" not in reply


def test_reset_clause_shaped_like_prose_is_dropped():
    """Only timestamp-shaped fragments are whitelisted out of the raw error."""
    reply = _gateway_provider_error_reply(
        "API call failed after 3 retries: You've hit your session limit; "
        "resets after we finish migrating account acct_88ff <internal-note>"
    )
    assert "session limit reached" in reply
    assert "internal-note" not in reply
    assert "acct_88ff" not in reply


def test_ordinary_rate_limit_keeps_its_own_advice():
    """A 429 is not a plan window — "wait a moment" stays correct for it."""
    reply = _gateway_provider_error_reply(
        "API call failed after 3 retries: HTTP 429 rate limit exceeded"
    )
    assert "rate-limiting" in reply
    assert "limit reached" not in reply


def test_auth_failure_is_not_mistaken_for_a_quota():
    reply = _gateway_provider_error_reply(
        "API call failed after 3 retries: HTTP 401 incorrect api key"
    )
    assert "authentication failed" in reply


@pytest.mark.parametrize(
    "text",
    [
        "Here is the summary you asked for.",
        "The context limit for this model is 200k tokens.",
        "I hit a wall on the parser, but the tests pass now.",
    ],
)
def test_ordinary_prose_is_not_a_quota_envelope(text):
    assert _gateway_usage_limit_reply(text) is None
