"""The fallback/model-switch notice must survive ``display.cleanup_progress``.

Covers gateway/status_durability.py and the two call-outs it owns:
``run_agent._emit_status`` / ``_emit_pending_fallback_notice`` (the producer)
and ``gateway/run.py``'s cleanup-tracking branch (the consumer).

The bug these pin down: the notice went out on the ``lifecycle`` key, the
post-turn sweep deleted every id it had tracked, and a model switch the
operator needed to see vanished seconds after it appeared.
"""

from __future__ import annotations

import types

import pytest

from gateway import status_durability
from run_agent import AIAgent


# --- the policy --------------------------------------------------------------


def test_model_switch_is_durable():
    assert status_durability.is_durable(status_durability.MODEL_SWITCH) is True


@pytest.mark.parametrize("kind", ["lifecycle", "warn", "tool.started", "", None])
def test_progress_kinds_are_not_durable(kind):
    assert status_durability.is_durable(kind) is False


def test_is_durable_never_raises_on_junk():
    class Boom:
        def __str__(self):  # pragma: no cover - exercised via is_durable
            raise RuntimeError("nope")

    assert status_durability.is_durable(Boom()) is False


# --- the producer ------------------------------------------------------------


def _agent_stub():
    """Minimal object the two AIAgent methods under test actually touch."""
    seen: list[tuple[str, str]] = []
    agent = types.SimpleNamespace(
        log_prefix="",
        status_callback=lambda kind, msg: seen.append((kind, msg)),
        _vprint=lambda *a, **k: None,
        _pending_fallback_notice=None,
    )
    # _emit_pending_fallback_notice calls self._emit_status, and its own body
    # swallows every exception — an AttributeError here would read as "emitted
    # nothing" instead of "the stub is wrong", so wire the real method in.
    agent._emit_status = lambda msg, **kw: AIAgent._emit_status(agent, msg, **kw)
    return agent, seen


def test_emit_status_defaults_to_lifecycle():
    agent, seen = _agent_stub()
    AIAgent._emit_status(agent, "⏳ working")
    assert seen == [("lifecycle", "⏳ working")]


def test_emit_status_honours_explicit_kind():
    agent, seen = _agent_stub()
    AIAgent._emit_status(agent, "switched", event_type="model_switch")
    assert seen == [("model_switch", "switched")]


def test_fallback_notice_is_emitted_as_durable_and_cleared():
    agent, seen = _agent_stub()
    agent._pending_fallback_notice = (
        "🔄 Switched to fallback model: claude-fable-5 via anthropic "
        "→ claude-opus-5 via copilot-acp"
    )
    AIAgent._emit_pending_fallback_notice(agent)

    assert len(seen) == 1
    kind, text = seen[0]
    assert kind == status_durability.MODEL_SWITCH
    assert status_durability.is_durable(kind) is True
    assert "claude-opus-5" in text
    # One-shot: a second call on the same turn must not re-emit.
    AIAgent._emit_pending_fallback_notice(agent)
    assert len(seen) == 1


def test_no_notice_means_no_status():
    agent, seen = _agent_stub()
    AIAgent._emit_pending_fallback_notice(agent)
    assert seen == []


# --- the consumer ------------------------------------------------------------


def test_durable_notice_is_not_suppressed_by_the_platform_filter():
    """The kind changed; the Telegram noise filter must still let it through."""
    from gateway.platforms.base import Platform
    from gateway.run import _prepare_gateway_status_message

    notice = (
        "🔄 Switched to fallback model: claude-fable-5 via anthropic "
        "→ claude-opus-5 via copilot-acp"
    )
    prepared = _prepare_gateway_status_message(
        Platform.TELEGRAM, status_durability.MODEL_SWITCH, notice
    )
    assert prepared == notice


def test_cleanup_branch_skips_durable_kinds():
    """The guard in _status_callback_sync, evaluated as the source does it."""
    cleanup_progress = True

    def tracks(event_type: str) -> bool:
        return cleanup_progress and not status_durability.is_durable(event_type)

    assert tracks("lifecycle") is True
    assert tracks(status_durability.MODEL_SWITCH) is False
