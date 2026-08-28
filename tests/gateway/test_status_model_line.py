"""End-to-end: the line ``/status`` actually prints for the model.

``tests/gateway/test_status_model_truth.py`` pins the precedence as a pure
function. This file drives the real ``_handle_status_command`` so the wiring is
covered too — the 2026-08-10 defect was not in a resolver, it was in which value
the handler handed to the "Model:" heading.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_state import AsyncSessionDB

SESSION_ID = "20260806_192739_81b9e209"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003948593463",
        chat_name="Home",
        chat_type="group",
        user_id="42",
        user_name="Oleg",
    )


def _runner(*, db_model: str, config_model: str, provider: str = "copilot-acp"):
    from gateway.run import GatewayRunner

    source = _source()
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id=SESSION_ID,
        created_at=datetime(2026, 8, 6, 19, 27),
        updated_at=datetime(2026, 8, 9, 7, 2),
        origin=source,
        display_name="secretary",
        platform=Platform.TELEGRAM,
        chat_type="group",
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner.session_store = MagicMock()
    runner.session_store._entries = {entry.session_key: entry}
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.get_model_override.return_value = None
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner._queue_depth = MagicMock(return_value=0)
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._agent_cache_lock.__enter__ = MagicMock(return_value=None)
    runner._agent_cache_lock.__exit__ = MagicMock(return_value=False)
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = "secretary"
    runner._session_db._db.get_session.return_value = {
        "id": SESSION_ID,
        "model": db_model,
        "billing_provider": provider,
        "input_tokens": 10,
        "output_tokens": 20,
    }

    config = {"model": {"default": config_model, "provider": provider}}
    return runner, entry, config


async def _status(monkeypatch, *, db_model: str, config_model: str) -> str:
    runner, entry, config = _runner(db_model=db_model, config_model=config_model)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: config_model)
    # Off the bridge for this test: the engine lookup would otherwise resolve
    # against the developer's own profile config.
    monkeypatch.setattr(
        "gateway.status_model_truth.engine_model", lambda config_model="": ""
    )
    return await runner._handle_status_command(
        MessageEvent(text="/status", source=entry.origin, message_id="1")
    )


@pytest.mark.asyncio
async def test_status_reports_the_configured_model_not_the_last_turns(monkeypatch):
    """The secretary case: /status said opus while /model said sonnet."""
    out = await _status(
        monkeypatch, db_model="claude-opus-5", config_model="claude-sonnet-5"
    )
    model_line = next(l for l in out.splitlines() if l.startswith("**Model:**"))
    assert "claude-sonnet-5" in model_line
    assert "claude-opus-5" not in model_line


@pytest.mark.asyncio
async def test_the_previous_turns_model_gets_its_own_labelled_line(monkeypatch):
    out = await _status(
        monkeypatch, db_model="claude-opus-5", config_model="claude-sonnet-5"
    )
    assert "**Last turn answered on:** `claude-opus-5`" in out


@pytest.mark.asyncio
async def test_no_history_line_when_the_two_agree(monkeypatch):
    out = await _status(
        monkeypatch, db_model="claude-sonnet-5", config_model="claude-sonnet-5"
    )
    assert "Last turn answered on" not in out
    assert "**Model:** `claude-sonnet-5`" in out


@pytest.mark.asyncio
async def test_a_session_override_is_what_status_shows(monkeypatch):
    runner, entry, config = _runner(
        db_model="claude-opus-5", config_model="claude-sonnet-5"
    )
    runner._session_model_overrides[entry.session_key] = {
        "model": "claude-fable-5",
        "provider": "copilot-acp",
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: "claude-sonnet-5")
    monkeypatch.setattr(
        "gateway.status_model_truth.engine_model", lambda config_model="": ""
    )
    out = await runner._handle_status_command(
        MessageEvent(text="/status", source=entry.origin, message_id="1")
    )
    assert "**Model:** `claude-fable-5`" in out


@pytest.mark.asyncio
async def test_status_survives_a_broken_overlay(monkeypatch):
    """The overlay is a call-out; if it raises, /status must still answer."""
    runner, entry, config = _runner(
        db_model="claude-opus-5", config_model="claude-sonnet-5"
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: "claude-sonnet-5")

    def _boom(**kwargs):
        raise RuntimeError("overlay exploded")

    monkeypatch.setattr("gateway.status_model_truth.resolve", _boom)
    out = await runner._handle_status_command(
        MessageEvent(text="/status", source=entry.origin, message_id="1")
    )
    assert "Hermes Gateway Status" in out
    assert "**Model:**" in out
