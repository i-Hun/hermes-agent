"""``/status`` must answer "what runs next", never "what ran last time".

The incident these pin down: 2026-08-10, secretary profile. ``/status`` printed
``claude-opus-5`` and ``/model`` printed ``claude-sonnet-5`` in the same minute.
The sessions row still carried the model that answered on 2026-08-09 07:02; the
profile had moved to sonnet at 08:37, and no turn had passed in that thread
since. Nothing was broken except the heading.
"""

from __future__ import annotations

import pytest

from gateway import status_model_truth as truth
# Bound before the autouse fixture below swaps the module attribute, so the two
# tests that exercise the real helper get the real helper.
from gateway.status_model_truth import engine_model as real_engine_model


@pytest.fixture(autouse=True)
def _no_engine(monkeypatch):
    """Default the profile to "not the ACP bridge".

    :func:`truth.engine_model` reaches into ``agent.acp_config`` for the real
    profile, which would make every assertion depend on the developer's own
    ``config.yaml``. Tests that care about the bridge opt back in explicitly.
    """
    monkeypatch.setattr(truth, "engine_model", lambda config_model="": "")


# --- the incident ------------------------------------------------------------

def test_stale_db_row_does_not_become_current_model():
    out = truth.resolve(db_model="claude-opus-5", config_model="claude-sonnet-5")
    assert out["model"] == "claude-sonnet-5"
    assert out["source"] == "config"


def test_the_stale_value_is_kept_but_relabelled():
    """The record is the only trace of who answered; it just isn't the answer."""
    out = truth.resolve(db_model="claude-opus-5", config_model="claude-sonnet-5")
    assert out["last_turn"] == "claude-opus-5"


def test_agreeing_db_row_adds_no_second_line():
    out = truth.resolve(db_model="claude-sonnet-5", config_model="claude-sonnet-5")
    assert out["last_turn"] == ""


@pytest.mark.parametrize(
    "db, config",
    [
        ("claude-opus-5", "anthropic/claude-opus-5"),
        ("CLAUDE-OPUS-5", "claude-opus-5"),
        ("claude-opus-5", "copilot-acp/claude-opus-5"),
    ],
)
def test_route_and_case_differences_are_not_a_disagreement(db, config):
    """Otherwise the extra line fires on every /status, which trains it away."""
    out = truth.resolve(db_model=db, config_model=config)
    assert out["last_turn"] == ""


# --- precedence --------------------------------------------------------------

def test_running_agent_wins():
    """A turn in flight is not a prediction — it is the process answering."""
    out = truth.resolve(
        is_running=True,
        agent_model="claude-fable-5",
        agent_provider="copilot-acp",
        override={"model": "claude-opus-5"},
        config_model="claude-sonnet-5",
        db_model="claude-haiku-4-5-20251001",
    )
    assert (out["model"], out["source"]) == ("claude-fable-5", "running")


def test_a_running_turn_suppresses_the_history_line():
    """It is answering right now; "last turn" next to it reads as a conflict."""
    out = truth.resolve(
        is_running=True, agent_model="claude-fable-5", db_model="claude-opus-5"
    )
    assert out["last_turn"] == ""


def test_session_override_beats_config():
    out = truth.resolve(
        override={"model": "claude-opus-5", "provider": "copilot-acp"},
        config_model="claude-sonnet-5",
    )
    assert (out["model"], out["provider"]) == ("claude-opus-5", "copilot-acp")
    assert out["source"] == "session_override"


def test_idle_cached_agent_does_not_beat_config():
    """The warm object is a leftover from the previous turn.

    On an ACP profile it is not even the deciding one: resolve_model re-reads
    config.yaml on every engine spawn, so a config edit lands on the next turn
    while the cached agent still names the old model.
    """
    out = truth.resolve(
        is_running=False,
        agent_model="claude-opus-5",
        config_model="claude-sonnet-5",
    )
    assert (out["model"], out["source"]) == ("claude-sonnet-5", "config")


def test_empty_override_map_is_not_an_override():
    out = truth.resolve(override={}, config_model="claude-sonnet-5")
    assert out["source"] == "config"


def test_override_without_a_model_is_not_an_override():
    out = truth.resolve(override={"provider": "openai"}, config_model="claude-sonnet-5")
    assert out["source"] == "config"


def test_non_dict_override_is_ignored():
    out = truth.resolve(override="claude-opus-5", config_model="claude-sonnet-5")
    assert out["model"] == "claude-sonnet-5"


# --- last resort -------------------------------------------------------------

def test_falls_back_to_the_cached_agent_when_nothing_is_configured():
    out = truth.resolve(agent_model="claude-opus-5")
    assert (out["model"], out["source"]) == ("claude-opus-5", "fallback")


def test_falls_back_to_the_db_row_when_there_is_nothing_else():
    """Better than a blank line — and the relabelling drops out on its own."""
    out = truth.resolve(db_model="claude-opus-5")
    assert out["model"] == "claude-opus-5"
    assert out["last_turn"] == ""


def test_nothing_anywhere_yields_an_empty_line_not_a_crash():
    out = truth.resolve()
    assert out["model"] == ""


# --- the bridge --------------------------------------------------------------

def test_acp_profile_shows_what_the_engine_will_spawn(monkeypatch):
    """Asked of acp_config, so /status cannot drift from the real resolver."""
    seen = {}

    def _engine(config_model=""):
        seen["config_model"] = config_model
        return "claude-opus-5"

    monkeypatch.setattr(truth, "engine_model", _engine)
    out = truth.resolve(config_model="claude-opus-5", config_provider="copilot-acp")
    assert (out["model"], out["source"]) == ("claude-opus-5", "engine")
    assert seen["config_model"] == "claude-opus-5"


def test_engine_model_is_empty_off_the_bridge(monkeypatch):
    """The real helper, not the fixture: a non-ACP profile gets no engine answer."""
    monkeypatch.setattr("agent.acp_config.is_acp_config", lambda **kw: False)
    assert real_engine_model("claude-opus-5") == ""


def test_engine_model_swallows_a_resolver_failure(monkeypatch):
    """A broken config must cost /status its engine line, never the command."""
    def _boom(*a, **kw):
        raise RuntimeError("unparseable config.yaml")

    monkeypatch.setattr("agent.acp_config.is_acp_config", lambda **kw: True)
    monkeypatch.setattr("agent.acp_config.resolve_model", _boom)
    assert real_engine_model("claude-opus-5") == ""


def test_a_broken_engine_lookup_still_yields_the_config_model(monkeypatch):
    monkeypatch.setattr(truth, "engine_model", lambda config_model="": "")
    out = truth.resolve(config_model="claude-sonnet-5", config_provider="copilot-acp")
    assert (out["model"], out["source"]) == ("claude-sonnet-5", "config")
