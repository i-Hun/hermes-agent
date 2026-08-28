"""The skill-nudge counter must survive the death of the agent object.

Regression suite for the 2026-08-10 finding on ``secretary``: no background
skill review since 07.08 while cron turns alone ran 54 + 73 + 28 engine tool
calls. Each of those turns was its own one-turn session and every interactive
turn had a gateway restart in front of it, so an object-local counter that only
ever saw the PREVIOUS turn's ledger could not get past ~9 of 20.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import acp_nudge


class FakeAgent:
    """Just the attributes the nudge accounting touches."""

    def __init__(self, provider="copilot-acp", interval=20, tools=("skill_manage",)):
        self.provider = provider
        self._skill_nudge_interval = interval
        self.valid_tool_names = list(tools)
        self._iters_since_skill = 0
        self._acp_native_tool_calls = []

    def run_engine_calls(self, n):
        self._acp_native_tool_calls.extend({"name": "Bash"} for _ in range(n))


def turn(agent, calls):
    """One ACP turn: loop start stamps the mark, engine works, turn finalizes."""
    agent._iters_since_skill += acp_nudge.skill_iterations_delta(agent)
    agent.run_engine_calls(calls)
    acp_nudge.finalize_turn(agent)
    return agent._iters_since_skill


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


def test_counter_file_lives_in_profile_state(tmp_path):
    agent = FakeAgent()
    turn(agent, 3)
    path = Path(tmp_path) / "state" / "skill-nudge.json"
    assert path.is_file()
    assert json.loads(path.read_text())["iters"] == 3


def test_single_turn_session_counts_its_own_calls():
    """The cron case: agent is built, does 54 calls, dies. That is 54, not 1."""
    cron = FakeAgent()
    assert turn(cron, 54) == 54
    assert acp_nudge.current_count() == 54


def test_counter_survives_agent_rebuild():
    """Three separate one-turn agents (gateway restart between each)."""
    for calls in (54, 73, 28):
        turn(FakeAgent(), calls)
    assert acp_nudge.current_count() == 155


def test_threshold_reached_across_sessions():
    """What the upstream check reads must cross the interval eventually."""
    a = FakeAgent()
    assert turn(a, 8) < a._skill_nudge_interval
    b = FakeAgent()
    assert turn(b, 15) >= b._skill_nudge_interval


def test_tool_free_turn_still_counts_one():
    agent = FakeAgent()
    assert turn(agent, 0) == 1
    assert turn(agent, 0) == 2


def test_warm_agent_does_not_double_count():
    """Two turns on one object count 5 + 7, not the ledger length twice."""
    agent = FakeAgent()
    assert turn(agent, 5) == 5
    assert turn(agent, 7) == 12


def test_async_review_calls_do_not_count_as_parent_work():
    """A review fork that reuses the parent's ACP client appends to the parent
    ledger AFTER the turn ended. Marking at turn start is what keeps the
    review's own work from re-triggering the review."""
    agent = FakeAgent()
    turn(agent, 4)
    agent.run_engine_calls(9)  # background review, between turns
    assert turn(agent, 2) == 6


def test_reset_clears_disk():
    agent = FakeAgent()
    turn(agent, 25)
    acp_nudge.note_reset(agent)
    assert acp_nudge.current_count() == 0
    # And the calls already booked are not counted a second time.
    assert turn(agent, 3) == 3


def test_non_acp_path_is_untouched(tmp_path):
    agent = FakeAgent(provider="anthropic")
    assert acp_nudge.skill_iterations_delta(agent) == 1
    agent.run_engine_calls(30)
    acp_nudge.finalize_turn(agent)
    acp_nudge.note_reset(agent)
    assert not (Path(tmp_path) / "state" / "skill-nudge.json").exists()


def test_no_accounting_without_skill_manage():
    """No tool to fire the review with — accumulating would only mean a burst
    of reviews the moment it reappears."""
    agent = FakeAgent(tools=("memory",))
    turn(agent, 12)
    assert acp_nudge.current_count() == 0


def test_review_fork_does_not_feed_its_own_counter():
    """The fork runs with interval 0 (background_review sets that)."""
    fork = FakeAgent(interval=0)
    turn(fork, 11)
    assert acp_nudge.current_count() == 0


def test_shrinking_ledger_is_not_negative():
    agent = FakeAgent()
    turn(agent, 6)
    agent._acp_native_tool_calls = []  # ledger replaced under us
    assert turn(agent, 2) == 8


def test_one_shot_process_defers_the_review(monkeypatch):
    """`hermes -z` / ask-hermes: the daemon review thread would die with the
    process, so the debt stays on disk for the next gateway turn."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    agent = FakeAgent()
    turn(agent, 30)
    assert acp_nudge.may_fire(agent) is False
    assert acp_nudge.current_count() == 30  # not spent


def test_gateway_process_may_fire(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    assert acp_nudge.may_fire(FakeAgent()) is True


def test_non_acp_may_fire_regardless_of_host(monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    assert acp_nudge.may_fire(FakeAgent(provider="anthropic")) is True


def test_corrupt_state_file_is_survivable(tmp_path):
    state = Path(tmp_path) / "state"
    state.mkdir(parents=True)
    (state / "skill-nudge.json").write_text("{not json")
    assert turn(FakeAgent(), 4) == 4
