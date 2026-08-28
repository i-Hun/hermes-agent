"""Regression tests for the ACP native-tools MCP bridge exposure list.

The bridge must expose skill_manage: it is the self-improvement path (the
system prompt tells the model to save skills with it), and the upstream
EXPOSED_TOOLS list omits it. Regression for the native-tools rollout that
silently dropped skill creation (2026-07-20 .. 2026-07-21).
"""

import json
import unittest

from agent.transports import hermes_acp_tools_server as bridge


def test_extra_tools_contains_skill_manage():
    assert "skill_manage" in bridge.EXTRA_TOOLS


def test_fallback_base_tools_contains_skill_manage():
    assert "skill_manage" in bridge._FALLBACK_BASE_TOOLS


def test_loop_tools_complete():
    assert set(bridge.LOOP_TOOLS) == {
        "memory", "session_search", "todo", "delegate_task"
    }


def test_exposure_list_dedupes_and_includes_all_layers():
    """Recompute the exposure list the way _build_server does (without the
    mcp dependency) and assert every layer survives, with no duplicates."""
    from model_tools import get_tool_definitions

    defs = {
        td["function"]["name"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }
    names, seen = [], set()
    for n in bridge._base_exposed_tools() + bridge.EXTRA_TOOLS + bridge.LOOP_TOOLS:
        if n in seen or n not in defs:
            continue
        seen.add(n)
        names.append(n)

    assert len(names) == len(set(names))
    assert "skill_manage" in names
    for loop_tool in bridge.LOOP_TOOLS:
        assert loop_tool in names, loop_tool


class RepeatGuardTests(unittest.TestCase):
    """A blind retry loop must be interrupted with an explanation.

    Hermes' own loop guard lives in the tool executor, which the engine
    bypasses when it calls these tools over MCP — so without this a stuck
    model repeats one call indefinitely (observed: eight identical attempts
    at the same blocked write).
    """

    def setUp(self) -> None:
        from agent.transports import hermes_acp_tools_server as srv

        self.srv = srv
        srv._repeat_state["key"] = None
        srv._repeat_state["count"] = 0

    def test_identical_calls_are_allowed_up_to_the_limit(self) -> None:
        args = {"action": "create", "name": "x"}
        for _ in range(self.srv._REPEAT_LIMIT):
            self.assertIsNone(self.srv._repeat_warning("skill_manage", args))

    def test_repeat_past_limit_returns_actionable_message(self) -> None:
        args = {"action": "create", "name": "x"}
        for _ in range(self.srv._REPEAT_LIMIT):
            self.srv._repeat_warning("skill_manage", args)
        warning = self.srv._repeat_warning("skill_manage", args)
        self.assertIsNotNone(warning)
        payload = json.loads(warning)
        self.assertEqual(payload["error"], "repeated_call")
        # The message must tell the model what to do instead, otherwise it
        # just retries again — the failure mode this exists to stop.
        self.assertIn("change the arguments", payload["message"])

    def test_a_different_call_clears_the_counter(self) -> None:
        args = {"action": "create", "name": "x"}
        for _ in range(self.srv._REPEAT_LIMIT + 1):
            self.srv._repeat_warning("skill_manage", args)
        self.assertIsNone(self.srv._repeat_warning("memory", {"q": "y"}))
        self.assertIsNone(self.srv._repeat_warning("skill_manage", args))

    def test_same_tool_different_arguments_is_not_a_repeat(self) -> None:
        for i in range(self.srv._REPEAT_LIMIT + 2):
            self.assertIsNone(
                self.srv._repeat_warning("memory", {"note": f"entry {i}"})
            )
