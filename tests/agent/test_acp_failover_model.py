"""A failed primary must not answer as a model nobody chose.

Local overlay test (not upstream).

``_acp_failover_model`` used to end in a hardcoded ``claude-opus-4-8``. Nothing
in any profile named it: it was simply where every ``RuntimeError`` /
``TimeoutError`` landed. On 2026-08-02 that carried 82k output of the
``founder`` profile — pinned to ``claude-sonnet-5`` precisely to spend less —
onto opus twice, and the only trace was one WARNING line; the token audit found
it days later, not the operator.

So the default is now the primary model itself, which
``run_prompt_with_failover`` treats as "no distinct fallback" and declines to
retry. A profile that wants a genuine downgrade names it in its own ``.env``.

The subtle half is the CLI path: there ``HERMES_ACP_MODEL`` is unset and the
primary comes from what Hermes requested. If this function resolved the primary
on its own it would land on the legacy default, compare unequal, and fire the
downgrade anyway — so the caller passes the model it actually used.
"""

from __future__ import annotations

import os
import unittest

from agent import acp_failover
from agent.acp_failover import primary_attempt_timeout
from agent.copilot_acp_client import _acp_failover_model

_MODEL_VARS = ("HERMES_ACP_FAILOVER", "HERMES_ACP_FALLBACK_MODEL", "HERMES_ACP_MODEL")


class _EnvCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in _MODEL_VARS}
        for k in _MODEL_VARS:
            os.environ.pop(k, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestResolution(_EnvCase):
    def test_an_explicit_fallback_wins(self) -> None:
        os.environ["HERMES_ACP_FALLBACK_MODEL"] = "claude-opus-4-8"
        self.assertEqual(
            _acp_failover_model("claude-sonnet-5"), "claude-opus-4-8"
        )

    def test_unset_means_the_primary_itself(self) -> None:
        self.assertEqual(
            _acp_failover_model("claude-sonnet-5"), "claude-sonnet-5"
        )

    def test_unset_never_reaches_a_model_nobody_chose(self) -> None:
        """The regression this file exists for."""
        for primary in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5"):
            with self.subTest(primary=primary):
                self.assertNotEqual(
                    _acp_failover_model(primary), "claude-opus-4-8"
                )

    def test_without_a_primary_it_resolves_the_engine_model(self) -> None:
        os.environ["HERMES_ACP_MODEL"] = "claude-opus-5"
        self.assertEqual(_acp_failover_model(), "claude-opus-5")

    def test_blank_env_value_is_not_a_choice(self) -> None:
        os.environ["HERMES_ACP_FALLBACK_MODEL"] = "   "
        self.assertEqual(_acp_failover_model("claude-opus-5"), "claude-opus-5")


class TestPrimaryTimeoutCap(_EnvCase):
    def test_no_cap_when_the_fallback_is_the_primary(self) -> None:
        """Capping the primary with nowhere to go just shortens the turn."""
        self.assertEqual(
            primary_attempt_timeout(
                1800.0,
                primary_model="claude-fable-5",
                failover_model="claude-fable-5",
                failover_enabled=True,
            ),
            1800.0,
        )

    def test_cap_still_applies_to_a_real_distinct_fallback(self) -> None:
        capped = primary_attempt_timeout(
            1800.0,
            primary_model="claude-fable-5",
            failover_model="claude-opus-5",
            failover_enabled=True,
        )
        self.assertLess(capped, 1800.0)


class _ScriptedClient:
    """Drives the REAL run_prompt_with_failover over scripted attempts."""

    def __init__(self, scripts) -> None:
        self._scripts = list(scripts)
        self.attempts: list[str | None] = []
        self.pulses: list[tuple] = []
        self._activity_agent = None
        self._hermes_cancelled = None

    def _fire_engine_activity(self, event_type, tool_name=None, preview=None, args=None):
        self.pulses.append((event_type, tool_name, preview, args))

    def _run_prompt(self, prompt_text, *, timeout_seconds, model_override=None):
        self.attempts.append(model_override)
        script = self._scripts.pop(0)
        exc = script.get("raises")
        if exc is not None:
            raise exc
        return script.get("text", ""), ""


class TestThroughTheRealTurn(_EnvCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["HERMES_ACP_FAILOVER"] = "1"
        self.rows: list[dict] = []
        self._real_record = acp_failover.record_usage
        acp_failover.record_usage = lambda **kw: self.rows.append(kw)
        self.addCleanup(
            lambda: setattr(acp_failover, "record_usage", self._real_record)
        )

    def test_a_dead_primary_raises_instead_of_downgrading(self) -> None:
        os.environ["HERMES_ACP_MODEL"] = "claude-sonnet-5"
        client = _ScriptedClient([
            {"raises": RuntimeError("weekly limit")},
            {"text": "answered by the wrong model"},
        ])
        with self.assertRaises(RuntimeError):
            acp_failover.run_prompt_with_failover(
                client, "prompt", 60.0, "claude-sonnet-5",
            )
        self.assertEqual(len(client.attempts), 1)
        self.assertEqual(self.rows[-1]["used_model"], "claude-sonnet-5")
        self.assertFalse(self.rows[-1]["failover"])

    def test_an_explicitly_named_fallback_still_takes_over(self) -> None:
        os.environ["HERMES_ACP_MODEL"] = "claude-sonnet-5"
        os.environ["HERMES_ACP_FALLBACK_MODEL"] = "claude-opus-4-8"
        client = _ScriptedClient([
            {"raises": TimeoutError("wedged")},
            {"text": "готово"},
        ])
        answer, _rail, used, fired = acp_failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-sonnet-5",
        )
        self.assertTrue(fired)
        self.assertEqual(used, "claude-opus-4-8")
        self.assertEqual(client.attempts, [None, "claude-opus-4-8"])
        self.assertEqual(answer, "готово")

    def test_a_fallback_equal_to_the_primary_is_not_a_fallback(self) -> None:
        """founder's own .env since 2026-08-02: sonnet-5 pinned on both."""
        os.environ["HERMES_ACP_MODEL"] = "claude-sonnet-5"
        os.environ["HERMES_ACP_FALLBACK_MODEL"] = "claude-sonnet-5"
        client = _ScriptedClient([
            {"raises": TimeoutError("wedged")},
            {"text": "second attempt that must not happen"},
        ])
        with self.assertRaises(TimeoutError):
            acp_failover.run_prompt_with_failover(
                client, "prompt", 60.0, "claude-sonnet-5",
            )
        self.assertEqual(len(client.attempts), 1)

    def test_the_cli_path_compares_the_same_resolution(self) -> None:
        """HERMES_ACP_MODEL unset: the primary is what Hermes requested.

        Resolving the fallback independently here lands on the legacy default,
        which is *different*, so the downgrade fires — on exactly the path that
        already substituted models silently once (2026-07-28).
        """
        client = _ScriptedClient([
            {"raises": RuntimeError("weekly limit")},
            {"text": "answered by the wrong model"},
        ])
        with self.assertRaises(RuntimeError):
            acp_failover.run_prompt_with_failover(
                client, "prompt", 60.0, "claude-opus-5",
            )
        self.assertEqual(len(client.attempts), 1)


if __name__ == "__main__":
    unittest.main()
