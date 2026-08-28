"""A tool event is not proof that the assistant's text block ended.

Regression suite for the fragmented-message bug (reported 2026-07-30): one
narration arrived in the Telegram progress bubble as two 💬 lines split
mid-word ("Ошибка окружения (н" … "ет Airflow Variables локально)") with tool
lines between them.

``claude-agent-acp`` surfaces a ``tool_call`` from the permission path *before*
the assistant message's tool_use block streams (its own comment in
``acp-agent.js``: "The SDK may invoke `canUseTool` … before the assistant
message's tool_use block streams to us"), and emits async ``tool_call_update``s
(``tool_progress`` keep-alives, terminal output/exit) that can land between two
text deltas. Flushing narration on those events cut a sentence in half — and
when the late update followed the final answer, it moved the reply itself out of
the answer and into the hidden reasoning rail.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.acp_session_update import handle_session_update  # noqa: E402
from gateway.progress_collapse import ProgressCollapser  # noqa: E402

M1 = "msg_011AAA"
M2 = "msg_011BBB"


class _FakeClient:
    """Just enough client for the overlay's narration bookkeeping."""

    def __init__(self) -> None:
        self.pulses: list[tuple] = []
        self.ledger: list[tuple[str, str]] = []

    def _record_native_tool_call(self, update: dict, kind: str) -> None:
        self.ledger.append((kind, str(update.get("toolCallId") or "")))

    def _fire_engine_activity(self, event_type, tool_name=None, preview=None, args=None):
        self.pulses.append((event_type, tool_name, preview, args))

    # convenience
    @property
    def narration_pulses(self) -> list[str]:
        return [p[1] for p in self.pulses if p[0] == "_thinking"]


class _Turn:
    """One ACP prompt: the buffers the client returns text/reasoning from."""

    def __init__(self) -> None:
        self.client = _FakeClient()
        self.text: list[str] = []
        self.reasoning: list[str] = []

    def send(self, update: dict) -> None:
        handle_session_update(self.client, update, self.text, self.reasoning)

    def chunk(self, text: str, message_id: str | None = None) -> None:
        u: dict = {"sessionUpdate": "agent_message_chunk", "content": {"text": text}}
        if message_id:
            u["messageId"] = message_id
        self.send(u)

    def tool_call(self, tool_id: str, title: str = "Bash", **raw) -> None:
        self.send({
            "sessionUpdate": "tool_call",
            "toolCallId": tool_id,
            "title": title,
            "rawInput": raw or {"command": "true"},
        })

    def tool_progress(self, tool_id: str) -> None:
        # What the engine sends while a tool runs: status only, no rawInput.
        self.send({
            "sessionUpdate": "tool_call_update",
            "toolCallId": tool_id,
            "status": "in_progress",
        })

    @property
    def answer(self) -> str:
        return "".join(self.text)

    @property
    def rail(self) -> str:
        return "".join(self.reasoning)


class TestNarrationBoundary(unittest.TestCase):
    # ── the reported case ────────────────────────────────────────────────
    def test_permission_tool_call_mid_stream_does_not_split_narration(self) -> None:
        t = _Turn()
        t.chunk("Ошибка окружения (н", M1)
        # canUseTool fired before the tool_use block finished streaming.
        t.tool_call("t1")
        t.chunk("ет Airflow Variables локально), не моей правки.", M1)
        # The streamed tool_use block finally arrives → refining update.
        t.tool_progress("t1")
        # Next assistant message: the real boundary.
        t.chunk("Готово.", M2)

        whole = "Ошибка окружения (нет Airflow Variables локально), не моей правки."
        self.assertIn(whole, t.rail)
        self.assertEqual(t.answer, "Готово.")
        # No half-sentence anywhere: the partial never became its own entry.
        self.assertNotIn("(н\n\n", t.rail)
        self.assertEqual(t.rail.count("Ошибка окружения"), 1)
        # The last live pulse carries the whole sentence.
        self.assertTrue(t.client.narration_pulses)
        self.assertEqual(t.client.narration_pulses[-1], whole)

    def test_async_update_after_final_answer_keeps_the_reply(self) -> None:
        # A tool_progress / terminal_exit update landing after the reply began
        # used to move the reply into the reasoning rail — the answer arrived
        # truncated or empty.
        t = _Turn()
        t.chunk("Разобрал: ", M2)
        t.tool_progress("t1")          # late keep-alive for an earlier tool
        t.chunk("причина в кэше.", M2)
        self.assertEqual(t.answer, "Разобрал: причина в кэше.")
        self.assertNotIn("Разобрал", t.rail)

    def test_answer_survives_a_trailing_tool_event(self) -> None:
        t = _Turn()
        t.chunk("Финальный ответ.", M2)
        t.tool_progress("t1")
        self.assertEqual(t.answer, "Финальный ответ.")

    # ── the behaviour that must NOT change ───────────────────────────────
    def test_real_boundary_still_reclassifies_narration(self) -> None:
        t = _Turn()
        t.chunk("Сначала разведка.", M1)
        t.tool_call("t1")
        t.chunk("Финальный ответ.", M2)
        self.assertEqual(t.answer, "Финальный ответ.")
        self.assertIn("Сначала разведка.", t.rail)
        self.assertNotIn("Сначала разведка.", t.answer)

    def test_narration_keeps_its_place_ahead_of_the_tool_label(self) -> None:
        t = _Turn()
        t.chunk("Проверяю конфиг.", M1)
        t.tool_call("t1", title="Read", path="/tmp/x")
        t.chunk("Готово.", M2)
        self.assertLess(t.rail.index("Проверяю конфиг."), t.rail.index("⚙️"))

    def test_engine_without_message_ids_still_flushes_on_boundary(self) -> None:
        # Older engines (and synthetic chunks such as "Compacting…") carry no
        # messageId; without that evidence a tool event stays a boundary.
        t = _Turn()
        t.chunk("Narration.")
        t.tool_call("t1")
        t.chunk("Answer.")
        self.assertEqual(t.answer, "Answer.")
        self.assertIn("Narration.", t.rail)

    def test_repeated_boundaries_pulse_once_per_growth(self) -> None:
        t = _Turn()
        t.chunk("Работаю.", M1)
        t.tool_call("t1")
        t.tool_progress("t1")
        t.tool_progress("t1")
        self.assertEqual(t.client.narration_pulses, ["Работаю."])

    def test_ledger_and_tool_pulses_unaffected(self) -> None:
        t = _Turn()
        t.chunk("Смотрю.", M1)
        t.tool_call("t1", title="Read", path="/tmp/a")
        t.tool_call("t2", title="Read", path="/tmp/b")
        self.assertEqual([k for k, _ in t.client.ledger], ["tool_call", "tool_call"])
        self.assertEqual(
            [p[0] for p in t.client.pulses],
            ["_thinking", "tool.started", "tool.started"],
        )


class TestNarrationCarriedAcrossAMessageSplit(unittest.TestCase):
    """The other half of the same report: the split is REAL, the sentence isn't.

    The model closes its API message at the tool-call boundary and continues the
    same sentence in the next one, under a new ``messageId``. Layer 2 must commit
    the run (it really ended) yet pulse the ACCUMULATED text, so the collapser's
    prefix test recognises a continuation instead of stacking a suffix line.
    """

    def test_mid_sentence_split_pulses_the_whole_utterance(self) -> None:
        t = _Turn()
        t.chunk("Ошибка окружения (н", M1)
        t.tool_call("t1")
        t.chunk("ет Airflow Variables локально), не моей правки.", M2)

        whole = "Ошибка окружения (нет Airflow Variables локально), не моей правки."
        self.assertEqual(t.client.narration_pulses[-1], "Ошибка окружения (н")
        # …until the continuation lands, which supersedes it as a prefix growth.
        t.tool_call("t2")
        self.assertEqual(t.client.narration_pulses[-1], whole)
        self.assertTrue(
            t.client.narration_pulses[-1].startswith(t.client.narration_pulses[0]),
            "the grown pulse must be a prefix superset or layer 3 cannot merge it",
        )

    def test_mid_sentence_split_leaves_one_paragraph_on_the_rail(self) -> None:
        t = _Turn()
        t.chunk("Ошибка окружения (н", M1)
        t.tool_call("t1")
        t.chunk("ет Airflow Variables локально), не моей правки.", M2)
        t.tool_call("t2")
        t.chunk("Готово.", "msg_011CCC")

        whole = "Ошибка окружения (нет Airflow Variables локально), не моей правки."
        self.assertIn(whole, t.rail)
        self.assertEqual(t.rail.count("Ошибка окружения"), 1)
        self.assertNotIn("(н\n\n", t.rail)
        self.assertEqual(t.answer, "Готово.")
        # The whole sentence keeps the prefix's place: ahead of its tool label.
        self.assertLess(t.rail.index(whole), t.rail.index("⚙️"))

    def test_split_on_a_space_does_not_glue_the_words(self) -> None:
        t = _Turn()
        t.chunk("Проверяю конфиг ", M1)
        t.tool_call("t1")
        t.chunk("и логи гейтвея", M2)
        t.tool_call("t2")
        self.assertEqual(t.client.narration_pulses[-1], "Проверяю конфиг и логи гейтвея")

    def test_three_way_split_keeps_accumulating(self) -> None:
        t = _Turn()
        t.chunk("Считаю карточки", M1)
        t.tool_call("t1")
        t.chunk(", потом воркеров", M2)
        t.tool_call("t2")
        t.chunk(", потом отчёт.", "msg_011CCC")
        t.tool_call("t3")
        self.assertEqual(
            t.client.narration_pulses[-1],
            "Считаю карточки, потом воркеров, потом отчёт.",
        )
        self.assertEqual(t.rail.count("Считаю карточки"), 1)

    # ── what must NOT be glued together ──────────────────────────────────
    def test_finished_sentence_starts_a_new_entry(self) -> None:
        t = _Turn()
        t.chunk("Сначала разведка.", M1)
        t.tool_call("t1")
        t.chunk("Теперь правка.", M2)
        t.tool_call("t2")
        self.assertEqual(
            t.client.narration_pulses, ["Сначала разведка.", "Теперь правка."]
        )
        self.assertIn("Сначала разведка.\n\n", t.rail)

    def test_colon_ends_the_utterance(self) -> None:
        t = _Turn()
        t.chunk("Проверяю два места:", M1)
        t.tool_call("t1")
        t.chunk("оба чистые.", M2)
        t.tool_call("t2")
        self.assertEqual(t.client.narration_pulses[-1], "оба чистые.")

    def test_closer_after_the_period_still_ends_it(self) -> None:
        t = _Turn()
        t.chunk("Готово (см. выше).", M1)
        t.tool_call("t1")
        t.chunk("Дальше правка.", M2)
        t.tool_call("t2")
        self.assertEqual(t.client.narration_pulses[-1], "Дальше правка.")

    def test_carry_does_not_leak_into_the_next_turn(self) -> None:
        client = _FakeClient()

        def prompt(text: str, mid: str, tool_id: str) -> list[str]:
            """One turn on the SAME client: fresh buffers, cut mid-utterance."""
            buf: list[str] = []
            rail: list[str] = []
            handle_session_update(
                client,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": text},
                    "messageId": mid,
                },
                buf,
                rail,
            )
            handle_session_update(
                client,
                {"sessionUpdate": "tool_call", "toolCallId": tool_id, "title": "Bash"},
                buf,
                rail,
            )
            return rail

        prompt("Обрыв на полу", M1, "t1")
        rail2 = prompt("Новый ответ.", M2, "t2")
        self.assertEqual(client.narration_pulses[-1], "Новый ответ.")
        self.assertNotIn("Обрыв", "".join(rail2))

    def test_engine_without_message_ids_is_protected_too(self) -> None:
        # No id ⇒ every tool event commits the run, so an id-less engine is the
        # WORST case for mid-word splits and the carry is its only defence. The
        # price is symmetric and accepted: two consecutive narration steps that
        # both lack terminal punctuation merge into one growing line.
        t = _Turn()
        t.chunk("Ошибка окружения (н")
        t.tool_call("t1")
        t.chunk("ет Airflow Variables локально).")
        t.tool_call("t2")
        self.assertEqual(
            t.client.narration_pulses[-1],
            "Ошибка окружения (нет Airflow Variables локально).",
        )
        # And the punctuated step after it still gets its own line.
        t.chunk("Дальше правка.")
        t.tool_call("t3")
        self.assertEqual(t.client.narration_pulses[-1], "Дальше правка.")

    def test_collapser_merges_the_carried_pulses_end_to_end(self) -> None:
        # Layer 2 output → layer 3 input, untouched: one line, not two.
        t = _Turn()
        t.chunk("Ошибка окружения (н", M1)
        t.tool_call("t1", title="Read", path="/tmp/dag.py")
        t.chunk("ет Airflow Variables локально), не моей правки.", M2)
        t.tool_call("t2", title="Read", path="/tmp/x.py")

        class _Adapter:
            supports_expandable_quote = True

        c = ProgressCollapser("replace", _Adapter())
        lines: list = []
        for kind, name, _prev, _args in t.client.pulses:
            lines = c.append(lines, f"💬 {name}" if kind == "_thinking" else f"💻 {name}")
        rendered = c.render(lines)
        self.assertEqual(rendered.count("Ошибка окружения"), 1)
        self.assertIn("не моей правки.", rendered)


class TestCollapserNarrationSupersede(unittest.TestCase):
    """Display-layer backstop: a partial 💬 is replaced, never stacked."""

    def _collapser(self) -> ProgressCollapser:
        class _Adapter:
            supports_expandable_quote = True

        return ProgressCollapser("replace", _Adapter())

    def test_grown_narration_replaces_the_partial(self) -> None:
        c = self._collapser()
        lines: list = []
        lines = c.append(lines, "💬 Ошибка окружения (н")
        lines = c.append(lines, "💻 Read sasrec infer dag")
        lines = c.append(lines, "💬 Ошибка окружения (нет Airflow Variables локально).")
        rendered = c.render(lines)
        self.assertEqual(rendered.count("Ошибка окружения"), 1)
        self.assertIn("нет Airflow Variables локально", rendered)
        self.assertNotIn("(н\n", rendered)

    def test_truncated_partial_is_also_recognised(self) -> None:
        c = self._collapser()
        lines = c.append([], "💬 Долгая мысль про кэш…")
        lines = c.append(lines, "💬 Долгая мысль про кэш и его инвалидацию.")
        rendered = c.render(lines)
        self.assertEqual(rendered.count("Долгая мысль"), 1)
        self.assertIn("инвалидацию", rendered)

    def test_distinct_narrations_both_survive(self) -> None:
        c = self._collapser()
        lines = c.append([], "💬 Первая мысль.")
        lines = c.append(lines, "💻 Read a")
        lines = c.append(lines, "💬 Вторая мысль.")
        rendered = c.render(lines)
        self.assertIn("Первая мысль.", rendered)
        self.assertIn("Вторая мысль.", rendered)

    def test_partial_only_pinned_is_still_superseded(self) -> None:
        # The partial can scroll out of the window and live on only as the
        # pinned line — replacing the window entry alone would leave it there.
        c = self._collapser()
        lines = c.append([], "💬 Начало фразы")
        for i in range(c.window + 2):
            lines = c.append(lines, f"💻 step {i}")
        lines = c.append(lines, "💬 Начало фразы и её конец.")
        rendered = c.render(lines)
        self.assertEqual(rendered.count("Начало фразы"), 1)
        self.assertIn("и её конец.", rendered)

    def test_non_narration_lines_untouched(self) -> None:
        c = self._collapser()
        lines = c.append([], "💻 Read a")
        lines = c.append(lines, "💻 Read a")
        self.assertEqual(len(lines), 2)


class _FailoverClient(_FakeClient):
    """A client whose ``_run_prompt`` replays scripted session updates.

    The narration state (``_acp_narration_carry`` and friends) lives on the
    CLIENT, while the text buffers are created per ``_run_prompt``. A failover
    or auth retry runs a SECOND prompt on the same client, so anything the dead
    attempt left behind is still there when the retry starts — that is the seam
    these tests cover.
    """

    def __init__(self, scripts) -> None:
        super().__init__()
        self._scripts = list(scripts)
        self.runs: list[tuple[str, list[str]]] = []
        self._hermes_cancelled = None
        self._activity_agent = None

    def _run_prompt(self, prompt_text, *, timeout_seconds, model_override=None):
        script = self._scripts.pop(0)
        text: list[str] = []
        rail: list[str] = []
        for update in script["updates"]:
            handle_session_update(self, update, text, rail)
        self.runs.append(("".join(text), list(rail)))
        exc = script.get("raises")
        if exc is not None:
            raise exc
        return "".join(text), "".join(rail)


def _chunk(text: str, message_id: str | None = None) -> dict:
    u: dict = {"sessionUpdate": "agent_message_chunk", "content": {"text": text}}
    if message_id:
        u["messageId"] = message_id
    return u


def _tool(tool_id: str) -> dict:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_id,
        "title": "Bash",
        "rawInput": {"command": "true"},
    }


class TestNarrationAcrossAnAttemptBoundary(unittest.TestCase):
    """A second producer writes to the same 💬 rail: failover and auth retry.

    ``agent/acp_failover`` fires its own ``_thinking`` lines ("primary model
    failed…", "auth token expired…") straight at ``_fire_engine_activity``,
    bypassing ``_narration_so_far``. It then re-runs the prompt on the same
    client. So a prefix carried by an attempt that DIED must not survive into
    the retry, and the notice must not be glued to the model's own sentence.
    """

    def setUp(self) -> None:
        import agent.acp_failover as failover

        self.failover = failover
        self._real_record = failover.record_usage
        failover.record_usage = lambda **kw: None
        self.addCleanup(self._restore)
        self._env = {
            k: os.environ.get(k)
            for k in ("HERMES_ACP_FAILOVER", "HERMES_ACP_FALLBACK_MODEL", "HERMES_ACP_MODEL")
        }
        os.environ["HERMES_ACP_FAILOVER"] = "1"
        os.environ["HERMES_ACP_FALLBACK_MODEL"] = "claude-opus-4-8"
        os.environ["HERMES_ACP_MODEL"] = "claude-opus-5"

    def _restore(self) -> None:
        self.failover.record_usage = self._real_record
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _render(self, client) -> str:
        class _Adapter:
            supports_expandable_quote = True

        c = ProgressCollapser("replace", _Adapter())
        lines: list = []
        for kind, name, _prev, _args in client.pulses:
            lines = c.append(lines, f"💬 {name}" if kind == "_thinking" else f"💻 {name}")
        return c.render(lines)

    # ── the dead attempt must not bleed into the retry ───────────────────
    def test_failover_retry_does_not_inherit_the_dead_attempts_prefix(self) -> None:
        client = _FailoverClient([
            {   # primary: cut mid-sentence by a tool call, then the model dies
                "updates": [_chunk("Проверяю конфиг", M1), _tool("t1"),
                            _chunk(" и логи", M2), _tool("t2")],
                "raises": TimeoutError("primary wedged"),
            },
            {   # retry on the backup model: a different sentence entirely
                "updates": [_chunk("Читаю ledger", "msg_R1"), _tool("t3"),
                            _chunk("Готово.", "msg_R2")],
            },
        ])
        answer, _reasoning, used, fired = self.failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-opus-5",
        )
        self.assertTrue(fired)
        self.assertEqual(used, "claude-opus-4-8")
        self.assertEqual(answer, "Готово.")
        # The retry's rail must contain only what the retry said.
        _retry_text, retry_rail = client.runs[-1]
        self.assertNotIn("Проверяю конфиг", "".join(retry_rail))
        # …and no pulse from the retry may re-state the dead attempt's prefix.
        after_notice = client.narration_pulses[
            client.narration_pulses.index(
                next(p for p in client.narration_pulses if "failing over" in p)
            ) + 1:
        ]
        self.assertTrue(after_notice, "the retry narrated nothing to check")
        for line in after_notice:
            self.assertNotIn("Проверяю конфиг", line)

    def test_carry_is_cleared_when_the_attempt_dies(self) -> None:
        client = _FailoverClient([
            {
                "updates": [_chunk("Считаю карточки", M1), _tool("t1"),
                            _chunk(", потом воркеров", M2), _tool("t2")],
                "raises": TimeoutError("primary wedged"),
            },
            {"updates": [_chunk("Готово.", "msg_R1")]},
        ])
        self.failover.run_prompt_with_failover(client, "prompt", 60.0, "claude-opus-5")
        self.assertFalse(getattr(client, "_acp_narration_carry", ""))
        self.assertFalse(getattr(client, "_acp_boundary_pending", False))
        # The slot reserved by the dead attempt points at a discarded rail.
        slot = getattr(client, "_acp_narration_carry_slot", None)
        if slot is not None:
            self.assertIsNot(slot[0], client.runs[-1][1])

    # ── the notice is its own event, not part of a sentence ──────────────
    def test_failover_notice_is_not_glued_to_a_half_sentence(self) -> None:
        # The attempt must die with a carry ARMED (a committed run that ended
        # mid-sentence), or the notice has nothing to be glued to and the test
        # passes vacuously — which is exactly how it read on the first draft.
        client = _FailoverClient([
            {
                "updates": [_chunk("Проверяю конфиг", M1), _tool("t1"),
                            _chunk(" и логи", M2), _tool("t2")],
                "raises": TimeoutError("primary wedged"),
            },
            {"updates": [_chunk("Готово.", "msg_R1")]},
        ])
        self.failover.run_prompt_with_failover(client, "prompt", 60.0, "claude-opus-5")
        self.assertTrue(client.runs[0][1], "the primary attempt narrated nothing")
        rendered = self._render(client)
        notice = next(p for p in client.narration_pulses if "failing over" in p)
        # The notice stands alone: it neither absorbs nor is absorbed by the
        # model's unfinished sentence.
        self.assertIn("failing over", rendered)
        self.assertTrue(notice.startswith("primary model failed"))
        self.assertNotIn("Проверяю конфиг", notice)
        self.assertEqual(rendered.count("failing over"), 1)

    def test_auth_retry_notice_takes_the_same_path(self) -> None:
        client = _FailoverClient([
            {
                "updates": [_chunk("Готовлю правку", M1), _tool("t1"),
                            _chunk(" в оверлее", M2), _tool("t2")],
                "raises": RuntimeError("Copilot ACP failed: 401 OAuth token has expired"),
            },
            {"updates": [_chunk("Готово.", "msg_R1")]},
        ])
        answer, _reasoning, used, fired = self.failover.run_prompt_with_failover(
            client, "prompt", 60.0, "claude-opus-5",
        )
        # Auth failure retries the SAME model — no downgrade (see acp_failover).
        self.assertFalse(fired)
        self.assertEqual(used, "claude-opus-5")
        self.assertEqual(answer, "Готово.")
        self.assertFalse(getattr(client, "_acp_narration_carry", ""))
        notice = next(p for p in client.narration_pulses if "auth token" in p)
        self.assertTrue(notice.startswith("auth token"))
        self.assertNotIn("Готовлю правку", notice)


if __name__ == "__main__":
    unittest.main()
