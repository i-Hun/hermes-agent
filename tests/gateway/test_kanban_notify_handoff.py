"""Kanban notification pings must not sever a worker's handoff mid-word.

Overlay behaviour (``hermes_cli/kanban_handoff.py``). The regression this
guards: a worker handed off a 399-char single-line summary and the operator
received it cut at exactly 200 characters, mid-word, with nothing marking the
cut — the report read as if the worker had stopped talking.

The seam tests matter as much as the unit tests: the helper is only useful if
the notifier actually calls it, and the notifier lives in an upstream file that
we patch with one-line call-outs.
"""

from __future__ import annotations

import inspect
import os
import unittest

from gateway import kanban_watchers
from hermes_cli.kanban_handoff import (
    DEFAULT_HANDOFF_CHARS,
    handoff_excerpt,
    handoff_limit,
)

# Verbatim payload of the incident (task t_415833af, 2026-08-07): one line,
# 399 chars. Upstream delivered the first 200 of it.
INCIDENT_SUMMARY = (
    "Все три пункта задачи 022 закрыты и задеплоены на стенд (192.168.1.63:8788): "
    "(1) to_info() больше не глотает ошибку Google Tasks — новое поле "
    "list_fetch_error + баннер «Повторить» вместо тихого «Списков не найдено»; "
    "(2) кнопки Перенести/Отключить не слипаются (block); (3) все шесть страниц "
    "/app/account/* подняты до max-w-3xl (ширина дерева), поля форм сужены до "
    "max-w-md. Коммиты e60058d и a05f3f0"
)


class HandoffExcerptTests(unittest.TestCase):
    def test_incident_summary_is_delivered_whole(self):
        """The reported regression: 399 chars must arrive intact, unmarked."""
        out = handoff_excerpt(INCIDENT_SUMMARY)
        self.assertEqual(out, INCIDENT_SUMMARY)
        self.assertNotIn("…", out)
        self.assertIn("Коммиты e60058d", out)

    def test_short_text_is_returned_verbatim(self):
        self.assertEqual(handoff_excerpt("Готово."), "Готово.")

    def test_multiline_summary_keeps_its_detail_lines(self):
        """Upstream kept only line 1; a headline+detail handoff lost the detail."""
        text = "Задача закрыта.\n- пункт один\n- пункт два"
        self.assertEqual(handoff_excerpt(text), text)

    def test_long_text_is_marked_as_cut(self):
        out = handoff_excerpt("a" * 40 + " " + "b" * 4000, limit=100)
        self.assertTrue(out.endswith("…"), out)
        self.assertLessEqual(len(out), 101)

    def test_cut_never_lands_mid_word(self):
        """The defect itself: '«Списк' instead of '«Списков'."""
        text = " ".join(f"слово{i}" for i in range(400))
        out = handoff_excerpt(text, limit=120)
        body = out.rstrip("…").rstrip()
        # every whitespace-separated token that survived must be a whole token
        tokens = set(text.split())
        for tok in body.split():
            self.assertIn(tok, tokens, f"severed token: {tok!r}")

    def test_cut_prefers_a_sentence_end(self):
        text = "Первое предложение закончилось. " + ("хвост " * 200)
        out = handoff_excerpt(text, limit=60)
        self.assertTrue(out.startswith("Первое предложение закончилось."), out)
        self.assertTrue(out.endswith("…"), out)

    def test_sentence_end_too_early_is_ignored(self):
        """A boundary at 5% of the budget would throw away the whole handoff."""
        text = "Ок. " + ("подробность " * 200)
        out = handoff_excerpt(text, limit=200)
        self.assertGreater(len(out), 100, out)

    def test_single_giant_token_falls_back_to_a_hard_cut(self):
        out = handoff_excerpt("x" * 5000, limit=50)
        self.assertEqual(out, "x" * 50 + "…")

    def test_dangling_punctuation_is_trimmed_before_the_marker(self):
        text = "начало, " + ("слово " * 200)
        out = handoff_excerpt(text, limit=40)
        self.assertNotIn(",…", out)
        self.assertNotIn(" …", out)

    def test_empty_inputs_stay_falsy(self):
        """Callers guard with ``if summary:`` — empty must not become a newline."""
        for value in (None, "", "   ", "\n\n"):
            self.assertEqual(handoff_excerpt(value), "")

    def test_non_string_payload_does_not_raise(self):
        self.assertEqual(handoff_excerpt(12345), "12345")


class HandoffLimitTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("HERMES_KANBAN_HANDOFF_CHARS", None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop("HERMES_KANBAN_HANDOFF_CHARS", None)
        if self._saved is not None:
            os.environ["HERMES_KANBAN_HANDOFF_CHARS"] = self._saved

    def test_default_is_generous_enough_for_a_paragraph(self):
        self.assertEqual(handoff_limit(), DEFAULT_HANDOFF_CHARS)
        self.assertGreater(DEFAULT_HANDOFF_CHARS, len(INCIDENT_SUMMARY))

    def test_env_override_is_honoured(self):
        os.environ["HERMES_KANBAN_HANDOFF_CHARS"] = "40"
        self.assertEqual(handoff_limit(), 40)
        self.assertTrue(handoff_excerpt(INCIDENT_SUMMARY).endswith("…"))

    def test_garbage_env_falls_back_instead_of_raising(self):
        for bad in ("", "abc", "0", "-5", "12.5"):
            os.environ["HERMES_KANBAN_HANDOFF_CHARS"] = bad
            self.assertEqual(handoff_limit(), DEFAULT_HANDOFF_CHARS, bad)


class NotifierSeamTests(unittest.TestCase):
    """The helper is dead code unless the upstream notifier calls it."""

    def setUp(self):
        self.src = inspect.getsource(kanban_watchers)

    def test_notifier_imports_the_overlay_helper(self):
        self.assertIn("from hermes_cli.kanban_handoff import handoff_excerpt", self.src)

    def test_no_hardcoded_truncation_survives_in_the_ping_builder(self):
        """Guards the exact literals the incident came from."""
        self.assertNotIn("[:200]", self.src)
        self.assertNotIn("[:160]", self.src)

    def test_completed_and_blocked_paths_both_use_the_helper(self):
        self.assertIn("handoff_excerpt(payload_summary)", self.src)
        self.assertIn("handoff_excerpt(task.result)", self.src)
        self.assertIn("handoff_excerpt(ev.payload['reason'])", self.src)
        # v2026.8.13 added a `review_requested` branch with its own hard
        # `[:200]` cut — a NEW mouth for the same incident. Upstream growing a
        # sixth truncation site is exactly what the literal guard above is for,
        # and this line names the site so the next one is a one-word diff.
        self.assertIn("handoff_excerpt(ev.payload['summary'])", self.src)

    def test_fallback_definition_exists_for_an_absent_overlay(self):
        """A missing overlay module must degrade, not crash the notifier loop."""
        self.assertIn("except Exception:", self.src.split("logger = logging")[0])

    def test_write_side_cap_is_routed_too(self):
        """A 900-char ping budget delivers 400 chars if the STORE caps at 400."""
        from hermes_cli import kanban_db

        src = inspect.getsource(kanban_db)
        # Anchor on the ASSIGNMENT, not on the argument name: v2026.8.13 renamed
        # the read side ``ev_summary`` → ``event_summary`` on the complete_task
        # path, so an argument-shaped anchor went red while the seam itself was
        # untouched. The count keeps BOTH write paths pinned.
        self.assertEqual(src.count("ev_summary = handoff_payload_excerpt("), 2)
        self.assertNotIn("splitlines()[0][:400]", src)

    def test_payload_budget_is_not_below_the_ping_budget(self):
        """The invariant the incident violated: store >= ping."""
        from hermes_cli.kanban_handoff import EVENT_PAYLOAD_CHARS

        self.assertGreaterEqual(EVENT_PAYLOAD_CHARS, DEFAULT_HANDOFF_CHARS)


# --------------------------------------------------------------------------
# End-to-end: drive the REAL notifier loop. The unit tests above prove the
# helper; only this proves the operator receives what it produced.
# --------------------------------------------------------------------------


class _RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)

    async def handle_message(self, event):  # pragma: no cover - unused
        pass


async def _one_tick(monkeypatch, runner):
    import asyncio

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _deliver(monkeypatch, tmp_path, db_name, summary):
    """Complete a task with ``summary`` and return the text actually sent."""
    import asyncio

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / db_name))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Strata 022 — проба выдачи", assignee="founder")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()

    adapter = _RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    asyncio.run(_one_tick(monkeypatch, runner))
    return adapter.sent


def test_notifier_delivers_the_incident_summary_whole(tmp_path, monkeypatch):
    sent = _deliver(monkeypatch, tmp_path, "handoff-whole.db", INCIDENT_SUMMARY)
    assert len(sent) == 1, sent
    text = sent[0]
    # the exact tail that used to be severed
    assert "Коммиты e60058d и a05f3f0" in text
    assert "«Списков не найдено»" in text
    assert "…" not in text
    assert text.startswith("✔ ")


def test_notifier_delivers_a_headline_plus_detail_handoff(tmp_path, monkeypatch):
    """Upstream kept line 1 only — a structured handoff arrived as its title."""
    summary = (
        "Задача 022 закрыта, три пункта.\n"
        "1) ошибка Google Tasks больше не глотается\n"
        "2) кнопки не слипаются\n"
        "3) ширина настроек выровнена по дереву"
    )
    sent = _deliver(monkeypatch, tmp_path, "handoff-lines.db", summary)
    assert len(sent) == 1, sent
    assert "3) ширина настроек выровнена по дереву" in sent[0]


def test_notifier_marks_an_oversized_summary_as_cut(tmp_path, monkeypatch):
    """The ceiling still exists — a transcript-sized handoff must not flood."""
    huge = "предложение номер один. " * 400
    sent = _deliver(monkeypatch, tmp_path, "handoff-huge.db", huge)
    assert len(sent) == 1, sent
    text = sent[0]
    assert text.endswith("…"), text[-40:]
    assert len(text) < 1200, len(text)


if __name__ == "__main__":
    unittest.main()
