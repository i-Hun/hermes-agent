"""Regression tests: collapsible progress history + live subagent tail.

Local overlay feature (ACP-native fleet): in "replace" tool-progress grouping
on platforms with expandable blockquotes (Telegram MarkdownV2 ``**> …||``),
older progress lines fold into a collapsed quote, the current line stays
visible, and while ``delegate_task`` children run, the tail of their live
transcripts (#67479, ``cache/delegation/live/<id>/task-N.log``) is surfaced
beneath the current line.

The logic lives in ``gateway/progress_collapse.py`` (extracted from
``gateway/run.py`` to keep the upstream-churn conflict surface small); these
tests import it directly.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gateway.progress_collapse import ProgressCollapser  # noqa: E402


class _QuoteAdapter:
    supports_expandable_quote = True


class _PlainAdapter:
    pass


@pytest.fixture()
def prog():
    return ProgressCollapser("replace", _QuoteAdapter())


def test_flatten_terminal_block_to_one_line():
    flat = ProgressCollapser.flatten_entry(
        "💻 terminal\n```\napt list --upgradable | head -5\n```"
    )
    assert flat == "💻 apt list --upgradable | head -5"


def test_render_collapses_history_keeps_current_and_tail(prog):
    prog.subagent_tail = ["📟 суб-агент 0 · tool⟩ -> web_search(x)"]
    out = prog.render(
        [
            "💻 terminal\n```\ncmd1\n```",
            "🔍 Searching...",
            "💻 terminal\n```\ncmd2\n```",
        ]
    )
    lines = out.splitlines()
    # History (all but last entry) folds into an expandable quote.
    assert lines[0] == "**>💻 cmd1"
    assert lines[1] == ">🔍 Searching...||"
    # Current entry keeps its fenced form; tail is appended last.
    assert "```" in out
    assert lines[-1].startswith("📟")


def test_pinned_narration_stays_visible_above_current_line(prog):
    """💬 narration outranks tool lines: it must not fold into history."""
    narration = "💬 Проверю по коду, как устроена надстройка EXTRA_TOOLS"
    prog.pinned_narrations = [narration]
    out = prog.render(
        [narration, "💻 terminal\n```\ncmd1\n```", "💻 terminal\n```\ncmd2\n```"]
    )
    lines = out.splitlines()
    # Folded history holds only the superseded terminal command…
    assert lines[0] == "**>💻 cmd1||"
    # …the narration sits visible (unquoted) right above the current entry.
    assert narration in lines
    idx = lines.index(narration)
    assert not lines[idx].startswith(">")
    assert "cmd2" in "\n".join(lines[idx + 1:])


def test_pinned_narration_not_duplicated_when_it_is_current(prog):
    narration = "💬 Думаю над планом"
    prog.pinned_narrations = [narration]
    out = prog.render(["💻 terminal\n```\ncmd1\n```", narration])
    assert out.count(narration) == 1
    assert out.splitlines()[-1] == narration


def test_pinned_narration_survives_window_scrollout(prog):
    """The pinned 💬 renders even after the rolling window dropped it."""
    narration = "💬 Анализирую логи гейтвея"
    prog.pinned_narrations = [narration]
    out = prog.render(
        ["💻 terminal\n```\ncmd8\n```", "💻 terminal\n```\ncmd9\n```"]
    )
    lines = out.splitlines()
    assert narration in lines
    assert lines.index(narration) > lines.index("**>💻 cmd8||")


def test_render_without_collapse_appends_tail_only():
    prog = ProgressCollapser("replace", _PlainAdapter())
    assert not prog.collapse_history
    prog.subagent_tail = ["📟 x"]
    out = prog.render(["a", "b"])
    assert out == "a\nb\n📟 x"


def test_append_pins_narration_and_keeps_rolling_window(prog):
    lines: list = []
    lines = prog.append(lines, "💬  Изучаю   код  ")
    assert prog.pinned_narrations == ["💬 Изучаю код"]
    for i in range(12):
        lines = prog.append(lines, f"💻 terminal\n```\ncmd{i}\n```")
    assert len(lines) == prog.window
    # Narration scrolled out of the window but stays pinned for render().
    assert prog.pinned_narrations == ["💬 Изучаю код"]
    assert "💬 Изучаю код" in prog.render(lines)


def test_every_narration_accumulates_and_none_is_cut(prog):
    """Three 💬 in a row: all visible, in order, none inside the quote."""
    said = ["💬 Первое", "💬 Второе", "💬 Третье"]
    lines: list = []
    for i, n in enumerate(said):
        lines = prog.append(lines, n)
        lines = prog.append(lines, f"💻 terminal\n```\ncmd{i}\n```")
    assert prog.pinned_narrations == said
    out = prog.render(lines)
    body = out.splitlines()
    # No narration hides inside the collapsed quote…
    quoted = [ln for ln in body if ln.startswith(">") or ln.startswith("**>")]
    assert quoted and not [q for q in quoted if "💬" in q]
    # …and all three sit visible, oldest first, above the current tool line.
    idx = [body.index(n) for n in said]
    assert idx == sorted(idx)
    assert idx[-1] == len(body) - 2 and body[-1].startswith("💻")


def test_bubble_cap_sheds_oldest_narrations_but_keeps_the_newest(prog):
    lines: list = []
    for i in range(6):
        lines = prog.append(lines, "💬 " + f"мысль{i} " * 200)
    out = prog.render(lines)
    assert len(out) <= prog.BUBBLE_CAP
    assert "мысль5" in out          # newest survives
    assert "мысль0" not in out      # oldest shed first


def test_on_content_bubble_resets_pin_and_tail_signature(prog):
    prog.pinned_narrations = ["💬 x", "💬 y"]
    prog._subagent_sig = "old"
    prog.on_content_bubble()
    assert prog.pinned_narrations == []
    assert prog._subagent_sig == ""


def test_collect_subagent_tail_reads_live_transcripts(prog, tmp_path, monkeypatch):
    d = tmp_path / "deleg_ab12cd34"
    d.mkdir()
    (d / "task-0.log").write_text(
        "=== Hermes subagent live transcript ===\n"
        "delegation: deleg_ab12cd34   task: 0\n"
        "goal: test goal\n"
        "started: now\n"
        "(append-only; streams while the subagent runs — tail -f me)\n"
        "========================================\n"
        "12:00:01 user     | kickoff: test goal\n"
        "12:00:05 tool     | -> terminal(ls -la /tmp)\n",
        encoding="utf-8",
    )
    (d / "task-1.log").write_text(
        "header\n12:00:07 final    | done: all good\n", encoding="utf-8"
    )
    stub = types.ModuleType("tools.delegation_live_log")
    stub.live_transcript_root = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "tools.delegation_live_log", stub)

    tail = prog.collect_subagent_tail()
    assert any("tool⟩ -> terminal(ls -la /tmp)" in t for t in tail)
    # Fresh "final" line lingers as ✅ completion feedback…
    assert any(t.startswith("✅") and "done: all good" in t for t in tail)

    # …but a final older than ~15s drops out entirely.
    old = time.time() - 60
    os.utime(d / "task-1.log", (old, old))
    tail2 = prog.collect_subagent_tail()
    assert not any("done: all good" in t for t in tail2)


class _Result:
    def __init__(self, success=True, message_id="m1", retryable=False):
        self.success = success
        self.message_id = message_id
        self.retryable = retryable


@pytest.mark.asyncio
async def test_refresh_tail_io_edits_existing_message(prog, monkeypatch):
    monkeypatch.setattr(prog, "collect_subagent_tail", lambda: ["📟 live"])
    edits: list = []

    async def edit_fn(mid, text):
        edits.append((mid, text))
        return _Result()

    async def send_fn(text):  # pragma: no cover — must not be called
        raise AssertionError("send_fn must not be called when a message exists")

    st = await prog.refresh_tail_io(
        ["line"],
        active=True,
        throttle_ok=True,
        message_id="m0",
        edit_fn=edit_fn,
        send_fn=send_fn,
    )
    assert st == ("m0", True)
    assert edits and "📟 live" in edits[0][1]

    # Unchanged tail → no second edit committed.
    prog._subagent_poll_ts = 0.0
    st2 = await prog.refresh_tail_io(
        ["line"],
        active=True,
        throttle_ok=True,
        message_id="m0",
        edit_fn=edit_fn,
        send_fn=send_fn,
    )
    assert st2 is None and len(edits) == 1


@pytest.mark.asyncio
async def test_refresh_tail_io_sends_new_message_only_when_tail_nonempty(prog, monkeypatch):
    monkeypatch.setattr(prog, "collect_subagent_tail", lambda: [])
    st = await prog.refresh_tail_io(
        [], active=True, throttle_ok=True, message_id=None,
        edit_fn=None, send_fn=None,
    )
    # Empty tail with no prior sig change → nothing committed.
    assert st is None

    monkeypatch.setattr(prog, "collect_subagent_tail", lambda: ["📟 live"])
    prog._subagent_poll_ts = 0.0
    sends: list = []

    async def send_fn(text):
        sends.append(text)
        return _Result(message_id="m9")

    st2 = await prog.refresh_tail_io(
        [], active=True, throttle_ok=True, message_id=None,
        edit_fn=None, send_fn=send_fn,
    )
    assert st2 == ("m9", True)
    assert sends and "📟 live" in sends[0]


def test_telegram_converter_keeps_multiline_expandable_quote():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class _Dummy:
        pass

    out = TelegramAdapter.format_message(
        _Dummy(), "**>💻 cmd1\n>🔍 grep foo\n>💻 make test||\n⏳ current"
    )
    lines = out.splitlines()
    # The || terminator on a continuation line must survive unescaped —
    # escaping it breaks collapsing of multi-line expandable quotes.
    assert lines[2].endswith("||") and "\\|\\|" not in out


# =========================================================================
# Flush left edge of the collapsed history (Oleg, 2026-08-06)
# =========================================================================
# Symptom: every progress line but the first was indented by one space in
# Telegram.  Cause: ">" is the whole marker — the space after it is CONTENT,
# not syntax, so "> x" renders as " x".  The first line looked fine only
# because the Bot API trims whitespace at the start of the message.


def _quote_content(payload_line: str) -> str | None:
    """What Telegram shows for one MarkdownV2 quote line ('' if not a quote)."""
    import re as _re

    m = _re.match(r'^(?:\*\*)?>{1,3}(.*)$', payload_line)
    return None if m is None else m.group(1)


def test_collapsed_history_lines_start_flush(prog):
    """No quote line may hand Telegram a leading space as content."""
    out = prog.render(
        [
            "💻 terminal\n```\ncmd1\n```",
            "🔍 Searching...",
            "💻 terminal\n```\ncmd2\n```",
        ]
    )
    quoted = [ln for ln in out.splitlines() if ln.lstrip("*").startswith(">")]
    assert quoted, "expected a collapsed history block"
    for ln in quoted:
        assert not _quote_content(ln).startswith(" "), ln


def test_rendered_bubble_survives_telegram_conversion_flush(prog):
    """End-to-end: collapser → adapter payload → what Telegram would show."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class _Dummy:
        pass

    lines = [f"💻 terminal\n```\ncmd{i}\n```" for i in range(1, 6)]
    payload = TelegramAdapter.format_message(_Dummy(), prog.render(lines))
    shown = [_quote_content(ln) for ln in payload.splitlines()]
    shown = [s for s in shown if s is not None]
    assert len(shown) >= 4
    # Every visible history line begins with its emoji, none with a space —
    # i.e. the left edge is straight, not ragged.
    for s in shown:
        assert s.startswith("💻"), repr(s)


def test_adapter_still_round_trips_a_spaced_quote():
    """Upstream/model-authored '> text' must keep its space verbatim."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class _Dummy:
        pass

    out = TelegramAdapter.format_message(_Dummy(), "> Line one\n> Line two")
    assert "> Line one" in out and "> Line two" in out
    assert "\\>" not in out


def test_adapter_accepts_flush_quote_marker():
    """'>text' (Telegram's own syntax) is a quote, not escaped prose."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class _Dummy:
        pass

    out = TelegramAdapter.format_message(_Dummy(), ">flush quote")
    assert out.startswith(">flush") and "\\>" not in out
