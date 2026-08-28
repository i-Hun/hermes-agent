"""Real ACP usage: engine transcript → ledger row (local overlay, 15.08.2026).

The ledger used to guess token counts at ~4 chars/token while the engine wrote
the real numbers to disk. These tests pin the seam between the place that KNOWS
and the place that guessed — and, above all, the deduplication: the engine
appends one line per CONTENT BLOCK, so summing lines instead of messages turns
a "real" figure into one wronger than the estimate it replaced.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from agent import acp_usage_ledger, acp_usage_transcript


def _assistant(msg_id: str, *, out: int, inp: int = 10, cache_read: int = 0,
               cache_creation: int = 0, sidechain: bool = False,
               model: str = "claude-opus-5") -> dict:
    return {
        "type": "assistant",
        "isSidechain": sidechain,
        "requestId": f"req_{msg_id}",
        "uuid": f"uuid-{msg_id}",
        "message": {
            "id": msg_id,
            "model": model,
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
        },
    }


def _write_transcript(root, cwd: str, session_id: str, records) -> None:
    d = root / "projects" / acp_usage_transcript._project_slug(cwd)
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{session_id}.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    return tmp_path / "claude"


def test_project_slug_matches_engine_mangling():
    # Every non-alphanumeric becomes a dash — the leading slash and the dot of
    # `.hermes` each produce one, hence the doubled dash. Observed on disk.
    assert acp_usage_transcript._project_slug("/home/hun/.hermes") == "-home-hun--hermes"
    assert acp_usage_transcript._project_slug(
        "/home/hun/.hermes/kanban/workspaces/t_02174822"
    ) == "-home-hun--hermes-kanban-workspaces-t-02174822"


def test_one_message_many_blocks_counts_once(claude_home):
    """The regression this whole module exists to avoid.

    Three lines, one message id, one usage — a naive line sum reports 3x the
    output tokens. Seen live: 16 lines for 6 messages.
    """
    rec = _assistant("msg_1", out=100, inp=7, cache_read=5000)
    _write_transcript(claude_home, "/w", "sess-a", [rec, dict(rec), dict(rec)])

    got = acp_usage_transcript.read_session_usage("sess-a", "/w")

    assert got["messages"] == 1
    assert got["completion_tokens"] == 100
    assert got["prompt_tokens"] == 7
    assert got["cache_read_tokens"] == 5000
    assert got["total_tokens"] == 5107


def test_fan_out_lives_in_a_sibling_directory_and_is_counted(claude_home):
    """The whole point of measuring: subagents are NOT in the session file.

    Verified live on 15.08.2026 — a founder turn's own file showed 45k output
    tokens while 11 `subagents/agent-*.jsonl` beside it held 349k more. A
    reader of the session file alone reports the cheap part of a fan-out.
    """
    _write_transcript(claude_home, "/w", "sess-b", [_assistant("msg_parent", out=50)])
    subs = claude_home / "projects" / acp_usage_transcript._project_slug("/w") \
        / "sess-b" / "subagents"
    subs.mkdir(parents=True)
    for i, out in enumerate((300, 200)):
        (subs / f"agent-a{i}.jsonl").write_text(
            json.dumps(_assistant(f"msg_sub{i}", out=out, sidechain=True)) + "\n",
            encoding="utf-8",
        )

    got = acp_usage_transcript.read_session_usage("sess-b", "/w")

    assert got["completion_tokens"] == 550
    assert got["subagent_completion_tokens"] == 500
    assert got["subagent_messages"] == 2
    assert got["subagents"] == 2


def test_inline_sidechain_still_counts(claude_home):
    """An older engine wrote sidechains into the session file itself.

    The layout is the engine's business and may change back; the flag keeps
    working either way, so a layout change degrades to "counted, in the same
    bucket" rather than to "silently missing".
    """
    _write_transcript(claude_home, "/w", "sess-i", [
        _assistant("p", out=10),
        _assistant("s", out=90, sidechain=True),
    ])

    got = acp_usage_transcript.read_session_usage("sess-i", "/w")

    assert got["completion_tokens"] == 100
    assert got["subagent_completion_tokens"] == 90
    assert "subagents" not in got


def test_found_by_glob_when_cwd_disagrees(claude_home):
    """The engine may resolve a different cwd (symlink, worktree) than we asked."""
    _write_transcript(claude_home, "/real/place", "sess-c", [_assistant("m", out=5)])

    assert acp_usage_transcript.read_session_usage("sess-c", "/asked/elsewhere")
    assert acp_usage_transcript.read_session_usage("sess-c") is not None


def test_missing_or_empty_transcript_returns_none(claude_home):
    assert acp_usage_transcript.read_session_usage("nope", "/w") is None
    _write_transcript(claude_home, "/w", "sess-d", [{"type": "user"}])
    assert acp_usage_transcript.read_session_usage("sess-d", "/w") is None


def test_failover_sums_both_engine_sessions_then_clears(claude_home):
    """A retry spawns a SECOND engine session; both burned tokens.

    Attributing only the attempt that answered would hide the cost of the one
    that failed — which is exactly the turn worth seeing.
    """
    _write_transcript(claude_home, "/w", "sess-1", [_assistant("m1", out=10, inp=1)])
    _write_transcript(claude_home, "/w", "sess-2", [_assistant("m2", out=20, inp=2)])
    client = SimpleNamespace()
    acp_usage_transcript.note_engine_session(client, "sess-1", "/w")
    acp_usage_transcript.note_engine_session(client, "sess-2", "/w")

    got = acp_usage_transcript.take_call_usage(client)

    assert got["completion_tokens"] == 30
    assert got["engine_sessions"] == ["sess-1", "sess-2"]
    # Cleared: the next turn must not be billed for this one.
    assert acp_usage_transcript.take_call_usage(client) is None


def test_stale_session_is_dropped(claude_home):
    """An interrupted turn raises past every ledger write, leaving its id behind."""
    _write_transcript(claude_home, "/w", "old", [_assistant("m", out=999)])
    _write_transcript(claude_home, "/w", "new", [_assistant("m2", out=7)])
    client = SimpleNamespace(_acp_engine_sessions=[
        {"id": "old", "cwd": "/w", "ts": time.time() - 3600},
    ])
    acp_usage_transcript.note_engine_session(client, "new", "/w")

    got = acp_usage_transcript.take_call_usage(client)

    assert got["completion_tokens"] == 7
    assert got["engine_sessions"] == ["new"]


def _last_row(path):
    return json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])


def test_ledger_row_prefers_real_over_estimate(tmp_path, monkeypatch):
    ledger = tmp_path / "acp-usage.jsonl"
    monkeypatch.setattr(acp_usage_ledger, "_LEDGER_PATH", ledger)

    acp_usage_ledger.record_usage(
        session_id="s", requested_model="claude-opus-5", used_model="claude-opus-5",
        prompt_text="x" * 4000, completion_text="y" * 400,
        real={"prompt_tokens": 11, "completion_tokens": 22, "cache_read_tokens": 33,
              "cache_creation_tokens": 44, "total_tokens": 110, "messages": 3,
              "subagent_completion_tokens": 9, "subagent_messages": 1},
    )

    row = _last_row(ledger)
    assert row["estimated"] is False
    assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 22)
    assert (row["cache_read_tokens"], row["cache_creation_tokens"]) == (33, 44)
    assert row["subagent_completion_tokens"] == 9
    # A reader that never learned the new columns must still get the better
    # number — the estimate is not left standing next to the truth.
    assert row["prompt_tokens_est"] == 11
    assert row["total_tokens_est"] == 110


def test_ledger_row_falls_back_to_estimate(tmp_path, monkeypatch):
    ledger = tmp_path / "acp-usage.jsonl"
    monkeypatch.setattr(acp_usage_ledger, "_LEDGER_PATH", ledger)

    acp_usage_ledger.record_usage(
        session_id="s", requested_model="claude-opus-5", used_model="claude-opus-5",
        prompt_text="x" * 400, completion_text="y" * 40, real=None,
    )

    row = _last_row(ledger)
    assert row["estimated"] is True
    assert row["prompt_tokens_est"] == 100
    assert "cache_read_tokens" not in row


def test_ledger_records_failure_class_and_message(tmp_path, monkeypatch):
    ledger = tmp_path / "acp-usage.jsonl"
    monkeypatch.setattr(acp_usage_ledger, "_LEDGER_PATH", ledger)

    acp_usage_ledger.record_usage(
        session_id="s", requested_model="claude-opus-5", used_model="claude-opus-5",
        prompt_text="p", completion_text=None, ok=False,
        error=TimeoutError("weekly limit\n  resets 3pm " + "z" * 500),
    )

    row = _last_row(ledger)
    assert row["ok"] is False
    assert row["error_type"] == "TimeoutError"
    assert row["error"].startswith("weekly limit resets 3pm")
    assert len(row["error"]) <= 200


def test_recording_never_raises_on_a_broken_real_dict(tmp_path, monkeypatch):
    """Accounting is best-effort: a broken witness must not fail a turn."""
    ledger = tmp_path / "acp-usage.jsonl"
    monkeypatch.setattr(acp_usage_ledger, "_LEDGER_PATH", ledger)

    acp_usage_ledger.record_usage(
        session_id="s", requested_model="m", used_model="m", prompt_text="p",
        completion_text="c", real={"messages": 1, "prompt_tokens": "not-a-number"},
    )
    # The row must still be there: a broken witness costs us the upgrade, not
    # the accounting. Sharing the writer's `try` is how one bad field would
    # delete the very row it was meant to improve.
    row = _last_row(ledger)
    assert row["estimated"] is True
    assert row["prompt_tokens_est"] == 1


def test_take_call_usage_survives_a_client_without_sessions():
    assert acp_usage_transcript.take_call_usage(SimpleNamespace()) is None
