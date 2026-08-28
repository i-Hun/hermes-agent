"""Real token usage for a Copilot-ACP call, read off the engine's own transcript.

The ACP JSON-RPC stream carries no token counts, so ``acp_usage_ledger`` has
always written a ~4-chars/token ESTIMATE. The engine, meanwhile, writes the
authoritative numbers to disk: every assistant message in
``~/.claude/projects/<slug>/<sessionId>.jsonl`` carries ``message.usage`` with
``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens`` /
``cache_creation_input_tokens``.

Two supports under one fact — one place KNOWS, the other GUESSES. This module
lets the one that knows tell the one that guesses (skill rule 5). The estimate
stays as the fallback for when the transcript is absent or unreadable.

Why the whole file is one call: the bridge issues ``session/new`` per API call
(one engine process per call), so a sessionId maps 1:1 to one Hermes turn.

Subagent (``Agent``/``Task``) traffic is NOT in that file. This engine writes
each subagent to a sibling directory, ``<project>/<sessionId>/subagents/
agent-*.jsonl`` — so a reader of the session file alone misses a fan-out
entirely, which is the one turn shape most worth measuring (verified live
15.08.2026: a founder turn showed 45k output tokens in its session file while
11 subagent files beside it held the bulk of the spend). Those files are read
too: counted into the totals because they are spend on the same account, and
reported separately so a fan-out is visible AS a fan-out.

Deduplication is mandatory, not defensive: the engine appends one line per
content block, so a single assistant message with text + two tool calls is
written three times with the SAME ``message.usage``. Summing lines instead of
messages overcounted by 2.25x across a day of real transcripts (7.1x on the
tool-heavy founder turn) — a "real" number wronger than the estimate it
replaced.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _projects_root() -> Path:
    """Where Claude Code keeps its per-session transcripts.

    Resolved per call, not at import: the engine's own ``CLAUDE_CONFIG_DIR``
    override (and the tests) must be able to move it, and a module-level
    constant would freeze whatever HOME the gateway happened to boot with.
    """
    base = os.environ.get("CLAUDE_CONFIG_DIR") or (
        Path(os.path.expanduser("~")) / ".claude"
    )
    return Path(base) / "projects"

# An aborted turn can leave a session id on the client with no ledger row to
# flush it (an interrupt raises past every record_* call). Anything older than
# this is dropped rather than billed to whichever turn comes next.
_MAX_SESSION_AGE_S = 15 * 60


def _project_slug(cwd: str) -> str:
    """Claude Code's project-directory mangling: every non-alphanumeric → '-'.

    ``/home/hun/.hermes`` → ``-home-hun--hermes`` (the leading slash and the
    dot each produce their own dash, hence the doubled one).
    """
    return re.sub(r"[^A-Za-z0-9]", "-", cwd or "")


def transcript_path(session_id: str, cwd: str | None = None) -> Path | None:
    """Locate the transcript for ``session_id``. Fast path first, glob second."""
    if not session_id:
        return None
    if cwd:
        p = _projects_root() / _project_slug(cwd) / f"{session_id}.jsonl"
        if p.is_file():
            return p
    # The engine may have resolved a different cwd than we asked for (symlinks,
    # a worktree). Session ids are uuids, so a name match is unambiguous.
    try:
        for p in _projects_root().glob(f"*/{session_id}.jsonl"):
            return p
    except OSError:
        pass
    return None


def subagent_paths(session_file: Path) -> list[Path]:
    """The fan-out transcripts that sit beside a session file.

    ``<project>/<sessionId>.jsonl`` is the parent turn; each subagent it spawned
    gets its own file under ``<project>/<sessionId>/subagents/``. Missing the
    directory is the normal case (most turns spawn nothing), not an error.
    """
    d = session_file.with_suffix("") / "subagents"
    try:
        return sorted(p for p in d.glob("agent-*.jsonl") if p.is_file())
    except OSError:
        return []


def _accumulate(path: Path, totals: dict, seen: set, models: set,
                *, subagent: bool) -> None:
    """Fold one transcript file into ``totals``. Silent on a missing file."""
    try:
        fh = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "assistant":
                continue
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            # One assistant message, N content blocks, N identical rows.
            key = str(msg.get("id") or d.get("requestId") or d.get("uuid") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            out = int(usage.get("output_tokens") or 0)
            totals["prompt_tokens"] += int(usage.get("input_tokens") or 0)
            totals["completion_tokens"] += out
            totals["cache_read_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
            totals["cache_creation_tokens"] += int(
                usage.get("cache_creation_input_tokens") or 0
            )
            totals["messages"] += 1
            # Trust the file it came from, but honour the flag too: an older
            # engine wrote sidechains inline, and this must not stop counting
            # them if the layout changes back.
            if subagent or d.get("isSidechain"):
                totals["subagent_completion_tokens"] += out
                totals["subagent_messages"] += 1
            if isinstance(msg.get("model"), str):
                models.add(msg["model"])


def read_session_usage(session_id: str, cwd: str | None = None) -> dict[str, Any] | None:
    """Sum the real usage of one engine session, fan-out included."""
    path = transcript_path(session_id, cwd)
    if path is None:
        return None
    seen: set[str] = set()
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "subagent_completion_tokens": 0,
        "messages": 0,
        "subagent_messages": 0,
    }
    models: set[str] = set()
    _accumulate(path, totals, seen, models, subagent=False)
    subs = subagent_paths(path)
    for sub in subs:
        _accumulate(sub, totals, seen, models, subagent=True)
    if subs:
        totals["subagents"] = len(subs)
    if not totals["messages"]:
        return None
    totals["total_tokens"] = (
        totals["prompt_tokens"]
        + totals["completion_tokens"]
        + totals["cache_read_tokens"]
        + totals["cache_creation_tokens"]
    )
    if models:
        totals["transcript_models"] = sorted(models)
    return totals


def note_engine_session(client, session_id: str, cwd: str | None = None) -> None:
    """Remember the engine session a call ran on. Never raises.

    Kept as a LIST: a failover or auth retry spawns a second engine session for
    the same Hermes turn, and both burned tokens. The ledger row for that turn
    should carry the sum, not just the attempt that happened to answer.
    """
    try:
        sessions = getattr(client, "_acp_engine_sessions", None)
        if not isinstance(sessions, list):
            sessions = []
            client._acp_engine_sessions = sessions
        now = time.time()
        sessions[:] = [s for s in sessions if now - s.get("ts", 0) <= _MAX_SESSION_AGE_S]
        if not any(s.get("id") == session_id for s in sessions):
            sessions.append({"id": session_id, "cwd": cwd, "ts": now})
    except Exception:
        pass


def take_call_usage(client) -> dict[str, Any] | None:
    """Real usage for the turn that is being recorded now; clears the tracker.

    Called from the ledger write, which happens exactly once per completion on
    both the success and the failure paths — so clearing here is what keeps one
    turn's engine sessions from being billed to the next.
    """
    try:
        sessions = getattr(client, "_acp_engine_sessions", None) or []
        client._acp_engine_sessions = []
        merged: dict[str, Any] | None = None
        ids: list[str] = []
        for s in sessions:
            got = read_session_usage(s.get("id"), s.get("cwd"))
            if not got:
                continue
            ids.append(s.get("id"))
            if merged is None:
                merged = got
                continue
            for k, v in got.items():
                if isinstance(v, int):
                    merged[k] = merged.get(k, 0) + v
                elif k == "transcript_models":
                    merged[k] = sorted(set(merged.get(k, [])) | set(v))
        if merged is not None:
            merged["engine_sessions"] = ids
        return merged
    except Exception:
        return None
