"""Per-agent / per-task usage ledger for the Copilot-ACP (Claude Code) bridge.

The ACP bridge drives the real Claude engine over stdio and gets back only text
-- the JSON-RPC stream carries no token counts, so ``CopilotACPClient`` reports
``usage = 0`` and ``hermes insights`` shows nothing. This module records a
per-call ledger so token spend can be attributed to a profile (agent) and
session (task).

Two kinds of row live here, and a reader must be able to tell them apart:

* ``"estimated": false`` -- REAL counts, read off the engine's own transcript
  (see ``acp_usage_transcript``): prompt/completion plus the cache-read and
  cache-creation figures an estimate cannot see at all.
* ``"estimated": true``  -- the ~4 chars/token heuristic, used only when no
  transcript could be found for the call.

The ``*_est`` fields carry the best available number in BOTH cases so existing
readers keep working unchanged; ``estimated`` says which method produced them,
and real rows additionally carry the cache columns. Recording must never break
a completion, so every public entry point swallows its own errors.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# Install root is stable regardless of per-profile HOME remapping: this file
# lives at <root>/hermes-agent/agent/acp_usage_ledger.py, so parents[2] is the
# real ~/.hermes install root shared by every profile gateway.
_INSTALL_ROOT = Path(__file__).resolve().parents[2]
_LEDGER_PATH = _INSTALL_ROOT / "logs" / "acp-usage.jsonl"

_write_lock = threading.Lock()


def estimate_tokens(text) -> int:
    """Rough token estimate for mixed English/Russian text.

    ~4 characters per token is the standard coarse approximation for Claude's
    tokenizer; good enough for relative attribution across agents/tasks. Not a
    billing figure.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _resolve_profile() -> str:
    """Best-effort active-profile (agent) name; never raises."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        home = os.environ.get("HERMES_HOME", "").strip()
        if not home:
            return "default"
        parts = Path(home).resolve().parts
        if "profiles" in parts:
            i = parts.index("profiles")
            if i + 1 < len(parts):
                return parts[i + 1]
        return "default"


def record_usage(
    *,
    session_id=None,
    requested_model=None,
    used_model=None,
    prompt_text=None,
    completion_text=None,
    reasoning_text=None,
    failover: bool = False,
    ok: bool = True,
    source=None,
    real=None,
    error=None,
) -> None:
    """Append one usage row to the ledger. Never raises.

    One row == one ACP completion == one agent turn ("task" granularity at the
    session level). ``requested_model`` vs ``used_model`` differ when the opus
    auto-failover fired.

    ``real`` is the transcript-derived usage dict (``acp_usage_transcript``)
    when the engine's own numbers could be read; absent it, the row falls back
    to the chars/4 estimate and says so. ``error`` is the exception that ended
    a failed turn -- its CLASS and first line, because "ok: false, see gateway
    log" made every failure look alike.
    """
    try:
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(completion_text) + estimate_tokens(
            reasoning_text
        )
        row: dict[str, Any] = {
            "ts": time.time(),
            "profile": _resolve_profile(),
            "session_id": session_id or None,
            "requested_model": requested_model or None,
            "used_model": used_model or requested_model or None,
            "failover": bool(failover),
            "ok": bool(ok),
            "prompt_tokens_est": prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "total_tokens_est": prompt_tokens + completion_tokens,
            "estimated": True,
        }
        # Own try: a malformed witness must cost us the upgrade, not the row.
        # A cross-check sharing the writer's try is how one bad field deletes
        # the accounting it was meant to improve.
        try:
            _merge_real(row, real)
        except Exception:
            pass
        if error is not None:
            row["error_type"] = type(error).__name__
            row["error"] = " ".join(str(error).split())[:200]
        if source:
            row["source"] = source
        line = json.dumps(row, ensure_ascii=False)
        with _write_lock:
            _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LEDGER_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        # Accounting is best-effort; a broken ledger must never fail a turn.
        pass


def _merge_real(row: dict[str, Any], real) -> None:
    """Replace the guessed numbers with the engine's own, in place.

    Builds the upgrade aside and commits it in one update: a half-applied
    merge would leave a row claiming ``estimated: false`` over numbers that
    are still guesses -- worse than an honest estimate.
    """
    if not isinstance(real, dict) or not real.get("messages"):
        return
    upgrade: dict[str, Any] = {
        "prompt_tokens": int(real.get("prompt_tokens") or 0),
        "completion_tokens": int(real.get("completion_tokens") or 0),
        "cache_read_tokens": int(real.get("cache_read_tokens") or 0),
        "cache_creation_tokens": int(real.get("cache_creation_tokens") or 0),
        "total_tokens": int(real.get("total_tokens") or 0),
        "estimated": False,
    }
    # Overwrite the `*_est` fields too — a reader that never learned about the
    # new columns should still get the better number, not the older one.
    upgrade["prompt_tokens_est"] = upgrade["prompt_tokens"]
    upgrade["completion_tokens_est"] = upgrade["completion_tokens"]
    upgrade["total_tokens_est"] = upgrade["total_tokens"]
    # A fan-out is spend the turn's own numbers hide: subagents answer inside
    # the same engine session and their tokens are already in the sums above.
    # Report them separately so 10 parallel agents are visible AS a fan-out,
    # not merely as one large turn.
    if real.get("subagent_messages"):
        upgrade["subagent_completion_tokens"] = int(
            real.get("subagent_completion_tokens") or 0
        )
        upgrade["subagent_messages"] = int(real.get("subagent_messages") or 0)
        upgrade["subagents"] = int(real.get("subagents") or 0)
    if real.get("engine_sessions"):
        upgrade["engine_sessions"] = list(real["engine_sessions"])
    if real.get("transcript_models"):
        upgrade["transcript_models"] = list(real["transcript_models"])
    row.update(upgrade)


def ledger_path() -> Path:
    return _LEDGER_PATH
