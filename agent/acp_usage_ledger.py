"""Per-agent / per-task usage ledger for the Copilot-ACP (Claude Code) bridge.

The ACP bridge drives the real Claude engine over stdio and gets back only text
-- the JSON-RPC stream carries no token counts, so ``CopilotACPClient`` reports
``usage = 0`` and ``hermes insights`` shows nothing. Until the bridge exposes
real usage, this module records an *estimated* per-call ledger so token spend
can be attributed to a profile (agent) and session (task).

Numbers here are ESTIMATES (heuristic ~4 chars/token), never billed exact
counts. Every row is flagged ``"estimated": true`` so a downstream reader never
mistakes them for authoritative usage. Recording must never break a completion,
so every public entry point swallows its own errors.
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
) -> None:
    """Append one estimated-usage row to the ledger. Never raises.

    One row == one ACP completion == one agent turn ("task" granularity at the
    session level). ``requested_model`` vs ``used_model`` differ when the opus
    auto-failover fired.
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


def ledger_path() -> Path:
    return _LEDGER_PATH
