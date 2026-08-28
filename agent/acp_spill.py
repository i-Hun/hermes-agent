"""Spill oversized MCP tool results to disk and hand the model a locator.

WHY THIS EXISTS
---------------
Upstream Hermes already has a three-layer spill (``tools/tool_result_storage``:
per-tool cap → ``maybe_persist_tool_result`` → ``enforce_turn_budget``), but
every layer of it is called from ``agent/tool_executor.py``. On the ACP bridge
the engine executes tools, so ``tool_executor`` is not in the path at all and
the whole mechanism sits idle for the five bridge profiles.

SCOPE — read this before believing it fixes more than it does
-------------------------------------------------------------
This covers the tools WE serve over MCP (``hermes-tools``). It cannot cover the
engine's own ``Bash``/``Grep``/``Read``: their results land in Claude Code's
context, and Hermes only ever sees a ``tool_call_update`` notification — by then
the other context is already filled. There is no seam on our side for those.

Locator lifetime: files live under ``$HERMES_HOME/state/acp-spill`` rather than
``/tmp`` so a path quoted in yesterday's transcript still resolves after a
reboot. They are reaped by age instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_CHARS = int(
    os.environ.get("HERMES_ACP_SPILL_THRESHOLD", "24000") or 24000
)
PREVIEW_CHARS = int(os.environ.get("HERMES_ACP_SPILL_PREVIEW", "1500") or 1500)
TTL_DAYS = float(os.environ.get("HERMES_ACP_SPILL_TTL_DAYS", "3") or 3)

# Tools whose result is a control answer, not a payload: spilling them would
# cost a round-trip to learn something that fits in a line. `job_output` is
# already tail-capped by acp_jobs.
#
# skill_* is here for a different reason, and it is the one worth stating:
# loading a skill IS a request for its whole text. Handing back a locator does
# not save the read, it only adds a round-trip before the same read happens.
# A 100 KB skill is a real cost — but the fix for it is a smaller skill, not a
# locator. Spill cannot make a needed payload unneeded.
NEVER_SPILL: frozenset[str] = frozenset({
    "todo", "memory", "kanban_complete", "kanban_block", "kanban_comment",
    "kanban_heartbeat", "kanban_create", "kanban_unblock", "kanban_link",
    "job_start", "job_kill", "job_list", "job_output",
    "skill_view", "skills_list", "skill_manage",
})

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

SPILL_TAG = "<spilled-output>"
SPILL_CLOSING_TAG = "</spilled-output>"


def _storage_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    path = Path(home) / "state" / "acp-spill"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _preview(content: str, max_chars: int) -> tuple[str, bool]:
    """Truncate on a line boundary. Upstream's helper when it is importable."""
    try:
        from tools.tool_result_storage import generate_preview
        return generate_preview(content, max_chars=max_chars)
    except Exception:
        if len(content) <= max_chars:
            return content, False
        cut = content[:max_chars]
        nl = cut.rfind("\n")
        if nl > max_chars // 2:
            cut = cut[:nl + 1]
        return cut, True


def _reap(directory: Path) -> int:
    cutoff = time.time() - TTL_DAYS * 86400
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for path in entries:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def threshold_for(tool_name: str) -> float:
    """Resolve the spill threshold for a tool.

    Honours the upstream per-tool registry when it is reachable (so a tool that
    declares itself unlimited stays unlimited) but never lets it raise our own
    ceiling — the point of this module is that the bridge's context is the
    scarce resource, not the tool's opinion of its own size.
    """
    if tool_name in NEVER_SPILL:
        return float("inf")
    try:
        from tools.budget_config import DEFAULT_BUDGET
        registry_value = DEFAULT_BUDGET.resolve_threshold(tool_name)
        if registry_value == float("inf"):
            return float("inf")
        return min(float(registry_value), float(DEFAULT_THRESHOLD_CHARS))
    except Exception:
        return float(DEFAULT_THRESHOLD_CHARS)


def _spill_message(preview: str, has_more: bool, size: int, path: Path,
                   tool_name: str) -> str:
    kb = size / 1024
    size_str = "%.1f MB" % (kb / 1024) if kb >= 1024 else "%.1f KB" % kb
    msg = SPILL_TAG + "\n"
    msg += (
        "The %s result was %d characters (%s) — too large to put in context "
        "whole.\n" % (tool_name, size, size_str)
    )
    msg += "Full output saved to: %s\n" % path
    msg += (
        "Read that file (Read / Grep with offset+limit) if you need more than "
        "the preview. Do NOT re-run the tool to see the rest.\n\n"
    )
    msg += "Preview (first %d chars):\n" % len(preview)
    msg += preview
    if has_more:
        msg += "\n…"
    msg += "\n" + SPILL_CLOSING_TAG
    return msg


def maybe_spill(tool_name: str, content: str,
                threshold: Optional[float] = None) -> str:
    """Return ``content`` untouched, or a locator block if it is oversized."""
    if not isinstance(content, str):
        return content
    limit = threshold if threshold is not None else threshold_for(tool_name)
    if limit == float("inf") or len(content) <= limit:
        return content

    directory = _storage_dir()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_tool = _UNSAFE.sub("_", tool_name)[:40] or "tool"
    path = directory / ("%s-%s-%s.txt" % (safe_tool, stamp, uuid.uuid4().hex[:8]))

    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        # Falling back to truncation is still better than blowing the context,
        # but say so plainly: a silent truncation is what spill exists to
        # replace.
        logger.warning("spill write failed for %s: %s", tool_name, exc)
        preview, _ = _preview(content, PREVIEW_CHARS)
        return (
            preview
            + "\n\n[Truncated: %d chars; the full output could NOT be saved "
              "(%s). Narrow the query and retry.]" % (len(content), exc)
        )

    _reap(directory)
    preview, has_more = _preview(content, PREVIEW_CHARS)
    logger.info(
        "spilled %s result: %d chars -> %s", tool_name, len(content), path
    )
    return _spill_message(preview, has_more, len(content), path, tool_name)


def stats() -> str:
    """Small self-report, useful from the doctor and from tests."""
    directory = _storage_dir()
    try:
        files = [p for p in directory.iterdir() if p.is_file()]
    except OSError:
        files = []
    return json.dumps({
        "dir": str(directory),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "threshold_chars": DEFAULT_THRESHOLD_CHARS,
        "ttl_days": TTL_DAYS,
    }, ensure_ascii=False)
