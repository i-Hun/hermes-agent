"""Text / ``--json`` output helpers shared by the ``hermes kanban`` CLI modules."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable, Iterable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workflow

_STATUS_ICONS = {c.key: c.icon for c in kanban_workflow.DEFAULT_WORKFLOW} | {kanban_workflow.ARCHIVED: "—"}

_TASK_DICT_FIELDS = (
    "id", "title", "body", "assignee", "status", "priority", "tenant",
    "workspace_kind", "workspace_path", "branch_name", "project_id",
    "created_by", "created_at", "started_at", "completed_at", "result",
    "skills", "max_runtime_seconds", "max_retries", "model_override", "provider_override",
    "session_id", "workflow_template_id", "current_step_key", "completion_contract", "last_failure_error",
)
_SHOW_RUN_FIELDS = (
    "id", "profile", "step_key", "status", "outcome", "summary", "error",
    "metadata", "worker_pid", "started_at", "ended_at",
)
_RUNS_RUN_FIELDS = (
    "id", "profile", "status", "outcome", "started_at", "ended_at",
    "summary", "error", "metadata", "worker_pid", "step_key",
)
_ATTACHMENT_FIELDS = ("id", "filename", "content_type", "size", "uploaded_by", "stored_path", "created_at")


def _fmt_ts(ts: Optional[int]) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else ""


def _print_json(obj: Any, *, ascii: bool = False) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=ascii))


def _json_out(args: argparse.Namespace, obj: Any, *, ascii: bool = False) -> bool:
    """Print ``obj`` as JSON and return True when ``--json`` was passed."""
    if not getattr(args, "json", False):
        return False
    _print_json(obj, ascii=ascii)
    return True


def _fmt_counts(counts: dict, empty: str = "") -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or empty


def _err(msg: str, rc: int = 1) -> int:
    print(msg, file=sys.stderr)
    return rc


def _bulk_apply(ids: Iterable[str], op: Callable[[str], Any],
                ok_msg: Callable[[str], str], fail_msg: Callable[[str], str]) -> int:
    """Run ``op(tid) -> bool`` per id, print ok/fail lines, exit 1 if any failed."""
    failed = False
    for tid in ids:
        if op(tid):
            print(ok_msg(tid))
        else:
            failed = True
            print(fail_msg(tid), file=sys.stderr)
    return 1 if failed else 0


def _fmt_duration(seconds: float) -> str:
    """Compact age: ``45s``, ``12m``, ``3h42m``, ``2d5h`` (no spaces, one field)."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, _secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours}h"


def _fmt_task_line(
    t: kb.Task, *, slot_seconds: Optional[int] = None, limit_seconds: Optional[int] = None,
) -> str:
    """One ``list`` row. ``slot_seconds``/``limit_seconds`` are passed for a
    ``running`` card only: how long the current run has held its dispatch slot and
    the wall-clock cap it will be held to, so "it has been sitting there for six
    hours" reads off the board instead of a ``ps`` investigation."""
    icon = _STATUS_ICONS.get(t.status, "?")
    assignee = t.assignee or "(unassigned)"
    tenant = f" [{t.tenant}]" if t.tenant else ""
    line = f"{icon} {t.id}  {t.status:8s}  {assignee:20s}{tenant}  {t.title}"
    if slot_seconds is not None:
        slot = f"slot {_fmt_duration(slot_seconds)}"
        slot += f"/{_fmt_duration(limit_seconds)}" if limit_seconds else "/no cap"
        line += f"  [{slot}]"
    return line


def _obj_dict(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in fields}


def _task_to_dict(t: kb.Task) -> dict[str, Any]:
    d = _obj_dict(t, _TASK_DICT_FIELDS)
    d["skills"] = list(t.skills) if t.skills else []
    return d
