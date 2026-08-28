"""Was the kanban worker's task actually left unfinished?

Local overlay module (not upstream).

Upstream's stop-guard (``agent/kanban_stop.py``) answers that question by
walking Hermes' own message list for an assistant ``tool_calls`` entry named
``kanban_complete`` / ``kanban_block``. That predicate is blind on the two
paths this fleet actually uses:

* **native ACP tool-calling** — the engine drives MCP tools inside one API
  call, so Hermes' message list carries no ``tool_calls`` at all (a worker
  that ran a hundred tools logs "0 tool calls");
* **the Bash CLI** — profiles are told to close cards with
  ``hermes kanban complete <id>``, which is a shell command, not a tool call,
  and therefore invisible to any transcript predicate by construction.

Observed 2026-08-04 in twelve worker logs across three profiles: every one of
those tasks reached ``done``/``archived``, and every one was nudged anyway
with "Task ... is still `running`" — a claim the guard never checked. The
nudge then cost three failed API calls, because completing the card also
``rmtree``s the scratch workspace that is the worker's own cwd (see
``kanban_db._cleanup_workspace``), so the next client construction died in
``os.getcwd()``.

The fix is the same rule the profiles themselves are held to: **a task is
closed when the board says so, and nothing else.** Ask the board.

Fail-open by design: any error here returns False, i.e. upstream's behaviour
(nudge). A guard that silently disables itself on a database hiccup is worse
than one that occasionally nudges twice.
"""

from __future__ import annotations

import os
from typing import Any

# ``blocked`` counts: ``kanban_block`` is a terminal move for the guard, and a
# card the dispatcher auto-blocked is not "still running" either.
_BOARD_TERMINAL = frozenset(
    {"done", "archived", "blocked", "failed", "cancelled"}
)

_TERMINAL_TOOLS = ("kanban_complete", "kanban_block")


def _task_id(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("HERMES_KANBAN_TASK") or "").strip()


def engine_called_terminal_tool(agent: Any) -> bool:
    """True if the ENGINE called a terminal board tool during this run.

    Reads the native-call ledger off the agent, not off ``agent.client``:
    per-request ACP clients are discarded when the call returns, so the client
    a caller can still reach usually recorded nothing (same trap as
    ``acp_native_calls``). MCP exposes the tool as ``kanban_complete`` or as
    ``mcp__<server>__kanban_complete`` depending on the naming mode, so match
    on the tail rather than on equality.
    """
    try:
        calls = list(getattr(agent, "_acp_native_tool_calls", None) or [])
        if not calls:
            client = getattr(agent, "client", None)
            calls = list(getattr(client, "_native_tool_calls", None) or [])
        for entry in calls:
            name = str(
                entry.get("name") if isinstance(entry, dict)
                else getattr(entry, "name", "")
            )
            tail = name.rsplit("__", 1)[-1]
            if tail in _TERMINAL_TOOLS:
                return True
    except Exception:
        return False
    return False


def board_says_terminal(task_id: str | None = None) -> bool:
    """True if the board already records this worker's task as terminal.

    Authoritative and path-independent: it is true whether the card was closed
    by an MCP tool, by ``hermes kanban complete`` in a shell, or by another
    process entirely.
    """
    tid = _task_id(task_id)
    if not tid:
        return False
    try:
        from hermes_cli.kanban_db import connect_closing, get_task

        with connect_closing() as conn:
            task = get_task(conn, tid)
        if task is None:
            return False
        return str(getattr(task, "status", "") or "").lower() in _BOARD_TERMINAL
    except Exception:
        return False


def worker_already_finished(agent: Any = None, task_id: str | None = None) -> bool:
    """Should the stop-guard stay quiet for this turn?

    The cheap in-memory check runs first so a healthy engine-driven completion
    costs no database read at all.
    """
    if not _task_id(task_id):
        return False
    return engine_called_terminal_tool(agent) or board_says_terminal(task_id)


__all__ = [
    "board_says_terminal",
    "engine_called_terminal_tool",
    "worker_already_finished",
]
