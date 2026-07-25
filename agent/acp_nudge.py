"""Counting "work done" for the self-improvement nudge on the ACP path.

Local overlay module (not upstream).

Hermes triggers its background skill review after N tool-calling iterations
(``skills.creation_nudge_interval``, default 15). On a normal provider the
agent loop runs one iteration per tool batch, so an active turn contributes
several and the threshold is reached within a turn or two.

On the native ACP path the engine drives its own tools inside a single API
call, so the loop sees exactly ONE iteration per turn no matter how much work
happened — a turn that ran thirty tools counts the same as "hello". The
threshold then needs 15 *turns of conversation*, and since the counter lives
on the cached agent it also resets whenever the agent is rebuilt (config
change, gateway restart). Result observed 2026-07-25: the review had never
fired once on the ACP fleet — no skills, no memory writes, nothing in the logs.

The ledger of native tool calls already records what the engine actually did,
so it is used here as the unit of work: the counter advances by the number of
engine tool calls made during the turn, which restores the original intent
("review after roughly this much activity") on both paths.
"""

from __future__ import annotations

from typing import Any

_MARK = "_acp_nudge_seen_calls"


def skill_iterations_delta(agent: Any) -> int:
    """How much to advance the skill-nudge counter for the iteration just run.

    Returns 1 on non-ACP paths (unchanged upstream behaviour). On the ACP path
    returns the number of engine tool calls recorded since the last check, and
    at least 1 so a tool-free turn still counts as progress exactly as before.
    Never raises: a miscount must not break the agent loop.
    """
    try:
        calls = getattr(agent, "_acp_native_tool_calls", None)
        if calls is None:
            return 1
        seen = int(getattr(agent, _MARK, 0) or 0)
        total = len(calls)
        if total < seen:  # ledger reset between runs
            seen = 0
        setattr(agent, _MARK, total)
        return max(1, total - seen)
    except Exception:
        return 1
