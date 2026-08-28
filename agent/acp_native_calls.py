"""Reading the native-tool ledger of an ACP run.

Local overlay module (not upstream). One function, but it encodes a fact that
is easy to get wrong: the ledger must be read off the AGENT, not off
``agent.client``. Per-request ACP clients are created with ``shared=False`` and
discarded when the call returns (run_agent._create_request_openai_client), so
the client a caller can reach afterwards is usually not the one that recorded
anything — verified 2026-07-25, when a background review created a skill
through the MCP bridge while ``agent.client``'s ledger read zero.

Kept out of agent/curator.py so that upstream file carries a two-line call
instead of the rationale plus the fallback chain.
"""

from __future__ import annotations

from typing import Any


def native_calls_for_report(agent: Any) -> list[dict[str, str]]:
    """Tool calls the engine made during this run, newest last.

    Returns ``[{"name": ..., "arguments": ...}]`` — the shape curator's report
    and its consolidation heuristics expect. Never raises: a missing ledger is
    an empty run, not an error.
    """
    try:
        calls = list(getattr(agent, "_acp_native_tool_calls", None) or [])
        if not calls:
            # Fallback for runs that never went through a per-request client.
            client = getattr(agent, "client", None)
            calls = list(getattr(client, "_native_tool_calls", None) or [])
    except Exception:
        return []
    out: list[dict[str, str]] = []
    for entry in calls:
        if isinstance(entry, dict) and entry.get("name"):
            out.append({
                "name": str(entry.get("name")),
                "arguments": entry.get("arguments") or "",
            })
    return out


# Action labels mirror what summarize_background_review_actions() produces for
# non-ACP providers, so the user sees the same wording either way.
_SKILL_TOOLS = ("skill_manage", "skill_create", "skill_edit")
_MEMORY_TOOLS = ("memory", "memory_write", "remember")


def native_review_actions(
    agent: Any,
    notification_mode: str = "on",
    parent: Any = None,
    since: int = 0,
) -> list[str]:
    """Summarise a background review from the engine's own tool calls.

    Upstream builds this summary by walking the review session for ``tool``
    role messages. On the native ACP path there are none — the engine runs
    tools inside a single API call — so a review that genuinely created a
    skill produced an empty summary and stayed silent (observed 2026-07-25:
    the skill ``memory-and-skill-hygiene`` was written to disk while the user
    saw nothing). The ledger records what actually ran, so it is used instead.

    Returns [] when the mode is "off" or nothing relevant happened.
    """
    if str(notification_mode or "on").lower() == "off":
        return []
    calls = native_calls_for_report(agent)
    if not calls and parent is not None:
        # Which object ends up holding the ledger depends on whether the fork
        # got its own ACP client or reused the parent's — both happen. Falling
        # back to the parent covers the second case; `since` is where its
        # ledger stood before the review started, so the caller's own earlier
        # tool calls are not reported as review work.
        parent_calls = native_calls_for_report(parent)
        calls = parent_calls[since:] if since <= len(parent_calls) else []
    actions: list[str] = []
    for call in calls:
        name = str(call.get("name") or "").lower()
        args = call.get("arguments") or ""
        if any(t in name for t in _SKILL_TOOLS):
            label = _skill_action_label(args)
            if label:
                actions.append(label)
        elif any(t in name for t in _MEMORY_TOOLS):
            actions.append("Memory updated")
    # Preserve order, drop repeats — the same skill touched twice is one action.
    return list(dict.fromkeys(actions))


def _skill_action_label(arguments: str) -> str:
    """"Skill 'name' created" from a skill_manage call, best effort."""
    import json as _json

    action = name = ""
    try:
        parsed = _json.loads(arguments) if arguments else {}
        if isinstance(parsed, dict):
            action = str(parsed.get("action") or parsed.get("op") or "")
            name = str(parsed.get("name") or parsed.get("skill") or
                       parsed.get("skill_name") or "")
    except Exception:
        pass
    verb = {
        "create": "created", "add": "created", "write": "created",
        "update": "updated", "edit": "updated", "patch": "updated",
        "delete": "archived", "archive": "archived",
    }.get(action.lower(), "")
    if not verb:
        # Unknown/absent action: still tell the user a skill was touched,
        # rather than dropping the notification entirely.
        return f"Skill '{name}' updated" if name else "Skill library updated"
    return f"Skill '{name}' {verb}" if name else f"Skill library {verb}"
