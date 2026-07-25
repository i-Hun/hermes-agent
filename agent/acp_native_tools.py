"""ACP native-tools bridge helpers — MCP wiring + engine activity rendering.

Local overlay module (not upstream). Two concerns that belong together and
neither of which needs the client object:

* whether the engine drives Hermes tools natively over MCP
  (``HERMES_ACP_NATIVE_TOOLS``) and the server spec handed to it;
* turning an ACP ``tool_call`` update into the fields Hermes' progress rail
  expects, so engine-internal work shows up in the chat progress bubble.

Kept out of ``copilot_acp_client.py`` on purpose: that file is upstream, and
local lines there are what a ``hermes update`` has to merge. See also
``agent/acp_permission_guard.py`` and ``gateway/progress_collapse.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

def _acp_native_tools_enabled() -> bool:
    return os.getenv("HERMES_ACP_NATIVE_TOOLS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _hermes_agent_root() -> str:
    """Absolute path of the hermes-agent package root (…/hermes-agent)."""
    return str(Path(__file__).resolve().parents[1])


def _hermes_tools_mcp_servers() -> list[dict[str, Any]]:
    """ACP session/new.mcpServers entry exposing Hermes' curated tool subset.

    Uses the running interpreter (sys.executable → the venv python) and the
    derived package root, so it keeps working across relocations/updates. The
    engine spawns this under the bridge's scrubbed env, so we forward exactly
    what the server needs to load Hermes config and dispatch tools.
    """
    import sys as _sys

    py = os.getenv("HERMES_ACP_MCP_PYTHON", "").strip() or _sys.executable
    root = _hermes_agent_root()
    # Imported lazily: copilot_acp_client imports THIS module, so a top-level
    # import back into it would be circular.
    from agent.copilot_acp_client import _resolve_home_dir

    home = _resolve_home_dir()
    env_pairs = [
        {"name": "PYTHONPATH", "value": root},
        {"name": "HOME", "value": home},
        {"name": "HERMES_QUIET", "value": "1"},
        {"name": "HERMES_REDACT_SECRETS", "value": "true"},
    ]
    # Forward profile/env context the server (and its delegate_task children)
    # needs: HERMES_HOME scopes the profile, HERMES_PROFILE labels it, PATH lets
    # spawned one-shot hermes children find their own toolchain, and the
    # delegation depth counter bounds recursive fan-out across processes.
    for var in ("HERMES_HOME", "HERMES_PROFILE", "PATH", "HERMES_ACP_DELEGATE_DEPTH"):
        val = os.getenv(var, "").strip()
        if val:
            env_pairs.append({"name": var, "value": val})
    # Extended server (adds stateless memory/session_search/todo/delegate_task
    # shims) is the default; HERMES_ACP_MCP_MODULE overrides, and the base
    # module remains available as a fallback.
    module = (
        os.getenv("HERMES_ACP_MCP_MODULE", "").strip()
        or "agent.transports.hermes_acp_tools_server"
    )
    return [
        {
            "name": "hermes-tools",
            "command": py,
            "args": ["-m", module],
            "env": env_pairs,
        }
    ]



def _acp_tool_started_fields(
    update: dict[str, Any],
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Map an ACP tool_call update onto Hermes tool-progress fields.

    Returns (tool_name, preview, args) for the gateway's
    ``tool_progress_callback("tool.started", ...)``.

    Progress lines describe the ACTION, not the mechanics — the same way
    OpenClaw's progress drafts read ("🛠️ check JS syntax for /tmp/app.js")
    rather than echoing raw argv. The Claude engine already ships a one-line
    ``description`` alongside every Bash call ("List first 3 entries in
    /tmp"), so that is the preferred preview; the command itself is dropped
    from the args so the gateway renders a plain line instead of a fenced
    code block. Without a description we fall back to the most identifying
    argument (path/query/url), and only as a last resort to the command.
    """
    if not isinstance(update, dict):
        return "tool", None, None

    raw = update.get("rawInput") or update.get("raw_input") or update.get("input")
    args = raw if isinstance(raw, dict) else None

    # Tool label. The engine sets ``title`` to the command line itself on
    # tool_call_update, which would echo the command in the label — so a
    # title that matches the command is discarded in favour of the kind.
    title = update.get("title")
    name = title.strip() if isinstance(title, str) and title.strip() else ""
    if args:
        _cmd_raw = args.get("command") or args.get("cmd")
        if isinstance(_cmd_raw, str) and name and name.strip() == _cmd_raw.strip():
            name = ""
    if not name:
        tk = update.get("kind") or update.get("toolKind")
        name = str(tk).strip() if tk else "tool"

    description = ""
    preview = ""
    if args:
        desc = args.get("description") or args.get("summary")
        if isinstance(desc, str) and desc.strip():
            description = desc.strip().splitlines()[0]
        for key in ("path", "file_path", "query", "url", "pattern", "command", "cmd"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                preview = val.strip().splitlines()[0]
                break

    if description:
        # Action description wins, and the command is withheld so the
        # gateway's terminal path (fenced block) is not triggered.
        safe_args = {k: v for k, v in args.items() if k not in ("command", "cmd")} if args else None
        return _ACTION_TOOL_LABEL, description, safe_args or None

    if args:
        _cmd = args.get("command") or args.get("cmd")
        if isinstance(_cmd, str) and _cmd.strip():
            # No description from the engine (rare — Bash calls carry one).
            # Show only the program being run, never the argv: progress lines
            # are a public side channel and raw commands leak paths, hosts and
            # occasionally secrets into the chat.
            return _ACTION_TOOL_LABEL, _program_name(_cmd), None

    return name, preview or None, args


def _program_name(command: str) -> str:
    """First real program name in a shell command, for a preview-safe label.

    Skips leading env assignments and sudo so ``FOO=1 sudo systemctl …``
    reads as "systemctl". Returns a bare basename — never arguments.
    """
    try:
        import shlex

        parts = shlex.split(command)
    except Exception:
        parts = command.split()
    for tok in parts:
        if "=" in tok and not tok.startswith("/") and not tok.startswith("-"):
            continue  # env assignment prefix
        if tok in ("sudo", "env", "command", "exec", "nohup", "time"):
            continue
        return tok.rstrip("/").rsplit("/", 1)[-1][:40]
    return "command"


# Label shown for engine-driven actions. Neutral on purpose: the useful part
# of the line is the description that follows it, not the tool's identity.
_ACTION_TOOL_LABEL = "terminal"


def _normalize_native_tool_name(name: str) -> str:
    """Reduce an engine tool title to a bare Hermes tool name.

    MCP-bridged tools may arrive prefixed ("mcp__hermes-tools__skill_manage",
    "hermes-tools:skill_manage"); downstream consumers (curator run
    classification) match on the bare name. Titles with spaces are display
    strings, not identifiers — returned as-is.
    """
    n = (name or "").strip()
    if not n or " " in n:
        return n or "tool"
    if "__" in n:
        n = n.split("__")[-1]
    for sep in (":", "/"):
        if sep in n:
            n = n.rsplit(sep, 1)[-1]
    return n or "tool"


def _acp_tool_activity_label(update: dict[str, Any]) -> str:
    """Build a compact 'what the engine is doing' label from a tool_call update.

    Used to surface ACP native-tool activity as interim commentary so the
    gateway keeps a live 'working…' indicator during the otherwise-silent gaps
    between text bursts while the Claude engine runs its own tools.
    """
    if not isinstance(update, dict):
        return ""
    title = update.get("title")
    if isinstance(title, str) and title.strip():
        label = title.strip()
    else:
        tk = update.get("kind") or update.get("toolKind")
        label = str(tk).strip() if tk else "tool"
    # Best-effort argument hint (command / path / query) without dumping raw JSON.
    raw = update.get("rawInput") or update.get("raw_input") or update.get("input")
    hint = ""
    if isinstance(raw, dict):
        for key in ("command", "cmd", "path", "file_path", "query", "url", "pattern"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                hint = val.strip().splitlines()[0]
                break
    if hint:
        if len(hint) > 60:
            hint = hint[:57] + "…"
        return f"{label}: {hint}"
    return label


