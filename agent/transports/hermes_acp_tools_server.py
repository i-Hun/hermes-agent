"""Extended hermes-tools MCP server for the ACP native-tools bridge.

Supersedes ``agent.transports.hermes_tools_mcp_server`` for sessions running
with HERMES_ACP_NATIVE_TOOLS=1: exposes the same curated Hermes tool subset
PLUS stateless implementations of the four agent-loop tools that
``model_tools.handle_function_call`` refuses to dispatch (they normally need
the running AIAgent):

  - memory          -> tools.memory_tool.memory_tool with a fresh on-disk
                       MemoryStore per call (load_on_disk_store). File locks +
                       drift detection in MemoryStore make cross-process
                       writes safe; the gateway agent picks changes up via its
                       own drift detection.
  - session_search  -> tools.session_search_tool.session_search, which builds
                       its own SessionDB when none is passed. The
                       current-session-lineage guard is off (we don't know the
                       gateway session id here) — acceptable: worst case the
                       model re-reads its own transcript.
  - todo            -> in-process TodoStore. The engine spawns ONE MCP server
                       per ACP session, so module state == per-session state,
                       same semantics as the AIAgent-owned store.
  - delegate_task   -> synchronous one-shot ``hermes -z`` subprocesses (the
                       ask-hermes mechanism). Children inherit this process'
                       environment (HERMES_HOME -> same profile). Batch tasks
                       run in a small thread pool. role/toolsets are accepted
                       but ignored; ``background: true`` is honoured since
                       2026-08-13 and hands the run to agent.acp_jobs. Depth is
                       bounded via HERMES_ACP_DELEGATE_DEPTH.

It also adds two things the bridge has no other place for:

  - job_* tools    -> agent.acp_jobs: durable background work in a transient
                      systemd unit OUTSIDE the gateway cgroup. The engine's own
                      background work dies with the turn and `setsid` is not
                      enough (cgroup cleanup on gateway restart kills it).
  - result spill   -> agent.acp_spill: oversized results are written to disk and
                      replaced with a locator. Upstream's three-layer spill in
                      tools/tool_result_storage is called from
                      agent/tool_executor.py, which the ACP path never enters.

Update-resilience: this file is intentionally UNTRACKED in the hermes-agent
git checkout (install method: git) — ``git pull`` / ``hermes update`` never
touch it. It rides only module-level APIs (model_tools.get_tool_definitions /
handle_function_call, tools.*_tool entry points) and imports the base server
module only for its EXPOSED_TOOLS list, with a frozen fallback copy. If an
update renames those APIs the server fails loudly at startup (engine logs),
not silently.

Run with: python -m agent.transports.hermes_acp_tools_server
Spawned by: copilot_acp_client._hermes_tools_mcp_servers() (module name
            overridable via HERMES_ACP_MCP_MODULE).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard (non-loop) tools: reuse the base module's curated list so upstream
# additions flow through automatically; freeze a fallback in case an update
# renames the constant.
# ---------------------------------------------------------------------------

_FALLBACK_BASE_TOOLS: tuple[str, ...] = (
    "web_search", "web_extract",
    "browser_navigate", "browser_click", "browser_type", "browser_press",
    "browser_snapshot", "browser_scroll", "browser_back",
    "browser_get_images", "browser_console", "browser_vision",
    "vision_analyze", "image_generate",
    "skill_view", "skills_list", "skill_manage", "text_to_speech",
    "kanban_complete", "kanban_block", "kanban_comment", "kanban_heartbeat",
    "kanban_show", "kanban_list", "kanban_create", "kanban_unblock",
    "kanban_link",
)


def _base_exposed_tools() -> tuple[str, ...]:
    try:
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        return tuple(EXPOSED_TOOLS)
    except Exception:
        logger.warning("base EXPOSED_TOOLS unavailable — using frozen fallback")
        return _FALLBACK_BASE_TOOLS


# ---------------------------------------------------------------------------
# Agent-loop tool shims (stateless / process-local)
# ---------------------------------------------------------------------------

LOOP_TOOLS: tuple[str, ...] = ("memory", "session_search", "todo", "delegate_task")

# Tools the upstream MCP list omits but the Hermes agent loop normally has.
# skill_manage is the self-improvement path (the system prompt tells the model
# to save skills with it) — without it the native-tools bridge silently breaks
# skill creation. Stateless (writes to the skills dir), safe to bridge as-is.
EXTRA_TOOLS: tuple[str, ...] = ("skill_manage",)

_TODO_STORE = None  # lazy; one per MCP server process == one per ACP session

# Tools that exist only on this bridge — no Hermes tool definition backs them,
# so they are assembled from their own specs rather than looked up in
# model_tools.get_tool_definitions().
try:
    from agent.acp_jobs import DISPATCH as _JOB_DISPATCH, TOOL_SPECS as _JOB_TOOL_SPECS
except Exception as _exc:  # pragma: no cover - defensive: never block startup
    logger.warning("acp_jobs unavailable — background job tools disabled: %s", _exc)
    _JOB_DISPATCH, _JOB_TOOL_SPECS = {}, ()

try:
    from agent.acp_spill import maybe_spill as _maybe_spill
except Exception as _exc:  # pragma: no cover - defensive
    logger.warning("acp_spill unavailable — large results stay inline: %s", _exc)

    def _maybe_spill(tool_name: str, content: str) -> str:  # type: ignore[misc]
        return content


def _dispatch_memory(args: dict[str, Any]) -> str:
    from tools.memory_tool import load_on_disk_store, memory_tool

    allowed = {
        k: args[k]
        for k in ("action", "target", "content", "old_text", "operations")
        if k in args
    }
    return memory_tool(store=load_on_disk_store(), **allowed)


def _dispatch_session_search(args: dict[str, Any]) -> str:
    from tools.session_search_tool import session_search

    allowed = {
        k: args[k]
        for k in (
            "query", "role_filter", "limit", "session_id",
            "around_message_id", "window", "sort", "profile",
        )
        if k in args
    }
    return session_search(**allowed)


def _dispatch_todo(args: dict[str, Any]) -> str:
    global _TODO_STORE
    from tools.todo_tool import TodoStore, todo_tool

    if _TODO_STORE is None:
        _TODO_STORE = TodoStore()
    return todo_tool(
        todos=args.get("todos"),
        merge=bool(args.get("merge", False)),
        store=_TODO_STORE,
    )


# --- delegate_task -----------------------------------------------------------

_DELEGATE_MAX_DEPTH = 2
_DELEGATE_MAX_WORKERS = 3
_DELEGATE_OUTPUT_CAP = 8000


def _hermes_bin() -> str:
    override = os.getenv("HERMES_BIN", "").strip()
    if override:
        return override
    sibling = Path(sys.executable).with_name("hermes")
    if sibling.exists():
        return str(sibling)
    return shutil.which("hermes") or "hermes"


def _delegate_child_env(depth: int) -> dict[str, str]:
    env = dict(os.environ)
    # The venv entry point must not see our PYTHONPATH (mirrors the
    # ~/.local/bin/hermes wrapper, which unsets both before exec).
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("HERMES_KANBAN_TASK", None)  # children are not kanban workers
    env["HERMES_ACP_DELEGATE_DEPTH"] = str(depth + 1)
    env.setdefault("HERMES_QUIET", "1")
    home = env.get("HOME", str(Path.home()))
    env.setdefault("PATH", home + "/.local/bin:/usr/local/bin:/usr/bin:/bin")
    return env


def _run_one_delegation(task: dict[str, Any], depth: int, timeout: float) -> dict[str, Any]:
    goal = str(task.get("goal") or "").strip()
    if not goal:
        return {"ok": False, "error": "task has no goal"}
    context = str(task.get("context") or "").strip()
    prompt = "[delegate_task] " + goal
    if context:
        prompt += "\n\nContext:\n" + context
    prompt += "\n\nРаботай автономно; результат — в финальном сообщении."

    cmd = [_hermes_bin(), "-z", prompt]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_delegate_child_env(depth),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "goal": goal[:200],
                "error": "subagent timed out after %ds" % int(timeout)}
    except OSError as exc:
        return {"ok": False, "goal": goal[:200],
                "error": "failed to spawn hermes: %s" % exc}

    out = (proc.stdout or "").strip()
    if len(out) > _DELEGATE_OUTPUT_CAP:
        out = "…[truncated head]\n" + out[-_DELEGATE_OUTPUT_CAP:]
    result: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "goal": goal[:200],
        "output": out,
    }
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        result["error"] = "exit code %d" % proc.returncode + (
            ": …" + err[-500:] if err else ""
        )
    return result


def _delegation_prompt(task: dict[str, Any]) -> str:
    goal = str(task.get("goal") or "").strip()
    context = str(task.get("context") or "").strip()
    prompt = "[delegate_task] " + goal
    if context:
        prompt += "\n\nContext:\n" + context
    prompt += "\n\nРаботай автономно; результат — в финальном сообщении."
    return prompt


def _run_background_delegations(tasks: list[dict[str, Any]], depth: int) -> str:
    """Honour ``background: true`` by handing each task to the job runtime."""
    from agent.acp_jobs import start_detached_delegation

    env = _delegate_child_env(depth)
    started: list[dict[str, Any]] = []
    for task in tasks:
        goal = str(task.get("goal") or "").strip()
        if not goal:
            started.append({"ok": False, "error": "task has no goal"})
            continue
        raw = start_detached_delegation(
            prompt=_delegation_prompt(task),
            label=goal[:24],
            env=env,
            hermes_bin=_hermes_bin(),
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": "job runtime returned non-JSON", "raw": raw[:400]}
        payload["goal"] = goal[:200]
        started.append(payload)

    return json.dumps({
        "background": True,
        "started": started,
        "note": (
            "Subagents run as durable background jobs (transient systemd units "
            "outside the gateway cgroup): they survive this turn and a gateway "
            "restart. Poll with job_output / job_list — nothing will be "
            "reported back to you automatically."
        ),
    }, ensure_ascii=False)


def _dispatch_delegate_task(args: dict[str, Any]) -> str:
    depth = 0
    try:
        depth = int(os.getenv("HERMES_ACP_DELEGATE_DEPTH", "0") or 0)
    except ValueError:
        depth = 0
    if depth >= _DELEGATE_MAX_DEPTH:
        return json.dumps({
            "error": (
                "Delegation depth limit reached (depth=%d, max=%d) on the "
                "ACP native-tools bridge." % (depth, _DELEGATE_MAX_DEPTH)
            )
        })

    tasks = args.get("tasks")
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except (json.JSONDecodeError, TypeError):
            tasks = None
    if not isinstance(tasks, list) or not tasks:
        tasks = [{"goal": args.get("goal"), "context": args.get("context")}]
    tasks = [t for t in tasks if isinstance(t, dict)][:8]
    if not tasks:
        return json.dumps({"error": "no valid tasks provided"})

    background = bool(args.get("background")) or any(
        bool(t.get("background")) for t in tasks
    )
    if background and _JOB_TOOL_SPECS:
        return _run_background_delegations(tasks, depth)

    try:
        timeout = float(os.getenv("HERMES_ACP_DELEGATE_TIMEOUT", "600") or 600)
    except ValueError:
        timeout = 600.0

    if len(tasks) == 1:
        results = [_run_one_delegation(tasks[0], depth, timeout)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(len(tasks), _DELEGATE_MAX_WORKERS)
        ) as pool:
            results = list(
                pool.map(lambda t: _run_one_delegation(t, depth, timeout), tasks)
            )

    return json.dumps({
        "results": results,
        "note": (
            "bridged delegate_task: subagents ran as synchronous one-shot "
            "hermes runs in this profile; role/toolsets/background fields "
            "are ignored"
        ),
    }, ensure_ascii=False)


_LOOP_DISPATCH = {
    "memory": _dispatch_memory,
    "session_search": _dispatch_session_search,
    "todo": _dispatch_todo,
    "delegate_task": _dispatch_delegate_task,
}

# Schema description amendments so the model knows the bridge semantics.
_DESCRIPTION_SUFFIX = {
    "delegate_task": (
        " NOTE (ACP bridge): by default subagents run synchronously as one-shot "
        "Hermes runs in this profile and the call returns when they finish. "
        "Pass background=true to hand them to the durable job runtime instead — "
        "they then outlive this turn and you poll them with job_output. "
        "role/toolsets are ignored."
    ),
    "todo": (
        " NOTE (ACP bridge): the list lives for this session only."
    ),
}

# Amending the tool's prose is not enough: upstream documents `background` as
# "DEPRECATED / IGNORED … setting this has no effect", which on this bridge is
# the exact opposite of the truth. A model reads the parameter, believes it,
# and never opts into the durable runtime. Two texts that contradict each other
# are worse than one that is merely terse, so the parameter is re-documented
# where the model actually looks.
_PARAM_DESCRIPTION_OVERRIDE: dict[str, dict[str, str]] = {
    "delegate_task": {
        "background": (
            "ACP bridge: NOT ignored here. false/omitted → subagents run "
            "synchronously and this call returns their results. true → each "
            "subagent becomes a durable background job (transient systemd unit "
            "outside the gateway cgroup) that outlives this turn and a gateway "
            "restart; the call returns job ids and NOTHING comes back on its "
            "own — poll with job_output."
        ),
    },
}


def _amend_param_docs(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Re-document parameters whose upstream text is false on this bridge.

    Copies before writing: the schema dict comes from the shared tool registry,
    and mutating it in place would leak bridge wording into the same process's
    non-bridge callers.
    """
    overrides = _PARAM_DESCRIPTION_OVERRIDE.get(name)
    if not overrides or not isinstance(schema.get("properties"), dict):
        return schema
    amended = dict(schema)
    props = {k: dict(v) if isinstance(v, dict) else v
             for k, v in schema["properties"].items()}
    for param, text in overrides.items():
        if param in props and isinstance(props[param], dict):
            props[param]["description"] = text
    amended["properties"] = props
    return amended


# ---------------------------------------------------------------------------
# Server assembly (mirrors the base module's low-level Server pattern)
# ---------------------------------------------------------------------------


# --- Repeat guard --------------------------------------------------------
# Hermes' own loop guard lives in agent/tool_executor.py, which the engine
# bypasses: it calls these tools over MCP, so a model stuck in a loop can
# repeat one call indefinitely. Observed 2026-07-25: eight identical attempts
# at the same blocked write in fifteen minutes. This is a deliberately small
# stand-in — it does not judge success or progress, it only notices that the
# exact same call is being made over and over and says so, which is enough to
# break a blind retry loop.

_REPEAT_LIMIT = int(os.environ.get("HERMES_ACP_REPEAT_LIMIT", "3") or 3)
_repeat_state: dict[str, Any] = {"key": None, "count": 0}


def _repeat_warning(name: str, arguments: dict[str, Any] | None) -> Optional[str]:
    """Warn once the identical call repeats past the limit, else None."""
    try:
        key = name + "|" + json.dumps(arguments or {}, sort_keys=True, default=str)
    except Exception:
        key = name
    if key == _repeat_state["key"]:
        _repeat_state["count"] += 1
    else:
        _repeat_state["key"] = key
        _repeat_state["count"] = 1
    if _repeat_state["count"] <= _REPEAT_LIMIT:
        return None
    return json.dumps({
        "error": "repeated_call",
        "tool": name,
        "attempts": _repeat_state["count"],
        "message": (
            f"This exact {name} call has now been made {_repeat_state['count']} "
            "times with identical arguments. Repeating it will not produce a "
            "different result — change the arguments, use a different tool, or "
            "tell the user what is blocking you."
        ),
    }, ensure_ascii=False)


def _build_server() -> Any:
    try:
        from mcp.server.lowlevel import Server
        import mcp.types as types
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            "hermes-acp-tools MCP server requires the 'mcp' package: %s" % exc
        ) from exc

    from model_tools import get_tool_definitions, handle_function_call

    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed: list[Any] = []
    seen: set[str] = set()
    for name in _base_exposed_tools() + EXTRA_TOOLS + LOOP_TOOLS:
        if name in seen:
            continue
        seen.add(name)
        spec = all_defs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered in this Hermes process", name)
            continue
        description = (spec.get("description") or ("Hermes %s tool" % name))
        description += _DESCRIPTION_SUFFIX.get(name, "")
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}
        params_schema = _amend_param_docs(name, params_schema)
        exposed.append(
            types.Tool(name=name, description=description, inputSchema=params_schema)
        )

    # Bridge-only tools: no Hermes tool definition backs them, so they carry
    # their own schema. Registered last so an upstream tool of the same name
    # would win — we are the guest here.
    for spec in _JOB_TOOL_SPECS:
        if spec["name"] in seen:
            logger.warning(
                "not registering bridge tool %s — a Hermes tool already owns "
                "that name", spec["name"],
            )
            continue
        seen.add(spec["name"])
        exposed.append(types.Tool(
            name=spec["name"],
            description=spec["description"],
            inputSchema=spec["schema"],
        ))

    exposed_by_name = {t.name: t for t in exposed}
    server = Server("hermes-tools")

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return list(exposed)

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        import anyio

        if name not in exposed_by_name:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "unknown tool %s" % name, "tool": name}),
            )]
        repeat = _repeat_warning(name, arguments)
        if repeat is not None:
            logger.warning("repeat guard tripped for %s", name)
            return [types.TextContent(type="text", text=repeat)]

        handler = _LOOP_DISPATCH.get(name) or _JOB_DISPATCH.get(name)
        spilled = True
        try:
            if handler is not None:
                result = await anyio.to_thread.run_sync(
                    lambda: handler(arguments or {})
                )
            else:
                result = await anyio.to_thread.run_sync(
                    lambda: handle_function_call(name, arguments or {})
                )
        except Exception as exc:
            logger.exception("tool %s raised", name)
            # An error payload is small and is the whole answer: spilling it
            # would make the model read a file to learn that something failed.
            result = json.dumps({"error": str(exc), "tool": name})
            spilled = False
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        if spilled:
            result = _maybe_spill(name, result)
        return [types.TextContent(type="text", text=result)]

    logger.info(
        "hermes-acp-tools MCP server registered %d tools (%d loop shims, %d job tools)",
        len(exposed),
        sum(1 for t in exposed if t.name in LOOP_TOOLS),
        sum(1 for t in exposed if t.name in _JOB_DISPATCH),
    )
    return server


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write("hermes-acp-tools MCP server cannot start: %s\n" % exc)
        return 2

    import anyio
    from mcp.server.stdio import stdio_server

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    try:
        anyio.run(_run)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-acp-tools MCP server crashed")
        sys.stderr.write("hermes-acp-tools MCP server error: %s\n" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
