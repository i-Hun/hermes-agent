"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend. Each request starts a short-lived ACP session, sends the formatted
conversation as a single prompt, collects text chunks, and converts the result
back into the minimal shape Hermes expects from an OpenAI client.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error, is_write_approval_required
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_log = logging.getLogger("agent.acp_bridge")


class ACPBridgeEvicted(RuntimeError):
    """This bridge died because Hermes killed it — not because the model failed.

    Local overlay (2026-07-26). Two Hermes-owned kills leave a dead ACP
    subprocess that is indistinguishable from a provider outage at the wire:
    the one-live-bridge eviction below, and an interrupt/redirect aborting the
    turn that owns the bridge. Both surface as `rc=-15` (our SIGTERM), and the
    failover layer used to read that as "primary model failed" and spend a
    second call on the backup model. Raising a distinct type lets the caller
    tell "we did this" from "the provider did this".
    """


# One-live-bridge-per-session invariant — a belt over agent/turn_lease.py. Two
# concurrent turns on the SAME hermes session must never both hold a live ACP
# bridge: interleaved writes corrupt the single engine transcript (the
# "Repaired N message-alternation violation(s)" symptom). Keyed by the hermes
# session id (agent.session_id); the newest spawn wins and terminates any stale
# bridge left behind by a /stop or a fail-open turn overlap.
_LIVE_BRIDGES: dict[str, "subprocess.Popen[str]"] = {}
_LIVE_BRIDGES_LOCK = threading.Lock()


def _bridge_slot(agent: object) -> object:
    """Registry key for this agent's bridge.

    Local overlay (2026-07-26). The background memory/skill review runs in a
    forked agent that INHERITS the parent's ``session_id``, so both claimed the
    same slot and "newest spawn wins" made them terminate each other mid-turn
    (`ACP process exited (rc=-15) during 'initialize'` → spurious failover to
    the backup model). The fork is a separate conversation with the engine and
    deserves its own slot; the one-live-bridge invariant still holds within
    each of the two lanes.
    """
    session_id = getattr(agent, "session_id", None)
    if not session_id:
        return None
    if getattr(agent, "_hermes_bg_review", False):
        return f"{session_id}:bg-review"
    return session_id


def _claim_session_bridge(session_id, proc) -> None:
    """Make ``proc`` the sole live bridge for ``session_id``.

    If a concurrent turn on the same hermes session still owns a live bridge
    (turn_lease failed open on a long turn, or a /stop left the old ACP process
    running), terminate it first — two turns interleaving writes into one engine
    transcript is what produces the "Repaired N message-alternation violation(s)"
    corruption. Newest spawn wins. Within a single turn's own failover this is a
    no-op: the primary bridge is closed + released before the retry spawns.
    """
    if not session_id:
        return
    with _LIVE_BRIDGES_LOCK:
        prev = _LIVE_BRIDGES.get(session_id)
        if prev is not None and prev is not proc and prev.poll() is None:
            # Local overlay (2026-07-26): stamp the victim BEFORE the signal so
            # the thread blocked in its `_request` loop reports an eviction
            # instead of "exited (rc=-15) — usually a revoked token", and so the
            # failover layer knows not to degrade the model over our own kill.
            setattr(prev, "_hermes_evicted", "a newer turn claimed this session's bridge")
            _log.warning(
                "ACP bridge evicted: slot=%s pid=%s superseded by pid=%s "
                "(one-live-bridge invariant)",
                session_id, prev.pid, proc.pid,
            )
            try:
                prev.terminate()
                prev.wait(timeout=2)
            except Exception:
                try:
                    prev.kill()
                except Exception:
                    pass
        _LIVE_BRIDGES[session_id] = proc


def _release_session_bridge(session_id, proc) -> None:
    """Drop ``proc`` from the registry, but only if it still owns the slot
    (a newer turn may have already replaced it)."""
    if not session_id:
        return
    with _LIVE_BRIDGES_LOCK:
        if _LIVE_BRIDGES.get(session_id) is proc:
            _LIVE_BRIDGES.pop(session_id, None)

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)

# Stderr fingerprint of the deprecated `gh copilot` CLI extension
# (https://github.blog/changelog/2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension).
# We require BOTH the literal product name ("gh-copilot") AND a deprecation
# marker, so generic stderr from the NEW `@github/copilot` CLI — whose repo
# is github.com/github/copilot-cli and which legitimately mentions "copilot-cli"
# in its own banners and error messages — doesn't get misclassified as the
# deprecated extension.
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""

    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    # Copilot ACP is a model-driving CLI executor: it legitimately needs LLM
    # provider credentials. Route through the central helper so Tier-1 secrets
    # (gateway bot tokens, GitHub auth, infra) are still stripped (#29157).
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _acp_env_with_model(model_override, requested=None):
    """Subprocess env for the ACP wrapper, with the engine model always pinned.

    The wrapper (claude-acp-run.sh) has its own resolution chain ending in a
    hardcoded default. Local overlay: pin the value Python resolved
    (agent/acp_config -- single source) on EVERY spawn, not just on a failover
    re-spawn, so the wrapper never has to guess and the two can't drift.
    ``model_override`` still forces a specific model for a per-call retry.
    """
    from agent.acp_config import resolve_model

    env = _build_subprocess_env()
    env["HERMES_ACP_MODEL"] = str(model_override or resolve_model(requested))
    return env


def _acp_failover_enabled():
    return os.getenv("HERMES_ACP_FAILOVER", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def _acp_failover_model(primary=None):
    """The model a failed primary attempt hands over to.

    ``model.acp_fallback`` in the profile's ``config.yaml`` when it names one
    -- until 2026-08-10 this was ``HERMES_ACP_FALLBACK_MODEL`` in ``.env``,
    still honoured with a deprecation warning so a profile that has not migrated
    does not silently lose its chosen downgrade. Otherwise **the primary model
    itself**, which ``run_prompt_with_failover`` reads as "no distinct fallback"
    and so declines to retry elsewhere.

    The old default was a hardcoded ``claude-opus-4-8``: a model no profile
    ever chose, reached whenever the primary raised. On 2026-08-02 that moved
    82k output of founder's work off its configured ``claude-sonnet-5`` and onto
    opus twice, leaving one WARNING as the only trace -- the token audit found
    it, not the operator. Failing loudly on the chosen model beats quietly
    answering on another one; a profile that wants a real downgrade names it
    in its own ``config.yaml``.

    ``primary`` is passed by the caller so the two sides compare the SAME
    resolution. Reading it here independently would diverge on the CLI path,
    where a session override may have chosen the primary -- the "equal" case
    would then miss and a downgrade fire anyway.
    """
    try:
        from agent.acp_config import fallback_model

        explicit = fallback_model()
    except Exception:  # pragma: no cover - overlay import guard
        explicit = os.getenv("HERMES_ACP_FALLBACK_MODEL", "").strip()
    if explicit:
        return explicit
    if primary:
        return str(primary)
    try:
        return _acp_effective_model()
    except Exception:
        return ""


def _acp_effective_model(requested=None):
    """The model the wrapper will run, resolved from the single source.

    Thin alias for ``agent.acp_config.resolve_model`` -- the chain, the
    ``config.yaml``-is-the-source rule and the fallback policy all live there.
    Passing
    ``requested`` (what Hermes asked for) matters: with nothing configured this
    used to land on a hardcoded ``claude-opus-4-8`` while Hermes recorded
    ``requested_model: claude-opus-5``, which is exactly the silent substitution
    found on the CLI path 2026-07-28.
    """
    from agent.acp_config import resolve_model

    return resolve_model(requested)


# --- Native MCP tool-calling (fable refusal fix) ------------------------------
# The legacy path injects a "<tool_call>{...} in OpenAI shape" instruction into
# the prompt and regex-parses tool calls back out of the model's TEXT. Claude
# Fable 5's frontier safeguard classifies that instruction as
# `reasoning_extraction` and REFUSES, so the engine silently falls back to opus.
#
# Native mode instead hands Hermes' tools to the real claude engine as a proper
# ACP MCP server (session/new.mcpServers) and lets the engine orchestrate with
# first-class tool-use — exactly how openclaw's claude-cli runtime works. No
# shim instruction, no refusal, and billing stays first-party (the bridge is
# already on the subscription OAuth token). Proven by scripts/acp_native_tools_probe.py.
#
# Gated OFF by default so the whole copilot-acp fleet is unaffected until a
# single agent is validated end-to-end. Enable per-agent with
# HERMES_ACP_NATIVE_TOOLS=1. Rides documented ACP + MCP interfaces (not binary
# internals), so it survives claude-agent-acp / claude CLI updates.

# --- Native-tools bridge (local overlay) ------------------------------------
# MCP wiring + engine-activity rendering live in agent/acp_native_tools.py;
# re-exported here so upstream call sites and tests keep their import path.
from agent.acp_native_tools import (  # noqa: E402,F401
    _acp_native_tools_enabled,
    _acp_tool_activity_label,
    _acp_tool_started_fields,
    _hermes_agent_root,
    _hermes_tools_mcp_servers,
    _normalize_native_tool_name,
)

# Optional per-profile MCP servers (agent/acp_extra_mcp.py), declared in the
# profile's config.yaml under acp.mcp_servers.
from agent.acp_extra_mcp import extra_mcp_servers  # noqa: E402


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


# --- Permission guard (local overlay) ---------------------------------------
# The approve/deny decision layer lives in agent/acp_permission_guard.py so
# this upstream file carries an import instead of ~600 local lines — every
# line kept here is conflict surface for the next `hermes update`. Names are
# re-exported because tests and other modules import them from here.
from agent.acp_permission_guard import (  # noqa: E402,F401
    _ACP_AUDIT_LOG_NAME,
    _CREDENTIAL_BASENAMES,
    _DENY_HINT,
    _DISCARD_SINKS,
    _acp_audit,
    _credential_basename_match,
    _curator_file_reason,
    _curator_permission_reason,
    _has_recursive_force_rm,
    _is_discard_sink,
    _operator_grant_for,
    _permission_allow,
    _permission_command_text,
    _permission_danger_reason,
    _permission_denied,
    _shell_credential_exfil,
    _shell_danger_reason,
    _shell_tokens,
    _strip_content_fields,
)


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    native_tools: bool = False,
) -> str:
    if native_tools:
        # Tools reach the engine as a native MCP server (session/new.mcpServers),
        # so we must NOT inject the <tool_call> shim — that instruction is what
        # trips Fable's reasoning_extraction refusal. Give a clean, neutral
        # preamble and let the engine drive tools itself.
        sections = [
            "You are being used as the active ACP agent backend for Hermes.",
            "You have access to native tools (including the `hermes-tools` MCP "
            "server). Use them directly to complete tasks; answer normally when "
            "no tool is needed.",
        ]
    else:
        sections = [
            "You are being used as the active ACP agent backend for Hermes.",
            "Use ACP capabilities to complete tasks.",
            "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
            "If no tool is needed, answer normally.",
        ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if not native_tools and isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if not native_tools and tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned



def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


# Hermes additions live in a mixin (agent/acp_client_mixin.py) so this file
# carries one line instead of ~105: the engine-activity relay, the native-call
# ledger and the curator-session flag are ours entirely and upstream has no
# opinion about them.
from agent.acp_client_mixin import HermesACPMixin  # noqa: E402


class CopilotACPClient(HermesACPMixin):
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "copilot-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        # Local overlay: os.getcwd() raises a path-less FileNotFoundError once
        # the directory is gone (a completed kanban card deletes its own
        # scratch workspace, which is the worker's cwd). Rationale and the
        # in-place repair live in agent/acp_cwd.
        from agent.acp_cwd import session_cwd

        self._acp_cwd = session_cwd(acp_cwd)
        self._native_tools = _acp_native_tools_enabled()
        self._mcp_servers = _hermes_tools_mcp_servers() if self._native_tools else []
        # Operator-declared servers come AFTER, and acp_extra_mcp refuses the
        # reserved name, so nothing can displace hermes-tools in the engine's
        # by-name map.
        self._mcp_servers += extra_mcp_servers()
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        # Set by ``abort_for_interrupt`` when Hermes itself cancels the turn
        # this client is serving (interrupt / redirect / steer). Reset per API
        # call in acp_failover.run_prompt_with_failover, because agent-scoped
        # (shared=True) clients outlive a single call.
        self._hermes_cancelled: str | None = None
        # Native-tools observability: every tool_call session/update the engine
        # reports is recorded here ({name, arguments}). On the native MCP path
        # tool calls never surface in the OpenAI-shaped response, so this list
        # is the only ground truth for reports (e.g. the curator's run.json).
        self._native_tool_calls: list[dict[str, str]] = []
        self._native_call_index: dict[str, int] = {}

    def abort_for_interrupt(self, reason: str = "interrupt") -> int:
        """Cancel the in-flight turn: kill the bridge and mark the kill as ours.

        Local overlay (2026-07-26). ``force_close_tcp_sockets`` — the generic
        stranger-thread abort — finds no sockets on an ACP client (the log line
        always reads ``tcp_force_closed=0``), so an interrupt or redirect left
        the engine generating for a turn nobody was waiting on. That zombie
        worker then raced the replacement turn for the one-live-bridge slot and
        SIGTERM'd it mid-``initialize``, which is what pushed live turns onto
        the failover model (researcher, 2026-07-26 14:29 / 14:44 / 20:47).

        Unlike TLS sockets, terminating a subprocess from another thread is
        safe — no FD is released here, ``Popen.wait`` in the owning thread does
        that. Returns the number of bridges terminated (0 or 1).
        """
        self._hermes_cancelled = reason
        with self._active_process_lock:
            proc = self._active_process
        if proc is None or proc.poll() is not None:
            return 0
        setattr(proc, "_hermes_evicted", f"turn cancelled by Hermes ({reason})")
        try:
            proc.terminate()
        except Exception:
            return 0
        return 1

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        # Drop this bridge from the per-session registry.
        _release_session_bridge(
            _bridge_slot(getattr(self, "_activity_agent", None)),
            proc,
        )
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        # Local overlay: remember what Hermes asked for, so a spawn with nothing
        # configured falls back to the caller's model instead of a constant
        # (agent/acp_config.resolve_model). _run_prompt has no other route to it.
        self._acp_requested_model = model
        # Local overlay: _format_messages_as_prompt keeps only the text parts,
        # so a multimodal message reached the engine as an instruction to
        # describe an image that was never attached. ACP carries images
        # natively — agent/acp_image_blocks. _run_prompt has no other route to
        # the original messages, same as the requested model above.
        from agent.acp_image_blocks import extract_image_blocks

        self._acp_prompt_images = extract_image_blocks(messages or [])
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            native_tools=self._native_tools,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        # Primary attempt + per-call model failover + usage accounting live
        # in the overlay; upstream owns everything around this call.
        from agent.acp_failover import (
            record_successful_call,
            run_prompt_with_failover,
        )

        response_text, reasoning_text, _used_model, _failover_fired = (
            run_prompt_with_failover(self, prompt_text, _effective_timeout, model)
        )

        if self._native_tools:
            # Engine orchestrated tools natively over MCP — nothing to parse out
            # of the text; the response IS the final answer.
            tool_calls, cleaned_text = [], (response_text or "").strip()
        else:
            tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=_used_model or model or "copilot-acp",
        )
        record_successful_call(
            self, prompt_text, cleaned_text, reasoning_text,
            requested_model=model, used_model=_used_model,
            failover=_failover_fired,
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model_override: str | None = None) -> tuple[str, str]:
        # Local overlay (2026-07-26): a cancelled turn must never spawn another
        # bridge. Its spawn would claim the session slot and evict the LIVE
        # turn's bridge — the failover retry of an already-interrupted call is
        # exactly how a healthy opus-5 turn got demoted to the backup model.
        if self._hermes_cancelled:
            raise ACPBridgeEvicted(
                "Refusing to start a Copilot ACP bridge for a cancelled turn "
                f"({self._hermes_cancelled})."
            )
        # Fresh tool-pulse dedup state per prompt — engine toolCallIds are
        # only unique within a session, and stale ids would leak memory.
        self._acp_fired_tool_ids = set()
        self._acp_pending_tools = {}
        try:
            # Hide the console the CLI child would otherwise flash on Windows
            # (#56747). Hide-only — stdio pipes stay intact for the ACP wire.
            from hermes_cli._subprocess_compat import windows_hide_flags

            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1,
                cwd=self._acp_cwd,
                # Upstream added the Windows console-hide flag here; we pass
                # the model-aware env instead of the plain one so a per-call
                # failover can re-spawn the bridge on another model. Both are
                # needed — keep them together.
                env=_acp_env_with_model(
                    model_override,
                    requested=getattr(self, "_acp_requested_model", None),
                ),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Copilot ACP command '{self._acp_command}'. "
                "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Copilot ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        # One-live-bridge-per-session invariant (belt over turn_lease).
        _claim_session_bridge(
            _bridge_slot(getattr(self, "_activity_agent", None)),
            proc,
        )

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0
        # Set from `initialize`; read by the mid-turn follow-up push below.
        # False until proven otherwise so a handshake that never completes
        # cannot make us write a second prompt into an unknown engine.
        prompt_queueing = False

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None, steerable: bool = False) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            # Local overlay (2026-08-07): mid-turn follow-up delivery.
            #
            # `outstanding` is every prompt id this call is still owed — our
            # own, plus any follow-up we push while the turn runs. It is a SET,
            # not a single id, because the engine hands the turn over on
            # queueing: the original id settles immediately with a zero-usage
            # `end_turn` while the model keeps working, and the answer text
            # streams under the LATER id. Returning on our own id (what this
            # loop did before) would therefore return an empty answer and kill
            # the bridge mid-sentence. Measured on a raw bridge 2026-08-07 —
            # see agent/acp_midturn_steer.
            outstanding = {request_id}
            last_result: Any = None

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break

                # A follow-up the operator sent while this turn is running:
                # push it as its own `session/prompt` on the SAME session, so
                # the engine hands it to the model at its next model call
                # instead of it waiting for a turn that already answered.
                if steerable and prompt_queueing:
                    from agent.acp_midturn_steer import pull_followup

                    followup = pull_followup(self)
                    if followup:
                        next_id += 1
                        followup_id = next_id
                        proc.stdin.write(json.dumps({
                            "jsonrpc": "2.0",
                            "id": followup_id,
                            "method": "session/prompt",
                            "params": {
                                **params,
                                "prompt": [{"type": "text", "text": followup}],
                            },
                        }) + "\n")
                        proc.stdin.flush()
                        outstanding.add(followup_id)
                        deadline = time.monotonic() + timeout_seconds
                        _log.info(
                            "ACP mid-turn follow-up delivered into the live turn "
                            "(%d chars, prompt id %d)",
                            len(followup),
                            followup_id,
                        )

                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Local overlay (2026-08-01): the deadline measures SILENCE, not
                # total elapsed time. `session/prompt` is not one API call — it
                # is the WHOLE agent turn, tools and all, so a healthy engine
                # legitimately holds this request for 30+ minutes while shouting
                # `session/update` the entire time. Measured on founder that day:
                # a turn that ran tools continuously from 18:06 to 18:36 was cut
                # at exactly 1800.0s, reported as `TimeoutError` and re-answered
                # by the failover model — the third total-time guillotine on the
                # same shape, after the stale detector (agent/acp_stale.py) and
                # the wedge-prone primary cap (agent/acp_failover.py), both of
                # which were converted to idle time in July for this very reason.
                # Any byte from the bridge proves it is alive, so any byte buys
                # another full window; genuine silence still dies on schedule
                # (and, in a gateway, dies EARLIER — the stale detector's idle
                # threshold is 600s against this one's 1800s).
                deadline = time.monotonic() + timeout_seconds

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue

                message_id = msg.get("id")
                if message_id not in outstanding:
                    continue
                if "error" in msg:
                    if message_id != request_id:
                        # A follow-up prompt was rejected. That is a lost
                        # steer, not a lost turn — the answer to the original
                        # prompt is still coming, so keep waiting for it.
                        outstanding.discard(message_id)
                        _log.warning(
                            "ACP mid-turn follow-up (id %s) rejected: %s",
                            message_id,
                            (msg.get("error") or {}).get("message") or msg.get("error"),
                        )
                        if outstanding:
                            continue
                        return last_result
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"Copilot ACP {method} failed: {err.get('message') or err}"
                    )
                last_result = msg.get("result")
                outstanding.discard(message_id)
                # Only leave once nothing is still owed: on a hand-off the
                # original id settles first and empty, and the real answer
                # arrives under the follow-up's id.
                if not outstanding:
                    return last_result
                continue

            # Local overlay (2026-07-26): before blaming the provider, check
            # whether WE killed this bridge (slot eviction or a cancelled
            # turn). Same rc, completely different meaning.
            _rc = proc.poll()
            _killed_by_us = getattr(proc, "_hermes_evicted", None) or (
                self._hermes_cancelled if _rc is not None else None
            )
            if _rc is not None and _killed_by_us:
                raise ACPBridgeEvicted(
                    f"Copilot ACP bridge terminated by Hermes (rc={_rc}) during "
                    f"'{method}': {_killed_by_us}. Not a provider failure — no "
                    "failover, the turn is retried or cancelled on the same model."
                )

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                if _is_gh_copilot_deprecation_message(stderr_text):
                    raise RuntimeError(
                        "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                        "(github.com/github/copilot-cli), but the binary it just "
                        "spawned is the deprecated `gh copilot` extension.\n\n"
                        "Install the new CLI:\n"
                        "  npm install -g @github/copilot\n"
                        "  # then verify with: copilot --help\n\n"
                        "If `copilot` already resolves to the new CLI but you still see this,\n"
                        "point Hermes at it explicitly:\n"
                        "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                        "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                        "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                        f"Original error:\n{stderr_text}"
                    )
                raise RuntimeError(f"Copilot ACP process exited early: {stderr_text}")
            # Local overlay (2026-07-26): a process that already exited is NOT a
            # timeout, even when it died too fast to say anything on stderr —
            # the common case is a revoked/absent subscription token, where the
            # bridge exits within milliseconds. Reporting that as TimeoutError
            # made Hermes blame the model and fail over to the fallback one,
            # sending users chasing a latency problem that never existed.
            if proc.poll() is not None:
                # A negative rc is a SIGNAL: something killed the bridge, it did
                # not decide to leave. Pointing at the subscription token there
                # cost a full evening of chasing a healthy token (2026-07-26) —
                # only rc >= 0 (the process exited on its own) earns that hint.
                if _rc is not None and _rc < 0:
                    raise RuntimeError(
                        f"Copilot ACP bridge killed by signal {-_rc} (rc={_rc}) "
                        f"during '{method}' without writing to stderr. A signal "
                        "means something terminated it — an interrupt, a "
                        "competing turn on this session, or the OOM killer. The "
                        "subscription token is not implicated; check "
                        "~/.hermes/logs/claude-acp.log and `dmesg -T | tail`."
                    )
                raise RuntimeError(
                    f"Copilot ACP process exited (rc={proc.poll()}) during "
                    f"'{method}' without writing to stderr — usually a revoked "
                    "or missing subscription token. Check "
                    "~/.hermes/logs/claude-acp.log and re-run "
                    "`claude setup-token > ~/.hermes/.claude-acp-token`."
                )
            raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}.")

        try:
            init_result = _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": True,
                            "writeTextFile": True,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            # Local overlay (2026-08-07): only push a follow-up into a live
            # turn on an engine that says it accepts one. Absent capability →
            # the text stays in `_pending_steer` and upstream's
            # `turn_finalizer` hands it back for next-turn delivery.
            from agent.acp_midturn_steer import note_initialize_result

            prompt_queueing = note_initialize_result(init_result)

            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": self._mcp_servers,
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Copilot ACP did not return a sessionId.")
            # Local overlay (2026-08-15): this id names the engine's own
            # transcript on disk, which carries the REAL token counts the
            # JSON-RPC stream never sends. Hand it to the ledger so the row
            # stops being a chars/4 guess.
            from agent.acp_usage_transcript import note_engine_session

            note_engine_session(self, session_id, self._acp_cwd)

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            # Local overlay: attach the images the caller sent as first-class
            # ACP blocks when the engine advertises promptCapabilities.image,
            # and refuse (rather than answer from nothing) when it does not.
            from agent.acp_image_blocks import (
                build_prompt_blocks,
                engine_accepts_images,
            )

            prompt_blocks = build_prompt_blocks(
                prompt_text,
                getattr(self, "_acp_prompt_images", None),
                accepts_images=engine_accepts_images(init_result),
            )
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": prompt_blocks,
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                steerable=True,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        except BaseException:
            # Local overlay (2026-08-07): a follow-up we already wrote onto
            # the wire dies with the bridge. It is no longer in
            # `_pending_steer` (we drained it), so without this it would
            # exist nowhere — not in the answer, not in the queue, not in
            # the transcript. Hand it back to the retry / leftover path.
            from agent.acp_midturn_steer import restore_undelivered

            restore_undelivered(self)
            raise
        finally:
            self.close()

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            # Everything we do with a session update — narration handling,
            # the native-call ledger, progress pulses — lives in the overlay
            # so this insertion stays one call long.
            from agent.acp_session_update import handle_session_update

            handle_session_update(
                self,
                (msg.get("params") or {}).get("update") or {},
                text_parts,
                reasoning_parts,
            )
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = _permission_allow(
                message_id,
                params,
                extra_reason_fn=(
                    _curator_permission_reason if self._curator_session() else None
                ),
            )
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if not denied and is_write_approval_required(str(path)):
                    # Approval-gated paths (e.g. ~/.ssh/config) are not
                    # hard-denied for interactive tools, but the ACP shim has
                    # no human channel to confirm the write. Fold them into the
                    # SAME deny funnel below so the one sanctioned way through
                    # (an operator grant) covers them too — a deny with no
                    # named way out is the one that gets routed around.
                    denied = (
                        f"Write denied: '{path}' requires interactive approval "
                        "and cannot be written through the ACP file bridge."
                    )
                if denied:
                    # The engine's own Write tool goes through
                    # session/request_permission, but this ACP fs capability
                    # does not — so the operator grant has to be honoured on
                    # BOTH paths, or the escape hatch works only by luck of
                    # which write path the engine picks.
                    _grant = _operator_grant_for(str(path))
                    if _grant is None:
                        raise PermissionError(denied)
                    _acp_audit(
                        "ALLOW-GRANT",
                        {"toolCall": {"title": f"fs/write_text_file {path}"}},
                        f"operator grant {_grant.get('id')} "
                        f"(reason: {_grant.get('reason') or '—'}; "
                        f"issuer: {_grant.get('issuer')})",
                    )
                if self._curator_session():
                    curator_denied = _curator_file_reason([str(path)])
                    if curator_denied:
                        raise PermissionError(curator_denied)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
