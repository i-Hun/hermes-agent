"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend. Each request starts a short-lived ACP session, sends the formatted
conversation as a single prompt, collects text chunks, and converts the result
back into the minimal shape Hermes expects from an OpenAI client.
"""

from __future__ import annotations

import json
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

from agent.file_safety import get_read_block_error, get_write_denied_error
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0


# One-live-bridge-per-session invariant — a belt over agent/turn_lease.py. Two
# concurrent turns on the SAME hermes session must never both hold a live ACP
# bridge: interleaved writes corrupt the single engine transcript (the
# "Repaired N message-alternation violation(s)" symptom). Keyed by the hermes
# session id (agent.session_id); the newest spawn wins and terminates any stale
# bridge left behind by a /stop or a fail-open turn overlap.
_LIVE_BRIDGES: dict[str, "subprocess.Popen[str]"] = {}
_LIVE_BRIDGES_LOCK = threading.Lock()


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


def _acp_env_with_model(model_override):
    """Subprocess env for the ACP wrapper, optionally forcing a model.

    The wrapper (claude-acp-run.sh) selects the engine model from
    ``$HERMES_ACP_MODEL`` (default claude-opus-4-8). Injecting it here lets a
    per-call retry re-spawn the bridge on a different model without touching
    any file -- this is what powers per-call auto-failover.
    """
    env = _build_subprocess_env()
    if model_override:
        env["HERMES_ACP_MODEL"] = str(model_override)
    return env


def _acp_failover_enabled():
    return os.getenv("HERMES_ACP_FAILOVER", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def _acp_failover_model():
    return os.getenv("HERMES_ACP_FALLBACK_MODEL", "").strip() or "claude-opus-4-8"


def _acp_effective_model():
    """The model the wrapper uses by default (env override or its opus default).

    Must agree with claude-acp-run.sh's ``${HERMES_ACP_MODEL:-claude-opus-4-8}``
    so failover comparison and the usage ledger label the real engine model.
    Fable's limits are exhausted, so opus is primary fleet-wide; when Fable
    returns, set ``HERMES_ACP_MODEL=claude-fable-5`` in the gateway env to flip
    back (that also re-arms opus failover, since primary != failover again).
    """
    return os.getenv("HERMES_ACP_MODEL", "").strip() or "claude-opus-4-8"


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


class CopilotACPClient:
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
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._native_tools = _acp_native_tools_enabled()
        self._mcp_servers = _hermes_tools_mcp_servers() if self._native_tools else []
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        # Native-tools observability: every tool_call session/update the engine
        # reports is recorded here ({name, arguments}). On the native MCP path
        # tool calls never surface in the OpenAI-shaped response, so this list
        # is the only ground truth for reports (e.g. the curator's run.json).
        self._native_tool_calls: list[dict[str, str]] = []
        self._native_call_index: dict[str, int] = {}

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
            getattr(getattr(self, "_activity_agent", None), "session_id", None),
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

        from agent.acp_usage_ledger import record_usage

        _session_id = getattr(
            getattr(self, "_activity_agent", None), "session_id", None
        )
        _requested_model = model
        _used_model = _acp_effective_model()
        _failover_fired = False
        # Request 3: when a fast-fail model is primary (Fable) with opus failover
        # armed, cap the PRIMARY attempt so a wedged primary yields to opus fast
        # instead of holding the session for the full timeout — this is what
        # narrows the turn_lease fail-open window. Opus-primary (no distinct
        # failover) keeps the full timeout; the failover retry always does too.
        from agent.acp_failover import primary_attempt_timeout

        _primary_timeout = primary_attempt_timeout(
            _effective_timeout,
            primary_model=_used_model,
            failover_model=_acp_failover_model(),
            failover_enabled=_acp_failover_enabled(),
        )
        try:
            response_text, reasoning_text = self._run_prompt(
                prompt_text,
                timeout_seconds=_primary_timeout,
            )
        except (RuntimeError, TimeoutError) as _primary_exc:
            _fo_model = _acp_failover_model()
            if (
                _acp_failover_enabled()
                and _fo_model
                and _fo_model != _used_model
            ):
                try:
                    self._fire_engine_activity(
                        "_thinking",
                        "primary model failed (%s); failing over to %s"
                        % (type(_primary_exc).__name__, _fo_model),
                    )
                except Exception:
                    pass
                try:
                    response_text, reasoning_text = self._run_prompt(
                        prompt_text,
                        timeout_seconds=_effective_timeout,
                        model_override=_fo_model,
                    )
                    _used_model = _fo_model
                    _failover_fired = True
                except (RuntimeError, TimeoutError):
                    record_usage(
                        session_id=_session_id,
                        requested_model=_requested_model,
                        used_model=_fo_model,
                        prompt_text=prompt_text,
                        completion_text=None,
                        failover=True,
                        ok=False,
                    )
                    raise
            else:
                record_usage(
                    session_id=_session_id,
                    requested_model=_requested_model,
                    used_model=_used_model,
                    prompt_text=prompt_text,
                    completion_text=None,
                    failover=False,
                    ok=False,
                )
                raise

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
        record_usage(
            session_id=_session_id,
            requested_model=_requested_model,
            used_model=_used_model,
            prompt_text=prompt_text,
            completion_text=cleaned_text,
            reasoning_text=reasoning_text,
            failover=_failover_fired,
            ok=True,
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model_override: str | None = None) -> tuple[str, str]:
        # Fresh tool-pulse dedup state per prompt — engine toolCallIds are
        # only unique within a session, and stale ids would leak memory.
        self._acp_fired_tool_ids = set()
        self._acp_pending_tools = {}
        try:
            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
                env=_acp_env_with_model(model_override),
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
            getattr(getattr(self, "_activity_agent", None), "session_id", None),
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

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
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

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"Copilot ACP {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

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
            raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}.")

        try:
            _request(
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

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()

    def _fire_engine_activity(
        self,
        event_type: str,
        tool_name: str | None = None,
        preview: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Relay engine-internal activity to the owning agent's progress rail.

        ``_activity_agent`` is attached at client construction (see
        ``create_openai_client``). The gateway re-binds
        ``agent.tool_progress_callback`` per turn, so we resolve it at call
        time; without an attached agent (aux clients, tests) this is a no-op.
        Runs on the ACP stdout-reader thread — the gateway callback is
        thread-safe (queue-backed) by design.
        """
        agent = getattr(self, "_activity_agent", None)
        if agent is not None:
            # Liveness stamp for the stale-call detector: a non-streaming ACP
            # call produces no bytes while the engine works, so without this
            # a long-but-healthy turn is killed at the stale threshold and
            # answered by the failover model (see agent/acp_stale.py).
            try:
                agent._acp_last_event_ts = time.time()
            except Exception:
                pass
        cb = getattr(agent, "tool_progress_callback", None) if agent is not None else None
        if cb is None:
            return
        try:
            cb(event_type, tool_name, preview, args)
        except Exception:
            pass

    def _curator_session(self) -> bool:
        """True when this client serves the background skill-curator fork."""
        agent = getattr(self, "_activity_agent", None)
        platform = getattr(agent, "platform", "") if agent is not None else ""
        return str(platform or "").strip().lower() == "curator"

    _NATIVE_CALL_CAP = 800
    _NATIVE_CALL_ARGS_CAP = 2000

    def _record_native_tool_call(self, update: dict[str, Any], kind: str) -> None:
        """Track engine tool calls ({name, arguments}) for post-run reports.

        The initial tool_call often arrives without rawInput (arguments land
        via tool_call_update), so entries are indexed by toolCallId and
        backfilled. Never raises.
        """
        try:
            calls = self._native_tool_calls
            index = self._native_call_index
        except AttributeError:
            calls = self._native_tool_calls = []
            index = self._native_call_index = {}
        # Mirror onto the owning agent. Per-REQUEST clients are created with
        # shared=False and discarded when the call returns (run_agent
        # _create_request_openai_client), so a ledger kept only on the client
        # dies with it — curator then reads agent.client._native_tool_calls and
        # sees [] even though the engine ran tools (verified 2026-07-25: a
        # background skill review wrote a skill via the MCP bridge while the
        # ledger read 0). The agent outlives every request, so it owns the
        # authoritative list; the client-local one stays for compatibility.
        agent = getattr(self, "_activity_agent", None)
        if agent is not None:
            try:
                shared = getattr(agent, "_acp_native_tool_calls", None)
                if shared is None:
                    shared = agent._acp_native_tool_calls = []
                if shared is not calls:
                    calls = self._native_tool_calls = shared
            except Exception:
                pass
        try:
            tool_id = str(
                update.get("toolCallId") or update.get("tool_call_id") or ""
            ) or None
            name, _preview, args = _acp_tool_started_fields(update)
            norm = _normalize_native_tool_name(name)
            args_json = ""
            if isinstance(args, dict):
                try:
                    args_json = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args_json = str(args)
                if len(args_json) > self._NATIVE_CALL_ARGS_CAP:
                    args_json = args_json[: self._NATIVE_CALL_ARGS_CAP] + "…"
            if tool_id is not None and tool_id in index:
                # Known call: backfill arguments/name from the update.
                entry = calls[index[tool_id]]
                if args_json and not entry.get("arguments"):
                    entry["arguments"] = args_json
                if entry.get("name") in ("", "tool") and norm not in ("", "tool"):
                    entry["name"] = norm
                return
            if kind == "tool_call" or tool_id is not None:
                if len(calls) >= self._NATIVE_CALL_CAP:
                    return
                calls.append({"name": norm, "arguments": args_json})
                if tool_id is not None:
                    index[tool_id] = len(calls) - 1
        except Exception:
            pass

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
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            elif kind in ("tool_call", "tool_call_update"):
                self._record_native_tool_call(update, kind)
                # Message text emitted BEFORE a tool call is the engine's
                # progress narration, not the final answer. Reclassify it as
                # reasoning so show_reasoning=false hides it; only the text
                # after the last tool call remains the user-facing reply.
                if text_parts:
                    flushed = "".join(text_parts).strip()
                    text_parts.clear()
                    if flushed:
                        if reasoning_parts is not None:
                            reasoning_parts.append(flushed + "\n\n")
                        # Live "thinking" pulse for the gateway progress
                        # bubble (💬 …, gated on display.thinking_progress).
                        first_line = flushed.splitlines()[0][:200]
                        self._fire_engine_activity("_thinking", first_line)
                if kind == "tool_call" and reasoning_parts is not None:
                    label = _acp_tool_activity_label(update)
                    if label:
                        reasoning_parts.append(f"\n⚙️ {label}\n")
                # Live tool pulse (⚙️ tool: …, gated on tool_progress).
                # The engine's initial tool_call often arrives WITHOUT
                # rawInput (the command/path lands later via
                # tool_call_update), so an eager pulse renders as a
                # preview-less "⚙️ Terminal..." line. Defer the pulse until
                # arguments are available and fire it exactly once per
                # toolCallId; id-less updates keep the eager behavior.
                fired = getattr(self, "_acp_fired_tool_ids", None)
                if fired is None:
                    fired = self._acp_fired_tool_ids = set()
                pending = getattr(self, "_acp_pending_tools", None)
                if pending is None:
                    pending = self._acp_pending_tools = {}
                tool_id = str(
                    update.get("toolCallId") or update.get("tool_call_id") or ""
                ) or None
                _tname, _tprev, _targs = _acp_tool_started_fields(update)
                _has_details = bool(_tprev or _targs)
                if kind == "tool_call":
                    if tool_id is None:
                        self._fire_engine_activity("tool.started", _tname, _tprev, _targs)
                    elif _has_details:
                        fired.add(tool_id)
                        self._fire_engine_activity("tool.started", _tname, _tprev, _targs)
                    else:
                        pending[tool_id] = _tname
                elif (
                    tool_id is not None
                    and tool_id in pending
                    and tool_id not in fired
                    and _has_details
                ):
                    orig_name = pending.pop(tool_id)
                    fired.add(tool_id)
                    if _tname == "tool" and orig_name:
                        # The update usually carries no title — re-map with
                        # the remembered one so shell tools still hit the
                        # "terminal" fenced-block rendering.
                        _tname, _tprev, _targs = _acp_tool_started_fields(
                            {**update, "title": orig_name}
                        )
                    self._fire_engine_activity("tool.started", _tname, _tprev, _targs)
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
                    content = path.read_text()
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
                if denied:
                    raise PermissionError(denied)
                if self._curator_session():
                    curator_denied = _curator_file_reason([str(path)])
                    if curator_denied:
                        raise PermissionError(curator_denied)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""))
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
