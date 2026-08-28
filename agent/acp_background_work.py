"""Engine background work that cannot outlive the ACP call that started it.

Local overlay module (not upstream).

The bridge is scoped to ONE API call: ``copilot_acp_client._run_prompt``
spawns it and ``close()`` terminates the process when the call returns
(copilot_acp_client.py, ``proc.terminate()``). Everything the engine parked
*outside* that call therefore dies with it. Its own subagent tool —
``Agent``/``Task``, which runs **asynchronously by default** — is exactly
that: the tool returns a task id immediately, looks successful, and the worker
is killed seconds later when the turn is answered. Synchronous is the opt-in
(``run_in_background: false``), so "no flag" is the dangerous case, not the
safe one.

Nothing here prevents the call. The engine's built-in tools never reach
``session/request_permission`` (the permission guard sees Bash and the MCP
bridge, never ``Agent``), so there is no seam to gate them at. What is
fixable is the SILENCE: from inside the turn "killed with the bridge" is
indistinguishable from "the job finished and printed nothing", and an agent
that cannot tell them apart reports success. On 2026-08-01 the ``founder``
profile hit this twice in one evening — the second time right after
diagnosing the first — and lost an hour of strata work each time.

So the loss is made loud, three ways:

* recorded at the moment of the call, from the UN-truncated ``rawInput``
  (the ledger caps arguments at 2000 chars and an ``Agent`` prompt blows
  straight past that, so the flag is often outside the stored slice);
* stated in the reply itself — which puts it in the transcript, so the model
  reads its own failed attempt on the next turn instead of trusting it;
* appended to ``logs/acp-background-work.jsonl`` so ``bridge-doctor`` can
  count it across the fleet rather than relying on someone noticing.

Loud is not the same as right. Async-by-default means a subagent is RECORDED
at spawn, but plenty of them come back before the reply is assembled and lose
nothing — and by then the engine has written the answer down as a
``tool_result`` in its transcript. So the record is a suspicion and the
transcript is the verdict: ``warning_text`` warns only about calls with no
result on disk, and journals every verdict so the doctor reads it instead of
re-deriving it (16.08.2026: ``personal`` fanned out 15 flagless subagents, all
15 delivered, all 15 reported lost — the detector was the only casualty).

Naming is taken from ``_meta.claudeCode.toolName`` (the bridge stamps the real
Claude Code tool name on every notification) rather than from ``title``:
``_acp_tool_started_fields`` deliberately rewrites the title to a neutral
action label, so by the time the ledger sees a call its identity is gone.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

# Same anchor the usage ledger uses: this file is <root>/hermes-agent/agent/…,
# so parents[2] is the install root shared by every profile gateway.
_INSTALL_ROOT = Path(__file__).resolve().parents[2]
_LOG_PATH = _INSTALL_ROOT / "logs" / "acp-background-work.jsonl"

_write_lock = threading.Lock()

# Claude Code spells the flag snake_case; accept the camelCase form too so a
# future rename on the engine side degrades to "still detected".
_BG_KEYS = ("run_in_background", "runInBackground")

# What the engine prints when a subagent was launched asynchronously. This is
# EVIDENCE rather than inference, so it outranks both the flag and the default
# below — see ``_is_background``.
_ASYNC_RESULT_MARKERS = ("async agent launched",)


# What the work IS decides what was lost, so the two classes are kept apart:
# a subagent lives inside the engine process and dies with it, a shell child
# is an OS process that survives while the handle to read it does not.
#
# The spawn tool is matched EXACTLY. It used to be a prefix, which quietly
# swept in every sibling that merely starts with the same word — ``TaskCreate``,
# ``TaskList``, ``TaskUpdate`` are todo bookkeeping, not a subagent spawn, and
# being async-by-default is a property of the one tool, not of its name's
# prefix. Shell keeps the prefix on purpose: ``BashOutput`` and friends are
# shell-shaped, and shell defaults to foreground anyway, so a wide net there
# cannot manufacture a warning.
_SUBAGENT_TOOLS = ("agent", "task")
_SHELL_TOOLS = ("bash", "shell", "terminal")

# A URL argument means the call fetches something and hands the result back
# inside the turn. ``WebFetch`` is ``{url, prompt}``: prompt-shaped, and yet
# entirely synchronous.
_FETCH_KEYS = ("url", "urls")
# What a spawn carries besides the prompt. Required by the shape path so that
# "has a prompt" alone can never stand in for "is a subagent". ``description``
# is in the list because ``subagent_type`` is optional on the engine's own
# ``Agent`` schema — without it a real degraded-mode spawn would slip past.
_SPAWN_KEYS = (
    "subagent_type",
    "subagentType",
    "agent_type",
    "agentType",
    "description",
)

_KIND_SUBAGENT = "subagent"
_KIND_SHELL = "shell"
_KIND_UNKNOWN = "unknown"


def _claude_tool_name(update: dict[str, Any]) -> str:
    """Real Claude Code tool name for an ACP tool_call update, or ""."""
    try:
        meta = update.get("_meta") or update.get("meta") or {}
        cc = meta.get("claudeCode") or meta.get("claude_code") or {}
        name = cc.get("toolName") or cc.get("tool_name") or ""
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    # Fallback for a bridge that stops stamping _meta: the pre-rewrite title.
    title = update.get("title")
    return title.strip() if isinstance(title, str) else ""


def _raw_args(update: dict[str, Any]) -> dict[str, Any] | None:
    raw = update.get("rawInput") or update.get("raw_input") or update.get("input")
    return raw if isinstance(raw, dict) else None


def _explicit_flag(args: dict[str, Any]) -> bool | None:
    """The caller's own answer, or None when it never said."""
    for key in _BG_KEYS:
        if key in args:
            return bool(args.get(key))
    return None


def _result_text(update: dict[str, Any]) -> str:
    """Flattened text of whatever the tool returned so far (lowercased)."""
    out: list[str] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or len(out) > 40:
            return
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for val in node.values():
                walk(val, depth + 1)
        elif isinstance(node, (list, tuple)):
            for val in node:
                walk(val, depth + 1)

    walk(update.get("content"))
    walk(update.get("rawOutput") or update.get("raw_output"))
    return " ".join(out).lower()


def _is_background(
    args: dict[str, Any], kind: str, update: dict[str, Any] | None = None
) -> bool:
    """Will this call's work be left running outside the API call?

    Three supports, strongest first — the ordering is the whole point:

    * what the engine SAID it did ("Async agent launched") — evidence;
    * what the caller explicitly asked for — a claim, but an honest one;
    * what the tool does when nobody asks. ``Agent``/``Task`` are async BY
      DEFAULT, so an omitted flag means background, not foreground. Missing
      this is what let the 2026-08-02 05:39 loss through: ``founder`` called
      ``Agent`` with no flag at all, the detector required a truthy one, and
      the turn was answered with a confident "запущена фоновым агентом" and
      no warning. A predicate that only recognises the explicit form checks
      the polite half of the failure.
    """
    if update is not None:
        blob = _result_text(update)
        if any(marker in blob for marker in _ASYNC_RESULT_MARKERS):
            return True
    flag = _explicit_flag(args)
    if flag is not None:
        return flag
    return kind == _KIND_SUBAGENT


def _looks_like_a_spawn(args: dict[str, Any]) -> bool:
    """Does this argument set identify a subagent spawn on its own?

    Used only when the tool name is missing. It has to be a positive
    signature, not a catch-all, because the kind it returns is what grants
    background-by-default: on 2026-08-02 06:57 "any ``prompt`` with no
    ``command``" handed that default to ``WebFetch(url, prompt)``, and four
    synchronous fetches were reported to the agent as lost work.
    """
    if not args.get("prompt"):
        return False
    if any(key in args for key in _FETCH_KEYS):
        return False
    return any(key in args for key in _SPAWN_KEYS)


def _classify(tool: str, args: dict[str, Any]) -> str:
    """Which kind of background work this is.

    The name decides when it is known; the argument shape is only consulted
    when the bridge stopped stamping ``_meta``, and then only for a signature
    specific enough to be a spawn (see ``_looks_like_a_spawn``). Anything left
    over is ``unknown``, which is background only if the caller said so.
    """
    low = tool.strip().lower()
    if any(t == low for t in _SUBAGENT_TOOLS):
        return _KIND_SUBAGENT
    if any(t == low or low.startswith(t) for t in _SHELL_TOOLS):
        return _KIND_SHELL
    if args.get("command") or args.get("cmd"):
        return _KIND_SHELL
    if _looks_like_a_spawn(args):
        return _KIND_SUBAGENT
    return _KIND_UNKNOWN


def _label(args: dict[str, Any]) -> str:
    """Short human handle for the call — never the prompt body."""
    for key in ("description", "summary", "command", "cmd"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            line = val.strip().splitlines()[0]
            return line[:80] + ("…" if len(line) > 80 else "")
    return ""


def reset_for_call(client: Any) -> None:
    """Start a fresh record for one turn. Never raises.

    Agent-scoped (``shared=True``) clients outlive a request, so without this
    a warning would be repeated on every later turn of the same session.
    Called once per turn from ``run_prompt_with_failover``, i.e. it spans the
    auth retry and the failover attempt on purpose: work spawned by an attempt
    that then failed over is lost just the same.
    """
    try:
        client._acp_background_calls = []
    except Exception:
        pass


def note_background_tool_call(client: Any, update: dict[str, Any]) -> None:
    """Record an engine tool call whose work cannot survive this turn.

    Called for every ``tool_call``/``tool_call_update``; deduplicated by
    ``toolCallId`` because the engine sends both for the same call. Never
    raises — observability must not be able to break a turn.
    """
    try:
        if not isinstance(update, dict):
            return
        args = _raw_args(update)
        if not args:
            return
        tool = _claude_tool_name(update)
        kind = _classify(tool, args)
        if not _is_background(args, kind, update):
            return
        calls = getattr(client, "_acp_background_calls", None)
        if calls is None:
            calls = client._acp_background_calls = []
        tool_id = str(
            update.get("toolCallId") or update.get("tool_call_id") or ""
        )
        if tool_id and any(c.get("id") == tool_id for c in calls):
            return
        entry = {
            "id": tool_id,
            "tool": tool or "?",
            "kind": kind,
            "label": _label(args),
            "asked": "explicit" if _explicit_flag(args) else "default",
        }
        calls.append(entry)
        _append_log(client, entry)
    except Exception:
        pass


def _append_log(client: Any, entry: dict[str, str]) -> None:
    """One jsonl row per detected call, for bridge-doctor. Never raises."""
    try:
        from agent.acp_usage_ledger import _resolve_profile

        agent = getattr(client, "_activity_agent", None)
        row = {
            "ts": time.time(),
            "profile": _resolve_profile(),
            "session_id": getattr(agent, "session_id", None),
            **entry,
        }
        line = json.dumps(row, ensure_ascii=False)
        with _write_lock:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass


def log_path() -> Path:
    return _LOG_PATH


_SUBAGENT_NOTE = (
    "was killed when this reply was sent — the ACP bridge is one process per "
    "API call and background subagents live inside it. Assume the work did "
    "NOT happen. Next time: `delegate_task(background=true)` hands the "
    "subagent to the durable job runtime (a transient systemd unit outside "
    "the gateway cgroup — it survives this turn AND a gateway restart; poll "
    "it with `job_output`). Re-doing it inside the turn with "
    "`run_in_background: false` also works — a turn has no length limit — and "
    "multi-day work still belongs on the Kanban board."
)
_DEFAULT_ASYNC_HINT = (
    "  ⚑ `Agent`/`Task` are asynchronous BY DEFAULT — omitting the flag does "
    "not make the work synchronous."
)
_SHELL_NOTE = (
    "was killed when this reply was sent — the engine tears its background "
    "shells down together with the bridge, so neither the process nor the "
    "`BashOutput` handle survives (measured 2026-08-02: a loop appending a "
    "timestamp every 2s wrote exactly one line). Use `job_start` instead: it "
    "runs the command in a transient systemd unit outside the gateway cgroup, "
    "so it outlives this turn AND a gateway restart, and you read it back "
    "with `job_output`. Do NOT reach for `setsid nohup` — a detached child "
    "keeps the gateway's cgroup, and `ExecStopPost=gateway.cgroup_cleanup` "
    "SIGKILLs it at the next restart (measured 2026-08-13, that is exactly "
    "how a detached helper died mid-edit)."
)
_UNKNOWN_NOTE = (
    "may not have survived this reply — the ACP bridge is one process per API "
    "call and anything parked outside the call dies with it. Verify before "
    "reporting it as done."
)


_DENY_LOG = Path.home() / ".hermes" / "state" / "subagent-cap-denials.log"
_DENY_LOOKBACK_LINES = 400


def _denied_tool_ids() -> set[str]:
    """`toolu_*` ids the engine's PreToolUse cap hook refused. Never raises.

    Only the tail is read: the log is append-only and an id from days ago
    cannot belong to this turn, so scanning further buys nothing.
    """
    try:
        with _DENY_LOG.open(encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-_DENY_LOOKBACK_LINES:]
    except OSError:
        return set()
    ids: set[str] = set()
    for line in tail:
        for field in line.split("\t"):
            if field.startswith("id=") and field[3:].startswith("toolu_"):
                ids.add(field[3:].strip())
    return ids


def _walk_result_ids(node: Any, ids: set[str], depth: int = 0) -> None:
    """Collect every ``tool_use_id`` a transcript row carries."""
    if depth > 8:
        return
    if isinstance(node, dict):
        val = node.get("tool_use_id")
        if isinstance(val, str) and val:
            ids.add(val)
        for sub in node.values():
            _walk_result_ids(sub, ids, depth + 1)
    elif isinstance(node, (list, tuple)):
        for sub in node:
            _walk_result_ids(sub, ids, depth + 1)


def _result_ids_in(path: Any, ids: set[str]) -> None:
    """Fold one transcript file's delivered ids into ``ids``. Silent on OSError."""
    try:
        fh = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            if "tool_use_id" not in line:
                continue
            try:
                _walk_result_ids(json.loads(line), ids)
            except ValueError:
                continue


def _delivered_tool_ids(client: Any) -> set[str]:
    """`toolu_*` ids whose RESULT is already on the engine's disk. Never raises.

    The predicate above answers "will this outlive the call" from the tool's
    DEFAULT, because at spawn time that is all there is. By the time the reply
    is built the engine has written the truth down: a subagent that came back
    left a ``tool_result`` block carrying its ``tool_use_id`` in the session
    transcript, and a subagent still running left nothing. Two supports under
    one fact — the one that KNOWS must correct the one that guesses.

    It matters because background-by-default is the COMMON case, not the rare
    one: on 16.08.2026 ``personal`` fanned out 15 subagents with no flag and
    every single result was delivered inside the turn, while ``bridge-doctor``
    reported all 15 as "the work did not happen". Nothing was lost — the
    detector was. A warning that cries wolf is the one people stop reading.

    The fan-out lives in ``<sessionId>/subagents/agent-*.jsonl``, beside the
    session file rather than inside it, so both are read.
    """
    try:
        from agent.acp_usage_transcript import subagent_paths, transcript_path
    except Exception:
        return set()
    ids: set[str] = set()
    try:
        # Read-only on purpose: `take_call_usage` CLEARS this list, and it runs
        # on the ledger write which comes after the answer is assembled.
        sessions = list(getattr(client, "_acp_engine_sessions", None) or [])
    except Exception:
        return set()
    for entry in sessions:
        try:
            path = transcript_path(str(entry.get("id") or ""), entry.get("cwd"))
            if path is None:
                continue
            _result_ids_in(path, ids)
            for sub in subagent_paths(path):
                _result_ids_in(sub, ids)
        except Exception:
            continue
    return ids


_OUTCOME_DELIVERED = "delivered"
_OUTCOME_DENIED = "denied"
_OUTCOME_LOST = "lost"


def _verdicts(client: Any, calls: list[dict[str, Any]]) -> dict[str, str]:
    """Per-call outcome at the last moment it can still be observed."""
    denied = _denied_tool_ids()
    delivered = _delivered_tool_ids(client)
    out: dict[str, str] = {}
    for call in calls:
        cid = str(call.get("id") or "")
        if cid and cid in denied:
            out[cid] = _OUTCOME_DENIED
        elif cid and cid in delivered:
            out[cid] = _OUTCOME_DELIVERED
        else:
            out[cid] = _OUTCOME_LOST
    return out


def _journal_verdicts(client: Any, calls: list[dict[str, Any]],
                      verdicts: dict[str, str]) -> None:
    """Append one outcome row per call so the doctor need not re-derive it.

    The retrospective check cannot repeat the transcript lookup: transcripts
    are rotated and the 168h window spans 4488 files / 1.7 GB. The turn that
    watched the call end is the only place the answer is cheap, so it writes
    it down. Idempotent — ``warning_text`` may be called more than once.
    """
    try:
        done = getattr(client, "_acp_bgwork_journalled", None)
        if not isinstance(done, set):
            done = set()
            client._acp_bgwork_journalled = done
    except Exception:
        done = set()
    for call in calls:
        cid = str(call.get("id") or "")
        if not cid or cid in done:
            continue
        done.add(cid)
        _append_log(client, {
            "id": cid,
            "outcome": verdicts.get(cid, _OUTCOME_LOST),
            "tool": str(call.get("tool") or ""),
            "kind": str(call.get("kind") or ""),
        })


def warning_text(client: Any) -> str:
    """The notice to append to the reply, or "" when nothing was backgrounded.

    Deliberately part of the ANSWER rather than a progress line: a progress
    line is not in the transcript, so the next turn would not see it and the
    same mistake repeats — which is exactly what happened twice on 2026-08-01.
    """
    try:
        calls = list(getattr(client, "_acp_background_calls", None) or [])
    except Exception:
        return ""
    if not calls:
        return ""
    # Two calls never lost work and must not be reported as damage: one the
    # cap hook REFUSED (it never ran — the hook logs its refusals to disk with
    # the same `toolu_*` id the engine sends us, verified 15.08.2026) and one
    # whose RESULT the engine already wrote to the transcript (it came back
    # inside the turn — verified 16.08.2026 on 15 flagless `personal` spawns,
    # all 15 delivered while the doctor called all 15 lost). Both are disk
    # witnesses, not guesses. Reporting damage that did not happen teaches the
    # reader to skip this warning, which is the one warning that must be read.
    verdicts = _verdicts(client, calls)
    _journal_verdicts(client, calls, verdicts)
    calls = [c for c in calls
             if verdicts.get(str(c.get("id") or "")) == _OUTCOME_LOST]
    if not calls:
        return ""
    lines: list[str] = []
    for kind, note in (
        (_KIND_SUBAGENT, _SUBAGENT_NOTE),
        (_KIND_SHELL, _SHELL_NOTE),
        (_KIND_UNKNOWN, _UNKNOWN_NOTE),
    ):
        group = [c for c in calls if c.get("kind") == kind]
        if not group:
            continue
        names = ", ".join(
            sorted({str(c.get("tool") or "?") for c in group})
        )
        what = f"`{names}` (background) ×{len(group)}"
        labels = [str(c.get("label") or "") for c in group if c.get("label")]
        detail = f" — {'; '.join(labels[:3])}" if labels else ""
        lines.append(f"• {what}{detail}: {note}")
        if any(c.get("asked") == "default" for c in group):
            lines.append(_DEFAULT_ASYNC_HINT)
    return "⚠️ **Background work did not survive this turn.**\n" + "\n".join(lines)


def append_warning(client: Any, response_text: str) -> str:
    """Response text with the notice appended when there is one."""
    note = warning_text(client)
    if not note:
        return response_text
    body = (response_text or "").rstrip()
    return f"{body}\n\n{note}" if body else note
