"""ACP permission guard — the decision layer for engine tool approvals.

Local overlay module (not upstream). The Claude engine drives its native
tools over ACP and asks this client to approve each one; Hermes is the
trusted operator, so approvals are automatic EXCEPT for a narrow denylist
of catastrophic / credential-exfil actions and the curator session policy.

Why it lives here rather than in ``copilot_acp_client.py``: that file is
upstream and every local line in it is conflict surface for the next
``hermes update``. All guard logic is pure functions over the ACP
permission payload, so it moves out wholesale and the upstream file keeps
only an import. Same reasoning as ``gateway/progress_collapse.py``.

Public surface (re-exported by copilot_acp_client for compatibility):
    _permission_allow(message_id, params, extra_reason_fn) -> response dict
    _permission_denied(message_id, params, reason)         -> response dict
    _permission_danger_reason(params)                      -> reason | None
    _curator_permission_reason(params)                     -> reason | None
"""

from __future__ import annotations

import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from agent.acp_operator_grant import operator_grant_for as _operator_grant_for
except Exception:  # pragma: no cover - fail CLOSED, never open
    # If the grant module is missing (partial restore from the survival bundle),
    # the guard must keep refusing rather than silently allowing: the escape
    # hatch disappearing is an inconvenience, the guard disappearing is not.
    def _operator_grant_for(path: str) -> dict[str, Any] | None:  # type: ignore[misc]
        return None

def _permission_denied(
    message_id: Any,
    params: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Refuse a permission request, telling the engine WHY when we can.

    Outcome choice matters for the model, not just for us. claude-agent-acp
    maps ``cancelled`` to ``throw new Error("Tool use aborted")`` — an opaque
    technical failure that reads like a crash, so the model retries the same
    goal with shuffled syntax (observed: 8 rewordings of one .env write).
    Selecting an advertised reject option instead yields
    ``{behavior: "deny", message: ...}``, which the model reads as a decision
    and routes around. We attach the guard's reason via the ACP-reserved
    ``_meta`` so the refusal is self-describing for anything that surfaces it.
    """
    options = (params or {}).get("options") or []
    chosen: Any = None
    for wanted in ("reject_once", "reject_always"):
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") == wanted:
                chosen = opt.get("optionId") or opt.get("option_id")
                if chosen:
                    break
        if chosen:
            break
    outcome: dict[str, Any] = (
        {"outcome": "selected", "optionId": chosen}
        if chosen
        else {"outcome": "cancelled"}
    )
    result: dict[str, Any] = {"outcome": outcome}
    if reason:
        result["_meta"] = {
            "hermes.guard": {
                "denied": True,
                "reason": reason,
                "hint": _DENY_HINT,
            }
        }
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


# Guidance attached to every guard refusal: the blocked shapes all have a
# supported alternative, and saying so up front is what stops the retry loop.
# The operator-grant sentence is load-bearing: without a NAMED way through, a
# model that genuinely needs the write (the operator just pasted a token in
# chat) has nothing left but to reword the same command — which is how the
# guard got routed around instead of respected. See agent/acp_operator_grant.py.
_DENY_HINT = (
    "Hermes guard refused this action (not a transient failure — retrying "
    "with different syntax will fail the same way). Since 2026-08-25 the guard "
    "refuses ONLY catastrophic, effectively-irreversible actions: filesystem "
    "wipes (mkfs, dd onto a raw disk), fork bombs, chmod -R 777 /, and "
    "recursive deletes aimed at /, $HOME or a system root. Credential files "
    "are no longer guarded — that half was removed by the operator. "
    "Do not reword this command: narrow the target to the exact path you mean, "
    "or ask the operator to run it."
)


# --- Selective auto-approve guard for ACP native tool calls -------------------
# The Claude engine drives native tools (Bash, file edits) over ACP and asks us
# to approve each. We auto-approve for fluent operation, but HARD-DENY a small,
# unambiguous set of catastrophic / credential-exfil actions that must never run
# unattended, and audit-log EVERY decision so approvals stay reviewable. This is
# a backstop, not a full sandbox: fs/read and fs/write already enforce redaction
# and the write-denylist; this covers shell tool calls that bypass those paths.

_ACP_AUDIT_LOG_NAME = "acp-permission-audit.log"

# Catastrophic, effectively-irreversible shell actions (checked against lowered
# command text). Kept deliberately narrow to avoid false-positives on normal dev
# work — routine deletes of project subdirs are NOT matched.
_DANGEROUS_COMMAND_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),          # fork bomb
    re.compile(r"\bmkfs(\.\w+)?\b"),                                         # format filesystem
    re.compile(r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|vd|mmcblk|disk|hd)"),        # raw disk write
    re.compile(r">\s*/dev/(sd|nvme|vd|mmcblk|disk|hd)"),                     # clobber raw disk
    re.compile(r"\bchmod\b[^\n]*-r[^\n]*\s777\s+/(?:\s|$)"),                 # chmod -R 777 /
)

# Absolute/system/home targets that make a recursive delete catastrophic.
# TEXT-level fallback only: used when the payload has no recognizable command
# field, or when quoting is unparseable. The authoritative check is structural
# (`_catastrophic_rm_target`) and judges the rm's OWN argv — this pattern fires
# on the token anywhere in the blob and is therefore the fail-closed half.
# Boundary classes include quotes/backslash because the command usually arrives
# JSON-serialized (e.g. rawInput={"command": "rm -rf /"}), so the target token is
# flanked by `"`.
_SYSTEM_ROOT_NAMES: tuple[str, ...] = (
    "home", "etc", "usr", "var", "boot", "bin", "sbin", "lib", "lib32", "lib64",
    "opt", "root", "srv", "mnt", "media", "data", "dev", "proc", "sys", "run",
    "snap", "nix",
)
# /tmp is deliberately absent: scratch dirs under it are ordinary work, and the
# depth rule below would otherwise refuse `rm -rf /tmp/<anything>`.
_CATASTROPHIC_RM_TARGET_RE = re.compile(
    r"""(?:^|[\s"'])(?:/|~|\$home|\$\{home\}|/\*)(?:[\s"'`;&|\\]|$)"""
    r"""|(?:^|[\s"'])(?:/(?:""" + "|".join(_SYSTEM_ROOT_NAMES) + r"""))"""
    r"""(?:/[^/\s"'`;&|\\*]*)?(?:/\*|/)?(?:[\s"'`;&|\\]|$)"""
)

# Credential files whose overwrite/exfiltration must never be auto-approved,
# plus the operators that indicate a write-out or ship-out of one.
#
# 25.08.2026 — КРЕДОВАЯ ПОЛОВИНА СНЯТА по прямому решению Олега («режь часть с
# кредами»). Катастрофическая половина (fork bomb / mkfs / dd of=/dev/sd* /
# recursive rm по корню и системным путям / chmod -R 777 /) остаётся в силе:
# она стоит между кривой строкой и стёртой машиной, и её никто не отменял.
#
# Почему снята. Заслон судил не действие, а ПОДСТРОКУ в тексте команды: чтение
# имён переменных грепом, запись собственного конфига плагина и даже текст
# kanban-карточки, где упомянут файл, уходили в тихий DENY, неотличимый от
# отказа оператора. Цена контроля оказалась выше пользы: он не мешал ничему
# опасному (движок работает от лица Олега на его же машине), но регулярно
# заставлял искать обходные пути — то есть учил обходить заслон, а не соблюдать.
#
# Прямые следствия, названные вслух:
#   * запись/чтение .env, id_rsa, .credentials.json больше НЕ требуют гранта;
#   * стор операторских грантов acp-operator-grants.json стал обычным файлом.
#     Выдачу грантов шеллом по-прежнему ловит _shell_issues_operator_grant()
#     (строка ~842) — эта проверка не тронута, но сами гранты теперь не нужны;
#   * аудит-лог пишется как писался: каждое решение по-прежнему видно.
#
# ВЕРНУТЬ: перенести имена из _CREDENTIAL_BASENAMES_DISABLED обратно в
# _CREDENTIAL_BASENAMES. Одна строка, остальной код кредовой ветки не удалён.
_CREDENTIAL_BASENAMES_DISABLED: tuple[str, ...] = (
    ".credentials.json", "credentials.json", "auth.json", ".claude-acp-token",
    "id_rsa", "id_ed25519", "id_ecdsa", ".env",
    # The operator-grant store. Listed here so every write path already covered
    # for credentials covers it too; `operator_grant_for()` refuses to return a
    # grant for its own store, so this entry is un-grantable by construction —
    # the engine cannot open its own door.
    "acp-operator-grants.json",
)
_CREDENTIAL_BASENAMES: tuple[str, ...] = ()
_EXFIL_TOKENS_RE = re.compile(
    r">>?|\btee\b|\bcp\b|\bmv\b|\bscp\b|\bcurl\b|\bwget\b|\bnc\b|\brsync\b|\bcat\b"
)
# Shell utilities that write out / ship out a file's contents, matched as argv
# token basenames only (openclaw-style structural check, never as substrings
# of arbitrary flattened text).
_EXFIL_BIN_NAMES: frozenset[str] = frozenset(
    {"cat", "cp", "mv", "tee", "scp", "curl", "wget", "nc", "rsync"}
)
# In-place editors and interpreters. Found 2026-07-30: the set above lists only
# ship-out binaries, so `sed -i` and `python3 - <<EOF` wrote a credential file
# straight past the guard — a control that refuses `cp` while allowing `sed -i`
# is not a control, it is a spelling test. Cost of including them: `sed -n p`,
# `awk` and `python3` READS of a credential file are refused too. Accepted —
# `cat` (also a read) has always been on the list, `grep`/`head`/`tail` still
# read fine, and the deny hint now names the sanctioned way through.
_INPLACE_WRITE_BIN_NAMES: frozenset[str] = frozenset(
    {
        "sed", "perl", "awk", "gawk", "mawk", "ed", "ex", "sponge", "patch",
        "truncate", "shred", "install", "dd", "chmod", "chown", "chgrp",
        "ln", "touch",
    }
)
_INTERPRETER_BIN_NAMES: frozenset[str] = frozenset(
    {"python", "python3", "ruby", "node", "deno", "bun", "php"}
)

# rawInput keys that carry a shell command / a target file path across the
# engine's native tools (Bash, Write, Edit, NotebookEdit). Key names vary by
# engine build (snake_case vs camelCase), so matching is done on a normalized
# form (lowercased, underscores stripped) — never on literal spellings.
_SHELL_COMMAND_KEYS: tuple[str, ...] = ("command", "cmd", "script")
_FILE_PATH_KEYS_NORM: frozenset[str] = frozenset(
    {
        "filepath", "path", "abspath", "notebookpath",
        "target", "destination", "dest", "uri", "file", "filename",
    }
)
# File-content fields: NEVER scanned for danger patterns. A document that
# merely mentions credential filenames or shell operators is not an action.
_FILE_CONTENT_KEYS_NORM: frozenset[str] = frozenset(
    {
        "content", "oldstring", "newstring", "oldtext", "newtext",
        "text", "newsource", "oldsource", "cells", "body", "diff", "edits",
    }
)


def _norm_key(key: str) -> str:
    return key.replace("_", "").lower()
# Tool-call kinds that only read; a credential path there is the read-redaction
# layer's job (get_read_block_error), not this guard's.
_READONLY_TOOL_KINDS: frozenset[str] = frozenset({"read", "search", "fetch", "think"})


def _permission_command_text(params: dict[str, Any] | None) -> str:
    """Flatten an ACP permission request into searchable text.

    Robust to schema nesting: pulls the toolCall title + rawInput, and falls
    back to the whole params blob (minus the options menu), so the pattern
    checks fire regardless of the engine's exact payload shape.
    """
    p = params or {}
    parts: list[str] = []
    tc = p.get("toolCall") or p.get("tool_call") or {}
    if isinstance(tc, dict):
        for key in ("title", "kind"):
            val = tc.get(key)
            if isinstance(val, str):
                parts.append(val)
        raw = tc.get("rawInput") or tc.get("raw_input") or tc.get("input")
        if raw is not None:
            try:
                parts.append(json.dumps(raw, ensure_ascii=False))
            except Exception:
                parts.append(str(raw))
    if not parts:
        try:
            trimmed = {k: v for k, v in p.items() if k != "options"}
            parts.append(json.dumps(trimmed, ensure_ascii=False))
        except Exception:
            parts.append(str(p))
    return "\n".join(parts)


def _has_recursive_force_rm(text: str) -> bool:
    """True if the text contains an `rm` invocation with both -r and -f flags."""
    for m in re.finditer(r"\brm\s+((?:-\S+\s+)+)", text):
        letters = m.group(1).replace("-", "")
        if "r" in letters and "f" in letters:
            return True
    return False


def _tool_call_meta(
    params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None]:
    """Return (kind, rawInput-dict-or-None) for an ACP permission request."""
    p = params or {}
    tc = p.get("toolCall") or p.get("tool_call") or {}
    kind = ""
    raw: dict[str, Any] | None = None
    if isinstance(tc, dict):
        kind = str(tc.get("kind") or "").lower()
        r = tc.get("rawInput") or tc.get("raw_input") or tc.get("input")
        if isinstance(r, dict):
            raw = r
    return kind, raw


def _extract_shell_command(raw: dict[str, Any] | None) -> str | None:
    if raw:
        for key in _SHELL_COMMAND_KEYS:
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return None


def _extract_file_paths(raw: dict[str, Any] | None) -> list[str]:
    paths: list[str] = []
    if raw:
        for key, val in raw.items():
            if _norm_key(key) in _FILE_PATH_KEYS_NORM and isinstance(val, str) and val.strip():
                paths.append(val)
    return paths


def _credential_basename_match(path: str) -> str | None:
    """Match a single path/token against the credential basenames."""
    base = path.rstrip("/").rsplit("/", 1)[-1].lstrip("@").lower()
    for cred in _CREDENTIAL_BASENAMES:
        if base == cred or base.startswith(cred + "."):
            return cred
    return None


# Characters that can only sit OUTSIDE a path — used to recover a filename that
# is embedded in a bigger token rather than standing alone as an argv word.
_PATH_TOKEN_BOUNDARY = "\"'`(),;:= \t\n"


def _embedded_credential_path(token: str) -> tuple[str, str] | None:
    """Find a credential path INSIDE a token, e.g. the body of `python3 -c`.

    ``python3 -c "open('/x/.env','w').write(...)"`` arrives as one argv word,
    so the basename check (which splits on ``/`` and compares the tail) sees
    ``.env','w').write(...)`` and matches nothing. That gap is exactly how an
    interpreter wrote a credential file past the guard. Returns
    (basename, recovered path) so a grant can still be matched by path.
    """
    low = token.lower()
    for cred in _CREDENTIAL_BASENAMES:
        idx = 0
        while True:
            i = low.find(cred, idx)
            if i < 0:
                break
            idx = i + 1
            end = i + len(cred)
            after = low[end:end + 1]
            if after and (after.isalnum() or after in "-_."):
                continue  # `.environment`, `auth.json.bak-old` — not this file
            before = low[i - 1:i]
            if before and (before.isalnum() or before in "-_"):
                continue  # `my.env` is someone else's file, not the token
            start = i
            while start > 0 and low[start - 1] not in _PATH_TOKEN_BOUNDARY:
                start -= 1
            return cred, token[start:end]
    return None


def _shell_tokens(command: str) -> list[str] | None:
    """Tokenize a shell command; operators become their own tokens."""
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        return list(lex)
    except ValueError:
        return None


# Redirect destinations that discard output — writing to them exfiltrates
# nothing, so they never count as the write half of a credential pair.
_DISCARD_SINKS: frozenset[str] = frozenset(
    {"/dev/null", "/dev/zero", "/dev/tty", "/dev/stdout", "/dev/stderr"}
)


def _is_discard_sink(target: str) -> bool:
    return target.strip().strip("\"'").lower() in _DISCARD_SINKS


# Statement separators for the credential check. `|` is deliberately NOT here:
# a pipe carries the data across, so `grep KEY .env | tee /tmp/out` is one
# exfil, not two innocent halves. The curator check splits on `|` because it
# judges mutation targets, not data flow — same-looking constant, different
# question.
_EXFIL_STATEMENT_SEPARATORS: frozenset[str] = frozenset({";", "&&", "||", "&"})


def _segment_credential_hit(
    tokens: list[str], depth: int
) -> tuple[str, str] | None:
    """(credential basename, path token) for ONE statement, else None."""
    has_op = False
    cred_hit: tuple[str, str] | None = None
    prev_was_redirect = False
    for tok in tokens:
        if not tok:
            continue
        if depth < 2 and any(ws in tok for ws in (" ", "\t", "\n")):
            # A quoted multi-word token is likely a nested command
            # (bash -c '...', ssh host '...'); check the pair inside it only.
            nested = _shell_credential_hit(tok, depth + 1)
            if nested:
                return nested
            continue
        # A redirect whose destination is a null/tty sink writes nothing, so
        # it must not arm the operator half of the pair. Without this,
        # `grep KEY ~/.hermes/.env 2>/dev/null` — a pure read — was refused
        # because the stderr redirect counted as a write (observed 2026-07-25).
        if prev_was_redirect:
            # This token is the destination of a bare ">"/">>" seen last pass.
            prev_was_redirect = False
            if _is_discard_sink(tok):
                continue
            has_op = True
        if ">" in tok:
            redirect_target = tok.split(">")[-1]
            if redirect_target:
                if not _is_discard_sink(redirect_target):
                    has_op = True
            else:
                # Bare ">"/">>": the destination is the next token.
                prev_was_redirect = True
        word = tok.lstrip("<>&|;")
        bin_name = word.rstrip("/").rsplit("/", 1)[-1].lower()
        if (
            bin_name in _EXFIL_BIN_NAMES
            or bin_name in _INPLACE_WRITE_BIN_NAMES
            or bin_name in _INTERPRETER_BIN_NAMES
        ):
            has_op = True
        if cred_hit is None:
            cred = _credential_basename_match(word)
            if cred:
                cred_hit = (cred, word)
            else:
                cred_hit = _embedded_credential_path(word)
    return cred_hit if has_op else None


def _shell_credential_hit(command: str, depth: int = 0) -> tuple[str, str] | None:
    """Structural check: does this command pair a write-out/ship-out operation
    with a credential-file token? Returns (basename, path token).

    Tokens are examined individually (argv words, redirect targets, nested
    command strings) and grouped per statement, so a credential filename in
    unrelated prose — or in a different statement of the same line — can no
    longer pair with an operator elsewhere in the flattened blob. The token is
    returned alongside the basename because an operator grant is issued for one
    PATH, and the basename alone can't tell `.env` files apart.
    """
    tokens = _shell_tokens(command)
    if tokens is None:
        # Unparseable quoting: fail closed with the legacy substring pair. No
        # token to grant against, so this shape can never be granted — the
        # operator's way out is to have the model write a parseable command.
        low = command.lower()
        if _EXFIL_TOKENS_RE.search(low):
            for cred in _CREDENTIAL_BASENAMES:
                if cred in low:
                    return cred, ""
        return None

    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _EXFIL_STATEMENT_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)

    for seg in segments:
        hit = _segment_credential_hit(seg, depth)
        if hit:
            return hit
    return None


def _shell_credential_exfil(command: str, depth: int = 0) -> str | None:
    """Back-compat wrapper: the credential basename only (no path token)."""
    hit = _shell_credential_hit(command, depth)
    return hit[0] if hit else None


# The grant-issuing command. It is reachable from a shell (`hermes env-grant`)
# as well as from chat, and a door the engine can open for itself is not a door.
# The store path is already protected as a credential basename; this closes the
# other half — invoking the issuer.
_GRANT_ISSUER_TOKENS: frozenset[str] = frozenset({"env-grant", "/env-grant", "env_grant"})
# ...but the same word is also a plugin NAME, and reading or administering the
# plugin is ordinary work (`hermes plugins enable env-grant`, `grep env-grant
# config.yaml`). What separates the two is POSITION, not vocabulary: a shell word
# runs something only in command position. Tokens that keep the next word in
# command position — the launcher itself, wrappers, flags, VAR=value assignments
# and a bare duration (`timeout 120 hermes …`).
_GRANT_TRANSPARENT_PREFIXES: frozenset[str] = frozenset(
    {
        "hermes", "sudo", "doas", "env", "command", "exec", "nohup", "xargs",
        "stdbuf", "nice", "ionice", "timeout", "time", "uv", "uvx", "poetry",
    }
)
_DURATION_ARG_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")


def _shell_issues_operator_grant(command: str, depth: int = 0) -> bool:
    """True only when the command actually INVOKES the grant issuer.

    Position is the whole test: `hermes env-grant …` runs it, `grep -i
    env-grant` merely names it. The first version fired on the token anywhere in
    the line, so reading the plugin's own name was refused — found 2026-07-30
    while verifying the plugin was enabled, one grep pattern apart from an
    identical command that passed. A guard that blocks reads teaches the model to
    reword instead of to obey, which is the exact failure the operator grant
    exists to end.
    """
    tokens = _shell_tokens(command)
    if tokens is None:
        # Unparseable quoting: fail closed on the bare mention, as before.
        return "env-grant" in command.lower()

    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in (";", "&&", "||", "|", "&"):
            segments.append([])
        else:
            segments[-1].append(tok)

    for seg in segments:
        in_command_position = True
        for tok in seg:
            if depth < 2 and any(ws in tok for ws in (" ", "\t", "\n")):
                # Nested command string (bash -c '…', ssh host '…').
                if _shell_issues_operator_grant(tok, depth + 1):
                    return True
                continue
            word = tok.strip().strip("\"'").lower()
            if not word:
                continue
            if in_command_position and word in _GRANT_ISSUER_TOKENS:
                return True
            base = word.rstrip("/").rsplit("/", 1)[-1]
            if (
                base in _GRANT_TRANSPARENT_PREFIXES
                or word.startswith("-")
                or _DURATION_ARG_RE.match(word)
                or ("=" in word and not word.startswith("/"))
            ):
                continue  # still in command position
            in_command_position = False
    return False


# --- Recursive deletes: judge the rm's OWN argv, not the whole line -------------
# The first version tested `has a recursive rm` AND `a catastrophic path appears
# somewhere in the text`. Those two facts need not belong to the same command:
# on 2026-08-02 `sudo rm -rf /var/lib/containerd.old-… ; df -h /` was refused
# because of the trailing `df -h /` — the same "match anywhere instead of in
# position" defect already fixed twice (grep of `env-grant`, `pgrep -f` matching
# its own command line). The scan below collects the arguments of each `rm` that
# actually carries a recursive flag, and judges only those.
#
# The second half is scope. The old target pattern required a boundary right
# after `/home`, so it caught the spellings nobody types (`rm -rf /home`) and
# let through the ones that kill the machine: `rm -rf /home/hun`, `rm -rf ~/*`,
# `rm -rf /var/lib`, `rm -rf /usr/lib` all passed. Depth is the rule now:
#   0 segments (`/`)                     → catastrophic
#   1 segment  (`/home`, `/etc`, `/opt`) → catastrophic (a whole top-level dir)
#   2 segments (`/home/hun`, `/var/lib`) → catastrophic iff under a system root
#   3+ segments                          → ordinary work, allowed
# so `/var/lib/containerd.old-…`, `/tmp/x/y` and `~/Downloads/old` stay allowed
# while `/home/hun` — 170G including Hermes itself — no longer does.
_RM_BIN_NAMES: frozenset[str] = frozenset({"rm"})
_SYSTEM_ROOT_DIRS: frozenset[str] = frozenset(_SYSTEM_ROOT_NAMES)
_CATASTROPHIC_RM_DEPTH = 2
# `>`, `>>`, `2>`, `>&`, `&>`, `<`: shlex(punctuation_chars=True) hands these back
# as their own tokens, and the word after a redirect is a destination, not a
# delete target.
_REDIRECT_TOKEN_RE = re.compile(r"^&?\d*[<>]+&?\d*$")
# Separators that only exist because the payload arrived JSON-serialized.
_JSON_PUNCTUATION_RE = re.compile(r"""["',\[\]]+""")
_GLOB_SEGMENTS: frozenset[str] = frozenset({"*", "**", "?", "."})
# Binaries that EXECUTE a quoted command string handed to them as one argument.
_NESTED_SHELL_RUNNERS: frozenset[str] = frozenset(
    {"bash", "sh", "zsh", "dash", "ksh", "fish", "ssh", "su", "flatpak-spawn"}
)


def _home_dir() -> str:
    return os.environ.get("HOME") or os.path.expanduser("~")


def _delete_target_path(token: str) -> str | None:
    """Absolute, normalized path a delete argument points at (None if relative).

    Relative paths are out of scope on purpose: without the shell's cwd they
    cannot be judged, and refusing them would refuse ordinary `rm -rf build`.
    """
    word = token.strip().strip("\"'`")
    if not word:
        return None
    home = _home_dir()
    if word == "~" or word.startswith("~/"):
        word = home + word[1:]
    elif word.startswith("$HOME"):
        word = home + word[5:]
    elif word.startswith("${HOME}"):
        word = home + word[7:]
    elif word.startswith("~"):
        expanded = os.path.expanduser(word)     # ~root, ~otheruser
        if expanded == word:
            return None
        word = expanded
    if not word.startswith("/"):
        return None
    segments: list[str] = []
    for seg in word.split("/"):
        if not seg or seg == ".":
            continue
        if seg in _GLOB_SEGMENTS or "*" in seg or "?" in seg:
            # `~/*`, `/home/*/data`: the glob's PARENT is what gets emptied, so
            # judge that and stop — everything past the wildcard is unknown.
            break
        if seg == "..":
            if segments:
                segments.pop()
            continue
        segments.append(seg)
    return "/" + "/".join(segments)


def _is_catastrophic_delete_target(path: str) -> bool:
    segments = [s for s in path.split("/") if s]
    if len(segments) <= 1:
        return True                                  # `/` or a whole top-level dir
    if len(segments) > _CATASTROPHIC_RM_DEPTH:
        return False
    return segments[0].lower() in _SYSTEM_ROOT_DIRS


def _statement_groups(command: str) -> list[list[str]] | None:
    """Per-statement argv groups, or None if quoting is unparseable.

    Lines are tokenized separately: shlex treats `\\n` as plain whitespace, so a
    two-line script would otherwise fuse into one statement and the second
    line's arguments would look like arguments of the first line's command.
    """
    groups: list[list[str]] = []
    for line in command.splitlines():
        if not line.strip():
            continue
        tokens = _shell_tokens(line)
        if tokens is None:
            return None
        current: list[str] = []
        for tok in tokens:
            if tok in (";", "&&", "||", "|", "&"):
                groups.append(current)
                current = []
            else:
                current.append(tok)
        groups.append(current)
    return [g for g in groups if g]


def _rm_recursive_targets(command: str, depth: int = 0) -> list[str]:
    """Every path a recursive `rm` in this command would delete."""
    groups = _statement_groups(command)
    if groups is None:
        return []
    targets: list[str] = []
    for seg in groups:
        i = 0
        in_command_position = True
        command_base = ""
        while i < len(seg):
            tok = seg[i]
            if depth < 2 and any(ws in tok for ws in (" ", "\t", "\n")):
                # A quoted multi-word token is a nested command only when this
                # statement RUNS one. `bash -c "rm -rf /"` deletes; `echo "rm
                # -rf /"` prints, and refusing it would be the same
                # match-anywhere mistake one level down.
                if in_command_position or command_base in _NESTED_SHELL_RUNNERS:
                    targets.extend(_rm_recursive_targets(tok, depth + 1))
                i += 1
                continue
            word = tok.strip().strip("\"'")
            if _REDIRECT_TOKEN_RE.match(word):
                i += 2                               # skip the destination too
                continue
            if not word:
                i += 1
                continue
            base = word.rstrip("/").rsplit("/", 1)[-1].lower()
            if in_command_position and base in _RM_BIN_NAMES:
                i = _collect_rm_argv(seg, i + 1, targets)
                continue
            if in_command_position and (
                base in _GRANT_TRANSPARENT_PREFIXES
                or word.startswith("-")
                or _DURATION_ARG_RE.match(word)
                or ("=" in word and not word.startswith("/"))
            ):
                i += 1
                continue                             # still in command position
            if in_command_position:
                command_base = base
            in_command_position = False
            i += 1
    return targets


def _collect_rm_argv(seg: list[str], start: int, targets: list[str]) -> int:
    """Consume one rm's arguments; append its targets iff it is recursive.

    Flags are read from anywhere in the argv (`rm /path -rf` is the same command
    as `rm -rf /path`), and only `--recursive` among the long options maps to r —
    `--force` happens to contain the letter and must not be mistaken for it.
    """
    flags = ""
    args: list[str] = []
    end_of_flags = False
    i = start
    while i < len(seg):
        word = seg[i].strip().strip("\"'")
        if _REDIRECT_TOKEN_RE.match(word):
            i += 2
            continue
        if not word:
            i += 1
            continue
        if not end_of_flags and word == "--":
            end_of_flags = True
            i += 1
            continue
        if not end_of_flags and word.startswith("--"):
            if word[2:].lower() == "recursive":
                flags += "r"
            i += 1
            continue
        if not end_of_flags and word.startswith("-") and len(word) > 1:
            flags += word[1:].lower()
            i += 1
            continue
        args.append(word)
        i += 1
    if "r" in flags:
        targets.extend(args)
    return i


def _catastrophic_rm_target(command: str) -> str | None:
    """The path a recursive delete in ``command`` would destroy, else None."""
    groups = _statement_groups(command)
    if groups is None:
        # Unparseable quoting: fail closed on the legacy text pair, which needs
        # no tokens. Deliberately weaker AND wider than the structural check.
        low = command.lower()
        if _has_recursive_force_rm(low) and _CATASTROPHIC_RM_TARGET_RE.search(low):
            return "<unparseable command>"
        return None
    for token in _rm_recursive_targets(command):
        path = _delete_target_path(token)
        if path is not None and _is_catastrophic_delete_target(path):
            return path
    return None


# --- Restarting a gateway that is carrying Kanban work -------------------------
# A Kanban worker is its own process, but it lives in the cgroup of whichever
# gateway holds .dispatcher.lock — NOT of the profile the card belongs to. That
# unit is KillMode=mixed, so restarting the gateway takes the whole cgroup with
# it and the card's run dies mid-flight. Twice now (2026-08-01, and 2026-08-02
# 08:18 when the lock had moved to `personal`) a gateway that looked idle by its
# own turn counter was carrying the strata worker, and both times the restart
# was typed straight into a turn. gateway-restart-when-idle.sh already waits for
# those workers; this rule makes the safe path the only path instead of a habit.
_GATEWAY_KILL_VERBS: frozenset[str] = frozenset(
    {"restart", "stop", "kill", "reload-or-restart", "try-restart"}
)
_GATEWAY_UNIT_STEM = "hermes-gateway"
_KANBAN_WORKER_MARK = "work kanban task"
_GATEWAY_SAFE_PATH_HINT = (
    "use ~/.hermes/scripts/gateway-restart-when-idle.sh (detached via "
    "systemd-run --user --collect --unit=…): it waits for the workers, then "
    "drains the gateway"
)


def _gateway_unit_from_profile(profile: str) -> str:
    """The default profile lives in ~/.hermes; its unit carries no suffix."""
    prof = profile.strip().strip("\"'")
    if not prof or prof == "default":
        return f"{_GATEWAY_UNIT_STEM}.service"
    return f"{_GATEWAY_UNIT_STEM}-{prof}.service"


def _gateway_kill_target(command: str, depth: int = 0) -> str | None:
    """Unit this command would restart/stop, or None — position decides.

    ``systemctl --user status hermes-gateway-personal`` and a grep for the name
    are reads and must pass: same lesson as the anti-self-grant rule, which
    once refused to let me read the word it was guarding.
    """
    tokens = _shell_tokens(command)
    if tokens is None:
        return None
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in (";", "&&", "||", "|", "&"):
            segments.append([])
        else:
            segments[-1].append(tok)
    for seg in segments:
        words: list[str] = []
        for tok in seg:
            if depth < 2 and any(ws in tok for ws in (" ", "\t", "\n")):
                nested = _gateway_kill_target(tok, depth + 1)
                if nested:
                    return nested
                continue
            words.append(tok.strip().strip("\"'"))
        if not words:
            continue
        lowered = [w.lower() for w in words]
        bases = [w.rstrip("/").rsplit("/", 1)[-1] for w in lowered]
        if not any(w in _GATEWAY_KILL_VERBS for w in lowered):
            continue
        if "systemctl" in bases:
            for word in lowered:
                if word.startswith(_GATEWAY_UNIT_STEM):
                    return word if word.endswith(".service") else f"{word}.service"
            continue
        if "hermes" in bases and "gateway" in lowered:
            profile = ""
            for flag in ("-p", "--profile"):
                if flag in lowered:
                    idx = lowered.index(flag)
                    if idx + 1 < len(words):
                        profile = words[idx + 1]
            return _gateway_unit_from_profile(profile)
    return None


def _live_kanban_workers(unit: str) -> int:
    """How many Kanban worker processes sit in ``unit``'s cgroup right now."""
    count = 0
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
                if _KANBAN_WORKER_MARK.encode() not in cmdline:
                    continue
                if unit in (entry / "cgroup").read_text(encoding="utf-8"):
                    count += 1
            except OSError:
                continue
    except OSError:
        return 0
    return count


def _shell_danger_verdict(command: str) -> tuple[str | None, dict[str, Any] | None]:
    """(deny reason, operator grant used) for a shell command string."""
    low = command.lower()
    unit = _gateway_kill_target(command)
    if unit:
        workers = _live_kanban_workers(unit)
        if workers:
            return (
                f"restarting {unit} would kill {workers} live Kanban worker(s) "
                f"in its cgroup (KillMode=mixed) — {_GATEWAY_SAFE_PATH_HINT}",
                None,
            )
    if _shell_issues_operator_grant(command):
        return (
            "operator grants are issued BY THE OPERATOR in chat "
            "(/env-grant), never by the engine",
            None,
        )
    for rx in _DANGEROUS_COMMAND_RES:
        if rx.search(low):
            return f"destructive command pattern /{rx.pattern[:48]}/", None
    rm_target = _catastrophic_rm_target(command)
    if rm_target:
        return (
            f"recursive rm targeting a root/system/home path ({rm_target})",
            None,
        )
    hit = _shell_credential_hit(command)
    if hit:
        cred, token = hit
        grant = _operator_grant_for(token) if token else None
        if grant is not None:
            return None, grant
        return f"shell write/exfil touching credential file '{cred}'", None
    return None, None


def _shell_danger_reason(command: str) -> str | None:
    """Danger checks that apply to a shell command string."""
    return _shell_danger_verdict(command)[0]


def _strip_content_fields(params: dict[str, Any] | None) -> dict[str, Any]:
    """Copy of params with file-content fields removed from rawInput, and the
    toolCall-level ``content`` blocks dropped entirely (ACP puts display diffs
    with oldText/newText there — display content, never an action)."""
    p = dict(params or {})
    key = "toolCall" if "toolCall" in p else "tool_call"
    tc = p.get(key)
    if isinstance(tc, dict):
        tc = dict(tc)
        tc.pop("content", None)
        for rk in ("rawInput", "raw_input", "input"):
            r = tc.get(rk)
            if isinstance(r, dict):
                tc[rk] = {
                    k: v for k, v in r.items()
                    if _norm_key(k) not in _FILE_CONTENT_KEYS_NORM
                }
        p[key] = tc
    return p


def _danger_verdict(
    params: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """(deny reason, operator grant used) for one permission request.

    Structural, openclaw-style dispatch instead of one flat substring scan:
    shell commands are tokenized and checked for operator+credential pairs;
    file writes are judged by DESTINATION PATH only. File content is never
    scanned, so writing a document that merely mentions credential filenames
    or shell operators is no longer a false positive.

    A live operator grant (agent/acp_operator_grant.py) for the exact path
    turns a credential refusal into an allow — and the grant comes back so the
    caller can say so in the audit log. Nothing else is grantable: catastrophic
    commands stay denied regardless.
    """
    kind, raw = _tool_call_meta(params)
    command = _extract_shell_command(raw)
    if command is not None:
        return _shell_danger_verdict(command)
    paths = _extract_file_paths(raw)
    if paths:
        if kind in _READONLY_TOOL_KINDS:
            return None, None
        for path in paths:
            cred = _credential_basename_match(path)
            if cred:
                grant = _operator_grant_for(path)
                if grant is not None:
                    return None, grant
                return (
                    f"file {kind or 'write'} targeting credential file '{cred}'",
                    None,
                )
        return None, None
    # Unknown payload shape: legacy conservative scan over the flattened call,
    # minus any file-content fields. No path is identified here, so a grant
    # cannot apply — an operator who hits this should ask for the write in a
    # shape the structural checks understand.
    text = _permission_command_text(_strip_content_fields(params))
    if not text:
        return None, None
    low = text.lower()
    for rx in _DANGEROUS_COMMAND_RES:
        if rx.search(low):
            return f"destructive command pattern /{rx.pattern[:48]}/", None
    # JSON punctuation is flattened to whitespace FIRST: an argv-list payload
    # (`{"args": ["rm", "-rf", "/home/…"]}`) reads as `rm", "-rf`, where the
    # flag scan finds no flags after `rm` and the whole pair goes quiet. This
    # path is the crude fail-closed half; it must not be defeated by a comma.
    flat = _JSON_PUNCTUATION_RE.sub(" ", low)
    if _has_recursive_force_rm(flat) and _CATASTROPHIC_RM_TARGET_RE.search(flat):
        return "recursive-force rm targeting a root/system/home path", None
    if _EXFIL_TOKENS_RE.search(low):
        for base in _CREDENTIAL_BASENAMES:
            if base.lower() in low:
                return f"write/exfil touching credential file '{base}'", None
    return None, None


def _permission_danger_reason(params: dict[str, Any] | None) -> str | None:
    """Return a reason if this tool call is catastrophic/exfil, else None."""
    return _danger_verdict(params)[0]


# --- Curator-session policy ----------------------------------------------------
# The background skill curator (AIAgent platform="curator") must restructure the
# library through skill_manage: that's the path that writes .archive/ backups,
# records absorbed_into for cron-reference migration, and feeds the run report.
# Its prompt already forbids shell archiving, but a prompt is not enforcement —
# the 2026-07-21 consolidation run did the whole restructure via shell and left
# .archive/ empty. These rules make the sanctioned flows the only ones that pass:
#   • reading anywhere — fine (base guard/redaction still apply);
#   • mkdir + mv/cp INTO a skill's references/, templates/, scripts/, assets/
#     subdir — the prompt's package-integrity demote flow;
#   • everything else that mutates the skills tree (rm/rmdir of skill content,
#     mv of skill roots, writes outside support dirs, anything touching
#     .archive/) — denied; the model must use skill_manage.
# Like the base guard this is a structural backstop, not a sandbox: it judges
# argv tokens and destination paths, never flattened text or file content.

_CURATOR_SUPPORT_DIRS: frozenset[str] = frozenset(
    {"references", "templates", "scripts", "assets"}
)
_CURATOR_DELETE_BINS: frozenset[str] = frozenset(
    {"rm", "rmdir", "shred", "unlink", "truncate"}
)
# tar/dd are deliberately absent: their argument order defeats the
# last-token-is-destination heuristic (tar's last arg is the SOURCE), and
# reading skills content into an archive elsewhere mutates nothing.
_CURATOR_MOVE_BINS: frozenset[str] = frozenset(
    {"mv", "cp", "rsync", "install", "ln", "rename"}
)
_CURATOR_INPLACE_BINS: frozenset[str] = frozenset({"sed", "perl"})
_CURATOR_USE_TOOL_HINT = (
    "curator-policy: mutate the skills library via skill_manage "
    "(action=delete archives with absorbed_into bookkeeping; "
    "action=patch/create/write_file edit content) — shell/file mutation "
    "of the skills tree is limited to moves into references/, templates/, "
    "scripts/, assets/"
)


def _curator_skills_roots() -> tuple[str, ...]:
    roots: list[str] = []
    try:
        from hermes_constants import get_hermes_home

        roots.append(str(get_hermes_home() / "skills"))
    except Exception:
        pass
    default = os.path.expanduser("~/.hermes/skills")
    if default not in roots:
        roots.append(default)
    return tuple(roots)


def _curator_skills_rel(token: str) -> str | None:
    """Path relative to a skills root if *token* points into it, else None."""
    if not token:
        return None
    t = token.strip().strip("\"'")
    if not t or t.startswith("-"):
        return None
    t = t.lstrip("<>&|;")
    home = os.path.expanduser("~")
    t = t.replace("${HOME}", home).replace("$HOME", home)
    t = os.path.expanduser(t)
    if "/" not in t:
        return None
    t = os.path.normpath(t)
    for root in _curator_skills_roots():
        root = os.path.normpath(root)
        if t == root:
            return ""
        if t.startswith(root + os.sep):
            return t[len(root) + 1:].replace(os.sep, "/")
    return None


def _curator_rel_allowed(rel: str) -> bool:
    """A skills-tree write destination is sanctioned only inside a support dir."""
    parts = [p for p in rel.split("/") if p]
    if ".archive" in parts:
        return False
    return any(p in _CURATOR_SUPPORT_DIRS for p in parts)


def _curator_shell_reason(command: str, depth: int = 0) -> str | None:
    """Structural check of a shell command against the curator policy."""
    tokens = _shell_tokens(command)
    if tokens is None:
        # Unparseable quoting near the skills tree: fail closed.
        low = command.lower()
        if ".hermes/skills" in low or ".hermes\\skills" in low:
            return _CURATOR_USE_TOOL_HINT + " (unparseable command)"
        return None

    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in (";", "&&", "||", "|", "&"):
            segments.append([])
        else:
            segments[-1].append(tok)

    for seg in segments:
        if not seg:
            continue
        if depth < 2:
            for tok in seg:
                if any(ws in tok for ws in (" ", "\t", "\n")):
                    nested = _curator_shell_reason(tok, depth + 1)
                    if nested:
                        return nested
        bin_name = seg[0].rsplit("/", 1)[-1].lower()
        rels = [(tok, _curator_skills_rel(tok)) for tok in seg[1:]]
        skills_toks = [(tok, rel) for tok, rel in rels if rel is not None]

        # Redirect targets writing into the skills tree.
        for i, tok in enumerate(seg):
            if tok in (">", ">>") and i + 1 < len(seg):
                rel = _curator_skills_rel(seg[i + 1])
                if rel is not None and not _curator_rel_allowed(rel):
                    return _CURATOR_USE_TOOL_HINT + " (shell redirect)"
            elif ">" in tok:
                rel = _curator_skills_rel(tok.split(">")[-1])
                if rel is not None and not _curator_rel_allowed(rel):
                    return _CURATOR_USE_TOOL_HINT + " (shell redirect)"

        if not skills_toks:
            continue
        if any(".archive" in [p for p in (rel or "").split("/")] for _, rel in skills_toks) \
                and bin_name not in ("ls", "cat", "head", "tail", "grep", "find", "diff", "stat"):
            return "curator-policy: .archive/ is managed by skill_manage action=delete"
        if bin_name in _CURATOR_DELETE_BINS:
            return _CURATOR_USE_TOOL_HINT + f" (delete via {bin_name})"
        if bin_name in _CURATOR_INPLACE_BINS and any(
            t == "-i" or t.startswith("-i") for t in seg[1:]
        ):
            return _CURATOR_USE_TOOL_HINT + f" (in-place {bin_name} -i)"
        if bin_name in _CURATOR_MOVE_BINS:
            # Destination = last non-flag path-ish token of the segment.
            dest_tok = None
            for tok in reversed(seg[1:]):
                if not tok.startswith("-"):
                    dest_tok = tok
                    break
            dest_rel = _curator_skills_rel(dest_tok) if dest_tok else None
            if dest_rel is not None and _curator_rel_allowed(dest_rel):
                continue  # sanctioned move into a support subdir
            if bin_name in ("mv", "rename"):
                return _CURATOR_USE_TOOL_HINT + " (mv outside support dirs)"
            if dest_rel is not None:
                return _CURATOR_USE_TOOL_HINT + f" ({bin_name} into skills tree)"
    return None


def _curator_file_reason(paths: list[str], kind: str = "") -> str | None:
    """Judge engine file writes/edits by destination path, curator rules."""
    if kind in _READONLY_TOOL_KINDS:
        return None
    for p in paths:
        rel = _curator_skills_rel(p)
        if rel is None:
            continue
        if not _curator_rel_allowed(rel):
            return _CURATOR_USE_TOOL_HINT + f" (file {kind or 'write'})"
    return None


def _curator_permission_reason(params: dict[str, Any] | None) -> str | None:
    kind, raw = _tool_call_meta(params)
    command = _extract_shell_command(raw)
    if command is not None:
        return _curator_shell_reason(command)
    return _curator_file_reason(_extract_file_paths(raw), kind)


def _acp_audit(decision: str, params: dict[str, Any] | None, reason: str = "") -> None:
    """Append a one-line audit record of a permission decision. Never raises."""
    try:
        tc = (params or {}).get("toolCall") or {}
        title = ""
        if isinstance(tc, dict):
            title = str(tc.get("title") or tc.get("kind") or "")
        if not title:
            title = _permission_command_text(params).replace("\n", " ")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{ts}\t{decision}\t{title[:200]}"
        if reason:
            line += f"\t{reason}"
        try:
            from hermes_constants import get_hermes_home

            log_dir = get_hermes_home() / "logs"
        except Exception:
            log_dir = Path(os.path.expanduser("~/.hermes/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / _ACP_AUDIT_LOG_NAME, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _permission_allow(
    message_id: Any,
    params: dict[str, Any] | None = None,
    extra_reason_fn: Any = None,
) -> dict[str, Any]:
    """Auto-approve an ACP session/request_permission from the Claude engine.

    The Claude Code engine drives its own native tools over ACP and asks the
    client (us) to approve each one. Hermes is the trusted operator of this
    bridge, so we pick an advertised "allow" option instead of cancelling —
    the old hardcoded cancel aborted every native tool call. If the engine
    offers no allow-kind option, we fall back to denying.

    Guard: a small denylist of catastrophic / credential-exfil actions is
    hard-denied even when an allow option is offered, and every decision is
    written to the ACP permission audit log.

    We deliberately pick "allow_once" over "allow_always": the engine
    remembers an "always" answer for the rest of the session and stops
    asking, which would take this guard (and any extra_reason_fn) out of the
    chain for every later call of that tool.
    """
    reason, grant = _danger_verdict(params)
    if not reason and extra_reason_fn is not None:
        # Session-scoped policy layer (e.g. curator-mode skills-tree rules).
        try:
            reason = extra_reason_fn(params)
        except Exception:
            reason = None
    if reason:
        _acp_audit("DENY", params, reason)
        return _permission_denied(message_id, params, reason)

    options = (params or {}).get("options") or []
    chosen: Any = None
    escalated = False
    for wanted in ("allow_once", "allow_always"):
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") == wanted:
                chosen = opt.get("optionId") or opt.get("option_id")
                if chosen:
                    break
        if chosen:
            escalated = wanted == "allow_always"
            break
    if chosen is None:
        _acp_audit("DENY", params, "no allow-kind option offered")
        return _permission_denied(
            message_id, params, "no allow-kind option offered"
        )

    # Only reachable when the engine offered no per-call option; log it so a
    # guard that fell out of the chain stays visible instead of going quiet.
    note = "session-scoped allow_always (no allow_once offered)" if escalated else ""
    if grant is not None:
        # A write that the base guard would have refused, allowed because the
        # operator opened a door. This MUST be distinguishable in the audit log
        # from a routine approval — "the operator asked for it" and "the model
        # routed around the guard" looking identical is the whole failure this
        # mechanism exists to end.
        grant_note = (
            f"operator grant {grant.get('id')} for {grant.get('path')} "
            f"(reason: {grant.get('reason') or '—'}; issuer: {grant.get('issuer')})"
        )
        _acp_audit("ALLOW-GRANT", params, f"{grant_note}; {note}" if note else grant_note)
    else:
        _acp_audit("ALLOW", params, note)
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "selected",
                "optionId": chosen,
            }
        },
    }

