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
_DENY_HINT = (
    "Hermes guard refused this action (not a transient failure — retrying "
    "with different syntax will fail the same way). Credential files "
    "(.env, .credentials.json, tokens, ssh keys) are read-only to the engine. "
    "For config/env changes use a systemd drop-in "
    "(~/.config/systemd/user/<unit>.service.d/*.conf) or ask the operator."
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

# Absolute/system/home targets that make a recursive-force delete catastrophic.
# Boundary classes include quotes/backslash because the command usually arrives
# JSON-serialized (e.g. rawInput={"command": "rm -rf /"}), so the target token is
# flanked by `"`.
_CATASTROPHIC_RM_TARGET_RE = re.compile(
    r"""(?:^|[\s"'])(?:/|~|\$home|\$\{home\}|/\*)(?:[\s"'`;&|\\]|$)"""
    r"""|(?:^|[\s"'])(?:/home|/etc|/usr|/var|/boot|/bin|/sbin|/lib|/opt|/root|/sys|/proc)"""
    r"""(?:/\*|/)?(?:[\s"'`;&|\\]|$)"""
)

# Credential files whose overwrite/exfiltration must never be auto-approved,
# plus the operators that indicate a write-out or ship-out of one.
_CREDENTIAL_BASENAMES: tuple[str, ...] = (
    ".credentials.json", "credentials.json", "auth.json", ".claude-acp-token",
    "id_rsa", "id_ed25519", "id_ecdsa", ".env",
)
_EXFIL_TOKENS_RE = re.compile(
    r">>?|\btee\b|\bcp\b|\bmv\b|\bscp\b|\bcurl\b|\bwget\b|\bnc\b|\brsync\b|\bcat\b"
)
# Shell utilities that write out / ship out a file's contents, matched as argv
# token basenames only (openclaw-style structural check, never as substrings
# of arbitrary flattened text).
_EXFIL_BIN_NAMES: frozenset[str] = frozenset(
    {"cat", "cp", "mv", "tee", "scp", "curl", "wget", "nc", "rsync"}
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


def _shell_credential_exfil(command: str, depth: int = 0) -> str | None:
    """Structural check: does this command pair a write-out/ship-out operation
    with a credential-file token?

    Tokens are examined individually (argv words, redirect targets, nested
    command strings), so a credential filename inside unrelated prose can no
    longer pair with an operator elsewhere in the same flattened blob.
    """
    tokens = _shell_tokens(command)
    if tokens is None:
        # Unparseable quoting: fail closed with the legacy substring pair.
        low = command.lower()
        if _EXFIL_TOKENS_RE.search(low):
            for cred in _CREDENTIAL_BASENAMES:
                if cred in low:
                    return cred
        return None
    has_op = False
    cred_hit: str | None = None
    prev_was_redirect = False
    for tok in tokens:
        if not tok:
            continue
        if depth < 2 and any(ws in tok for ws in (" ", "\t", "\n")):
            # A quoted multi-word token is likely a nested command
            # (bash -c '...', ssh host '...'); check the pair inside it only.
            nested = _shell_credential_exfil(tok, depth + 1)
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
        if word.rstrip("/").rsplit("/", 1)[-1].lower() in _EXFIL_BIN_NAMES:
            has_op = True
        if cred_hit is None:
            cred_hit = _credential_basename_match(word)
    return cred_hit if has_op else None


def _shell_danger_reason(command: str) -> str | None:
    """Danger checks that apply to a shell command string."""
    low = command.lower()
    for rx in _DANGEROUS_COMMAND_RES:
        if rx.search(low):
            return f"destructive command pattern /{rx.pattern[:48]}/"
    if _has_recursive_force_rm(low) and _CATASTROPHIC_RM_TARGET_RE.search(low):
        return "recursive-force rm targeting a root/system/home path"
    cred = _shell_credential_exfil(command)
    if cred:
        return f"shell write/exfil touching credential file '{cred}'"
    return None


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


def _permission_danger_reason(params: dict[str, Any] | None) -> str | None:
    """Return a reason if this tool call is catastrophic/exfil, else None.

    Structural, openclaw-style dispatch instead of one flat substring scan:
    shell commands are tokenized and checked for operator+credential pairs;
    file writes are judged by DESTINATION PATH only. File content is never
    scanned, so writing a document that merely mentions credential filenames
    or shell operators is no longer a false positive.
    """
    kind, raw = _tool_call_meta(params)
    command = _extract_shell_command(raw)
    if command is not None:
        return _shell_danger_reason(command)
    paths = _extract_file_paths(raw)
    if paths:
        if kind in _READONLY_TOOL_KINDS:
            return None
        for path in paths:
            cred = _credential_basename_match(path)
            if cred:
                return f"file {kind or 'write'} targeting credential file '{cred}'"
        return None
    # Unknown payload shape: legacy conservative scan over the flattened call,
    # minus any file-content fields.
    text = _permission_command_text(_strip_content_fields(params))
    if not text:
        return None
    low = text.lower()
    for rx in _DANGEROUS_COMMAND_RES:
        if rx.search(low):
            return f"destructive command pattern /{rx.pattern[:48]}/"
    if _has_recursive_force_rm(low) and _CATASTROPHIC_RM_TARGET_RE.search(low):
        return "recursive-force rm targeting a root/system/home path"
    if _EXFIL_TOKENS_RE.search(low):
        for base in _CREDENTIAL_BASENAMES:
            if base.lower() in low:
                return f"write/exfil touching credential file '{base}'"
    return None


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
    """
    reason = _permission_danger_reason(params)
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
    for wanted in ("allow_always", "allow_once"):
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") == wanted:
                chosen = opt.get("optionId") or opt.get("option_id")
                if chosen:
                    break
        if chosen:
            break
    if chosen is None:
        _acp_audit("DENY", params, "no allow-kind option offered")
        return _permission_denied(
            message_id, params, "no allow-kind option offered"
        )

    _acp_audit("ALLOW", params)
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

