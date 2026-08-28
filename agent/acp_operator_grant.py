"""Operator grants — the sanctioned escape hatch from the credential guard.

Local overlay module (not upstream). ``agent/acp_permission_guard.py`` hard-denies
every engine write that targets a credential file (``.env``, ``auth.json``, ssh
keys...). That rule is right almost always and wrong exactly when the operator
has just handed over a secret in chat and asked for it to be installed — the
2026-07-30 founder-bot token being the case that forced this module. Until now
the only way through was for the agent to route around its own guard
(``sed -i`` / ``python3`` were not covered), which is the worst possible
outcome: the control is bypassed *and* the bypass is invisible.

So: one narrow, explicit, expiring, audited door.

    operator issues  ->  /env-grant <abs-path> [minutes] [reason]   (slash command)
    guard consults   ->  operator_grant_for(path)                   (this module)
    audit records    ->  ALLOW-GRANT <id> in acp-permission-audit.log

Properties that matter:

* **Path-scoped.** A grant covers one absolute path, nothing else.
* **Time-boxed.** Default 15 minutes, hard max 120; expiry is checked on read,
  and expired entries are pruned on the next write.
* **Operator-issued.** The only sanctioned issuer is the ``/env-grant`` slash
  command, which is dispatched from an *inbound operator message* in the
  gateway — a channel the agent cannot emit into. See
  ``~/.hermes/plugins/env-grant/``.
* **Not self-grantable.** The grant store itself is a protected control file:
  the guard refuses engine writes to it and no grant can ever cover it.

**This is not a security boundary** — same caveat as
``agent/file_safety.get_read_block_error``. The engine runs as the same OS user
with a shell, so a determined model can still write any file. What this buys is
that the *sanctioned* path is explicit and logged, so "the operator asked for
it" and "the model routed around its guard" stop looking identical in the audit
log.

Store: ``<hermes-root>/state/acp-operator-grants.json`` — one file for the whole
fleet (grants are path-scoped, and a profile boundary would only add a way for a
grant to silently not apply).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

GRANT_STORE_NAME = "acp-operator-grants.json"
DEFAULT_TTL_MINUTES = 15
MAX_TTL_MINUTES = 120


def _hermes_root() -> Path:
    """The FLEET root (``~/.hermes``), not the active profile's home.

    Under a profile, ``get_hermes_home()`` returns ``<root>/profiles/<name>``;
    grants live above that so one grant works no matter which profile's engine
    asks. Same widening shape as the read/write denylists.
    """
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
        if home.parent.name == "profiles":
            return home.parent.parent
        return home
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def grant_store_path() -> Path:
    return _hermes_root() / "state" / GRANT_STORE_NAME


def normalize_target(path: str) -> str:
    """Canonical form used for BOTH sides of a grant match.

    Grants are issued by a human typing a path (``~/.hermes/.../.env``) and
    consulted against whatever the engine put in its tool call
    (``/home/hun/.hermes/.../.env``), so both go through one function or the
    match silently never fires.
    """
    if not path:
        return ""
    p = str(path).strip().strip("\"'")
    if not p:
        return ""
    p = os.path.expandvars(os.path.expanduser(p))
    if not os.path.isabs(p):
        # A bare/relative token can't be matched to a grant: we don't know the
        # engine's cwd here, and guessing would make a grant cover a file the
        # operator never named.
        return ""
    return os.path.normpath(os.path.realpath(p))


def is_grant_store(path: str) -> bool:
    """True if *path* is the grant store — the one file no grant may cover."""
    target = normalize_target(path)
    if not target:
        return False
    return target == normalize_target(str(grant_store_path()))


def _load() -> dict[str, Any]:
    try:
        with open(grant_store_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("grants"), list):
            return data
    except Exception:
        pass
    return {"version": 1, "grants": []}


def _save(data: dict[str, Any]) -> None:
    store = grant_store_path()
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, store)


def _live(grants: list[dict[str, Any]], now: float | None = None) -> list[dict[str, Any]]:
    ts = time.time() if now is None else now
    out = []
    for g in grants:
        if not isinstance(g, dict):
            continue
        try:
            if float(g.get("expires_at") or 0) > ts:
                out.append(g)
        except Exception:
            continue
    return out


def issue_grant(
    path: str,
    minutes: float | None = None,
    reason: str = "",
    issuer: str = "",
) -> dict[str, Any]:
    """Record an operator grant for *path*. Raises ValueError on bad input."""
    target = normalize_target(path)
    if not target:
        raise ValueError(
            "grant needs an ABSOLUTE path (a relative one can't be matched "
            "against the engine's tool call)"
        )
    if is_grant_store(target):
        raise ValueError("the grant store itself can never be granted")
    try:
        mins = float(minutes) if minutes else DEFAULT_TTL_MINUTES
    except (TypeError, ValueError):
        raise ValueError(f"minutes must be a number, got {minutes!r}")
    if mins <= 0:
        raise ValueError("minutes must be positive")
    mins = min(mins, MAX_TTL_MINUTES)

    now = time.time()
    grant = {
        "id": "g-" + uuid.uuid4().hex[:8],
        "path": target,
        "issued_at": now,
        "expires_at": now + mins * 60.0,
        "minutes": mins,
        "reason": (reason or "").strip()[:200],
        "issuer": (issuer or "slash-command").strip()[:120],
    }
    data = _load()
    # Prune expired AND any live grant for the same path — re-issuing replaces,
    # so /env-grant <path> 1 genuinely shortens an earlier long grant instead of
    # leaving it in force underneath.
    data["grants"] = [
        g for g in _live(data.get("grants") or [], now)
        if normalize_target(str(g.get("path") or "")) != target
    ]
    data["grants"].append(grant)
    _save(data)
    return grant


def operator_grant_for(path: str) -> dict[str, Any] | None:
    """Return the live grant covering *path*, else None. Never raises."""
    try:
        target = normalize_target(path)
        if not target or is_grant_store(target):
            return None
        for g in _live(_load().get("grants") or []):
            if normalize_target(str(g.get("path") or "")) == target:
                return g
    except Exception:
        pass
    return None


def list_grants() -> list[dict[str, Any]]:
    return _live(_load().get("grants") or [])


def revoke(path: str | None = None) -> int:
    """Revoke the grant for *path*, or every live grant when *path* is None."""
    data = _load()
    live = _live(data.get("grants") or [])
    if path is None:
        removed = len(live)
        data["grants"] = []
    else:
        target = normalize_target(path)
        keep = [g for g in live if normalize_target(str(g.get("path") or "")) != target]
        removed = len(live) - len(keep)
        data["grants"] = keep
    _save(data)
    return removed


def describe(grant: dict[str, Any]) -> str:
    left = max(0.0, float(grant.get("expires_at") or 0) - time.time())
    mins = int(left // 60)
    secs = int(left % 60)
    reason = grant.get("reason") or "—"
    return (
        f"{grant.get('id')}  {grant.get('path')}\n"
        f"    осталось {mins}м{secs:02d}с · причина: {reason} · "
        f"выдал: {grant.get('issuer')}"
    )
