"""Extra MCP servers handed to the engine on ``session/new``.

Local overlay module (not upstream). ``acp_native_tools._hermes_tools_mcp_servers``
builds the one server Hermes always ships — its own curated tool subset. This
module builds the *optional* ones an operator declares per profile, so a single
agent can reach a documentation/API MCP endpoint the others do not pay for.

Why a module and not ``hermes mcp add``: the bridge hardcodes the server list it
sends (``copilot_acp_client`` builds ``mcpServers`` itself), so nothing Hermes'
own MCP registry knows about ever reaches the engine. And why not a ``.mcp.json``
in the repo — the file is read at *session start* from the session's cwd, and a
Kanban worker starts in an empty scratch dir (``kanban/workspaces/<id>``), not in
the repository. It would cover the interactive topic and miss exactly the
sessions that write code.

Configuration lives in the profile's ``config.yaml`` — deliberately NOT in
``.env``, which ``acp_permission_guard`` holds read-only, so adding a server does
not need an operator ``/env-grant``::

    acp:
      mcp_servers:
        nuxt:
          type: http
          url: https://nuxt.com/mcp
          # headers: {Authorization: "Bearer …"}   # optional
          # enabled: false                          # keep the entry, skip it

A list of entries carrying their own ``name`` is accepted too. ``stdio`` entries
(``command``/``args``/``env``) work as well, but note they inherit the bridge's
scrubbed environment — anything they need must be listed in ``env``.

Two hard rules, both learned the expensive way:

* **Never raise.** This runs while constructing the client for a live turn. A
  typo in yaml must cost the servers it describes, not the turn.
* **Never shadow ``hermes-tools``.** The engine keys servers by name
  (``acp-agent.js``: ``mcpServers[server.name] = …``), so a duplicate name would
  silently replace Hermes' own tools with a stranger's.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Name the bridge reserves for Hermes' own tool server.
RESERVED_NAMES = frozenset({"hermes-tools"})

#: Kill switch. Set to a falsy value to send only ``hermes-tools`` for a run.
ENABLE_VAR = "HERMES_ACP_EXTRA_MCP"

_REMOTE_TYPES = frozenset({"http", "sse"})
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FALSY = {"0", "false", "no", "off", ""}

#: (path, mtime_ns, size) -> parsed servers. The config file is read once per
#: client construction, i.e. once per API call; re-parsing yaml each time is
#: cheap but pointless, and the mtime key means an edit is picked up by the next
#: session without a restart.
_CACHE: dict[tuple[str, int, int], list[dict[str, Any]]] = {}


def _enabled() -> bool:
    raw = os.getenv(ENABLE_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSY


def config_path() -> Path:
    """Profile config that declares the extra servers."""
    from agent.acp_config import hermes_home

    return hermes_home() / "config.yaml"


def _warn(fmt: str, *args: Any) -> None:
    log.warning("ACP extra MCP: " + fmt, *args)


def _name_value_pairs(raw: Any, name: str, field: str) -> list[dict[str, str]] | None:
    """ACP wants ``[{name, value}]``; yaml authors want a mapping. Accept both."""
    if raw in (None, "", {}, []):
        return None
    pairs: list[dict[str, str]] = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, dict) and "name" in entry:
                items.append((entry.get("name"), entry.get("value")))
            else:
                _warn("server %r: skipping unreadable %s entry %r", name, field, entry)
        if not items:
            return None
    else:
        _warn(
            "server %r: %r must be a mapping or list, got %s", name, field, type(raw).__name__
        )
        return None
    for key, value in items:
        key = str(key or "").strip()
        if not key:
            continue
        pairs.append({"name": key, "value": str(value if value is not None else "")})
    return pairs or None


def _normalize(name: Any, spec: Any) -> dict[str, Any] | None:
    """One declared server -> an ACP ``mcpServers`` entry, or ``None`` if unusable."""
    name = str(name or "").strip()
    if not _NAME_RE.match(name):
        _warn("skipping server with unusable name %r", name)
        return None
    if name in RESERVED_NAMES:
        _warn("skipping server %r: that name belongs to Hermes' own tool server", name)
        return None
    if not isinstance(spec, dict):
        _warn("skipping server %r: expected a mapping, got %s", name, type(spec).__name__)
        return None
    if not _truthy(spec.get("enabled", True)):
        return None

    kind = str(spec.get("type") or "").strip().lower()
    url = str(spec.get("url") or "").strip()
    command = str(spec.get("command") or "").strip()
    # Type is optional when it is implied: a url means remote, a command stdio.
    if not kind:
        kind = "http" if url else ("stdio" if command else "")

    if kind in _REMOTE_TYPES:
        if not url.lower().startswith(("http://", "https://")):
            _warn("skipping server %r: 'url' must be http(s), got %r", name, url)
            return None
        # ``headers`` is REQUIRED by the wire schema (zod.gen.js: zMcpServerHttp
        # / zMcpServerSse), and a failing entry is dropped from the array without
        # a word (``vecSkipError``) — session/new still succeeds and the server
        # simply never exists. Always send the key, empty when unused.
        entry: dict[str, Any] = {
            "name": name,
            "type": kind,
            "url": url,
            "headers": _name_value_pairs(spec.get("headers"), name, "headers") or [],
        }
        return entry

    if kind == "stdio":
        if not command:
            _warn("skipping server %r: stdio needs a 'command'", name)
            return None
        args = spec.get("args") or []
        if not isinstance(args, list):
            _warn("skipping server %r: 'args' must be a list", name)
            return None
        # No "type" key: the engine treats a typeless entry as stdio, and an
        # explicit type="stdio" falls through its `!("type" in server)` branch
        # and is dropped. Verified against acp-agent.js.
        # ``args`` and ``env`` are required by zMcpServerStdio for the same
        # reason as ``headers`` above: omit one and the entry is skipped in
        # silence.
        entry = {
            "name": name,
            "command": command,
            "args": [str(a) for a in args],
            "env": _name_value_pairs(spec.get("env"), name, "env") or [],
        }
        return entry

    _warn("skipping server %r: unknown type %r (want http, sse or stdio)", name, kind)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSY


def _declared(path: Path) -> list[Any] | dict[str, Any]:
    """The raw ``acp.mcp_servers`` block, or an empty list."""
    try:
        import yaml
    except Exception:  # pragma: no cover - yaml ships with Hermes
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return []
    except Exception as exc:  # unreadable file, yaml syntax error, bad encoding…
        # Deliberately broad: this runs while constructing the client for a live
        # turn, so a typo must cost the servers it describes, not the turn.
        _warn("cannot read %s: %s", path, exc)
        return []
    if not isinstance(raw, dict):
        return []
    acp = raw.get("acp")
    if not isinstance(acp, dict):
        return []
    declared = acp.get("mcp_servers")
    if declared in (None, "", {}, []):
        return []
    if not isinstance(declared, (dict, list)):
        _warn("acp.mcp_servers must be a mapping or list, got %s", type(declared).__name__)
        return []
    return declared


def extra_mcp_servers(*, path: Path | None = None) -> list[dict[str, Any]]:
    """Operator-declared MCP servers for this profile, ready for ``session/new``.

    Returns ``[]`` on anything unexpected: no config, no block, bad yaml, kill
    switch off. Callers append this to the Hermes tool server list.
    """
    if not _enabled():
        return []
    cfg = path or config_path()
    try:
        stat = cfg.stat()
    except OSError:
        return []
    key = (str(cfg), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return [dict(entry) for entry in cached]

    declared = _declared(cfg)
    items = (
        list(declared.items())
        if isinstance(declared, dict)
        else [(entry.get("name") if isinstance(entry, dict) else None, entry) for entry in declared]
    )
    servers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, spec in items:
        entry = _normalize(name, spec)
        if entry is None:
            continue
        if entry["name"] in seen:
            _warn("skipping duplicate server name %r", entry["name"])
            continue
        seen.add(entry["name"])
        servers.append(entry)

    _CACHE.clear()
    _CACHE[key] = servers
    return [dict(entry) for entry in servers]


def describe(*, path: Path | None = None) -> str:
    """One-line summary for diagnostics (bridge-doctor, probes)."""
    if not _enabled():
        return f"disabled ({ENABLE_VAR}=0)"
    servers = extra_mcp_servers(path=path)
    if not servers:
        return "none"
    return ", ".join(
        f"{s['name']}={s.get('url') or s.get('command', '?')}" for s in servers
    )
