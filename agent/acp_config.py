"""Single source of truth for which model the ACP bridge runs: ``config.yaml``.

Local overlay module (not upstream).

The value that decides *which engine answers* used to live in the profile's
``.env`` as ``HERMES_ACP_MODEL``, while ``model.default`` in ``config.yaml`` was
demoted to a "label" that decided nothing. Two names for one setting, kept equal
by discipline. That arrangement is what this module now exists to undo, and the
reasons are all measured:

* **Everything else in Hermes already reads ``config.yaml``.** ``/model`` prints
  ``model.default`` as "Current model", the picker highlights it, ``/status``
  falls back to it, ``hermes config`` edits it. Only the engine read ``.env``. So
  every UI in the product could agree with itself and still disagree with the
  process that answers — the 2026-08-09 researcher case (sonnet answered while
  the profile said opus) and the 2026-08-10 secretary case both.
* **A ``/model`` press writes ``config.yaml`` for free.** The gateway persists
  ``model.default`` + ``model.provider`` on any ``--global`` switch. Making that
  write authoritative removes the second write instead of adding a third.
* **Two sources cannot be guarded, only reconciled.** The previous guard
  compared them and was blind to the case that actually happened: one of the two
  simply not existing. One source has nothing to compare.

So:

1. **One source.** ``model.default`` in the profile's ``config.yaml``
   (``$HERMES_HOME/config.yaml``). :func:`resolve_model` reads it; nothing here
   writes ``.env`` any more.
2. **``HERMES_ACP_MODEL`` keeps exactly one job**: the spawn-time channel from
   Python to ``claude-acp-run.sh``. ``_acp_env_with_model()`` writes the
   *resolved* value into the child's environment on every spawn, so the shell
   never guesses. It is an output of the resolution, never an input to it.
3. **A leftover ``HERMES_ACP_MODEL`` in ``.env`` is now a defect**, not a
   config. :func:`divergences` reports it — fatal when it names a different
   model than ``config.yaml`` (a real second source, the failure being undone),
   a warning when it merely duplicates it (dead line, delete it).

What still legitimately lives in ``.env``: ``HERMES_ACP_NATIVE_TOOLS``. It is a
gate, not a model, and it has no ``config.yaml`` counterpart — the
``unsourced``/``conflict`` checks below still cover it.

**One trap to know about this module's signatures.** Two families of readers
both take a keyword ``path``, and they mean different files:

* ``config_model`` / ``config_provider`` / ``fallback_model`` / ``is_acp_config``
  / ``explicit_choice`` / ``set_model`` → a **config.yaml** path;
* ``read_dotenv`` / ``legacy_pin`` / ``divergences`` / ``enforce`` / ``snapshot``
  → a **.env** path.

Crossing them is silent, not loud: handing ``.env`` to a config reader makes
``yaml.safe_load`` fail on the credential file, which returns "cannot tell",
which every caller here treats as "leave the source in charge". That is a
downgrade, never an error — it cost one real ``fatal`` on 2026-08-10.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Written into the *child* environment by ``_acp_env_with_model`` so the shell
#: wrapper never resolves anything itself. Read here only to detect a leftover.
MODEL_VAR = "HERMES_ACP_MODEL"
#: Engine-level failover target, deprecated alongside ``MODEL_VAR``. Its
#: replacement is :data:`FALLBACK_KEY`, not "nothing": a profile that wants a
#: real downgrade on a failed attempt must still be able to say so, and moving
#: the primary to ``config.yaml`` while leaving this one in ``.env`` would have
#: made the gateway refuse to start on a variable the failover path still reads.
FALLBACK_VAR = "HERMES_ACP_FALLBACK_MODEL"
#: ``model.acp_fallback`` in ``config.yaml`` — where that knob lives now. Unset
#: means "no distinct fallback": ``_acp_failover_model`` then answers with the
#: primary, which ``run_prompt_with_failover`` reads as "do not retry
#: elsewhere". Failing loudly on the chosen model beats quietly answering on
#: another one (founder, 2026-08-02: a hardcoded default moved 82k of output
#: onto opus twice, and only the token audit noticed).
FALLBACK_KEY = "acp_fallback"
#: Top-level ``fallback_model:`` block — Hermes' provider-level failover, a
#: different mechanism from :data:`FALLBACK_KEY` and deliberately named apart
#: from it: one picks another *engine model* for the bridge, the other picks
#: another *backend* for Hermes.
ROUTE_FALLBACK_KEY = "fallback_model"
NATIVE_VAR = "HERMES_ACP_NATIVE_TOOLS"
#: Env names that used to configure the model and must no longer appear in
#: ``.env``. Kept as a list rather than folded into the checks so
#: ``bridge-doctor`` and the tests name the same set.
LEGACY_MODEL_VARS = (MODEL_VAR, FALLBACK_VAR)
#: Escape hatch for a deliberate one-off ``HERMES_ACP_MODEL=… hermes -p x -z``:
#: downgrades every divergence to a warning instead of a hard stop.
OVERRIDE_VAR = "HERMES_ACP_ALLOW_ENV_OVERRIDE"

#: Last-resort model, kept only so behaviour is defined when nothing is
#: configured anywhere. Must stay equal to ``claude-acp-run.sh``'s own default.
LEGACY_DEFAULT_MODEL = "claude-opus-4-8"

#: Provider slugs whose engine is the local ACP bridge.
ACP_PROVIDERS = frozenset({"copilot-acp"})

#: Engine models the bridge is known to serve, most capable first. Two readers
#: and one list on purpose: the ``/model`` picker builds its bridge row from it,
#: and ``bridge-doctor``'s "nobody names a model that does not exist" check
#: judges the config against it — those two disagreeing is how a typo becomes a
#: button. Not a gate: ``_ENGINE_MODEL_RE`` decides what may be *run*, so a
#: config outside this tuple still works; being absent here only costs it a
#: button and earns a warning.
KNOWN_ENGINE_MODELS = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
)

_TRUTHY = ("1", "true", "yes", "on")

# Engine model names the bridge may be pointed at. Deliberately narrow: the aux
# path hands us provider labels ("copilot-acp") as if they were models, and
# spawning the engine with one of those is worse than falling back.
_ENGINE_MODEL_RE = re.compile(r"^claude-[a-z0-9][a-z0-9._-]*$", re.IGNORECASE)

# (path, mtime_ns, size) -> {var: value}; the gateway is long-lived and `.env`
# is hot-reloadable, so the stat is the cache key rather than the value.
_DOTENV_CACHE: dict[tuple[str, int, int], dict[str, str]] = {}

# (path, mtime_ns, size) -> {"model": str, "provider": str}; same stat-keyed
# shape and for a sharper reason: config.yaml is now the source, a long-lived
# gateway must pick up a `/model` write on the very next turn, and a value cache
# would serve the pre-switch model until restart.
_CONFIG_CACHE: dict[tuple[str, int, int], dict[str, str]] = {}

# Keys already warned about in this process, so a per-turn resolve doesn't spam.
_WARNED: set[str] = set()


class ACPConfigError(RuntimeError):
    """Raised when something other than ``config.yaml`` decides the model."""


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def dotenv_path() -> Path:
    return hermes_home() / ".env"


def config_path() -> Path:
    return hermes_home() / "config.yaml"


def read_dotenv(var: str, *, path: Path | None = None) -> str | None:
    """Value of ``var`` as written in ``.env``, independent of ``os.environ``.

    Reading the file rather than the process env is the whole point: it is what
    lets us tell "the file still carries a dead pin" apart from "something
    upstream of us exported one".
    """
    env_path = path or dotenv_path()
    try:
        stat = env_path.stat()
    except OSError:
        return None
    key = (str(env_path), stat.st_mtime_ns, stat.st_size)
    parsed = _DOTENV_CACHE.get(key)
    if parsed is None:
        parsed = {}
        try:
            text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            name, _, value = line.partition("=")
            name = name.strip()
            if not name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            parsed[name] = value
        _DOTENV_CACHE.clear()
        _DOTENV_CACHE[key] = parsed
    return parsed.get(var)


def normalize_model(value: Any) -> str | None:
    """A usable engine model name, or ``None`` if this isn't one.

    Accepts a provider-qualified reference (``anthropic/claude-opus-5``) and
    returns the bare engine name the ACP wrapper wants.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text:
        text = text.rsplit("/", 1)[-1].strip()
    return text if _ENGINE_MODEL_RE.match(text) else None


def _empty_config() -> dict[str, str]:
    """The "cannot tell" answer, fresh each time so no caller can poison it."""
    return {
        "model": "",
        "provider": "",
        "fallback": "",
        "route_fallback_provider": "",
        "route_fallback_model": "",
    }


def _read_config(path: Path | None = None) -> dict[str, str]:
    """The profile's model facts as *written* in ``config.yaml``.

    Keys: ``model``/``provider``/``fallback`` come from the ``model:`` section
    (the last one is ``model.acp_fallback``, the bridge's own downgrade);
    ``route_fallback_*`` come from the TOP-LEVEL ``fallback_model:`` block,
    which is a different mechanism — Hermes' provider-level failover, not the
    engine's. Both are read here rather than in two readers so the picker and
    ``resolve_model`` cannot disagree about one file.

    Empty strings for anything missing or unreadable — every one of those means
    "cannot tell", and no caller here may turn "cannot tell" into a switch.
    """
    cfg_path = path or config_path()
    try:
        stat = cfg_path.stat()
    except OSError:
        return _empty_config()
    key = (str(cfg_path), stat.st_mtime_ns, stat.st_size)
    cached = _CONFIG_CACHE.get(key)
    if cached is None:
        model = provider = fallback = ""
        route_provider = route_model = ""
        try:
            import yaml  # lazy: bare-python callers of this module have no deps

            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            model_cfg = raw.get("model")
            if isinstance(model_cfg, str):
                model = normalize_model(model_cfg) or ""
            elif isinstance(model_cfg, dict):
                model = normalize_model(
                    model_cfg.get("default") or model_cfg.get("model")
                ) or ""
                provider = str(model_cfg.get("provider") or "").strip()
                fallback = normalize_model(model_cfg.get(FALLBACK_KEY)) or ""
            route_cfg = raw.get(ROUTE_FALLBACK_KEY)
            if isinstance(route_cfg, dict):
                route_provider = str(route_cfg.get("provider") or "").strip()
                route_model = normalize_model(route_cfg.get("model")) or ""
        except Exception:
            return _empty_config()
        cached = {
            "model": model,
            "provider": provider,
            "fallback": fallback,
            "route_fallback_provider": route_provider,
            "route_fallback_model": route_model,
        }
        _CONFIG_CACHE.clear()
        _CONFIG_CACHE[key] = cached
    return cached


def config_model(*, path: Path | None = None) -> str | None:
    """``model.default`` from the profile's ``config.yaml`` — the source.

    ``None`` when the file is missing, unparseable, or names something that is
    not an engine model.
    """
    return _read_config(path)["model"] or None


def config_provider(*, path: Path | None = None) -> str:
    """``model.provider`` from the profile's ``config.yaml``, or ``""``."""
    return _read_config(path)["provider"]


def fallback_model(*, path: Path | None = None) -> str:
    """``model.acp_fallback`` — the engine-level failover target, or ``""``.

    ``""`` means "no distinct fallback"; see :data:`FALLBACK_KEY`. The
    deprecated ``HERMES_ACP_FALLBACK_MODEL`` is still honoured when config says
    nothing, so a profile that never migrated keeps its chosen downgrade instead
    of silently losing it — but it earns a warning and a ``divergences`` entry.
    """
    from_config = _read_config(path)["fallback"]
    if from_config:
        return from_config
    # ``path`` is a *config* path here (the convention of every reader named
    # config_*), so it must not be forwarded to the dotenv reader — that mix-up
    # is what made divergences() yaml-parse the credential file and downgrade a
    # real conflict to a warning.
    stale = str(os.getenv(FALLBACK_VAR) or "").strip() or (
        read_dotenv(FALLBACK_VAR) or ""
    ).strip()
    if stale:
        _warn_once(
            "acp-fallback-legacy",
            "%s=%r is deprecated; move it to model.%s in %s.",
            FALLBACK_VAR, stale, FALLBACK_KEY, config_path(),
        )
    return stale


def route_fallback_provider(*, path: Path | None = None) -> str:
    """``fallback_model.provider`` from ``config.yaml``, or ``""``.

    Not :func:`fallback_model`: that one is the engine's own downgrade inside
    the bridge, this one is the backend Hermes falls over to when the main
    provider fails. A profile can have the bridge in exactly one of the two.
    """
    return _read_config(path)["route_fallback_provider"]


def route_fallback_model(*, path: Path | None = None) -> str | None:
    """``fallback_model.model``, when it names an engine model — else ``None``."""
    return _read_config(path)["route_fallback_model"] or None


def is_acp_config(*, path: Path | None = None) -> bool:
    """Whether this profile's engine is the bridge, per ``config.yaml``."""
    return config_provider(path=path).lower() in ACP_PROVIDERS


#: Back-compat alias. The old name called this "the label" because it decided
#: nothing; it decides everything now, so the name is wrong — kept only so an
#: unported caller keeps working.
config_label = config_model


def explicit_choice(requested: Any, *, path: Path | None = None) -> str | None:
    """The requested model, when it is not simply the profile's own config.

    Still worth naming after the inversion, but only for *reporting*: a request
    that differs from ``config.yaml`` came from a session ``/model``, a ``-m``
    flag, a cron job's ``model=`` or a subagent's ``model=``. :func:`resolve_model`
    no longer needs the discriminator (a request equal to the config resolves to
    the same string either way) — ``bridge-doctor`` and the logs do.
    """
    wanted = normalize_model(requested)
    if not wanted:
        return None
    current = config_model(path=path)
    if not current:
        return None
    if current.casefold() == wanted.casefold():
        return None
    return wanted


def legacy_pin(*, path: Path | None = None) -> str:
    """A ``HERMES_ACP_MODEL`` that still exists somewhere it shouldn't.

    File first, then the process env: a value in ``.env`` is a line somebody has
    to delete, a value only in the environment is an export or a systemd
    ``Environment=``. Both are reported; neither is obeyed.
    """
    from_file = (read_dotenv(MODEL_VAR, path=path) or "").strip()
    if from_file:
        return from_file
    return str(os.getenv(MODEL_VAR) or "").strip()


def resolve_model(requested: Any = None) -> str:
    """The model the ACP bridge will actually run.

    Order: the model explicitly requested for this run → ``model.default`` in
    the profile's ``config.yaml`` → the legacy default. Whatever comes out is
    written into the wrapper's environment by ``_acp_env_with_model``, so Python
    and the shell still cannot disagree.

    Note what is *not* in that chain: ``HERMES_ACP_MODEL``. A request that is
    simply the profile's own default resolves to the same string either way, so
    the common turn is unchanged; a request that differs is a deliberate
    per-run choice (session ``/model``, ``-m``, cron ``model=``) and wins, which
    is what it did before too. What changed is that a global switch now takes
    effect by writing one file that the whole product already reads.
    """
    wanted = normalize_model(requested)
    if wanted:
        chosen = explicit_choice(wanted)
        if chosen:
            _warn_once(
                f"acp-model-explicit:{chosen}",
                "%r was explicitly chosen for this run (session /model, -m, cron "
                "model= or a subagent's model=) over %s's %r — answering on %r.",
                chosen, config_path(), config_model() or "(unset)", chosen,
            )
        if wanted.casefold() not in {m.casefold() for m in KNOWN_ENGINE_MODELS}:
            _warn_once(
                f"acp-model-unknown:{wanted}",
                "%r is not a model the bridge is known to serve %s — running it "
                "anyway, but check the name if the engine refuses.",
                wanted, list(KNOWN_ENGINE_MODELS),
            )
        return wanted
    current = config_model()
    if current:
        return current
    stale = legacy_pin()
    if stale:
        _warn_once(
            "acp-model-legacy-pin",
            "model.default in %s names no engine model; falling back to the "
            "deprecated %s=%r. Move it into config.yaml — .env is no longer read "
            "for the model.",
            config_path(), MODEL_VAR, stale,
        )
        return stale
    _warn_once(
        "acp-model-default",
        "model.default in %s names no engine model and nothing was requested; "
        "using the legacy default %r.",
        config_path(), LEGACY_DEFAULT_MODEL,
    )
    return LEGACY_DEFAULT_MODEL


def native_tools_enabled() -> bool:
    """Whether the engine drives Hermes tools natively over MCP."""
    return _truthy(os.getenv(NATIVE_VAR, "0"))


def set_model(model: Any, *, path: Path | None = None) -> str:
    """Write ``model.default`` into ``config.yaml``; return what it reads back.

    The gateway's own ``/model --global`` path already persists this key through
    ``hermes_cli.config.save_config`` and does not call here — this exists for
    the shell-side callers (``acp-set-model.py``, fleet scripts) that used to
    rewrite ``.env`` and must not become the second writer they just replaced.

    Line-based on purpose. ``config.yaml`` is hand-maintained and full of
    comments; a yaml round-trip through ``safe_load``/``safe_dump`` would drop
    every one of them. Same atomicity contract as before: temp file in the same
    directory, mode preserved, ``os.replace`` — a crash leaves the old file, not
    a truncated one.
    """
    wanted = normalize_model(model)
    if not wanted:
        raise ValueError(f"not an engine model name: {model!r}")
    cfg_path = path or config_path()
    original = cfg_path.read_text(encoding="utf-8")
    mode = cfg_path.stat().st_mode & 0o777
    lines = original.splitlines(keepends=True)

    def _indent(text: str) -> int:
        return len(text) - len(text.lstrip(" "))

    start = None
    for i, line in enumerate(lines):
        if line.rstrip("\n").rstrip() == "model:" and _indent(line) == 0:
            start = i
            break

    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["model:\n", f"  default: {wanted}\n"]
    else:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            line = lines[j]
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if _indent(line) == 0:
                end = j
                break
        for j in range(start + 1, end):
            stripped = lines[j].strip()
            if stripped.startswith("default:"):
                lines[j] = " " * _indent(lines[j]) + f"default: {wanted}\n"
                break
        else:
            lines.insert(start + 1, f"  default: {wanted}\n")

    import tempfile

    fd, tmp = tempfile.mkstemp(dir=str(cfg_path.parent), prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("".join(lines))
        os.chmod(tmp, mode)
        os.replace(tmp, cfg_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    return config_model(path=cfg_path) or ""


def _warn_once(key: str, fmt: str, *args: Any) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    log.warning(fmt, *args)


def divergences(*, path: Path | None = None) -> list[dict[str, str]]:
    """Everything that could decide the model besides ``config.yaml``.

    ``legacy_pin``
        ``HERMES_ACP_MODEL`` still set, in ``.env`` or in the process env.
        ``fatal`` when it names a *different* model than ``config.yaml`` — that
        is a genuine second source and exactly the confusion being removed;
        ``warn`` when it merely duplicates it, because a dead line that agrees
        today is the one that silently disagrees after the next ``/model``.

    ``unsourced`` / ``conflict``
        Unchanged, and still only about ``HERMES_ACP_NATIVE_TOOLS``: a gate with
        no ``config.yaml`` counterpart, whose single source is still ``.env``.
        ``unsourced`` (the env decides, the file is silent) is the one that
        actually happens, since ``load_hermes_dotenv()`` overrides.
    """
    out: list[dict[str, str]] = []

    for var in LEGACY_MODEL_VARS:
        file_value = (read_dotenv(var, path=path) or "").strip()
        env_value = str(os.getenv(var) or "").strip()
        stale = file_value or env_value
        if not stale:
            continue
        where = str(path or dotenv_path()) if file_value else "the process environment"
        if var == MODEL_VAR:
            current, key = config_model(), "model.default"
        else:
            # A leftover FALLBACK_VAR is still *obeyed* when config is silent
            # (fallback_model), so it is only a second source once config names
            # its own. Marking it fatal regardless would refuse to start a
            # gateway over a variable the failover path was still reading.
            # NB: no ``path`` here. It is the *dotenv* path — handing it to a
            # config reader silently yaml-parses the credential file, which
            # fails, which reads as "config names no fallback", which downgrades
            # every real conflict to a warning.
            current, key = (_read_config()["fallback"] or None), f"model.{FALLBACK_KEY}"
        differs = bool(current) and normalize_model(stale) != current
        out.append({
            "var": var,
            "kind": "legacy_pin",
            "severity": "fatal" if differs else "warn",
            "env": env_value,
            "dotenv": file_value,
            "detail": (
                f"{var}={stale!r} in {where}. This setting now lives in {key} "
                f"in {config_path()} ({current or 'unset'})"
                + (
                    " — these name different models, so something other than "
                    "config.yaml is trying to decide. Delete the line."
                    if differs
                    else ". Move it there and delete the line."
                )
            ),
        })

    env_value = str(os.getenv(NATIVE_VAR) or "").strip()
    file_value = (read_dotenv(NATIVE_VAR, path=path) or "").strip()
    if env_value and file_value and env_value != file_value:
        out.append({
            "var": NATIVE_VAR,
            "kind": "conflict",
            "severity": "fatal",
            "env": env_value,
            "dotenv": file_value,
            "detail": (
                f"{NATIVE_VAR}={env_value!r} in the process environment but "
                f"{file_value!r} in {path or dotenv_path()}."
            ),
        })
    elif env_value and not file_value:
        out.append({
            "var": NATIVE_VAR,
            "kind": "unsourced",
            "severity": "fatal",
            "env": env_value,
            "dotenv": "",
            "detail": (
                f"{NATIVE_VAR}={env_value!r} was decided outside "
                f"{path or dotenv_path()}, which says nothing. This process uses "
                f"{env_value!r}; CLI runs of the same profile (ask-hermes, "
                f"hermes -p … -z) resolve their own answer."
            ),
        })
    return out


def enforce(context: str = "cli", *, path: Path | None = None) -> list[dict[str, str]]:
    """Fail fast on a second source of truth; return what was found.

    ``context="gateway"`` raises :class:`ACPConfigError` on a fatal divergence —
    a long-lived service that answers on a model nobody chose is worse than one
    that refuses to start and says why. Every other context logs instead, so a
    one-off CLI override stays possible; ``HERMES_ACP_ALLOW_ENV_OVERRIDE=1``
    downgrades the gateway too.
    """
    found = divergences(path=path)
    fatal = [d for d in found if d["severity"] == "fatal"]
    for d in found:
        if d["severity"] != "fatal":
            _warn_once(f"acp-config-warn:{d['var']}", "ACP config: %s", d["detail"])
    if not fatal:
        return found
    allowed = _truthy(os.getenv(OVERRIDE_VAR))
    message = (
        "ACP bridge model has more than one source of truth:\n  "
        + "\n  ".join(d["detail"] for d in fatal)
        + f"\nSingle source is model.default in {config_path()}. Remove the "
        f"duplicate (a .env line, a systemd drop-in Environment=, a shell "
        f"export), or set {OVERRIDE_VAR}=1 for a deliberate one-off override."
    )
    if context == "gateway" and not allowed:
        log.critical("%s", message)
        raise ACPConfigError(message)
    log.error("%s", message)
    return found


def snapshot(*, path: Path | None = None) -> dict[str, Any]:
    """Machine-readable state for ``bridge-doctor`` and tests."""
    return {
        "hermes_home": str(hermes_home()),
        "config": str(config_path()),
        "dotenv": str(path or dotenv_path()),
        "model": {
            "config": config_model() or "",
            "provider": config_provider(),
            "legacy_pin": legacy_pin(path=path),
            "resolved": resolve_model(),
        },
        "native_tools": {
            "env": str(os.getenv(NATIVE_VAR) or "").strip(),
            "dotenv": (read_dotenv(NATIVE_VAR, path=path) or "").strip(),
            "resolved": native_tools_enabled(),
        },
        "divergences": divergences(path=path),
    }
