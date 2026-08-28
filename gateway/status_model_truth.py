"""What ``/status`` may call "Model", and what it must call history instead.

Local overlay module (not upstream).

``/status`` used to resolve its model line as: live-or-cached agent → the
``model`` column of the ``sessions`` row → config. That middle step is the bug.
That column is written by ``_sync_session_model_from_agent`` *after every turn*
— it is a protocol record of **who answered last time**, and nothing reads it
back to decide anything. Printing it under the heading "Model" turns an audit
trail into a claim about the future.

Measured, 2026-08-10: ``/status`` on the secretary profile said
``claude-opus-5`` while ``/model`` said ``claude-sonnet-5``. Both were reading
honestly. The last turn in that thread ran on 2026-08-09 07:02; the profile
moved to sonnet at 08:37 the same day. No turn had passed since, so the column
still held the pre-switch model — a day and a half stale, presented as current.
Two commands disagreeing about the single most load-bearing fact in the system
is how an operator stops trusting either.

So the rule here: **the model line answers "what runs next", and it is computed
from the same two things ``/model`` reads** — the session override, then
``model.default`` in ``config.yaml``. The DB column keeps its value; it just
gets its own line, and only when it disagrees.

One case still outranks both: a turn *in flight*. Then the live agent is not a
prediction, it is the thing currently answering, and it wins.

The cached (idle) agent is deliberately NOT authoritative. It is a warm object
from a previous turn, and on an ACP profile it is not even the deciding one —
``acp_config.resolve_model`` re-reads ``config.yaml`` on every engine spawn, so
a config edit takes effect on the next turn while the cached agent still carries
the old name. Letting it speak would reintroduce the same lie with a shorter
staleness window. It is used only when nothing else can tell.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _same_model(left: Any, right: Any) -> bool:
    """Whether two model names denote the same model.

    Case- and route-insensitive: ``model.default`` may carry a
    provider-qualified ``anthropic/claude-opus-5`` while the session DB records
    the bare engine name, and a "last turn answered on" line that fires on that
    pair is a false alarm on every single ``/status``.
    """
    def norm(value: Any) -> str:
        text = _clean(value).casefold()
        return text.rsplit("/", 1)[-1] if "/" in text else text

    left_n, right_n = norm(left), norm(right)
    return bool(left_n) and left_n == right_n


def engine_model(config_model: Any = "") -> str:
    """The model the ACP bridge would spawn on, or ``""`` off the bridge.

    Asked of ``acp_config`` rather than reproduced here so ``/status`` cannot
    drift from the resolver the engine actually uses — that drift is the entire
    subject of this module. Degrades to ``""`` (i.e. "use ``config_model``") on
    any import or resolution failure.
    """
    try:
        from agent.acp_config import is_acp_config, resolve_model
    except Exception:  # pragma: no cover - overlay import guard
        return ""
    try:
        if not is_acp_config():
            return ""
        return _clean(resolve_model(config_model or None))
    except Exception:  # pragma: no cover - defensive
        return ""


def resolve(
    *,
    is_running: bool = False,
    agent_model: Any = "",
    agent_provider: Any = "",
    agent_base_url: Any = "",
    override: Any = None,
    config_model: Any = "",
    config_provider: Any = "",
    config_base_url: Any = "",
    db_model: Any = "",
    db_provider: Any = "",
    db_base_url: Any = "",
) -> dict[str, str]:
    """``{model, provider, base_url, source, last_turn}`` for the status lines.

    ``last_turn`` is the session DB's recorded model, and it is non-empty only
    when it names something other than what ``model`` says — i.e. only when
    there is genuinely a second fact worth showing. ``source`` is for tests and
    logs, not for display.

    Pure: every input is passed in, nothing is read from ``self`` or the world
    except :func:`engine_model`'s look at the profile config. That is what lets
    the precedence be tested without standing up a gateway, and the precedence
    is the whole point of the module.
    """
    override_map = override if isinstance(override, dict) else {}
    over_model = _clean(override_map.get("model"))

    model = provider = base_url = ""
    source = ""

    if is_running and _clean(agent_model):
        # A turn is in flight: this is not a prediction about the next turn, it
        # is the process answering right now.
        model = _clean(agent_model)
        provider = _clean(agent_provider)
        base_url = _clean(agent_base_url)
        source = "running"
    elif over_model:
        # /model --session. The next turn is built from this, config or not.
        model = over_model
        provider = _clean(override_map.get("provider"))
        base_url = _clean(override_map.get("base_url"))
        source = "session_override"
    else:
        engine = engine_model(config_model)
        model = engine or _clean(config_model)
        provider = _clean(config_provider)
        base_url = _clean(config_base_url)
        source = "engine" if engine else "config"

    if not model:
        # Nothing configured and nothing running: the warm agent, then the DB
        # record, are better than an empty line — but only here, last.
        model = _clean(agent_model) or _clean(db_model)
        provider = provider or _clean(agent_provider) or _clean(db_provider)
        base_url = base_url or _clean(agent_base_url) or _clean(db_base_url)
        source = "fallback"

    last_turn = _clean(db_model)
    if not last_turn or _same_model(last_turn, model) or source == "running":
        last_turn = ""

    return {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "source": source,
        "last_turn": last_turn,
    }
