"""What ``/model`` means on a profile whose engine is the ACP bridge.

Local overlay module (not upstream).

Three things go wrong when the generic model switcher meets an ACP profile, and
they are independent — fixing any two of them leaves ``/model`` broken:

**1. The route.** ``claude-*`` is a name the catalog knows, so
``model_switch.switch_model()`` resolves it to the ``anthropic`` provider at
``https://api.anthropic.com`` and hands the session a direct-API client. On a
subscription OAuth login that endpoint answers ``400 Third-party apps now draw
from your extra usage, not your plan limits`` — measured 2026-08-09 17:39:51,
three of them per turn, after which the turn slid back onto the bridge and
answered on the *pinned* model. So the switch both failed and looked like it
had worked. :func:`coerce_switch` keeps a ``claude-*`` target on the bridge.

**2. The ack.** This seam shrank twice and is now one job:
:func:`filter_switch_warnings` drops a warning that fires on the *healthy* case.
It used to also write a second copy of the model into ``.env`` (removed
2026-08-10, when ``config.yaml`` became the single source) and to narrate which
scope had moved — cut 2026-08-10 as duplication: the gateway already prints
``gateway.model.saved_global`` / ``session_only_hint`` right below, localized,
and the ack said the same thing in English. The leftover-``.env``-pin note went
with it: :func:`acp_config.enforce` refuses to start a gateway over that pin, so
a line in a ``/model`` reply was the weaker of two reporters, not the only one.

**3. The buttons.** Fixing 1 and 2 fixed the *typed* command only. The picker
hands ``switch_model`` an ``explicit_provider`` on every press — that is how a
row identifies itself — so :func:`coerce_switch` correctly stands aside and the
press lands wherever the row said. Measured 2026-08-09 on the default profile:
the picker offered ``moa``, ``copilot`` and ``anthropic``. There was no bridge
row at all (the provider's curated model list is the placeholder
``['copilot-acp']``, so the empty-row filter dropped it), and ``claude-opus-5``,
the model most of the fleet runs on, was on no button anywhere. The only row
leading to a claude model led to the direct API — the 400 again. There was no
correct button to press. :func:`picker_rows` adds the bridge's own row and takes
the ids it serves off the other rows, which is the payload-level dedup upstream
already applies to routing aggregators (#45954), for the reason it states there:
two buttons for one model, one of which silently breaks the call.

The per-session case needs no writing at all: ``acp_config.explicit_choice``
already reads "the agent's model differs from the profile's config label" as
"someone chose this for this run", which is precisely what a ``--session``
override produces.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Provider slugs served by the ACP bridge rather than by an HTTP endpoint.
ACP_PROVIDERS = frozenset({"copilot-acp"})


def is_acp_provider(provider: Any) -> bool:
    return str(provider or "").strip().lower() in ACP_PROVIDERS


def _engine_model(value: Any) -> str | None:
    """The bare engine name, or ``None``. Import-guarded: this module is a
    call-out from an upstream file and must degrade to "do nothing"."""
    try:
        from agent.acp_config import normalize_model

        return normalize_model(value)
    except Exception:  # pragma: no cover - overlay import guard
        return None


def coerce_switch(
    raw_input: Any, explicit_provider: Any, current_provider: Any
) -> tuple[str, str]:
    """``(provider_to_force, error)`` for one ``switch_model`` call.

    Forces ``copilot-acp`` when an ACP profile is switching to an engine model
    and the operator did not name a provider. An explicit ``--provider`` always
    wins: leaving the bridge on purpose stays possible, it just has to be said.

    Never raises and never invents an error — the empty tuple means "upstream
    resolution is correct here", which is every non-ACP profile and every
    non-``claude`` target.
    """
    if str(explicit_provider or "").strip():
        return "", ""
    if not is_acp_provider(current_provider):
        return "", ""
    if not _engine_model(raw_input):
        return "", ""
    return "copilot-acp", ""


def filter_switch_warnings(
    *,
    new_model: Any,
    target_provider: Any,
    current_provider: Any,
    warnings: list[str] | None = None,
    catalog_note: str = "",
) -> list[str]:
    """The warning list to show after a successful switch. Adds nothing routine.

    One warning upstream assembles is wrong here: ``validate_requested_model``
    checks the target against *GitHub Copilot's* curated catalog, and the bridge
    is not Copilot — it runs ``claude-agent-acp``. Measured 2026-08-10, a plain
    ``/model claude-fable-5`` came back "``claude-fable-5`` was not found in the
    GitHub Copilot ACP curated catalog … the model may still work", three lines,
    on a switch that worked. A warning that fires on the healthy case is one
    people learn to scroll past, so it is dropped for engine models on the ACP
    route (the real validator there is ``acp_config._ENGINE_MODEL_RE``). Dropped
    by *equality* with the message upstream computed, never by matching text.

    The one line it still adds fires only when a switch **leaves** the bridge,
    which is the trap the whole overlay exists for: the direct API answers 400
    on a subscription login, the turn falls back, and nothing in the ack would
    otherwise say the route changed.

    Never raises: this runs inside ``switch_model``'s success path, and an
    exception there would turn a completed switch into a reported failure. It
    writes nothing — the caller persists ``model.default``, which is the source
    ``acp_config.resolve_model`` reads.
    """
    notes: list[str] = list(warnings or [])
    if (
        catalog_note
        and catalog_note in notes
        and is_acp_provider(target_provider)
        and _engine_model(new_model)
    ):
        notes = [n for n in notes if n != catalog_note]
    if is_acp_provider(current_provider) and not is_acp_provider(target_provider):
        notes.append(
            "this profile runs on the ACP bridge — switching to "
            f"{str(target_provider or '?')} leaves it, so this session stops "
            "going through the subscription login"
        )
    return notes


# --- the buttons -------------------------------------------------------------

#: Slug and label of the virtual picker row for the bridge. The slug is not
#: cosmetic: it is what a press hands back as ``explicit_provider``, so it has
#: to be the provider slug ``switch_model`` resolves, not a display name.
ACP_PICKER_SLUG = "copilot-acp"
ACP_PICKER_NAME = "Claude (ACP bridge)"


def _profile_model() -> str | None:
    """``model.default`` from the profile's config — the source, not a session."""
    try:
        from agent.acp_config import config_model

        return config_model()
    except Exception:  # pragma: no cover - overlay import guard
        return None


def _route_fallback() -> tuple[str, str | None]:
    """``(fallback_model.provider, fallback_model.model)`` for this profile."""
    try:
        from agent.acp_config import route_fallback_model, route_fallback_provider

        return route_fallback_provider(), route_fallback_model()
    except Exception:  # pragma: no cover - overlay import guard
        return "", None


def is_acp_profile(current_provider: Any = "") -> bool:
    """Whether the bridge can serve this profile at all.

    Deliberately not the same question as ``is_acp_provider(current_provider)``,
    and the difference is the trap door: a session that already fell onto the
    direct API reports ``anthropic`` as its current provider (the gateway reads
    the session override before opening the picker), so asking only that would
    withhold the bridge row from exactly the session that needs it to get back.
    ``model.provider`` in ``config.yaml`` is the profile-level fact, so a profile
    configured for the bridge gets the row wherever this session happens to be
    routed right now.

    The third question is ``fallback_model.provider``, and it is the one this
    file was missing. Measured 2026-08-19: moving ``secretary`` onto a corporate
    HTTP endpoint left the bridge as its failover — turns that endpoint refused
    were in fact answered by ``claude-sonnet-5`` through the bridge — yet the
    picker lost the bridge row, because "is this the bridge" was asked only of
    ``model.provider``. A backend that already answers for this profile must
    have a button; withholding it also hands the direct-API row the only
    ``claude-*`` ids left, which is the 400 this file exists to prevent.
    """
    if is_acp_provider(current_provider):
        return True
    try:
        from agent.acp_config import is_acp_config

        if is_acp_config():
            return True
    except Exception:  # pragma: no cover - overlay import guard
        pass
    return is_acp_provider(_route_fallback()[0])


def engine_models(current_model: Any = "") -> list[str]:
    """The bridge's model ids: current, profile, failover, then the known list.

    Order is "what this profile actually runs" before "what exists", so the
    first button is the one it is already on. All three are included even when
    absent from ``KNOWN_ENGINE_MODELS``, because that tuple is a display list
    and not a gate — a profile configured for something newer than it must not
    lose its own model from the picker.

    ``fallback_model.model`` earns third place for the same reason the first two
    have theirs: on a profile whose main provider is not the bridge, it is the
    only engine model that has actually answered here, so it is the button most
    likely to be wanted — and it must not be the one the list happens to omit.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        model = _engine_model(value)
        if model and model.casefold() not in seen:
            seen.add(model.casefold())
            out.append(model)

    add(current_model)
    add(_profile_model())
    add(_route_fallback()[1])
    try:
        from agent.acp_config import KNOWN_ENGINE_MODELS

        for known in KNOWN_ENGINE_MODELS:
            add(known)
    except Exception:  # pragma: no cover - overlay import guard
        pass
    return out


def picker_rows(
    providers: Any,
    *,
    current_provider: Any = "",
    current_model: Any = "",
    max_models: Any = None,
) -> Any:
    """The ``/model`` picker payload for an ACP profile.

    Two edits, and they are one idea: every engine model gets exactly one button,
    and it is the one that works.

    * the bridge gets a row, first, built from :func:`engine_models`;
    * every other row loses the ids the bridge serves, matched by *exact* id.
      Exact on purpose: OpenRouter carries ``anthropic/claude-opus-5``, a
      genuinely different route that genuinely works, and it survives untouched
      precisely because it is not the same id. A user's own configured endpoint
      is skipped as well — upstream learned that one at #47077: the specific
      provider is not a duplicate of anything, and gutting it empties its row.

    Rows emptied by the dedup are dropped by ``list_picker_providers``' own
    empty-row filter downstream; ``total_models`` follows upstream's aggregator
    dedup and reports what is visible.

    Never raises and returns the input unchanged on any doubt: a picker that
    fails to open is worse than a picker missing a row.
    """
    try:
        if not is_acp_profile(current_provider):
            return providers
        models = engine_models(current_model)
        if not models:
            return providers
        try:
            limit = int(max_models) if max_models is not None else None
        except (TypeError, ValueError):
            limit = None
        row = {
            "slug": ACP_PICKER_SLUG,
            "name": ACP_PICKER_NAME,
            # Offered is not the same as current, and the picker prints a
            # "current" tag off this flag: a profile that merely *falls back*
            # to the bridge gets the row (see is_acp_profile) but must not be
            # told it is running on it. The row's position is not at stake —
            # it is prepended either way.
            "is_current": is_acp_provider(current_provider),
            "is_user_defined": False,
            "models": models[:limit] if limit is not None else list(models),
            "total_models": len(models),
            "source": "virtual",
            "authenticated": True,
            "auth_type": "subscription",
            "warning": (
                "Runs through the local claude-agent-acp bridge on the "
                "subscription login — not the direct Anthropic API."
            ),
        }
        served = {m.casefold() for m in models}
        rows = [row]
        for provider in list(providers or []):
            if not isinstance(provider, dict):
                rows.append(provider)
                continue
            if str(provider.get("slug", "")).strip().lower() in ACP_PROVIDERS:
                continue  # never two bridge rows
            if provider.get("is_user_defined"):
                rows.append(provider)
                continue
            original = list(provider.get("models") or [])
            kept = [m for m in original if str(m).casefold() not in served]
            if len(kept) != len(original):
                provider = dict(provider)
                provider["models"] = kept
                provider["total_models"] = len(kept)
            rows.append(provider)
        return rows
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("ACP picker row build failed: %s", exc)
        return providers


def ungrouped_fold(fold: Any, providers: Any) -> Any:
    """Wrap ``group_providers`` so the bridge keeps its own top-level button.

    Upstream folds ``copilot-acp`` into the GitHub Copilot family, and on a
    stock install that is right — the slug means "the ``copilot --acp``
    process". Here it does not: ``HERMES_COPILOT_ACP_COMMAND`` points the same
    slot at ``claude-agent-acp``, which is the entire premise of this overlay.
    Folded, the fleet's own models sit two taps deep behind a button naming the
    wrong vendor, while the button that *reads* like the right one (Anthropic)
    is the direct API. Adding a correct row and then hiding it is not a fix.

    Returns ``fold`` untouched off an ACP profile, and degrades to it on any
    error: a picker that raises here shows no keyboard at all.
    """
    try:
        if not any(
            isinstance(p, dict) and str(p.get("slug", "")).strip().lower() in ACP_PROVIDERS
            for p in (providers or [])
        ):
            return fold
    except Exception:  # pragma: no cover - defensive
        return fold

    def _fold(slugs: Any) -> Any:
        rows = fold(slugs)
        try:
            out: list = []
            promoted = False
            for row in rows:
                if row.get("kind") != "group":
                    out.append(row)
                    continue
                members = [
                    m for m in row.get("members", [])
                    if str(m).strip().lower() not in ACP_PROVIDERS
                ]
                if len(members) == len(row.get("members", [])):
                    out.append(row)
                    continue
                promoted = True
                if len(members) == 1:
                    out.append({"kind": "single", "slug": members[0]})
                elif members:
                    out.append(dict(row, members=members))
            if not promoted:
                return rows
            return [{"kind": "single", "slug": ACP_PICKER_SLUG}] + out
        except Exception:  # pragma: no cover - defensive
            return rows

    return _fold
