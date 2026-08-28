"""How long the PRIMARY attempt gets before failover takes over.

Local overlay module (not upstream).

Background. Failover exists for models that *wedge*: Fable 5 would stop
producing output without erroring, so the turn hung until the full timeout
expired. Capping the primary attempt short made it yield to opus quickly. The
cap was armed whenever primary != failover model, which was true by definition
whenever failover was configured at all.

That condition stopped matching reality once a slow-but-healthy model became
primary. An ACP turn where the engine drives its own tools legitimately runs
5-30 minutes; with a 120s cap every such turn was killed mid-flight and
re-run on the failover model. Measured 2026-07-25: 28 of 114 calls (25%) on
one profile failed over, so a quarter of the answers came from the fallback
model after burning two minutes of the primary's work first.

So the cap is now tied to the property it was invented for — models known to
wedge — instead of to "failover is configured". Anything else gets the full
timeout and fails over only on a real error.
"""

from __future__ import annotations

import logging
import os

from agent.acp_background_work import (
    append_warning as append_background_work_warning,
)
from agent.acp_background_work import (
    reset_for_call as reset_background_work,
)
from agent.acp_usage_ledger import record_usage
from agent.acp_usage_transcript import take_call_usage

# Failover used to be invisible outside logs/acp-usage.jsonl — absent from
# agent.log, errors.log and the journal — so tracking one down cost more than
# diagnosing it (2026-07-26). Every decision this module makes now says so.
_log = logging.getLogger("agent.acp_failover")

# Models that stall instead of erroring. Matched as a substring of the model
# id, lowercased, so version suffixes don't need listing.
_WEDGE_PRONE = ("fable",)

# Cap for a wedge-prone primary. Long enough for a normal answer to complete,
# short enough that a stall doesn't hold the session.
_DEFAULT_CAP_S = 120.0

# An engine 401 is NOT a model failure. Every model behind the subscription
# shares one bearer, so the backup model answers the same 401 — the downgrade
# buys nothing and hides the real cause. What repairs it is a NEW bridge: each
# _run_prompt spawns one, and the spawn re-runs the token helper (resolve →
# refresh → persist). So the cure is "retry the same model", not "try another".
#
# 2026-07-29, secretary: the bridge spawned at 16:02:52 held a bearer with nine
# minutes of life left; the turn outlived it, the engine returned
# "401 OAuth access token has expired", and Hermes silently demoted opus-5 to
# opus-4-8 for the rest of the answer. Third instance of the same lesson in
# this file (ACPBridgeEvicted, deliberate interrupt, now auth): not every
# exception out of the primary attempt is the model's fault.
_AUTH_MARKERS = (
    "401",
    "failed to authenticate",
    "invalid authentication credentials",
    "oauth token",
)
# Sub-kinds change only the wording of the alert, never the action: both are
# worth exactly one retry on a fresh bridge, because a refresh repairs an
# expired access token AND a revoked one (the refresh token is a separate
# credential). Only a dead refresh token needs a human, and that shows up as
# the retry failing too.
_TOKEN_EXPIRED_MARKERS = ("has expired", "token expired", "re-authenticate")


def _configured_cap() -> float:
    raw = os.getenv("HERMES_ACP_PRIMARY_TIMEOUT", "").strip()
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CAP_S
    return val if val > 0 else _DEFAULT_CAP_S


def auth_failure_kind(exc) -> str | None:
    """Classify an ACP failure as an auth problem, or ``None`` if it isn't.

    ``"expired"`` — the access token ran out mid-flight (self-healing: the next
    bridge spawn refreshes it). ``"rejected"`` — any other 401. Both take the
    same repair path; the kind only names the cause in the log and the chat.
    """
    text = " ".join(str(exc or "").split()).lower()
    if not any(marker in text for marker in _AUTH_MARKERS):
        return None
    if any(marker in text for marker in _TOKEN_EXPIRED_MARKERS):
        return "expired"
    return "rejected"


def is_wedge_prone(model: str | None) -> bool:
    """True if *model* is known to stall rather than fail."""
    name = str(model or "").lower()
    return any(tag in name for tag in _WEDGE_PRONE)


def primary_attempt_timeout(
    effective_timeout: float,
    *,
    primary_model: str | None,
    failover_model: str | None,
    failover_enabled: bool,
) -> float:
    """Seconds to allow the primary attempt before failing over.

    Full timeout unless the primary is wedge-prone AND a *different* failover
    model is actually available to take over — capping with nowhere to fail
    over to would just shorten the turn.

    Set ``HERMES_ACP_PRIMARY_TIMEOUT`` to override the cap; it applies only
    where the cap applies at all.
    """
    if not failover_enabled:
        return effective_timeout
    if not failover_model or failover_model == primary_model:
        return effective_timeout
    if not is_wedge_prone(primary_model):
        return effective_timeout
    return min(effective_timeout, _configured_cap())


# Reasons ``_close_request_client_once`` passes when HERMES cancels the
# request on purpose (interrupt / redirect / steer all route through the
# interrupt check). Everything else it can pass — stale-call kills, TTFB
# kills, the worker's own ``request_complete`` — is either a real failure or a
# normal end of call, and must keep its failover.
_DELIBERATE_CLOSE_REASONS = ("interrupt_abort", "stream_interrupt_abort")


def _cancel_label(agent) -> str:
    """What to call this cancellation, read while the state still says so."""
    try:
        if getattr(agent, "_pending_redirect", None):
            return "user correction (redirect)"
        if getattr(agent, "_pending_steer", None):
            return "user correction (steer)"
    except Exception:
        pass
    return "interrupt"


def note_request_close(agent, client, reason) -> None:
    """Record — on the client itself — that WE are killing this request.

    Called from ``_close_request_client_once`` at the moment of the kill.

    Two bugs, one cause (researcher, 2026-07-28 12:14): the generic
    stranger-thread abort is socket-based, and an ACP client has no sockets
    (``tcp_force_closed=0`` in every log line). So an interrupt neither stopped
    the engine nor left any trace on the request, and the only evidence that
    the kill was deliberate lived in agent state — ``_pending_redirect`` (drained
    in conversation_loop before the worker unwinds) and ``_interrupt_requested``
    (already consumed by the interrupt handler that ordered this very close).
    By the time the orphaned worker's exception reached the failover layer,
    both were gone: the cancel read as a provider failure and the turn was
    demoted to the backup model.

    So the fact is stamped here, at the event, on the object that dies with the
    call — ``abort_for_interrupt`` sets the sticky ``_hermes_cancelled`` and
    SIGTERMs the bridge, which also stops an engine that would otherwise keep
    generating for a turn nobody is waiting on (and then race the replacement
    turn for the one-live-bridge slot). Same lesson as the disk witness: read
    the fact written when it happened, not state someone may have tidied since.

    Non-ACP clients have no ``abort_for_interrupt`` and are left untouched.
    """
    if not reason or not any(t in str(reason) for t in _DELIBERATE_CLOSE_REASONS):
        return
    abort = getattr(client, "abort_for_interrupt", None)
    if not callable(abort):
        return
    label = _cancel_label(agent)
    try:
        killed = abort(label)
    except Exception as exc:
        _log.debug("ACP abort_for_interrupt failed (%s): %s", reason, exc)
        return
    _log.info(
        "ACP request cancelled on purpose (%s → %s); bridges terminated=%s — "
        "not a provider failure, no failover.",
        reason, label, killed,
    )


def _arm_for_call(client) -> None:
    """Clear the sticky cancel flag so it describes THIS call only.

    Agent-scoped (``shared=True``) clients outlive a single request; without
    this reset one cancelled turn would mark the client cancelled forever and
    ``_run_prompt`` would refuse to spawn a bridge ever again.
    """
    if getattr(client, "_hermes_cancelled", None) is None:
        return
    try:
        client._hermes_cancelled = None
    except Exception:
        pass


def _deliberate_interrupt(client) -> str | None:
    """Reason, if the in-flight call was killed on purpose rather than failing.

    ``redirect()`` (a mid-run correction from the user) and ``interrupt()``
    both abort the active model request — the ACP bridge sees the subprocess
    die and raises exactly like a real timeout. Without telling the two apart,
    a user typing a clarification makes the turn "fail over" to the backup
    model: the correction is applied, but the answer comes from the weaker
    model and the chat shows a scary "primary model failed (TimeoutError)"
    (reported 2026-07-26 — the failover always followed a
    "Redirected current run" message).

    The sticky flag written by ``note_request_close`` is the primary signal —
    it was recorded at the kill and cannot be raced. The agent-state probe
    below stays as a fallback for kills that never went through the request
    close path, but it is the one that failed on 2026-07-28: by the time this
    runs, the flags it reads have usually been cleared.
    """
    sticky = getattr(client, "_hermes_cancelled", None)
    if sticky:
        return str(sticky)
    agent = getattr(client, "_activity_agent", None)
    if agent is None:
        return None
    try:
        if getattr(agent, "_pending_redirect", None):
            return "user correction (redirect)"
        if getattr(agent, "_pending_steer", None):
            return "user correction (steer)"
        if getattr(agent, "_interrupt_requested", False) or getattr(
            agent, "is_interrupted", False
        ):
            return "interrupt"
    except Exception:
        return None
    return None


def run_prompt_with_failover(client, prompt_text, effective_timeout, requested_model):
    """Run the prompt, falling over to the backup model on a real failure.

    Returns ``(response_text, reasoning_text, used_model, failover_fired)``.

    Lives here rather than inside the upstream client method because all of
    it is ours: the timeout policy above, the per-call model re-spawn, and
    the usage ledger entries for both outcomes.
    """
    # Lazy: copilot_acp_client imports this module, so a top-level import
    # back into it would be circular.
    from agent.copilot_acp_client import (
        ACPBridgeEvicted,
        _acp_effective_model,
        _acp_failover_enabled,
        _acp_failover_model,
    )

    # The sticky cancel flag describes one call; arm it for this one.
    _arm_for_call(client)
    # Same reason, different record: what the engine parks in the background
    # belongs to THIS turn only. Reset spans the auth retry and the failover
    # attempt on purpose — work spawned by an attempt that then failed over
    # is lost just the same, so it still has to be reported.
    reset_background_work(client)

    _session_id = getattr(
        getattr(client, "_activity_agent", None), "session_id", None
    )
    _requested_model = requested_model
    # Pass what Hermes asked for: with HERMES_ACP_MODEL unset the resolver
    # honours the request instead of a hardcoded default, so the ledger's
    # requested/used pair stays honest on the CLI path (agent/acp_config).
    _used_model = _acp_effective_model(requested_model)
    _failover_fired = False
    # Request 3: when a fast-fail model is primary (Fable) with opus failover
    # armed, cap the PRIMARY attempt so a wedged primary yields to opus fast
    # instead of holding the session for the full timeout — this is what
    # narrows the turn_lease fail-open window. Opus-primary (no distinct
    # failover) keeps the full timeout; the failover retry always does too.
    from agent.acp_failover import primary_attempt_timeout

    _primary_timeout = primary_attempt_timeout(
        effective_timeout,
        primary_model=_used_model,
        failover_model=_acp_failover_model(_used_model),
        failover_enabled=_acp_failover_enabled(),
    )
    try:
        response_text, reasoning_text = client._run_prompt(
            prompt_text,
            timeout_seconds=_primary_timeout,
        )
    except (RuntimeError, TimeoutError) as _primary_exc:
        if isinstance(_primary_exc, ACPBridgeEvicted):
            # Hermes killed this bridge (slot eviction / cancelled turn). The
            # model never got a chance to fail, so degrading to the backup one
            # punishes the wrong party — re-raise and let the loop retry on the
            # same model. The distinct type existed since 2026-07-26 but nobody
            # read it; that is how it kept reaching the failover branch.
            _log.info(
                "ACP bridge killed by Hermes, not the provider — no failover: %s",
                " ".join(str(_primary_exc).split())[:160],
            )
            raise
        _interrupt_reason = _deliberate_interrupt(client)
        if _interrupt_reason:
            # Not a provider failure: the request was cancelled on purpose.
            # Re-raise so the agent loop can apply the correction on the
            # primary model, instead of spending a second call on the backup.
            _log.info(
                "primary attempt cancelled on purpose (%s) — no failover.",
                _interrupt_reason,
            )
            raise
        _auth_kind = auth_failure_kind(_primary_exc)
        if _auth_kind:
            _auth_reason = " ".join(str(_primary_exc).split())[:160]
            _log.warning(
                "ACP auth failure (%s) on %s — the bearer is shared by every "
                "model, so this is not a model failure: refreshing the token "
                "and retrying the SAME model. %s",
                _auth_kind, _used_model, _auth_reason,
            )
            try:
                client._fire_engine_activity(
                    "_thinking",
                    "auth token %s — refreshing and retrying on %s"
                    % ("expired" if _auth_kind == "expired" else "rejected",
                       _used_model),
                )
            except Exception:
                pass
            try:
                # model_override pins the SAME model explicitly: the fresh
                # spawn must not re-resolve to anything else.
                response_text, reasoning_text = client._run_prompt(
                    prompt_text,
                    timeout_seconds=effective_timeout,
                    model_override=_used_model,
                )
            except (RuntimeError, TimeoutError) as _retry_exc:
                _log.error(
                    "ACP auth retry on %s failed too (%s) — the subscription "
                    "login itself is likely dead; re-login Claude Code. %s",
                    _used_model,
                    type(_retry_exc).__name__,
                    " ".join(str(_retry_exc).split())[:160],
                )
                record_usage(
                    session_id=_session_id,
                    requested_model=_requested_model,
                    used_model=_used_model,
                    prompt_text=prompt_text,
                    completion_text=None,
                    failover=False,
                    ok=False,
                    real=take_call_usage(client),
                    error=_retry_exc,
                )
                raise
            else:
                _log.info(
                    "ACP auth retry on %s succeeded — no model downgrade.",
                    _used_model,
                )
                return response_text, reasoning_text, _used_model, False

        _fo_model = _acp_failover_model(_used_model)
        if (
            _acp_failover_enabled()
            and _fo_model
            and _fo_model != _used_model
        ):
            try:
                # Name the reason, not just the exception class: "primary model
                # failed (TimeoutError)" sent us hunting a latency problem when
                # the real cause was a dead ACP bridge (2026-07-26).
                _reason = " ".join(str(_primary_exc).split())[:160]
                # 2026-07-29: the notice below is a chat progress line, so a
                # live downgrade left NO trace in agent.log/errors.log — the
                # only record was a ledger row written after the retry ended,
                # minutes later. Say it in the log at the moment it happens.
                _log.warning(
                    "primary model %s failed (%s) — failing over to %s. %s",
                    _used_model, type(_primary_exc).__name__, _fo_model, _reason,
                )
                client._fire_engine_activity(
                    "_thinking",
                    "primary model failed (%s%s); failing over to %s"
                    % (
                        type(_primary_exc).__name__,
                        f": {_reason}" if _reason else "",
                        _fo_model,
                    ),
                )
            except Exception:
                pass
            try:
                response_text, reasoning_text = client._run_prompt(
                    prompt_text,
                    timeout_seconds=effective_timeout,
                    model_override=_fo_model,
                )
                _used_model = _fo_model
                _failover_fired = True
            except (RuntimeError, TimeoutError) as _fo_exc:
                record_usage(
                    session_id=_session_id,
                    requested_model=_requested_model,
                    used_model=_fo_model,
                    prompt_text=prompt_text,
                    completion_text=None,
                    failover=True,
                    ok=False,
                    real=take_call_usage(client),
                    error=_fo_exc,
                )
                raise
        else:
            # Say WHY nothing took over. Without this the turn just raises and
            # "no fallback configured" is indistinguishable in the log from
            # "the fallback also failed" — and since 2026-08-02 the default
            # fallback IS the primary, so this branch is the normal one.
            _log.info(
                "no failover from %s (%s): fallback=%r, enabled=%s — "
                "raising on the pinned model instead of downgrading.",
                _used_model, type(_primary_exc).__name__,
                _fo_model, _acp_failover_enabled(),
            )
            record_usage(
                session_id=_session_id,
                requested_model=_requested_model,
                used_model=_used_model,
                prompt_text=prompt_text,
                completion_text=None,
                failover=False,
                ok=False,
                real=take_call_usage(client),
                error=_primary_exc,
            )
            raise

    # Say it in the ANSWER, not on the progress rail. A progress line is not
    # part of the transcript, so the next turn would not see it and the agent
    # would keep believing its own "started in the background" report — which
    # is how the same hour was lost twice on 2026-08-01.
    response_text = append_background_work_warning(client, response_text)
    return response_text, reasoning_text, _used_model, _failover_fired


def record_successful_call(
    client, prompt_text, completion_text, reasoning_text,
    *, requested_model, used_model, failover,
):
    """Ledger entry for a call that produced an answer."""
    record_usage(
        session_id=getattr(
            getattr(client, "_activity_agent", None), "session_id", None
        ),
        requested_model=requested_model,
        used_model=used_model,
        prompt_text=prompt_text,
        completion_text=completion_text,
        reasoning_text=reasoning_text,
        failover=failover,
        ok=True,
        real=take_call_usage(client),
    )
