"""Counting "work done" for the self-improvement nudge on the ACP path.

Local overlay module (not upstream).

Hermes triggers its background skill review after N tool-calling iterations
(``skills.creation_nudge_interval``, default 15). On a normal provider the
agent loop runs one iteration per tool batch, so an active turn contributes
several and the threshold is reached within a turn or two.

On the native ACP path the engine drives its own tools inside a single API
call, so the loop sees exactly ONE iteration per turn no matter how much work
happened — a turn that ran thirty tools counts the same as "hello". The
threshold then needs 15 *turns of conversation*, and since the counter lives
on the cached agent it also resets whenever the agent is rebuilt (config
change, gateway restart). Result observed 2026-07-25: the review had never
fired once on the ACP fleet — no skills, no memory writes, nothing in the logs.

The ledger of native tool calls already records what the engine actually did,
so it is used here as the unit of work: the counter advances by the number of
engine tool calls made during the turn, which restores the original intent
("review after roughly this much activity") on both paths.

Where the counter LIVES (2026-08-10)
------------------------------------
The first version of this module kept advancing ``_iters_since_skill``, which
lives on the agent OBJECT, and read the ledger at the START of an iteration.
Both choices leak work:

* the ledger delta read at iteration start describes the PREVIOUS turn, so a
  turn's own calls only ever count if the same agent object survives to handle
  another turn — a one-turn session (every agentic cron) contributes exactly 1
  no matter how much it did;
* the object dies on every gateway restart / agent rebuild, so the count
  restarts from zero.

Measured on ``secretary`` 2026-08-10: no skill review since 07.08 19:36 while
three cron turns alone ran 54 + 73 + 28 engine tool calls. Every one of those
turns was its own session, and each of the four interactive turns in between
had a gateway restart in front of it, so the counter never got past ~9 of 20.
The memory nudge does not have this hole — ``turn_context`` re-hydrates
``_turns_since_memory`` from persisted history — which is exactly why memory
review was the one that kept working.

So on the ACP path the counter is durable and profile-wide
(``<HERMES_HOME>/state/skill-nudge.json``) and the accounting happens at the
END of the turn:

* ``skill_iterations_delta`` stamps the ledger mark at turn start and adds
  nothing (returning 0 keeps a single accounting point, and the mark taken at
  turn start is what keeps an ASYNC background review — which appends to the
  parent's ledger when the fork reuses the parent's ACP client — from counting
  as the parent's work);
* ``finalize_turn`` adds this turn's own calls and publishes the durable total
  onto ``_iters_since_skill``, so the upstream threshold check right below it
  is unchanged;
* ``note_reset`` zeroes the durable counter wherever upstream zeroes the
  in-memory one (review fired, or ``skill_manage`` was used).

Non-ACP paths keep the upstream "+1 per tool iteration" behaviour untouched.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_MARK = "_acp_nudge_seen_calls"
_STATE_NAME = "skill-nudge.json"


def _is_acp(agent: Any) -> bool:
    """True when this agent talks to the engine over the ACP bridge."""
    try:
        from agent.acp_model_choice import is_acp_provider

        return bool(is_acp_provider(getattr(agent, "provider", "") or ""))
    except Exception:
        return False


def _counts(agent: Any) -> bool:
    """Whether nudge accounting applies at all for this agent.

    Mirrors the upstream trigger's own preconditions: a profile with the nudge
    switched off, or a turn where ``skill_manage`` is not on the table, can
    never fire a review — accumulating for it would only mean a burst of
    reviews the moment the tool reappears. The review fork sets the interval to
    0, so this also keeps a review from feeding the counter that spawned it.
    """
    try:
        if int(getattr(agent, "_skill_nudge_interval", 0) or 0) <= 0:
            return False
        return "skill_manage" in (getattr(agent, "valid_tool_names", None) or ())
    except Exception:
        return False


def _state_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "state" / _STATE_NAME


def _update(fn: Callable[[int], int]) -> int:
    """Read-modify-write the durable counter under an exclusive lock.

    Two sessions of the same profile finish turns concurrently inside one
    gateway, so the read-modify-write needs the lock; the file is held open for
    the whole cycle rather than rewritten via a temp file so the lock and the
    write cover the same bytes.
    """
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+", encoding="utf-8") as fh:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass  # no flock (or not POSIX): last writer wins, still a counter
            fh.seek(0)
            raw = fh.read()
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            new = max(0, int(fn(int(data.get("iters") or 0))))
            import datetime as _dt

            data["iters"] = new
            data["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(data, ensure_ascii=False))
            fh.flush()
        return new
    except Exception as e:
        logger.debug("skill-nudge counter update failed: %s", e)
        return 0


def _ledger_len(agent: Any) -> int:
    try:
        return len(getattr(agent, "_acp_native_tool_calls", None) or [])
    except Exception:
        return 0


def skill_iterations_delta(agent: Any) -> int:
    """How much to advance the in-memory skill-nudge counter, at iteration start.

    Non-ACP: 1, exactly as upstream. ACP: 0 — the ledger mark is stamped here
    and the turn's own calls are counted by :func:`finalize_turn` once the turn
    is actually over. Never raises: a miscount must not break the agent loop.
    """
    try:
        if not _is_acp(agent):
            return 1
        setattr(agent, _MARK, _ledger_len(agent))
        return 0
    except Exception:
        return 1


def finalize_turn(agent: Any) -> None:
    """Book this turn's engine tool calls and publish the durable total.

    Called from ``turn_finalizer`` immediately before the upstream skill-nudge
    threshold check, so that check keeps reading ``_iters_since_skill`` and
    needs no changes. A tool-free turn still counts as 1, the way an upstream
    iteration does. No-op off the ACP path.
    """
    try:
        if not _is_acp(agent) or not _counts(agent):
            return
        total = _ledger_len(agent)
        seen = int(getattr(agent, _MARK, 0) or 0)
        if total < seen:  # ledger replaced between turns
            seen = 0
        setattr(agent, _MARK, total)
        agent._iters_since_skill = _update(lambda cur: cur + max(1, total - seen))
    except Exception as e:
        logger.debug("skill-nudge finalize failed: %s", e)


def note_reset(agent: Any) -> None:
    """Zero the durable counter alongside the in-memory one. No-op off ACP."""
    try:
        if not _is_acp(agent):
            return
        setattr(agent, _MARK, _ledger_len(agent))
        _update(lambda _cur: 0)
    except Exception as e:
        logger.debug("skill-nudge reset failed: %s", e)


def may_fire(agent: Any) -> bool:
    """Whether a review spawned right now would live long enough to finish.

    The review runs in a DAEMON thread, so it dies with the process. In the
    gateway that is fine — the process outlives any turn. A one-shot run
    (``hermes -z``, every ``ask-hermes`` call between agents) exits the moment
    the answer is printed, which would kill the review mid-flight AND zero the
    counter that earned it: work spent, nothing written, and the health guard
    still sees a frozen ``last_run_at``.

    Before the counter went durable this was theoretical — a one-shot process
    started from zero and never reached the threshold. Now it would happen
    routinely, so a one-shot defers instead: the debt stays on disk and the
    next gateway-hosted turn (telegram, cron) spends it.

    Off the ACP path the upstream behaviour is left exactly as it was.
    """
    try:
        if not _is_acp(agent):
            return True
        return os.environ.get("_HERMES_GATEWAY") == "1"
    except Exception:
        return True


def current_count() -> int:
    """Durable counter as it stands on disk (diagnostics, health guard)."""
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return int(data.get("iters") or 0) if isinstance(data, dict) else 0
    except Exception:
        return 0
