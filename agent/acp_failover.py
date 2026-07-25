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

import os

# Models that stall instead of erroring. Matched as a substring of the model
# id, lowercased, so version suffixes don't need listing.
_WEDGE_PRONE = ("fable",)

# Cap for a wedge-prone primary. Long enough for a normal answer to complete,
# short enough that a stall doesn't hold the session.
_DEFAULT_CAP_S = 120.0


def _configured_cap() -> float:
    raw = os.getenv("HERMES_ACP_PRIMARY_TIMEOUT", "").strip()
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CAP_S
    return val if val > 0 else _DEFAULT_CAP_S


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
