"""Deciding when a non-streaming ACP call is actually stuck.

Local overlay module (not upstream).

The stale-call detector kills a non-streaming request once TOTAL elapsed time
passes ``HERMES_API_CALL_STALE_TIMEOUT`` (600s here). For a normal provider
that is sound: the whole answer arrives in one response, so ten minutes of
nothing means the connection is dead.

The ACP bridge breaks that assumption. The engine drives its own tools inside
a single API call, so a legitimate turn can run 15-30 minutes while producing
no bytes on the response — and it gets killed at 600s, retried, and finally
answered by the failover model. Measured 2026-07-25 on the researcher profile:
a turn with latency 896s was killed mid-flight; the user saw "primary model
failed (TimeoutError); failing over to claude-opus-4-8" while the engine had
been working the whole time.

Upstream already solves this shape for Codex: it tracks the timestamp of the
last stream event and kills on IDLE time rather than total time. Same idea
here — the ACP client stamps ``_acp_last_event_ts`` on every session update
(tool calls, narration, content chunks), so an engine that is visibly working
keeps the call alive, while one that has genuinely gone silent still dies on
schedule.
"""

from __future__ import annotations

from typing import Any


def should_kill_stale(agent: Any, elapsed: float, threshold: float) -> bool:
    """True if a non-streaming call should be killed as stale.

    ``elapsed`` is total time since the call started; ``threshold`` is the
    configured limit. For providers that report engine activity we measure
    idle time since the last reported event instead, so a long-but-active
    call survives and a silent one is still cut off after the same threshold.
    """
    if elapsed <= threshold:
        return False
    try:
        import time

        last_event = getattr(agent, "_acp_last_event_ts", None)
        if last_event is None:
            return True  # no activity signal — total-time rule applies
        idle = time.time() - float(last_event)
        return idle > threshold
    except Exception:
        return True
