"""Reconcile provably-abandoned cron attempts from any execution path.

Local overlay module (ACP-native fleet).

Measured defect (2026-07-28): a cron attempt whose owner process died outside a
scheduler lifecycle — an interrupted ``hermes cron run``, a direct run cut
short by a gateway restart — stayed ``running`` in ``cron/executions.db``
indefinitely. ``recover_interrupted_executions()`` existed but was wired only
to ``CronScheduler.recover_interrupted`` (scheduler start/loop), and
non-terminal rows are exempt from retention pruning, so nothing else ever
reconciled them. "Killed by a restart" and "still running" were
indistinguishable in history.

Why the logic lives here instead of in ``cron/executions.py``
-------------------------------------------------------------
The first version of this fix restructured
``recover_interrupted_executions()`` upstream — extracting the body into a
``_recover_interrupted_unlocked(conn)`` so ``create_execution`` could reuse it
on its own connection. That is a +53/-23 rewrite of an upstream function, and
v2026.8.3 rewrote the same function (it now also emits execution-state events),
so the two collided.

Reuse is not possible without that rewrite: ``recover_interrupted_executions``
opens its own transaction, while the ledger's contract is *one connection per
operation* — ``tests/cron/test_execution_ledger.py`` asserts
``len(opened) == 6`` for six calls, so nesting it inside ``create_execution``
would break upstream's own test. Hence a small, deliberate duplicate of the
reconciliation SQL that runs on a caller-owned connection. The predicate it
duplicates (``status IN ('claimed','running')`` AND not this process AND owner
not live) is the stable part of that function; what upstream keeps changing
around it — retention, event emission — is exactly what we do NOT duplicate.

The upstream call sites (scheduler start/loop) are untouched and remain the
path that emits state events. This one is bookkeeping attached to the insert
it precedes: idempotent, liveness-proven, and it never blocks that insert.
"""

from __future__ import annotations

import sqlite3

#: Written into the recovered row so history says *how* it was reconciled.
RECOVERY_ERROR = (
    "This execution's owner exited before a durable terminal state "
    "(detected by a later attempt on the same job ledger); whether side "
    "effects ran is unknown."
)


def reconcile_abandoned_unlocked(conn: sqlite3.Connection) -> int:
    """Mark abandoned attempts ``unknown`` on a caller-owned connection.

    Returns the number of rows changed. Never raises: audit hygiene must not
    be able to block the attempt it precedes — a failure here would turn a
    bookkeeping nicety into a cron outage.
    """
    try:
        from cron.executions import (
            _PROCESS_ID,
            _hermes_now,
            _owner_is_live,
            _prune_unlocked,
        )
    except Exception:  # pragma: no cover - module shape changed upstream
        return 0

    try:
        now = _hermes_now().isoformat()
        changed = 0
        rows = conn.execute(
            """SELECT id, process_id, pid, process_started_at FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            cur = conn.execute(
                """UPDATE executions SET status='unknown', finished_at=?, error=?
                   WHERE id=? AND status IN ('claimed','running')""",
                (now, RECOVERY_ERROR, row["id"]),
            )
            changed += cur.rowcount
        if changed:
            _prune_unlocked(conn)
        return changed
    except Exception:
        return 0
