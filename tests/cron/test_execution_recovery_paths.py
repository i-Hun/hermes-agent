"""Recovery of abandoned cron attempts no longer needs a scheduler restart.

Local overlay tests (not upstream). Measured defect (2026-07-28): a cron
attempt whose owner process died outside a scheduler lifecycle — an
interrupted ``hermes cron run``, a direct run cut short by a gateway restart —
stayed ``running`` in ``cron/executions.db`` indefinitely.
``recover_interrupted_executions()`` existed but was only wired to
``CronScheduler.recover_interrupted`` (scheduler start/loop), and non-terminal
rows are exempt from retention pruning, so nothing else ever reconciled them.
"Killed by a restart" and "still running" were indistinguishable in history.

The fix hooks the same liveness-proven recovery into the two paths a human or
a job actually takes: creating the next attempt, and reading the history.
"""

from __future__ import annotations

import sqlite3


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions


def _orphan(executions, job_id: str) -> str:
    """An attempt left ``running`` by an owner that is provably gone.

    ``process_started_at=-1`` can never match a live PID's real start time, so
    ``_owner_is_live`` proves death without depending on a reaped PID.
    """
    record = executions.create_execution(job_id, source="direct")
    executions.mark_execution_running(record["id"])
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute(
            "UPDATE executions SET process_id=?, process_started_at=? WHERE id=?",
            ("killed-cli-run", -1, record["id"]),
        )
    assert executions.latest_execution(job_id)["status"] == "running"
    return record["id"]


def test_next_attempt_reconciles_an_abandoned_run(monkeypatch, tmp_path):
    """The regression: nothing between the kill and the next scheduler start."""
    executions = _point_ledger(monkeypatch, tmp_path)
    orphan_id = _orphan(executions, "wiki-ingest")

    # Any later attempt — same job or not — does the reconciliation.
    executions.create_execution("unrelated-job", source="builtin")

    row = [r for r in executions.list_executions(limit=50) if r["id"] == orphan_id][0]
    assert row["status"] == "unknown"
    assert row["finished_at"]
    assert "unknown" in row["error"].lower()


def test_history_read_reconciles_before_printing(monkeypatch, tmp_path, capsys):
    """`hermes cron runs` is where a human looks — it must not show a phantom."""
    executions = _point_ledger(monkeypatch, tmp_path)
    _orphan(executions, "cli-job")

    from hermes_cli.cron import cron_runs

    cron_runs("cli-job", limit=10)
    out = capsys.readouterr().out

    assert "unknown" in out
    assert "running" not in out


def test_live_attempt_is_never_reconciled_by_either_path(monkeypatch, tmp_path):
    """A genuinely in-flight attempt of another live process stays running."""
    import os

    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("in-flight", source="builtin")
    executions.mark_execution_running(record["id"])
    # Foreign import, but the PID is this (alive) process → not provably dead.
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute(
            "UPDATE executions SET process_id=?, pid=?, process_started_at=NULL "
            "WHERE id=?",
            ("another-import", os.getpid(), record["id"]),
        )

    executions.create_execution("some-other-job", source="builtin")
    from hermes_cli.cron import cron_runs

    cron_runs("in-flight", limit=10)

    assert executions.latest_execution("in-flight")["status"] == "running"


def test_recovery_failure_never_blocks_the_attempt(monkeypatch, tmp_path):
    """Audit hygiene is best-effort: a broken recovery must not lose a run."""
    executions = _point_ledger(monkeypatch, tmp_path)

    import cron.acp_execution_recovery as recovery

    def _boom(_conn):
        raise RuntimeError("ledger recovery exploded")

    # create_execution imports the helper inside its own body, so patching the
    # defining module is what the call actually resolves.
    monkeypatch.setattr(recovery, "reconcile_abandoned_unlocked", _boom)
    record = executions.create_execution("must-still-run", source="direct")

    assert record["status"] == "claimed"


def test_recovery_never_opens_a_second_connection(monkeypatch, tmp_path):
    """The ledger's contract is one connection per operation.

    Upstream asserts it too (``test_ledger_operations_close_every_connection``
    counts six for six calls); this pins the reason the overlay duplicates the
    reconciliation SQL instead of calling ``recover_interrupted_executions``,
    which would open a second one and turn that upstream test red.
    """
    executions = _point_ledger(monkeypatch, tmp_path)
    _orphan(executions, "abandoned")

    opened = []
    real_connect = sqlite3.connect

    def _tracking_connect(*args, **kwargs):
        opened.append(1)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(executions.sqlite3, "connect", _tracking_connect)
    executions.create_execution("next-attempt", source="direct")

    assert len(opened) == 1
    assert executions.latest_execution("abandoned")["status"] == "unknown"
