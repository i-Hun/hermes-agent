"""Invariant: a worker that outlives its own terminal transition is still
reachable and gets reaped by the dispatcher (issue #111791).

``complete_task`` wipes ``tasks.worker_pid`` and the ``running``-only reclaim
sweeps never look at a ``done`` card again, so a worker that called
``kanban_complete`` and then hung (holding deleted ``state.db`` sidecar fds)
was invisible to every command. The closed ``task_runs`` row now keeps the pid
and its spawn fingerprint, and ``reap_terminal_workers`` ends such a worker on
the next tick — never a recycled PID, never a legacy row without a fingerprint.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _completed_card_with_worker(conn, proc, *, ended_ago: int = 600) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    # Default: the run closed long enough ago that the reaper's grace window has passed.
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def test_worker_alive_after_completion_is_reaped_on_dispatch_tick(conn):
    proc = _sleeper()
    try:
        tid, run_id = _completed_card_with_worker(conn, proc)
        # The evidence the running-only sweeps lost lives on the closed run row.
        run = conn.execute("SELECT worker_pid, worker_started_at, ended_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["ended_at"] is not None and run["worker_pid"] == proc.pid and run["worker_started_at"] is not None

        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

        assert result.reaped_terminal_workers == [tid]
        assert proc.wait(timeout=10) is not None
        run = conn.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] is None and run["worker_started_at"] is None
        kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
        assert "terminal_worker_reaped" in kinds
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_recycled_pid_and_legacy_row_are_never_signalled(conn):
    stranger = _sleeper()
    legacy = _sleeper()
    try:
        _tid, run_id = _completed_card_with_worker(conn, stranger)
        # PID reuse: the recorded fingerprint belongs to a process that no longer exists.
        conn.execute("UPDATE task_runs SET worker_started_at = worker_started_at - 1000000 WHERE id=?", (run_id,))
        _, legacy_run = _completed_card_with_worker(conn, legacy)
        conn.execute("UPDATE task_runs SET worker_started_at = NULL WHERE id=?", (legacy_run,))

        assert kbd.reap_terminal_workers(conn) == []

        assert stranger.poll() is None and legacy.poll() is None
        # Stale evidence is dropped so the row is not rescanned; the legacy row is left alone.
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()["worker_pid"] is None
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (legacy_run,)).fetchone()["worker_pid"] == legacy.pid
    finally:
        for p in (stranger, legacy):
            p.kill()
            p.wait()


def test_fresh_terminal_run_is_left_alone_until_grace_passes(conn):
    """A worker is still alive for a moment after its own kanban_complete returns
    (final turn, session persistence): a just-closed run is not signalled."""
    proc = _sleeper()
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        os.kill(pid, sig)

    try:
        tid, run_id = _completed_card_with_worker(conn, proc, ended_ago=0)

        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == []

        assert signals == [] and proc.poll() is None
        run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] == proc.pid  # evidence kept for a later tick
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
            (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id),
        )
        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == [tid]
        assert [pid for pid, _ in signals] == [proc.pid]
    finally:
        proc.kill()
        proc.wait()


def test_one_failing_row_does_not_abort_the_sweep(conn):
    """A signal failure on one run is logged and skipped; the other rows are still reaped."""
    broken, healthy = _sleeper(), _sleeper()
    try:
        _completed_card_with_worker(conn, broken)
        tid, _ = _completed_card_with_worker(conn, healthy)

        def signal_fn(pid, sig):
            if pid == broken.pid:
                raise RuntimeError("boom")
            healthy.kill()

        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == [tid]
    finally:
        for p in (broken, healthy):
            p.kill()
            p.wait()


# --- Whole-scope reap: background children outlive the worker's main pid ------

import shutil


def _closed_run(conn, *, ended_ago: int = 600) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def _unit(tid: str, run_id: int) -> str:
    return f"hermes-worker-kanban-{tid}-run-{run_id}.scope"


@pytest.fixture
def fake_scopes(monkeypatch):
    """Fake user systemd: a set of live units; ``stop`` removes one."""
    live: set[str] = set()
    stopped: list[str] = []
    monkeypatch.setattr(kbd, "_list_worker_scope_units", lambda: sorted(live))
    monkeypatch.setattr(kbd, "_worker_scope_tasks_current", lambda unit: 3)

    def stop(unit):
        stopped.append(unit)
        live.discard(unit)
        return True

    monkeypatch.setattr(kbd, "_stop_worker_scope", stop)
    return live, stopped


def test_closed_run_scope_is_stopped_with_event(conn, fake_scopes):
    live, stopped = fake_scopes
    tid, run_id = _closed_run(conn)
    live.add(_unit(tid, run_id))

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

    assert stopped == [_unit(tid, run_id)]
    assert tid in result.reaped_terminal_workers
    ev = conn.execute(
        "SELECT payload, run_id FROM task_events WHERE task_id=? AND kind='terminal_worker_reaped'", (tid,),
    ).fetchone()
    assert ev is not None and ev["run_id"] == run_id
    assert '"tasks_current": 3' in ev["payload"] and _unit(tid, run_id) in ev["payload"]


def test_open_unknown_foreign_and_fresh_scopes_are_never_stopped(conn, fake_scopes):
    live, stopped = fake_scopes
    running = kb.create_task(conn, title="running", assignee="coder")
    kb.claim_task(conn, running, claimer=kb._claimer_id())
    running_run = kb._current_run_id(conn, running)
    fresh_tid, fresh_run = _closed_run(conn, ended_ago=0)
    foreign_tid, foreign_run = _closed_run(conn)
    conn.execute("UPDATE task_runs SET claim_lock='other-host:1' WHERE id=?", (foreign_run,))
    _, other_run = _closed_run(conn)
    live.update({
        _unit(running, running_run),          # open run
        _unit(fresh_tid, fresh_run),          # closed inside the grace window
        _unit(foreign_tid, foreign_run),      # claimed by another host
        _unit("t_nosuchtask", 987654),        # no task_runs row at all
        _unit("t_mismatch", other_run),       # row exists, task id does not match
        "hermes-worker-kanban-garbage.scope",  # unparseable
    })

    assert kbd.reap_terminal_worker_scopes(conn) == []
    assert stopped == []


def test_failed_stop_records_nothing_and_own_scope_is_skipped(conn, fake_scopes, monkeypatch):
    live, stopped = fake_scopes
    tid, run_id = _closed_run(conn)
    own_tid, own_run = _closed_run(conn)
    live.update({_unit(tid, run_id), _unit(own_tid, own_run)})
    monkeypatch.setattr(kbd, "_in_own_cgroup", lambda unit: unit == _unit(own_tid, own_run))
    monkeypatch.setattr(kbd, "_stop_worker_scope", lambda unit: stopped.append(unit) and False)

    assert kbd.reap_terminal_worker_scopes(conn) == []
    assert stopped == [_unit(tid, run_id)]  # own scope never attempted
    assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind='terminal_worker_reaped'").fetchone()[0] == 0


def test_stops_per_tick_are_bounded(conn, fake_scopes):
    live, _ = fake_scopes
    for _ in range(kbd._WORKER_SCOPE_MAX_STOPS_PER_TICK + 2):
        live.add(_unit(*_closed_run(conn)))
    first = kbd.reap_terminal_worker_scopes(conn)
    assert len(first) == kbd._WORKER_SCOPE_MAX_STOPS_PER_TICK
    assert len(kbd.reap_terminal_worker_scopes(conn)) == 2 and not live


def test_in_own_cgroup_matches_whole_path_segment(monkeypatch, tmp_path):
    cg = tmp_path / "cgroup"
    cg.write_text(encoding="utf-8", data="0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
                  "hermes-worker-kanban-t_ab-run-12.scope\n")
    real_path = kbd.Path
    monkeypatch.setattr(kbd, "Path", lambda p: cg if p == "/proc/self/cgroup" else real_path(p))
    assert kbd._in_own_cgroup("hermes-worker-kanban-t_ab-run-12.scope")
    assert not kbd._in_own_cgroup("hermes-worker-kanban-t_ab-run-1.scope")


def test_unreachable_systemctl_lists_nothing_and_never_raises(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=1)
    monkeypatch.setattr(kbd.subprocess, "run", boom)
    assert kbd._list_worker_scope_units() == []
    assert kbd._worker_scope_tasks_current("x.scope") is None


def test_no_user_systemd_is_a_silent_noop(conn, monkeypatch):
    tid, _ = _closed_run(conn)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert kbd.reap_terminal_worker_scopes(conn) == []
    monkeypatch.setattr(kbd, "_systemctl_user", lambda *a: None)
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)
    assert tid not in result.reaped_terminal_workers


def _user_scope_available() -> bool:
    if not (shutil.which("systemd-run") and shutil.which("systemctl")):
        return False
    from tools.process_registry import systemd_user_bus_env
    probe = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True,
                           env=systemd_user_bus_env(), check=False)
    return probe.returncode in (0, 1) and bool(probe.stdout.strip())


def _scope_active(unit: str) -> bool:
    from tools.process_registry import systemd_user_bus_env
    out = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True,
                         env=systemd_user_bus_env(), check=False)
    return out.stdout.strip() == "active"


@pytest.mark.live_system_guard_bypass  # stops only the two throwaway units this test creates
@pytest.mark.skipif(not _user_scope_available(), reason="needs a reachable user systemd manager")
def test_real_scope_with_background_child_is_stopped_after_main_exits(conn, monkeypatch):
    """End-to-end on a real user manager: the scope's main process exits and
    leaves a backgrounded ``sleep`` behind; a closed run's scope is stopped on
    the next tick, a still-open run's scope is not."""
    from tools.process_registry import systemd_user_bus_env

    tid, run_id = _closed_run(conn)
    live = kb.create_task(conn, title="still running", assignee="coder")
    kb.claim_task(conn, live, claimer=kb._claimer_id())
    live_run = kb._current_run_id(conn, live)
    # Unique task ids keep these test units disjoint from any real worker scope.
    units = {}
    for task_id, rid in ((tid, run_id), (live, live_run)):
        unit = _unit(task_id, rid)
        subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", unit.removesuffix(".scope"),
             "--", "/bin/sh", "-c", "sleep 300 & exit 0"],
            env=systemd_user_bus_env(), check=True, timeout=15,
        )
        units[task_id] = unit
    # Real systemctl list/show/stop, but the tick only ever sees this test's own
    # units — never a live worker scope that happens to share the host.
    real_list = kbd._list_worker_scope_units
    monkeypatch.setattr(kbd, "_list_worker_scope_units",
                        lambda: [u for u in real_list() if u in units.values()])
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not all(_scope_active(u) for u in units.values()):
            time.sleep(0.1)
        assert all(_scope_active(u) for u in units.values()), "main exited but the sleep should keep both scopes alive"

        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

        assert not _scope_active(units[tid])
        assert _scope_active(units[live])
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='terminal_worker_reaped'", (tid,),
        ).fetchone()
        assert ev is not None and units[tid] in ev["payload"]
    finally:
        for unit in units.values():
            subprocess.run(["systemctl", "--user", "stop", unit], env=systemd_user_bus_env(),
                           capture_output=True, check=False)
