"""Wall-clock cap per run + priority preemption in the kanban dispatcher.

Two failures found on the live board 17.10.2026:

* A card that names no ``max_runtime_seconds`` ran FOREVER: a live heartbeat kept
  its dispatch slot, so one priority-25 card held one of four slots for 6 h 12 min.
  The cap is now the card's own value, else a class default (90 min; 150 min for a
  card whose ``completion_contract`` publishes a PR and honestly waits on CI), and
  the ``timed_out`` event names the rule that fired via ``limit_source``.
* The dispatcher reclaimed wedged work but never merely LOWER-priority work, so a
  priority-96 card (the CI fix the whole queue depended on) could not start while a
  priority-25 card held the slot. ``preempt_lower_priority_running`` reclaims a run
  once it has lasted ``min_runtime_seconds`` AND a ready card sits at least
  ``margin`` priority points above it — then the lane order hands it the slot.

The acceptance repro for the second half is
``test_preemption_hands_the_slot_to_the_high_priority_card``: priority 10 holds the
slot for 100 min while priority 90 waits; after the change the dispatcher reclaims
the low card, writes the reason into the event log and spawns the high one.
"""

from __future__ import annotations

import os
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated board in a throwaway HERMES_HOME (config falls back to defaults)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def caps(monkeypatch):
    """Pin the class defaults so the host's own config can never move them."""
    monkeypatch.setattr(kb, "max_runtime_defaults", lambda: (5400, 9000))
    return 5400, 9000


@pytest.fixture
def dead_worker(monkeypatch):
    """Every liveness probe answers "gone": a SIGTERM is a successful reclaim."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    return sent


FAKE_PID = 4_000_001  # never a live process: a tick-level test must not signal itself


def _running(
    conn, *, priority: int = 0, contract: str | None = None, max_runtime: int | None = None,
    age_seconds: int = 3600, title: str | None = None, pid: int | None = None,
) -> str:
    """A card claimed and running since ``age_seconds`` ago (fresh heartbeat)."""
    tid = kb.create_task(
        conn, title=title or f"card p{priority}", assignee="default", priority=priority,
        completion_contract=contract, max_runtime_seconds=max_runtime,
    )
    assert kb.claim_task(conn, tid) is not None
    kbd._set_worker_pid(conn, tid, pid or os.getpid())
    now = int(time.time())
    old = now - age_seconds
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = ?, claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
            (old, now + 3600, now, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


@pytest.fixture
def tick(monkeypatch):
    """Tick-level preemption: the worker reads as gone, so the SIGTERM "works", and
    the crash sweep is neutralised — it reclaims dead PIDs first and would hide the
    path under test (it has its own tests)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(kbd, "detect_crashed_workers", lambda conn, board=None: [])


def _ready(conn, *, priority: int, assignee: str = "default", title: str | None = None) -> str:
    return kb.create_task(
        conn, title=title or f"ready p{priority}", assignee=assignee, priority=priority,
    )


def _events(conn, tid: str, kind: str) -> list:
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _runs(conn, tid: str) -> list:
    return kb.list_runs(conn, tid)


# ---------------------------------------------------------------------------
# Part 1 — a card that names no cap is still bounded
# ---------------------------------------------------------------------------


def test_default_cap_closes_a_run_that_named_no_cap(board, caps, dead_worker):
    """90 min default: the card that held a slot for 6 h is closed, not kept alive."""
    tid = _running(board, max_runtime=None, age_seconds=91 * 60)

    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: dead_worker.append((p, s))) == [tid]

    assert kb.get_task(board, tid).status == "ready"
    event = _events(board, tid, "timed_out")[-1]
    assert event.payload["limit_seconds"] == 5400
    assert event.payload["limit_source"] == "default"
    assert event.payload["elapsed_seconds"] >= 91 * 60
    assert _runs(board, tid)[-1].outcome == "timed_out"


def test_ci_waiting_card_gets_the_longer_default(board, caps, dead_worker):
    """A PR contract legitimately waits on CI: 150 min, and only past that."""
    tid = _running(board, contract="i-Hun/recsys-2026", age_seconds=100 * 60)
    def signal(pid, sig):
        dead_worker.append((pid, sig))

    assert kbd.enforce_max_runtime(board, signal_fn=signal) == []
    assert kb.get_task(board, tid).status == "running"

    with kb.write_txn(board):
        board.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (int(time.time()) - 151 * 60, tid),
        )

    assert kbd.enforce_max_runtime(board, signal_fn=signal) == [tid]
    event = _events(board, tid, "timed_out")[-1]
    assert event.payload["limit_seconds"] == 9000
    assert event.payload["limit_source"] == "default_ci"


def test_explicit_card_cap_wins_over_the_default(board, caps, dead_worker):
    tid = _running(board, max_runtime=30, age_seconds=60)

    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: None) == [tid]
    event = _events(board, tid, "timed_out")[-1]
    assert event.payload["limit_seconds"] == 30
    assert event.payload["limit_source"] == "task"


def test_run_cap_is_the_one_recorded_at_claim_time(board, caps, monkeypatch, dead_worker):
    """A config edit mid-run must not change the rule an attempt was opened with."""
    tid = _running(board, age_seconds=120)
    assert _runs(board, tid)[-1].max_runtime_seconds == 5400
    # Operator now lowers the default to 30 s: the running attempt keeps its 90 min.
    monkeypatch.setattr(kb, "max_runtime_defaults", lambda: (30, 30))

    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, tid).status == "running"


def test_operator_lowered_cap_applies_to_the_running_attempt(board, caps, dead_worker):
    """Lowering the cap on a runaway card is an operator rescue: it acts NOW, not next run."""
    tid = _running(board, age_seconds=120)  # opened with the 5400 s default
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET max_runtime_seconds = 60 WHERE id = ?", (tid,))

    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: None) == [tid]
    event = _events(board, tid, "timed_out")[-1]
    assert event.payload["limit_seconds"] == 60
    assert event.payload["limit_source"] == "task"


def test_explicit_zero_falls_back_to_the_default(board, caps, dead_worker):
    """0 on the card is not "unbounded": a card may not opt out of the slot budget."""
    tid = _running(board, max_runtime=0, age_seconds=91 * 60)

    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: None) == [tid]
    event = _events(board, tid, "timed_out")[-1]
    assert event.payload["limit_seconds"] == 5400
    assert event.payload["limit_source"] == "default"


def test_default_cap_can_be_disabled_by_config(board, monkeypatch, dead_worker):
    """``kanban.default_max_runtime_seconds: 0`` = no default (explicit caps still apply)."""
    monkeypatch.setattr(kb, "max_runtime_defaults", lambda: (None, None))
    tid = _running(board, age_seconds=10 * 3600)

    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, tid).status == "running"


def test_claim_stores_the_effective_cap_on_the_run(board, caps):
    """``hermes kanban runs`` shows the wall-clock rule the attempt ran under."""
    tid = kb.create_task(board, title="pr card", assignee="default",
                         completion_contract="https://github.com/i-Hun/recsys-2026/pull/7")
    assert kb.claim_task(board, tid) is not None
    assert _runs(board, tid)[-1].max_runtime_seconds == 9000

    plain = kb.create_task(board, title="local card", assignee="default")
    assert kb.claim_task(board, plain) is not None
    assert _runs(board, plain)[-1].max_runtime_seconds == 5400

    explicit = kb.create_task(board, title="capped", assignee="default", max_runtime_seconds=45)
    assert kb.claim_task(board, explicit) is not None
    assert _runs(board, explicit)[-1].max_runtime_seconds == 45


# ---------------------------------------------------------------------------
# Part 2 — preemption by priority
# ---------------------------------------------------------------------------


def test_preemption_reclaims_the_long_low_priority_run(board, caps, dead_worker):
    low = _running(board, priority=10, age_seconds=100 * 60)
    high = _ready(board, priority=90)

    preempted = kbd.preempt_lower_priority_running(
        board, signal_fn=lambda p, s: dead_worker.append((p, s)),
    )

    assert preempted == [low]
    # Nothing is lost: the card returns to ready with its workspace intact.
    assert kb.get_task(board, low).status == "ready"
    assert kb.get_task(board, low).worker_pid is None
    # The reason is on the board, not in the dispatcher's stdout.
    event = _events(board, low, "preempted")[-1]
    assert event.payload["reason"] == "higher_priority_ready"
    assert event.payload["preempted_by"] == high
    assert event.payload["candidate_priority"] == 90
    assert event.payload["running_priority"] == 10
    assert event.payload["elapsed_seconds"] >= 100 * 60
    assert _runs(board, low)[-1].outcome == "preempted"
    # A priority policy is not a failure: the breaker must not walk the card up.
    assert kb.get_task(board, low).consecutive_failures == 0
    assert _events(board, low, "gave_up") == []


def test_preemption_hands_the_slot_to_the_high_priority_card(board, caps, tick):
    """ACCEPTANCE REPRO: prio 10 held the ONLY slot 100 min while prio 90 waited."""
    low = _running(board, priority=10, age_seconds=100 * 60, title="owner panel",
                   pid=FAKE_PID, contract="i-Hun/recsys-2026")
    high = _ready(board, priority=90, title="CI fix")
    spawned = []

    def _spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    result = kbd.dispatch_once(board, spawn_fn=_spawn, max_spawn=1, max_in_progress=8)

    # The dispatcher reclaimed the low card, wrote the reason, and spawned the high one.
    assert result.preempted == [low]
    assert low not in [s[0] for s in result.spawned]
    assert result.spawned and result.spawned[0][0] == high
    assert spawned == [high]
    assert _events(board, low, "preempted")[-1].payload["preempted_by"] == high


def test_no_preemption_while_a_slot_is_free(board, caps, tick):
    """A free slot means nothing has to be taken away: spawn the high card, keep the run."""
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    high = _ready(board, priority=90)

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242, max_spawn=2)

    assert result.preempted == []
    assert [s[0] for s in result.spawned] == [high]
    assert kb.get_task(board, low).status == "running"
    assert _events(board, low, "preempted") == []


def test_one_high_card_clears_at_most_one_low_run(board, caps, tick):
    """One waiting card needs one slot: it must not empty the board."""
    low_a = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                     contract="i-Hun/recsys-2026")
    low_b = _running(board, priority=11, age_seconds=100 * 60, pid=FAKE_PID + 1,
                     contract="i-Hun/recsys-2026")
    _ready(board, priority=95)

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242, max_spawn=2)

    assert len(result.preempted) == 1
    assert result.preempted == [low_a]  # the lowest-priority offender goes first
    assert kb.get_task(board, low_b).status == "running"


def test_no_preemption_when_a_review_handoff_would_take_the_slot(board, caps, tick, monkeypatch):
    """The freed slot would be reserved for the review; the named card never runs."""
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    _ready(board, priority=95)
    review = kb.create_task(board, title="review handoff", assignee="default")
    kb.request_review(board, review, summary="ready for review")

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242, max_spawn=1)

    assert result.preempted == []
    assert kb.get_task(board, low).status == "running"


def test_capped_candidate_costs_no_preemption_in_a_tick(board, caps, tick):
    """Profile already at its cap: the lane would skip the candidate, so the tick
    must not kill a run (and then re-spawn that same run) for it."""
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    _ready(board, priority=95)

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242,
                               max_spawn=1, max_in_progress_per_profile=1)

    assert result.preempted == []
    assert result.spawned == []
    assert kb.get_task(board, low).status == "running"


def test_uncapped_board_never_preempts(board, caps, tick):
    """No configured cap = no scarcity to arbitrate, however long the run is."""
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    high = _ready(board, priority=95)

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242)

    assert result.preempted == []
    assert [s[0] for s in result.spawned] == [high]
    assert kb.get_task(board, low).status == "running"


def test_memory_pressure_blocks_preemption(board, caps, tick, monkeypatch):
    """Under critical memory pressure nothing may start, so taking a slot is pointless."""
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda sample=None: "critical")
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    _ready(board, priority=95)

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242, max_spawn=1)

    assert result.preempted == []
    assert kb.get_task(board, low).status == "running"


def test_freed_slot_goes_to_the_capped_candidates_profile(board, caps, tick, monkeypatch):
    """With per-profile caps, a preemption is only right when the CANDIDATE's profile
    has room: then the freed slot must go to it — not to the card just preempted."""
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: name in {"default", "spare"}))
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    high = _ready(board, priority=95, assignee="spare")

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242,
                               max_spawn=1, max_in_progress_per_profile=1)

    assert result.preempted == [low]
    assert [s[0] for s in result.spawned] == [high]
    assert kb.get_task(board, low).status == "ready"  # re-queued, not re-spawned


def test_dry_run_never_preempts(board, caps, tick):
    """``dispatch --dry-run`` is a preview: it must not SIGTERM a worker."""
    low = _running(board, priority=10, age_seconds=100 * 60, pid=FAKE_PID,
                   contract="i-Hun/recsys-2026")
    _ready(board, priority=95)

    result = kbd.dispatch_once(board, spawn_fn=lambda t, w, board=None: 4242,
                               max_spawn=1, dry_run=True)

    assert result.preempted == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_respects_the_priority_margin(board, caps, dead_worker):
    """+15 is not +20: a merely slightly-more-important card does not interrupt."""
    low = _running(board, priority=10, age_seconds=3 * 3600)
    _ready(board, priority=25)

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_waits_out_the_minimum_runtime(board, caps, dead_worker):
    """A 30-minute run is nowhere near wedged: preemption leaves it alone."""
    low = _running(board, priority=10, age_seconds=30 * 60)
    _ready(board, priority=90)

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_skips_a_candidate_that_cannot_spawn(board, caps, dead_worker):
    """Preempting for a card that will not run destroys work and frees nothing."""
    low = _running(board, priority=10, age_seconds=3 * 3600)
    _ready(board, priority=90, assignee="ghost-profile-does-not-exist")

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_skips_an_unassigned_candidate(board, caps, dead_worker):
    """An unassigned card only spawns through ``kanban.default_assignee``; without
    proof it will run, it must not cost the running card its slot."""
    low = _running(board, priority=10, age_seconds=3 * 3600)
    kb.create_task(board, title="orphan high", priority=95)  # no assignee at all

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_skips_a_candidate_at_its_per_profile_cap(board, caps, dead_worker):
    """A profile already at ``max_in_progress_per_profile`` cannot take the slot, so
    preempting would kill a run and re-spawn it in the same tick."""
    low = _running(board, priority=10, age_seconds=3 * 3600)
    _ready(board, priority=95)

    assert kbd.preempt_lower_priority_running(
        board, signal_fn=lambda p, s: None,
        per_profile_cap=1, per_profile_running={"default": 1},
    ) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_skips_a_foreign_claim(board, caps, dead_worker):
    """Another host's claim is never released: its PID means nothing here."""
    low = _running(board, priority=10, age_seconds=3 * 3600)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET claim_lock = 'otherhost:999' WHERE id = ?", (low,))
    _ready(board, priority=95)

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_skips_a_respawn_guarded_candidate(board, caps, dead_worker):
    """A card the guard holds back this tick must not cost another card its slot."""
    low = _running(board, priority=10, age_seconds=3 * 3600)
    guarded = kb.create_task(board, title="just finished", assignee="default", priority=90)
    assert kb.claim_task(board, guarded) is not None
    kb.complete_task(board, guarded, summary="done")
    # Back in the ready lane (a requeue), but its completion is inside the guard
    # window: the ready loop would skip it, so preemption must skip it too.
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (guarded,))
    assert kbd.check_respawn_guard(board, guarded, lane="ready") == "recent_success"

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_margin_zero_turns_it_off(board, caps, dead_worker, monkeypatch):
    monkeypatch.setattr(kbd, "preemption_policy", lambda: (0, 5400))
    low = _running(board, priority=10, age_seconds=5 * 3600)
    _ready(board, priority=95)

    assert kbd.preempt_lower_priority_running(board, signal_fn=lambda p, s: None) == []
    assert kb.get_task(board, low).status == "running"


def test_preemption_policy_reads_config(board, monkeypatch):
    """Operator knobs: margin and minimum runtime come from ``kanban.*``."""
    cfg = {"kanban": {"preempt_priority_margin": 5, "preempt_min_runtime_seconds": 600}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    assert kbd.preemption_policy() == (5, 600)

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"kanban": {}})
    assert kbd.preemption_policy() == (kbd.PREEMPT_PRIORITY_MARGIN, kbd.PREEMPT_MIN_RUNTIME_SECONDS)


# ---------------------------------------------------------------------------
# Part 3 — visibility: the slot age is on the board
# ---------------------------------------------------------------------------


def test_list_shows_slot_hold_time_and_cap(board, caps):
    from hermes_cli import kanban as kc

    _running(board, priority=25, age_seconds=100 * 60, title="owner panel")
    text = kc.run_slash("list")

    assert "slot 1h40m/1h30m" in text


def test_row_renders_no_cap_when_both_rules_are_disabled():
    """Both the card field and the class default may be off; say so, don't print 0."""
    from hermes_cli.kanban_output import _fmt_task_line

    class _Row:  # the two fields ``_fmt_task_line`` reads besides the slot numbers
        id, status, assignee, tenant, title = "t_x", "running", "default", None, "uncapped"

    assert "[slot 1h40m/no cap]" in _fmt_task_line(_Row(), slot_seconds=6000, limit_seconds=None)
    assert "[slot 1h40m/1h30m]" in _fmt_task_line(_Row(), slot_seconds=6000, limit_seconds=5400)


def test_list_leaves_non_running_rows_alone(board, caps):
    from hermes_cli import kanban as kc

    _ready(board, priority=5, title="waiting")

    text = kc.run_slash("list")
    assert "slot " not in text
