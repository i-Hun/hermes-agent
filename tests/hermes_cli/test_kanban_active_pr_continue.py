"""``active_pr`` must not be a life sentence: an explicit "continue" lifts it.

The guard exists so a re-queued card does not spawn a second worker that opens a
duplicate PR (#111910). It used to recognise only a handoff to a DIFFERENT
profile, plus ``changes_requested`` / ``review_reopened`` — which left a
single-profile board (implementer and reviewer are the same agent) with no way
at all to resume its own open PR: the card sat in ``ready`` behind
``active_pr`` for the whole 24 h window while the operator's
``reassign <id> <same-profile> --reason ...`` was recorded as a no-op.

What these tests pin down:

* ``reassign`` to the SAME profile releases the guard, with or without
  ``--reason``: on a single-profile board the command IS the "continue on this
  PR" signal, and requiring an extra flag left the operator with no working
  wording at all.
* ``unblock --reason`` carries the same meaning; a silent unblock (the cron
  alarms and the failure breaker call it that way) does not.
* What the guard still protects: implicit re-queues — a silent unblock, an
  unassign, the dispatcher's own ``default_assignee`` fill-in.
* The native review lane never takes the guard, reassignment or not.
* A status comment that merely re-quotes the PR already published (the
  worker's "pushed to #817", the operator's note) does not re-arm the guard —
  previously it moved the watermark forward and froze the card again after
  every handoff. A genuinely new PR URL does re-arm it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_817 = "Opened https://github.com/example/repo/pull/817 for review."
PR_818 = "Superseded by https://github.com/example/repo/pull/818"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return home


@pytest.fixture
def spawnable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every named profile exists and the review lane is on: the only thing
    holding a card back in these tests is the respawn guard itself."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}})


def _backdate_comments(conn, tid: str, seconds: int = 60) -> None:
    """Second granularity: put the PR comment strictly before the events that
    follow it, so "handoff after the comment" is deterministic."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, tid),
        )


def _fail_twice(conn, tid: str) -> None:
    """Two ended runs with outcome ``timed_out`` — the budget wall that parks a
    card without stamping a breaker error."""
    now = int(time.time())
    with kb.write_txn(conn):
        for i in (1, 2):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'dev', 'timed_out', 'timed_out', ?, ?)",
                (tid, now - 900 + i * 60, now - 840 + i * 60),
            )


def _card_with_open_pr(conn, *, assignee: str = "dev") -> str:
    tid = kb.create_task(conn, title="own pr", assignee=assignee)
    kb.add_comment(conn, tid, author=assignee, body=PR_817)
    _backdate_comments(conn, tid)
    _fail_twice(conn, tid)
    assert kb.block_task(conn, tid, reason="budget 400/400")
    return tid


def test_same_profile_reassign_with_reason_lifts_active_pr_and_spawns(
    kanban_home: Path, spawnable: None,
) -> None:
    """The reported freeze, end to end: PR open → timed out twice → blocked →
    unblock → operator ``reassign same-profile --reason`` → the worker runs."""
    with kbc.connect() as conn:
        tid = _card_with_open_pr(conn)
        assert kb.unblock_task(conn, tid) is True
        ready = kb.get_task(conn, tid)
        assert ready is not None and ready.status == "ready"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        res = kbd.dispatch_once(conn, dry_run=True)
        assert dict(res.respawn_guarded).get(tid) == "active_pr"
        assert tid not in [s[0] for s in res.spawned]

        assert kb.reassign_task(
            conn, tid, "dev", reason="continue the open PR #817 after the budget wall",
        ) is True
        assert kbd.check_respawn_guard(conn, tid) is None

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]
        assert tid not in dict(res.respawn_guarded)


def test_same_profile_reassign_without_a_reason_lifts_active_pr_and_spawns(
    kanban_home: Path, spawnable: None,
) -> None:
    """The reported freeze with the command the operator actually types.

    PR #817 open, two 400/400 timeouts, blocked, alarm-unblock -> the card sat in
    ``ready`` behind ``active_pr``; ``reassign <id> <same-profile>`` was recorded
    as ``{from: dev, to: dev}`` and ignored, so the only ways out were a decoy
    profile round-trip or a fake ``request-changes`` (which corrupts the
    review-round statistics). Here the plain reassign must release the guard and
    the worker must start on the very next dispatcher tick.
    """
    with kbc.connect() as conn:
        tid = _card_with_open_pr(conn)
        assert kb.unblock_task(conn, tid) is True
        ready = kb.get_task(conn, tid)
        assert ready is not None and ready.status == "ready"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        res = kbd.dispatch_once(conn, dry_run=True)
        assert dict(res.respawn_guarded).get(tid) == "active_pr"
        assert tid not in [s[0] for s in res.spawned]

        assert kb.reassign_task(conn, tid, "dev") is True
        after = kb.get_task(conn, tid)
        assert after is not None and after.status == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]
        assert tid not in dict(res.respawn_guarded)


def test_a_silent_unblock_and_an_unassign_still_guard_the_implementer(
    kanban_home: Path, spawnable: None,
) -> None:
    """The operator's assignment is the signal; bookkeeping is not.

    An alarm's unblock (no reason) and an unassign both move the card without
    asking for it to run, so the implementer must not be re-spawned against the
    PR it opened."""
    with kbc.connect() as conn:
        silent = _card_with_open_pr(conn)
        assert kb.unblock_task(conn, silent) is True
        assert kbd.check_respawn_guard(conn, silent) == "active_pr"

        unassigned = _card_with_open_pr(conn)
        assert kb.unblock_task(conn, unassigned) is True
        assert kb.reassign_task(conn, unassigned, None) is True
        assert kbd.check_respawn_guard(conn, unassigned) == "active_pr"


def test_the_review_lane_is_untouched_by_the_reassign_rule(
    kanban_home: Path, spawnable: None,
) -> None:
    """A review card quotes the PR it reviews by construction: the guard never
    applies to its lane, whether or not anyone reassigned it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        kb.add_comment(conn, tid, author="worker", body=PR_817)
        assert kb.request_review(
            conn, tid, summary="PR ready", expected_run_id=claimed.current_run_id,
        )
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        assert kb.reassign_task(conn, tid, "reviewer") is True
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]


def test_operator_unblock_with_reason_lifts_active_pr(
    kanban_home: Path, spawnable: None,
) -> None:
    """``unblock --reason`` is the same "continue" signal as ``reassign
    --reason``; the silent unblock every alarm and the breaker use is not."""
    with kbc.connect() as conn:
        silent = _card_with_open_pr(conn)
        assert kb.unblock_task(conn, silent) is True
        assert kbd.check_respawn_guard(conn, silent) == "active_pr"

        spoken = _card_with_open_pr(conn)
        assert kb.unblock_task(conn, spoken, reason="continue on the open PR") is True
        assert kbd.check_respawn_guard(conn, spoken) is None


def test_status_comment_quoting_the_same_pr_does_not_rearm(
    kanban_home: Path, spawnable: None,
) -> None:
    """A comment that repeats the PR already published is not a new
    publication: after a handoff the card must keep running."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="closer owns pr", assignee="dev")
        kb.add_comment(conn, tid, author="dev", body=PR_817)
        _backdate_comments(conn, tid)
        assert kb.assign_task(conn, tid, "closer") is True
        assert kbd.check_respawn_guard(conn, tid) is None

        kb.add_comment(conn, tid, author="closer", body="Pushed a fixup to " + PR_817.split()[-1])
        assert kbd.check_respawn_guard(conn, tid) is None


def test_a_second_pr_url_after_the_handoff_guards_again(
    kanban_home: Path, spawnable: None,
) -> None:
    """Fail closed on real ambiguity: a PR URL first mentioned AFTER the
    handoff may be a duplicate implementation, so the guard returns."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="second pr", assignee="dev")
        kb.add_comment(conn, tid, author="dev", body=PR_817)
        _backdate_comments(conn, tid)
        assert kb.assign_task(conn, tid, "closer") is True
        assert kbd.check_respawn_guard(conn, tid) is None

        kb.add_comment(conn, tid, author="closer", body=PR_818)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
