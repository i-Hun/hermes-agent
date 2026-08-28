"""Tests for the ACP bridge's durable job runtime and result spill.

Both modules are bridge-local (agent/acp_jobs.py, agent/acp_spill.py) and have
no upstream seam; these tests live with them so an upstream update cannot
silently delete the coverage along with the feature.

The systemd path is exercised for real where it is cheap (status resolution,
env allowlist, registry) and stubbed where a real transient unit would make the
suite depend on a user manager. The one claim that CANNOT be stubbed — that a
job outlives the gateway cgroup — is verified live in the change report, not
here: a unit test that starts and stops units is a flake in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from agent import acp_jobs, acp_spill


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    (tmp_path / "profile").mkdir()
    return tmp_path / "profile"


# --------------------------------------------------------------------------
# acp_jobs — registry
# --------------------------------------------------------------------------


def test_registry_roundtrip_and_recovers_from_corruption(home):
    reg = acp_jobs._Registry()
    assert reg.read() == {"jobs": {}}

    reg.mutate(lambda d: d["jobs"].update({"a": {"label": "x", "started_at": 1}}))
    assert reg.read()["jobs"]["a"]["label"] == "x"

    reg.path.write_text("{ not json", encoding="utf-8")
    assert reg.read() == {"jobs": {}}, "a corrupt registry must not wedge the tool"


def test_env_allowlist_never_forwards_secrets(home, monkeypatch):
    """Unit properties are readable via `systemctl show`, so the child env is
    an allowlist, not a copy of os.environ."""
    monkeypatch.setenv("GITLAB_IMAGES_TOKEN", "glpat-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    env = acp_jobs._child_env({"MY_FLAG": "1"})
    assert "GITLAB_IMAGES_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["MY_FLAG"] == "1"
    assert env["HERMES_HOME"] == str(home)


def test_child_env_drops_kanban_marker(home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    assert "HERMES_KANBAN_TASK" not in acp_jobs._child_env(None)


# --------------------------------------------------------------------------
# acp_jobs — status resolution
# --------------------------------------------------------------------------


def test_status_prefers_rc_file_over_unit(home, monkeypatch):
    """The rc file is the durable record: --collect unloads the unit the moment
    it stops, so asking systemd a second later answers 'not-found'."""
    entry = {"isolation": "systemd-run", "unit": "hermes-job-x.service"}
    monkeypatch.setattr(acp_jobs, "_unit_state", lambda unit: {"LoadState": "not-found"})

    _, rc_path, _, _ = acp_jobs._job_paths("j1")
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    rc_path.write_text("0", encoding="utf-8")
    assert acp_jobs._status_of("j1", entry) == {"status": "done", "exit_code": 0}

    rc_path.write_text("3", encoding="utf-8")
    assert acp_jobs._status_of("j1", entry) == {"status": "failed", "exit_code": 3}


def test_status_stopped_when_unit_gone_without_rc(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "_unit_state", lambda unit: {"LoadState": "not-found",
                                                              "ActiveState": "inactive"})
    out = acp_jobs._status_of("gone", {"isolation": "systemd-run", "unit": "u"})
    assert out["status"] == "stopped"


def test_status_running_while_unit_active(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "_unit_state",
                        lambda unit: {"ActiveState": "active", "LoadState": "loaded"})
    assert acp_jobs._status_of("j", {"isolation": "systemd-run", "unit": "u"})["status"] == "running"


# --------------------------------------------------------------------------
# acp_jobs — guards
# --------------------------------------------------------------------------


def test_job_start_requires_command_and_real_cwd(home):
    assert "error" in json.loads(acp_jobs.job_start({"command": "  "}))
    out = json.loads(acp_jobs.job_start({"command": "true", "cwd": "/no/such/dir"}))
    assert "cwd does not exist" in out["error"]


def test_a_label_with_no_ascii_left_falls_back_instead_of_becoming_dashes(home, monkeypatch):
    """Live regression: a Cyrillic goal produced the job id `delegate---16588518`.

    The id is what the model quotes back, what the unit is named, and what a
    human greps for — punctuation identifies nothing.
    """
    assert acp_jobs._sanitize_label("Ответь одной строкой") == "job"
    assert acp_jobs._sanitize_label("delegate-Ответь") == "delegate"
    assert acp_jobs._sanitize_label("build atc target") == "build-atc-target"
    assert acp_jobs._sanitize_label("") == "job"

    # Stub the spawn, per this file's own rule at the top: a real transient unit
    # must not become a suite dependency. It already had: `scripts/run_tests.sh`
    # blanks the environment, so `systemd-run --user` finds no bus and job_start
    # returns an error instead of an id — this test was red on the CANONICAL
    # runner since 13.08 while staying green under a bare `pytest` that happened
    # to inherit a login session. The claim here is the MINTED ID, not the unit.
    monkeypatch.setattr(acp_jobs, "_systemd_run_available", lambda: True)
    monkeypatch.setattr(
        acp_jobs.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )
    jid = json.loads(acp_jobs.job_start({
        "command": "true", "label": "delegate-Ответь одной строкой",
    }))["job_id"]
    assert jid.startswith("delegate-") and "---" not in jid


def test_job_start_enforces_concurrency_cap(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "MAX_RUNNING", 2)
    monkeypatch.setattr(acp_jobs, "_status_of", lambda jid, e: {"status": "running"})
    reg = acp_jobs._Registry()
    reg.mutate(lambda d: d["jobs"].update({
        "a": {"started_at": time.time()}, "b": {"started_at": time.time()},
    }))
    out = json.loads(acp_jobs.job_start({"command": "true"}))
    assert "too many running jobs (2/2)" in out["error"]
    assert sorted(out["running"]) == ["a", "b"]


def test_job_output_and_kill_reject_unknown_id(home):
    assert "unknown job_id" in json.loads(acp_jobs.job_output({"job_id": "nope"}))["error"]
    assert "unknown job_id" in json.loads(acp_jobs.job_kill({"job_id": "nope"}))["error"]


def test_job_kill_is_a_noop_on_a_finished_job(home, monkeypatch):
    reg = acp_jobs._Registry()
    reg.mutate(lambda d: d["jobs"].update({"j": {"started_at": time.time(),
                                                 "isolation": "systemd-run", "unit": "u"}}))
    monkeypatch.setattr(acp_jobs, "_status_of", lambda jid, e: {"status": "done", "exit_code": 0})
    out = json.loads(acp_jobs.job_kill({"job_id": "j"}))
    assert out["status"] == "done" and "nothing to kill" in out["note"]


# --------------------------------------------------------------------------
# acp_jobs — real end-to-end through the setsid arm (no systemd needed)
# --------------------------------------------------------------------------


def test_setsid_arm_runs_records_exit_code_and_says_it_is_weaker(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "_systemd_run_available", lambda: False)
    out = json.loads(acp_jobs.job_start({
        "command": "echo marker-line; exit 7", "label": "e2e",
    }))
    assert out["status"] == "running"
    assert "WEAKER" in out["note"], "the weak fallback must announce itself"

    jid = out["job_id"]
    for _ in range(100):
        status = json.loads(acp_jobs.job_output({"job_id": jid}))
        if status["status"] != "running":
            break
        time.sleep(0.1)
    assert status["status"] == "failed"
    assert status["exit_code"] == 7
    assert "marker-line" in status["output"]


def test_job_output_tail_flags_truncation(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "_systemd_run_available", lambda: False)
    jid = json.loads(acp_jobs.job_start({
        "command": "for i in $(seq 1 50); do echo line$i; done",
    }))["job_id"]
    for _ in range(100):
        out = json.loads(acp_jobs.job_output({"job_id": jid, "tail": 5}))
        if out["status"] != "running":
            break
        time.sleep(0.1)
    assert out["output"].count("\n") <= 4
    assert "truncated" in out


# --------------------------------------------------------------------------
# acp_jobs — reaper
# --------------------------------------------------------------------------


def test_reaper_drops_old_finished_jobs_but_never_a_running_one(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "TTL_DAYS", 1)
    old = time.time() - 3 * 86400
    statuses = {"finished": {"status": "done", "exit_code": 0},
                "busy": {"status": "running"}}
    monkeypatch.setattr(acp_jobs, "_status_of", lambda jid, e: statuses[jid])

    log, rc, script, cmd = acp_jobs._job_paths("finished")
    for path in (log, rc, script, cmd):
        path.write_text("x", encoding="utf-8")

    data = {"jobs": {"finished": {"started_at": old}, "busy": {"started_at": old}}}
    removed = acp_jobs._reap(data)

    assert removed == 1
    assert list(data["jobs"]) == ["busy"]
    assert not any(p.exists() for p in (log, rc, script, cmd))


def test_reaper_keeps_recent_jobs(home, monkeypatch):
    monkeypatch.setattr(acp_jobs, "_status_of", lambda jid, e: {"status": "done"})
    data = {"jobs": {"fresh": {"started_at": time.time()}}}
    assert acp_jobs._reap(data) == 0
    assert "fresh" in data["jobs"]


# --------------------------------------------------------------------------
# acp_spill
# --------------------------------------------------------------------------


def test_small_result_passes_through_untouched(home):
    assert acp_spill.maybe_spill("web_extract", "short") == "short"


def test_large_result_is_written_and_replaced_by_a_locator(home):
    payload = "line\n" * 20000
    out = acp_spill.maybe_spill("web_extract", payload)

    assert acp_spill.SPILL_TAG in out and acp_spill.SPILL_CLOSING_TAG in out
    assert len(out) < len(payload) / 10, "a locator that is not smaller saves nothing"

    path = Path([ln.split(": ", 1)[1] for ln in out.splitlines()
                 if ln.startswith("Full output saved to: ")][0])
    assert path.read_text(encoding="utf-8") == payload
    assert "line" in out, "the preview must still carry a sample"


def test_control_answers_are_never_spilled(home):
    payload = "x" * 200000
    for name in ("todo", "job_output", "kanban_create"):
        assert acp_spill.maybe_spill(name, payload) == payload


def test_non_string_results_pass_through(home):
    obj = {"a": 1}
    assert acp_spill.maybe_spill("web_extract", obj) is obj


def test_write_failure_truncates_and_says_so(home, monkeypatch):
    def _boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_text", _boom)
    out = acp_spill.maybe_spill("web_extract", "y" * 50000)
    assert "could NOT be saved" in out
    assert "disk full" in out
    assert len(out) < 5000


def test_spill_reaps_old_files(home, monkeypatch):
    monkeypatch.setattr(acp_spill, "TTL_DAYS", 1)
    directory = acp_spill._storage_dir()
    stale = directory / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    os.utime(stale, (time.time() - 5 * 86400,) * 2)

    acp_spill.maybe_spill("web_extract", "z" * 50000)
    assert not stale.exists()


def test_threshold_never_exceeds_our_own_ceiling(home):
    """A tool may register a 100K cap upstream; the bridge's context is the
    scarce resource, so our ceiling wins."""
    assert acp_spill.threshold_for("web_extract") <= acp_spill.DEFAULT_THRESHOLD_CHARS
    assert acp_spill.threshold_for("todo") == float("inf")


# --------------------------------------------------------------------------
# wiring into the MCP server
# --------------------------------------------------------------------------


def test_server_module_registers_the_job_tools():
    from agent.transports import hermes_acp_tools_server as srv

    names = {spec["name"] for spec in srv._JOB_TOOL_SPECS}
    assert names == {"job_start", "job_list", "job_output", "job_kill"}
    assert set(srv._JOB_DISPATCH) == names


def test_background_param_is_redocumented_against_upstreams_deprecated_text():
    """Upstream documents `background` as "DEPRECATED / IGNORED … no effect".

    On this bridge it is the switch that decides whether the work survives the
    turn. A model reads the parameter text, believes it, and never opts in — so
    the amendment has to land on the parameter, not only on the tool's prose.
    """
    from agent.transports import hermes_acp_tools_server as srv

    upstream = {
        "type": "object",
        "properties": {
            "background": {"type": "boolean",
                           "description": "DEPRECATED / IGNORED. No effect."},
            "goal": {"type": "string", "description": "keep me"},
        },
    }
    amended = srv._amend_param_docs("delegate_task", upstream)

    assert "DEPRECATED" not in amended["properties"]["background"]["description"]
    assert "job_output" in amended["properties"]["background"]["description"]
    assert amended["properties"]["goal"]["description"] == "keep me"
    # The registry dict is shared with non-bridge callers in this process.
    assert upstream["properties"]["background"]["description"].startswith("DEPRECATED")


def test_amend_param_docs_leaves_other_tools_alone():
    from agent.transports import hermes_acp_tools_server as srv

    schema = {"type": "object", "properties": {"background": {"description": "x"}}}
    assert srv._amend_param_docs("web_search", schema) is schema


def test_delegate_task_background_goes_to_the_job_runtime(home, monkeypatch):
    from agent.transports import hermes_acp_tools_server as srv

    calls = []

    def _fake(prompt, label, env, hermes_bin):
        calls.append({"prompt": prompt, "label": label})
        return json.dumps({"job_id": "delegate-x-1234", "status": "running"})

    monkeypatch.setattr(acp_jobs, "start_detached_delegation", _fake)

    out = json.loads(srv._dispatch_delegate_task({
        "background": True,
        "tasks": [{"goal": "measure the thing"}],
    }))

    assert out["background"] is True
    assert out["started"][0]["job_id"] == "delegate-x-1234"
    assert out["started"][0]["goal"] == "measure the thing"
    assert len(calls) == 1 and "measure the thing" in calls[0]["prompt"]


def test_delegate_task_stays_synchronous_without_the_flag(home, monkeypatch):
    from agent.transports import hermes_acp_tools_server as srv

    monkeypatch.setattr(srv, "_run_one_delegation",
                        lambda task, depth, timeout: {"ok": True, "goal": task["goal"],
                                                      "output": "done"})
    out = json.loads(srv._dispatch_delegate_task({"goal": "quick question"}))
    assert "background" not in out
    assert out["results"][0]["output"] == "done"


def test_pipefail_is_on_so_a_broken_pipeline_is_not_reported_as_success(home, monkeypatch):
    """Dogfooding this runtime on its own test suite produced rc=0 while pytest
    had refused to start: plain bash reports `failing | tail` as success, and
    for a background job the exit code is the only status the caller gets."""
    monkeypatch.setattr(acp_jobs, "_systemd_run_available", lambda: False)
    jid = json.loads(acp_jobs.job_start({
        "command": "false | tail -1", "label": "pipe",
    }))["job_id"]
    for _ in range(100):
        out = json.loads(acp_jobs.job_output({"job_id": jid}))
        if out["status"] != "running":
            break
        time.sleep(0.1)
    assert out["status"] == "failed", "a failing pipeline stage must fail the job"
