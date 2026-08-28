"""Durable background jobs for the ACP bridge.

WHY THIS EXISTS
---------------
The bridge is one engine process per API call: ``copilot_acp_client`` opens
``claude-agent-acp``, streams one turn, and closes it. Everything the engine
started in the background — its own ``Task`` subagents, its ``Bash
run_in_background`` shells — dies with that process. Measured over 168h on this
host: 39 subagents killed mid-flight, 33 of them in one profile.

``setsid nohup`` does NOT fix it, and that is the trap this module exists to
close. A detached process keeps the *parent's cgroup*, so it stays inside
``hermes-gateway*.service``; every gateway unit here carries
``ExecStopPost=… gateway.cgroup_cleanup``, which SIGKILLs whatever is left in
that cgroup. A job that outlives the turn still dies at the next restart —
observed 2026-08-13, a detached helper killed in the window between ``stop``
and the file edit it was supposed to make.

The only escape is a **transient systemd unit**: ``systemd-run --user`` places
the job under ``user@.service/app.slice/<unit>``, a *sibling* of the gateway,
not a child. Verified on this host:

    gateway : /user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service
    job     : /user.slice/user-1000.slice/user@1000.service/app.slice/hermes-job-*.service

WHAT THIS IS NOT
----------------
It does not make the engine's own ``Task``/``Bash`` background work survive —
those live inside the engine's process and are out of our reach. This is a
second, parallel mechanism the model must choose deliberately (``job_start``).
The banner about killed background work stays correct for the engine's tools.

Update-resilience: new local module, zero upstream seams. Wired only from
``agent/transports/hermes_acp_tools_server.py``, which is itself untracked.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --- knobs -------------------------------------------------------------------

MAX_RUNNING = int(os.environ.get("HERMES_ACP_JOB_MAX_RUNNING", "8") or 8)
TTL_DAYS = float(os.environ.get("HERMES_ACP_JOB_TTL_DAYS", "7") or 7)
DEFAULT_MEMORY_MAX = os.environ.get("HERMES_ACP_JOB_MEMORY_MAX", "16G").strip()
LOG_TAIL_CAP = int(os.environ.get("HERMES_ACP_JOB_TAIL_CAP", "20000") or 20000)

# Environment passed to a job. A transient unit's properties are world-readable
# via `systemctl --user show`, so the whole os.environ (tokens included) must
# NOT be forwarded. Callers can add their own via the `env` argument.
_ENV_ALLOWLIST = (
    "HERMES_HOME", "HOME", "PATH", "LANG", "LC_ALL", "TZ",
    "XDG_RUNTIME_DIR", "SHELL", "USER",
)

_UNSAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")


def _state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    path = Path(home) / "state" / "acp-jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _registry_path() -> Path:
    return _state_dir() / "jobs.json"


def _sanitize_label(raw: str) -> str:
    """A label the unit name can carry — and that a human can still read.

    Substituting every unsafe character one-for-one is fine for ASCII and
    useless for anything else: a Cyrillic goal became ``delegate---16588518``
    (observed live, 2026-08-13), a job id that identifies nothing. Unsafe runs
    collapse to a single dash, and a label left with no word characters at all
    is dropped rather than kept as punctuation.
    """
    label = _UNSAFE_LABEL.sub("-", raw).strip("-")[:40].strip("-")
    return label if re.search(r"[A-Za-z0-9]", label) else "job"


def _profile_tag() -> str:
    home = os.environ.get("HERMES_HOME") or ""
    name = Path(home).name if home else ""
    if name and name != ".hermes":
        return _UNSAFE_LABEL.sub("-", name)[:24] or "default"
    return "default"


# --- registry (flock-guarded, one file, whole-file rewrite) ------------------


class _Registry:
    """Tiny JSON registry under an advisory lock.

    Whole-file rewrite is deliberate: the file holds tens of entries, not
    thousands, and a rewrite under flock cannot half-apply the way an in-place
    edit can.
    """

    def __init__(self) -> None:
        self.path = _registry_path()
        self.lock_path = self.path.with_suffix(".lock")

    def _load_unlocked(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {"jobs": {}}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("job registry unreadable (%s) — starting a fresh one", exc)
            return {"jobs": {}}
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
            return {"jobs": {}}
        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        tmp.replace(self.path)

    def _lock(self):
        import fcntl

        fh = self.lock_path.open("a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return fh

    def read(self) -> dict[str, Any]:
        fh = self._lock()
        try:
            return self._load_unlocked()
        finally:
            fh.close()

    def mutate(self, fn) -> Any:
        """Run ``fn(data)`` under the lock and persist whatever it leaves."""
        fh = self._lock()
        try:
            data = self._load_unlocked()
            out = fn(data)
            self._save_unlocked(data)
            return out
        finally:
            fh.close()


# --- systemd plumbing --------------------------------------------------------


def _systemd_run_available() -> bool:
    if not shutil.which("systemd-run"):
        return False
    # A user manager needs a runtime dir; without it systemd-run --user fails
    # with a confusing "Failed to connect to bus".
    return bool(os.environ.get("XDG_RUNTIME_DIR")) or Path(
        "/run/user/%d" % os.getuid()  # windows-footgun: ok — systemd-run is Linux-only
    ).exists()


def _child_env(extra: Optional[dict[str, Any]]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in _ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val:
            env[key] = val
    env.setdefault(
        "PATH",
        env.get("HOME", str(Path.home())) + "/.local/bin:/usr/local/bin:/usr/bin:/bin",
    )
    env.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())  # windows-footgun: ok — systemd-run --user is Linux-only
    # A job must never be mistaken for a Kanban worker or inherit our delegate
    # depth accounting.
    env.pop("HERMES_KANBAN_TASK", None)
    if isinstance(extra, dict):
        for key, val in extra.items():
            if not isinstance(key, str) or not key:
                continue
            env[key] = str(val)
    return env


def _unit_name(job_id: str) -> str:
    return "hermes-job-%s-%s" % (_profile_tag(), job_id)


def _unit_state(unit: str) -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "show", unit,
             "-p", "ActiveState", "-p", "SubState", "-p", "LoadState",
             "-p", "Result", "-p", "ExecMainStatus"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            out[key] = val
    return out


# --- job lifecycle -----------------------------------------------------------


def _job_paths(job_id: str) -> tuple[Path, Path, Path, Path]:
    base = _state_dir()
    return (base / ("%s.log" % job_id), base / ("%s.rc" % job_id),
            base / ("%s.sh" % job_id), base / ("%s.cmd" % job_id))


def _wrapper_script(cmd_path: Path, rc_path: Path) -> str:
    """Runner that records the exit code where systemd cannot lose it.

    ``--collect`` unloads a transient unit as soon as it stops, so ``systemctl
    show`` is useless a second later. The rc file is the durable record; the
    unit is only consulted while the job is still alive.

    The command runs as a SEPARATE script, not interpolated into this one.
    Inlining it looked simpler and was wrong twice over: a command ending in
    ``exit`` terminated the wrapper before it could record anything (caught by
    the test, not by review), and the command text had to survive being pasted
    into another script's syntax.
    """
    return (
        "#!/bin/bash\n"
        "printf '=== hermes job start %%s ===\\n' \"$(date -Is)\"\n"
        "/bin/bash %s\n"
        "rc=$?\n"
        "printf '%%s' \"$rc\" > %s\n"
        "printf '=== hermes job end %%s rc=%%s ===\\n' \"$(date -Is)\" \"$rc\"\n"
        "exit $rc\n"
    ) % (shlex.quote(str(cmd_path)), shlex.quote(str(rc_path)))


def _reap(data: dict[str, Any]) -> int:
    """Drop finished jobs older than TTL and their artefacts."""
    cutoff = time.time() - TTL_DAYS * 86400
    removed = 0
    for job_id in list(data["jobs"].keys()):
        entry = data["jobs"][job_id]
        started = float(entry.get("started_at") or 0)
        if started > cutoff:
            continue
        status = _status_of(job_id, entry)["status"]
        if status == "running":
            continue
        for path in _job_paths(job_id):
            try:
                path.unlink()
            except OSError:
                pass
        data["jobs"].pop(job_id, None)
        removed += 1
    return removed


def _status_of(job_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Resolve a job's status from the rc file first, the unit second."""
    _, rc_path, _, _ = _job_paths(job_id)
    try:
        rc_text = rc_path.read_text(encoding="utf-8").strip()
    except OSError:
        rc_text = ""
    if rc_text:
        try:
            rc = int(rc_text)
        except ValueError:
            rc = -1
        return {
            "status": "done" if rc == 0 else "failed",
            "exit_code": rc,
        }

    if entry.get("isolation") == "systemd-run":
        state = _unit_state(entry.get("unit") or "")
        active = state.get("ActiveState", "")
        if active in ("active", "activating", "deactivating", "reloading"):
            return {"status": "running", "unit_state": active}
        if state.get("LoadState") == "not-found" or active in ("inactive", "failed", ""):
            # No rc file and the unit is gone: the job was stopped or killed
            # before its wrapper could record anything.
            return {"status": "stopped", "unit_state": active or "gone"}
        return {"status": "unknown", "unit_state": active}

    pid = entry.get("pid")
    if isinstance(pid, int) and pid > 0:
        return {"status": "running" if _pid_alive(pid) else "stopped"}
    return {"status": "unknown"}


def _pid_alive(pid: int) -> bool:
    """True only if the pid is a live process, not a zombie.

    ``os.kill(pid, 0)`` succeeds on a zombie, and in the setsid arm the wrapper
    IS our direct child — so a finished job would have reported itself running
    forever if the rc file were ever missing.
    """
    try:
        stat = Path("/proc/%d/stat" % pid).read_text(encoding="utf-8", errors="replace")
    except OSError:
        try:
            os.kill(pid, 0)  # windows-footgun: ok — liveness probe for a systemd transient unit (Linux-only)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False
    # "pid (comm) state ..." — comm may contain spaces and parens.
    tail = stat.rpartition(")")[2].strip()
    return bool(tail) and tail.split()[0] != "Z"


def _public(job_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    out = {
        "job_id": job_id,
        "label": entry.get("label"),
        "command": (entry.get("command") or "")[:300],
        "started_at": entry.get("started_at_iso"),
        "isolation": entry.get("isolation"),
        "log": entry.get("log"),
    }
    out.update(_status_of(job_id, entry))
    return out


def job_start(args: dict[str, Any]) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        return json.dumps({"error": "command is required"})

    label = _sanitize_label(str(args.get("label") or "job"))
    cwd = str(args.get("cwd") or "").strip() or str(Path.home())
    if not Path(cwd).is_dir():
        return json.dumps({"error": "cwd does not exist: %s" % cwd})
    memory_max = str(args.get("memory_max") or DEFAULT_MEMORY_MAX).strip()

    registry = _Registry()
    running = [
        jid for jid, e in registry.read()["jobs"].items()
        if _status_of(jid, e)["status"] == "running"
    ]
    if len(running) >= MAX_RUNNING:
        return json.dumps({
            "error": "too many running jobs (%d/%d)" % (len(running), MAX_RUNNING),
            "running": running,
            "hint": "job_kill one of them, or wait — this cap is HERMES_ACP_JOB_MAX_RUNNING",
        })

    job_id = "%s-%s" % (label, uuid.uuid4().hex[:8])
    log_path, rc_path, script_path, cmd_path = _job_paths(job_id)
    # pipefail, deliberately: for a background job the exit code is the ONLY
    # status signal, and plain bash reports `pytest … | tail` as success
    # whenever `tail` succeeds. Caught by dogfooding this runtime on its own
    # test suite — the job said rc=0 while pytest had refused to start. The
    # trade is a SIGPIPE-shaped false red on `… | head`, which is rarer and
    # visible in the log.
    cmd_path.write_text("set -o pipefail\n" + command + "\n", encoding="utf-8")
    script_path.write_text(_wrapper_script(cmd_path, rc_path), encoding="utf-8")
    script_path.chmod(0o700)
    log_path.touch()

    env = _child_env(args.get("env") if isinstance(args.get("env"), dict) else None)
    unit = _unit_name(job_id)
    started = time.time()

    if _systemd_run_available():
        cmd = [
            "systemd-run", "--user", "--collect", "--quiet",
            "--unit=%s" % unit,
            "--description=hermes job %s" % label,
            "--property=WorkingDirectory=%s" % cwd,
            "--property=StandardOutput=append:%s" % log_path,
            "--property=StandardError=append:%s" % log_path,
            # Host-freeze insurance: the pair is what protects the machine.
            # MemorySwapMax alone only changes WHO gets swapped out; MemoryMax
            # alone lets the job thrash. OOMPolicy=continue is not a default
            # for services — without it a cgroup OOM stops the whole unit.
            "--property=MemorySwapMax=0",
            "--property=OOMPolicy=continue",
        ]
        if memory_max and memory_max.lower() not in ("none", "infinity", "0"):
            cmd.append("--property=MemoryMax=%s" % memory_max)
        for key, val in env.items():
            cmd.append("--setenv=%s=%s" % (key, val))
        cmd.append(str(script_path))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return json.dumps({"error": "systemd-run failed: %s" % exc})
        if proc.returncode != 0:
            return json.dumps({
                "error": "systemd-run exited %d" % proc.returncode,
                "stderr": (proc.stderr or "")[-800:],
            })
        entry = {
            "label": label, "command": command, "cwd": cwd,
            "unit": unit, "isolation": "systemd-run",
            "log": str(log_path), "started_at": started,
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "memory_max": memory_max,
        }
        note = (
            "Running as transient unit %s, OUTSIDE the gateway cgroup: it "
            "survives this turn, a gateway restart and a reconnect." % unit
        )
    else:
        # No user systemd (bare `hermes -z` in a terminal, container, ssh
        # without a session). setsid is honest here — there is no gateway
        # cgroup to be cleaned out of — but it is strictly weaker.
        try:
            popen = subprocess.Popen(
                [str(script_path)],
                cwd=cwd, env=env, start_new_session=True,
                stdout=log_path.open("ab"), stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            return json.dumps({"error": "failed to spawn job: %s" % exc})
        entry = {
            "label": label, "command": command, "cwd": cwd,
            "pid": popen.pid, "isolation": "setsid",
            "log": str(log_path), "started_at": started,
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        }
        note = (
            "systemd --user is unavailable, so this job runs detached via "
            "setsid (pid %d). WEAKER: if this process is a gateway, its "
            "cgroup cleanup on restart will kill the job." % popen.pid
        )

    def _apply(data: dict[str, Any]) -> None:
        data["jobs"][job_id] = entry
        _reap(data)

    registry.mutate(_apply)

    return json.dumps({
        "job_id": job_id,
        "status": "running",
        "log": str(log_path),
        "note": note,
        "next": "job_output(job_id) for progress, job_list() for status, job_kill(job_id) to stop",
    }, ensure_ascii=False)


def job_list(args: dict[str, Any]) -> str:
    registry = _Registry()
    data = registry.read()
    jobs = [_public(jid, e) for jid, e in data["jobs"].items()]
    jobs.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    if args.get("running_only"):
        jobs = [j for j in jobs if j["status"] == "running"]
    return json.dumps({"jobs": jobs, "count": len(jobs)}, ensure_ascii=False)


def job_output(args: dict[str, Any]) -> str:
    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return json.dumps({"error": "job_id is required"})
    entry = _Registry().read()["jobs"].get(job_id)
    if entry is None:
        return json.dumps({"error": "unknown job_id %s" % job_id})

    log_path = Path(entry.get("log") or _job_paths(job_id)[0])
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return json.dumps({"error": "cannot read log: %s" % exc, "log": str(log_path)})

    try:
        tail_lines = int(args.get("tail") or 200)
    except (TypeError, ValueError):
        tail_lines = 200
    lines = text.splitlines()
    clipped_lines = len(lines) > tail_lines
    body = "\n".join(lines[-tail_lines:])
    clipped_chars = False
    if len(body) > LOG_TAIL_CAP:
        body = body[-LOG_TAIL_CAP:]
        clipped_chars = True

    out = {"job_id": job_id, "log": str(log_path), "output": body}
    out.update(_status_of(job_id, entry))
    if clipped_lines or clipped_chars:
        out["truncated"] = (
            "showing the tail only — the full log is the file at 'log', read it directly"
        )
    return json.dumps(out, ensure_ascii=False)


def job_kill(args: dict[str, Any]) -> str:
    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return json.dumps({"error": "job_id is required"})
    entry = _Registry().read()["jobs"].get(job_id)
    if entry is None:
        return json.dumps({"error": "unknown job_id %s" % job_id})

    status = _status_of(job_id, entry)["status"]
    if status != "running":
        return json.dumps({"job_id": job_id, "status": status,
                           "note": "job was not running; nothing to kill"})

    if entry.get("isolation") == "systemd-run":
        try:
            proc = subprocess.run(
                ["systemctl", "--user", "stop", entry.get("unit") or ""],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return json.dumps({"error": "systemctl stop failed: %s" % exc})
        ok = proc.returncode == 0
        detail = (proc.stderr or "").strip()[-400:]
    else:
        pid = entry.get("pid")
        ok, detail = False, "no pid recorded"
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)  # windows-footgun: ok — Linux-only job control
                ok, detail = True, "SIGTERM sent to process group"
            except (ProcessLookupError, PermissionError, OSError) as exc:
                detail = str(exc)

    result = {"job_id": job_id, "killed": ok}
    if detail:
        result["detail"] = detail
    result.update(_status_of(job_id, entry))
    return json.dumps(result, ensure_ascii=False)


# --- MCP tool surface --------------------------------------------------------

TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "job_start",
        "description": (
            "Start a shell command as a DURABLE background job and return "
            "immediately. Use this instead of the built-in background Bash or a "
            "background subagent whenever work must outlive this turn: on the "
            "ACP bridge the engine process ends with the turn and takes its own "
            "background work with it. The job runs in a transient systemd unit "
            "outside the gateway's cgroup, so it also survives a gateway "
            "restart. Poll with job_output, list with job_list, stop with "
            "job_kill. Memory is capped (MemoryMax, MemorySwapMax=0) so a "
            "runaway job cannot freeze the host."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run (bash, with pipefail on — a failing "
                        "stage of a pipeline makes the job fail)."
                    ),
                },
                "label": {"type": "string", "description": "Short slug used in the job id and unit name."},
                "cwd": {"type": "string", "description": "Working directory. Defaults to $HOME."},
                "memory_max": {
                    "type": "string",
                    "description": "systemd MemoryMax for the job, e.g. '8G'. Default %s." % DEFAULT_MEMORY_MAX,
                },
                "env": {
                    "type": "object",
                    "description": "Extra environment variables. The job otherwise gets a minimal allowlist, NOT the full environment.",
                },
            },
            "required": ["command"],
        },
        "handler": job_start,
    },
    {
        "name": "job_list",
        "description": "List background jobs started with job_start, with their status and exit codes.",
        "schema": {
            "type": "object",
            "properties": {
                "running_only": {"type": "boolean", "description": "Only jobs still running."},
            },
        },
        "handler": job_list,
    },
    {
        "name": "job_output",
        "description": (
            "Read the tail of a background job's combined stdout/stderr, plus its "
            "current status and exit code."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "tail": {"type": "integer", "description": "Lines from the end. Default 200."},
            },
            "required": ["job_id"],
        },
        "handler": job_output,
    },
    {
        "name": "job_kill",
        "description": "Stop a running background job (systemctl stop / SIGTERM to its process group).",
        "schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        "handler": job_kill,
    },
)

DISPATCH = {spec["name"]: spec["handler"] for spec in TOOL_SPECS}


def start_detached_delegation(prompt: str, label: str, env: dict[str, str],
                              hermes_bin: str) -> str:
    """Launch ``hermes -z`` as a durable job — the honest `background: true`.

    ``delegate_task`` has always accepted a ``background`` field and always
    ignored it. Ignoring it was the safe choice while the only way to detach
    was setsid; with a real job runtime it can mean what it says.
    """
    command = "%s -z %s" % (shlex.quote(hermes_bin), shlex.quote(prompt))
    return job_start({
        "command": command,
        "label": "delegate-%s" % label,
        "env": env,
    })
