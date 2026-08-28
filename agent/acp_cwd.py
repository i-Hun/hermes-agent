"""Resolving the session cwd when the directory underneath has been deleted.

Local overlay module (not upstream).

``CopilotACPClient.__init__`` resolves ``os.getcwd()`` for every request (a
fresh client is built per API call). If the process's own working directory
has been removed meanwhile, that call raises

    FileNotFoundError: [Errno 2] No such file or directory

with ``filename=None`` — no path, no context. Hermes' retry layer sees an
unclassifiable exception, burns three attempts with backoff and announces
"Provider unreachable — switching to fallback provider". Nothing is wrong with
the provider; the process is standing in a hole.

This is not hypothetical. A kanban worker's cwd IS its scratch workspace, and
closing the card ``rmtree``s exactly that directory
(``kanban_db._cleanup_workspace``), so any worker that completes its task and
then makes one more API call lands here. Twelve worker logs across three
profiles carried the signature before 2026-08-04.

Two things matter in the repair:

* **Recreate, do not relocate.** Falling back to ``$HOME`` would silently
  widen ``_ensure_path_within_cwd``'s containment root from one scratch
  directory to the whole home directory. The deleted path is still readable
  from ``/proc/self/cwd`` on Linux, so the original root can be restored
  instead of traded away.
* **If it cannot be repaired, say so.** A bare ``FileNotFoundError`` is
  indistinguishable from a transport fault; the raised error below names the
  directory and the cause, which is what the log needed and never had.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

_DELETED_SUFFIX = " (deleted)"


class SessionCwdDeleted(RuntimeError):
    """The working directory of this process no longer exists."""


def _deleted_cwd_path() -> str | None:
    """The path of the vanished cwd, read from /proc (Linux only).

    The kernel appends ``" (deleted)"`` to the link target once the directory
    is unlinked. A real directory whose name happens to end that way is
    checked for first, so the suffix is only stripped when it is a marker.
    """
    try:
        raw = os.readlink("/proc/self/cwd")
    except OSError:
        return None
    if Path(raw).is_dir():
        return raw
    if raw.endswith(_DELETED_SUFFIX):
        candidate = raw[: -len(_DELETED_SUFFIX)]
        return candidate or None
    return None


def session_cwd(explicit: str | None = None) -> str:
    """Absolute cwd for an ACP session, repairing a deleted one in place."""
    if explicit:
        return str(Path(explicit).resolve())
    try:
        return str(Path(os.getcwd()).resolve())
    except FileNotFoundError:
        pass

    missing = _deleted_cwd_path()
    if missing:
        try:
            Path(missing).mkdir(parents=True, exist_ok=True)
            os.chdir(missing)
            _log.warning(
                "Session cwd %s had been deleted underneath this process "
                "(a completed kanban card removes its own scratch workspace); "
                "recreated it so the run keeps its containment root.",
                missing,
            )
            return str(Path(missing).resolve())
        except OSError as exc:
            raise SessionCwdDeleted(
                f"Working directory {missing!r} was deleted and could not be "
                f"recreated ({exc}). This is not a provider failure."
            ) from exc

    raise SessionCwdDeleted(
        "Working directory of this process was deleted and its path could not "
        "be recovered. This is not a provider failure."
    )


__all__ = ["SessionCwdDeleted", "session_cwd"]
