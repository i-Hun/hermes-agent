"""Suppress a completed-event wake only with exact same-session provenance."""
import sqlite3

from gateway.kanban_watchers_common import logger


def is_self_completion(event, sub, runner, *, is_push: bool, source=None) -> bool:
    if event.kind != "completed":
        return False
    origin = (event.payload or {}).get("completion_origin")
    if not isinstance(origin, dict) or not origin.get("session_id"):
        return False
    # No profile heuristics: legacy/unknown ownership must still deliver.
    if not origin.get("profile") or origin["profile"] != sub.get("notifier_profile"):
        return False
    if not is_push:
        return origin["session_id"] == sub.get("chat_id")
    store = getattr(runner, "session_store", None)
    if source is None or not origin.get("session_key") or store is None:
        return False
    try:
        # Resolve the actual delivery source with the same store/config as inbound
        # routing. Raw user IDs differ in shared threads but not their session key;
        # per-user isolation, alternate IDs and multiplex namespaces still apply.
        key = runner._session_key_for_source(source)
        if key != origin["session_key"]:
            return False
        entry = store.lookup_by_session_key(key)
    except (OSError, sqlite3.Error):
        logger.exception("kanban: cannot verify completion actor session; retaining wake")
        return False
    # /new or /resume between completion and delivery is NOT a self echo.
    return entry is not None and entry.session_id == origin["session_id"]
