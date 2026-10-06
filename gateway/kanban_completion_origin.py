"""Suppress a completed-event wake only with exact same-session provenance."""
import sqlite3

from gateway.kanban_watchers_common import logger


def is_self_completion(event, sub, runner, *, is_push: bool) -> bool:
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
    metadata = sub.get("delivery_metadata") or {}
    for key in ("platform", "chat_id", "thread_id", "user_id", "chat_type"):
        if (origin.get(key) or "") != (sub.get(key) or metadata.get(key) or ""):
            return False
    scope = metadata.get("scope_id") or metadata.get("guild_id") or metadata.get("slack_team_id") or metadata.get("team_id") or ""
    if (origin.get("scope_id") or "") != scope:
        return False
    if (origin.get("parent_chat_id") or "") != (metadata.get("parent_chat_id") or ""):
        return False
    key = origin.get("session_key")
    store = getattr(runner, "session_store", None)
    if not key or store is None:
        return False
    try:
        entry = store.lookup_by_session_key(key)
    except (OSError, sqlite3.Error):
        logger.exception("kanban: cannot verify completion actor session; retaining wake")
        return False
    # /new or /resume between completion and delivery is NOT a self echo.
    return entry is not None and entry.session_id == origin["session_id"]
