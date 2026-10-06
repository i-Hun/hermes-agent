"""Session provenance of the completion actor (never the task's creator)."""
from gateway.session_context import _UNSET, _VAR_MAP, get_session_env, session_context_engaged


def _actor_var(name: str) -> str:
    # Off-turn work in a multi-session process must not borrow the last
    # agent's os.environ mirror. A fresh CLI process may inherit it.
    if session_context_engaged():
        value = _VAR_MAP[name].get()
        return "" if value is _UNSET else value
    return get_session_env(name, "")


def completion_origin() -> dict:
    """Use turn-local bindings; terminal CLI children inherit the same bindings.

    Missing identity stays unknown. In particular, never infer the actor from
    assignee, created_by, or the task's originating session.
    """
    names = {
        "session_id": "ID", "session_key": "KEY", "profile": "PROFILE",
        "platform": "PLATFORM", "chat_id": "CHAT_ID", "thread_id": "THREAD_ID",
        "chat_type": "CHAT_TYPE", "user_id": "USER_ID", "scope_id": "SCOPE_ID",
        "parent_chat_id": "PARENT_CHAT_ID",
    }
    origin = {key: _actor_var("HERMES_SESSION_" + name) for key, name in names.items()}
    if origin["session_id"] and not origin["profile"]:
        from hermes_cli.profiles import get_active_profile_name
        origin["profile"] = get_active_profile_name()
    return origin if origin["session_id"] and origin["profile"] else {}
