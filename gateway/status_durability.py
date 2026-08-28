"""Which agent status messages must survive ``display.cleanup_progress``.

Local overlay module (not upstream).

``cleanup_progress: true`` exists because a turn's progress bubbles are noise
once the answer lands: tool-progress lines, "⏳ Working — N min", retry chatter.
The gateway collects their message ids while the turn runs and deletes them all
after the final response (``gateway/run.py``, ``_cleanup_msg_ids``).

That sweep is indiscriminate, and one message it swept was not noise. When the
primary model fails and the fallback chain answers instead,
``chat_completion_helpers.try_activate_fallback`` records a one-shot notice and
``run_agent._emit_pending_fallback_notice`` emits it on the success path —
written, in that function's own docstring, so "the user always sees the switch
once". It went out on the ``lifecycle`` key like every progress bubble, so
cleanup deleted it seconds later, every time. Measured on the default profile
2026-08-09: three activations (19:56:13, 19:57:52, 20:05:41) of
``claude-fable-5 → claude-opus-5 (copilot-acp)`` recorded in ``agent.log``,
nothing visible in the chat — the operator spent the session believing his
``/model`` pick was live.

A model switch is not progress; it is a durable state change that outlives the
turn describing it. So it gets its own status kind, and the cleanup sweep is
taught to skip that kind. Two consequences follow from the same choice, both
wanted:

* the notice is not deleted after the final response;
* Telegram's ``send_or_update_status`` keys its edit-in-place cache on the
  status kind, so a durable notice no longer overwrites (or gets overwritten
  by) an ordinary ``lifecycle`` bubble.

Kept as a set rather than a boolean so a second durable kind (a credential
expiry, a provider demotion) is one line here and no change at either call
site.
"""

from __future__ import annotations

#: Status kind carried by the fallback/model-switch notice.
MODEL_SWITCH = "model_switch"

#: Status kinds the post-turn cleanup sweep must leave alone.
DURABLE_STATUS_KINDS = frozenset({MODEL_SWITCH})


def is_durable(event_type) -> bool:
    """Whether a status of this kind must survive post-turn cleanup.

    Never raises: this is consulted on the status-callback path, where an
    exception would cost the message it is trying to protect.
    """
    try:
        return str(event_type or "").strip() in DURABLE_STATUS_KINDS
    except Exception:  # pragma: no cover - defensive
        return False
