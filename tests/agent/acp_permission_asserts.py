"""Shared assertions for ACP permission tests (local overlay).

Lives outside both test files so the upstream one and the overlay one can use
it without either owning the other's imports.
"""

from __future__ import annotations


def refusal_outcome(response: dict) -> str:
    """Normalize a permission response to "cancelled" when it refuses.

    The guard refuses by SELECTING an advertised reject option rather than
    returning a bare ``cancelled``: claude-agent-acp turns ``cancelled`` into
    "Tool use aborted" (an opaque failure the model retries around), while a
    reject selection becomes ``{behavior: "deny", message: ...}`` the model
    treats as a decision. Tests assert the refusal, not that wire shape.
    """
    outcome = ((response.get("result") or {}).get("outcome") or {})
    kind = outcome.get("outcome")
    if kind == "selected" and str(outcome.get("optionId", "")).startswith("reject"):
        return "cancelled"
    return kind
