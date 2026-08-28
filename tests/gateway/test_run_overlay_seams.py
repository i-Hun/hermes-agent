"""Every overlay hook in ``gateway/run.py`` must actually be in the installed file.

Why a source-level test, and why it is not pretending to be a behaviour test:

``gateway/run.py`` is upstream's file. Our eleven hooks reach it through
``patches/hermes-agent-acp.diff``. Measured 2026-08-04 while porting the overlay
to v2026.8.3: ``git apply --3way`` can silently drop hooks — exit 0, no conflict
marker, no ``.rej``, and ``git diff --stat`` still shows a plausible hunk count.
A partially patched file reads as a successfully patched one.

The progress-rail suites do not catch that. ``test_telegram_progress_edit_transient``
says so in its own words — it *simulates the relevant block* from
``send_progress_messages`` rather than driving it — so reverting
``_prog.render(progress_lines)`` back to ``"\\n".join(progress_lines)`` inside
``run.py`` leaves all 72 of those tests green (measured, same day).

So this file answers exactly one question: **is every hook still in the file?**
That is the question a partially applied patch answers wrongly. It deliberately
asserts nothing about behaviour — the overlay modules own that, and they have
their own tests.

Anchors are chosen to hold across upstream restructuring: functions are located
by name through the AST (``send_progress_messages`` went from a nested ``def`` at
line 20749 to a ``TurnRunner`` method at line 3854 between our pinned base and
v2026.8.3 — same name, and the seams inside it are the same), never by line
number and never by surrounding comment text.
"""

from __future__ import annotations

import ast
import io
import pathlib
import textwrap
import tokenize

import pytest


def _run_py() -> pathlib.Path:
    import gateway.run as gateway_run

    return pathlib.Path(gateway_run.__file__)


@pytest.fixture(scope="module")
def source() -> str:
    return _run_py().read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


def _function_nodes(tree: ast.Module, name: str) -> list[ast.AST]:
    """Every function called ``name``, at any nesting depth."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _function_sources(source: str, tree: ast.Module, name: str) -> list[str]:
    """Source of every function called ``name``, at any nesting depth."""
    lines = source.split("\n")
    return [
        "\n".join(lines[node.lineno - 1 : node.end_lineno])
        for node in _function_nodes(tree, name)
    ]


def _code_only(body: str) -> str:
    """``body`` with comments stripped.

    A hook whose whole point is "we deliberately do NOT call X" documents that in
    a comment naming X. Asserting on the raw text then fails on our own prose,
    and a real regression — the call coming back — is indistinguishable from it.
    Measured: this is exactly how the first run of this file failed.
    """
    body = textwrap.dedent(body)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(body).readline))
    except (tokenize.TokenError, IndentationError) as exc:  # pragma: no cover
        raise AssertionError(f"could not tokenize the extracted body: {exc}") from exc
    return " ".join(t.string for t in toks if t.type != tokenize.COMMENT)


def _the_function(source: str, tree: ast.Module, name: str) -> str:
    bodies = _function_sources(source, tree, name)
    assert bodies, f"gateway/run.py no longer defines {name}() — the overlay hook has no anchor"
    return "\n\n".join(bodies)


# ---------------------------------------------------------------------------
# Hook 1 — usage-limit envelope (module level)
# ---------------------------------------------------------------------------


def test_usage_limit_reply_hook_is_defined_and_called(source: str, tree: ast.Module) -> None:
    from gateway import overlay_hooks

    assert callable(getattr(overlay_hooks, "usage_limit_reply_safe", None))
    body = _the_function(source, tree, "_gateway_provider_error_reply")
    assert "usage_limit_reply_safe(" in body, (
        "the usage-limit branch is gone from _gateway_provider_error_reply — a hit "
        "quota window would be buried under the generic rate-limit reply"
    )


def test_overlay_hooks_are_not_defined_inside_run_py(source: str) -> None:
    """Definitions in run.py collide with upstream; call sites do not.

    2026-08-09: upstream inserted a module-level function at exactly the anchor
    the steer/edit wrappers used, and ``git apply --3way`` conflicted over blank
    lines. The wrappers moved to ``gateway/overlay_hooks.py``; run.py keeps only
    call sites, each on a line the overlay already owns. Re-introducing a
    definition here re-opens that conflict class.
    """
    for name in (
        "usage_limit_reply_safe",
        "acp_steer_lands_this_turn",
        "telegram_edit_absorbed",
    ):
        assert f"def {name}(" not in source, (
            f"{name}() is defined inside gateway/run.py again — keep overlay hook "
            "bodies in gateway/overlay_hooks.py so upstream churn cannot collide "
            "with them"
        )


# ---------------------------------------------------------------------------
# Hook 2 — terminal-block header must not be suppressed under "replace"
# ---------------------------------------------------------------------------


def test_terminal_block_header_respects_replace_grouping(source: str, tree: ast.Module) -> None:
    body = _the_function(source, tree, "progress_callback")
    assert 'progress_grouping != "replace"' in body


# ---------------------------------------------------------------------------
# Hooks 3-8 — the progress rail
# ---------------------------------------------------------------------------

_RAIL_SEAMS = (
    ("from gateway.progress_collapse import ProgressCollapser", "collapser import"),
    ("ProgressCollapser(", "collapser construction"),
    ("from gateway.progress_flood import", "flood-budget import"),
    ("ProgressFloodBudget()", "flood-budget construction"),
    ("_prog.on_content_bubble()", "content-bubble reset"),
    ("_prog.append(", "line accumulation through the collapser"),
    ("_flood.interval", "edit cadence from the budget, not a fixed 1.5s"),
    ("_prog.render(", "bubble rendering through the collapser"),
    ("_flood.classify(", "edit-failure classification in the overlay"),
    ("_flood.on_flood(", "flood hold"),
    ("_flood_retry_after(", "retry-after parsing"),
    ("_prog.refresh_tail_io(", "live subagent tail"),
)


@pytest.mark.parametrize("needle,what", _RAIL_SEAMS, ids=[w for _, w in _RAIL_SEAMS])
def test_progress_rail_seam_present(source: str, tree: ast.Module, needle: str, what: str) -> None:
    body = _the_function(source, tree, "send_progress_messages")
    assert needle in body, f"overlay hook missing from send_progress_messages(): {what}"


def test_collapser_renders_both_bubble_paths(source: str, tree: ast.Module) -> None:
    """Edit path and first-send path both go through the collapser.

    Two call sites, not one: if only the edit path is patched the first bubble of
    every run renders uncollapsed and then silently changes shape on the first
    edit.
    """
    body = _the_function(source, tree, "send_progress_messages")
    assert body.count("_prog.render(") >= 2


@pytest.mark.parametrize(
    "upstream_form",
    ['"\\n".join(progress_lines)', "_PROGRESS_EDIT_INTERVAL"],
)
def test_upstream_form_did_not_survive(source: str, upstream_form: str) -> None:
    """The replaced upstream forms must be gone, not merely shadowed.

    This is the half a "did the hook land?" check misses: a 3-way merge can add
    our line *and* keep upstream's, leaving the rail rendering the uncollapsed
    text on a path nobody reads until it is the one that runs.
    """
    assert upstream_form not in source


# ---------------------------------------------------------------------------
# Hooks 9-10 — background-review notices are not conversational
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ["_deliver_bg_review_message", "_bg_review_send"])
def test_bg_review_notice_is_not_gated_on_run_generation(
    source: str, tree: ast.Module, fn: str
) -> None:
    """See the hook's own comment: the review finishes a minute or more after the
    answer, so every new user turn would discard it as stale."""
    body = _code_only(_the_function(source, tree, fn))
    assert "_run_still_current" not in body


# ---------------------------------------------------------------------------
# Hook 11 — a gateway must not start on a model nobody chose
# ---------------------------------------------------------------------------


def test_main_enforces_acp_config(source: str, tree: ast.Module) -> None:
    body = _the_function(source, tree, "main")
    assert "acp_config" in body
    assert "78" in body, "the EX_CONFIG exit is gone — a second source of truth would boot silently"
