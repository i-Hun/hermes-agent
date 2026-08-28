"""Every overlay seam OUTSIDE ``gateway/run.py`` must still be in its file.

Sibling of ``tests/gateway/test_run_overlay_seams.py``, same single question and
the same deliberate refusal to assert behaviour: **is the hook still there?**

Why this file exists. The 04.08.2026 port taught that a `git apply --3way` can
drop a hook without a conflict marker, so run.py got a seam test. The 15.08.2026
port to v2026.8.13 taught the second half: the seams that actually broke were in
the OTHER six patched files, and nothing was watching them. Seven files
conflicted, thirteen regions were merged by hand, and a behaviour suite would
have stayed green for most of them — upstream had refactored around the seam
(``self._busy_input_mode`` → ``effective_busy_input_mode``,
``_est_ctx`` absorbed into ``_report_stale_nonstream_kill``,
``promote_task`` displaced by a newly inserted ``request_review``).

Anchors are AST function names, never line numbers and never comment text — the
same rule as the run.py seam test, for the same measured reason.

Each seam is asserted in BOTH directions where we replaced upstream code: our
form present, upstream's form gone. A 3-way merge can add our line and leave
theirs shadowing it right next to it, and that reads as success everywhere else.
"""

from __future__ import annotations

import ast
import importlib
import io
import pathlib
import textwrap
import tokenize

import pytest


def _module_source(dotted: str) -> str:
    mod = importlib.import_module(dotted)
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


def _code_only(body: str) -> str:
    """``body`` with comments stripped and the CODE TEXT otherwise untouched.

    Our seams document themselves in comments that name the very symbol they
    replaced, so a raw-text assertion would pass on the prose alone — and a real
    regression would be indistinguishable from it.

    Deliberately not ``" ".join(token.string ...)``: that spelling turns
    ``foo(bar)`` into ``foo ( bar )``, so every call-shaped or subscript-shaped
    anchor silently stops matching and the assertion passes on nothing. Measured
    on this file's first draft — the "upstream form is gone" check for
    ``_ev_lines[0][:400]`` could never have fired. Comments are removed by
    truncating each line at the comment token's own column instead.
    """
    body = textwrap.dedent(body)
    lines = body.split("\n")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(body).readline))
    except (tokenize.TokenError, IndentationError):
        toks = []
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    return "\n".join(lines)


def _function_sources(source: str, name: str) -> list[str]:
    """Source of every function called ``name``, at any nesting depth."""
    tree = ast.parse(source)
    lines = source.split("\n")
    return [
        "\n".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _one_function(source: str, name: str) -> str:
    bodies = _function_sources(source, name)
    assert bodies, f"function {name!r} is gone — the seam has no anchor left"
    return "\n".join(bodies)


# (dotted module, function name, symbols that must be present)
#
# Anchors are CALL-shaped (``name(``) wherever the symbol is also imported into
# the same function: an ``import`` line alone satisfied a bare-name anchor while
# the call itself was gone — measured on this file's first draft, where reverting
# ``should_kill_stale`` left the presence check green.
PRESENT = [
    # Disk witness of what a background review actually wrote: the ledger and
    # the message walk are both blind on the native ACP path.
    ("agent.background_review", "_run_review_in_thread", ["snapshot_review_targets("]),
    ("agent.background_review", "_run_review_in_thread", ["_pass_kind", "_pass_error"]),
    # ACP is judged on IDLE time, not total time — a native turn legitimately
    # runs longer than the stale timeout while the engine reports activity.
    ("agent.chat_completion_helpers", "interruptible_api_call", ["should_kill_stale("]),
    # Failover bookkeeping must see every request close, on all four paths.
    ("agent.chat_completion_helpers", "_abort_active_request", ["note_request_close("]),
    ("agent.chat_completion_helpers", "direct_api_call", ["note_request_close("]),
    (
        "agent.chat_completion_helpers",
        "interruptible_streaming_api_call",
        ["note_request_close("],
    ),
    # The operator grant must be honoured on the fs/write_text_file path too —
    # the engine's own Write tool goes through session/request_permission and
    # this capability does not.
    ("agent.copilot_acp_client", "_handle_server_message", ["_operator_grant_for("]),
    ("agent.copilot_acp_client", "_handle_server_message", ["_curator_file_reason("]),
    # Worker handoff arrived chopped mid-word (t_415833af); the chat line and
    # the synthetic wake turn must carry the same excerpt.
    (
        "gateway.kanban_watchers",
        "_kanban_notifier_watcher",
        # Both replacement sites, each call-shaped, rather than one anchor on
        # the assignment that follows them: upstream initialises
        # ``wake_handoff = ""`` at the top of the watcher, so an anchor on the
        # bare assignment is satisfied by the initialisation while both of our
        # seams are gone. Naming ``h =``/``r =`` also keeps the two branches
        # (event payload summary / task result) independently pinned — a 3-way
        # that lands one and drops the other would otherwise read as success.
        ["h = handoff_excerpt(", "r = handoff_excerpt("],
    ),
    ("hermes_cli.kanban_db", "complete_task", ["handoff_payload_excerpt("]),
    ("hermes_cli.kanban_db", "edit_completed_task_result", ["handoff_payload_excerpt("]),
    # promote is the only cheap door back for a card the loop breaker escalated
    # into triage.
    ("hermes_cli.kanban_db", "promote_task", ["awaiting_human_after_block_loop("]),
    ("hermes_cli.kanban_db", "promote_task", ["'todo', 'blocked', 'triage'"]),
    # The skills index LRU revalidates its mtime/size manifest on every hit —
    # without it a skill written by another process (skill_manage under ACP)
    # stayed invisible in available_skills until the gateway restarted.
    (
        "agent.prompt_builder",
        "build_skills_system_prompt",
        ["live_manifest = _build_skills_manifest(", "(result, live_manifest)"],
    ),
    # FastAPI/Starlette proxies report as {"detail": "..."} — a key none of the
    # upstream extractors knew. Without it a descriptive 400 read as blank and
    # the "generic 400 + large session" heuristic pushed a healthy session into
    # the compression loop until auto-reset. Anchored on all three readers: the
    # is_generic verdict (_classify_400) is what kills sessions, but a port that
    # lands only that one leaves pattern matching and the operator-facing
    # message blind to the same body.
    ("agent.error_classifier", "_classify_400", ["_detail_message(body)"]),
    ("agent.error_classifier", "classify_api_error", ["_detail_message(body)"]),
    ("agent.error_classifier", "_extract_message", ["_detail_message(body)"]),
    # Soft ceiling advisory on SKILL.md growth, on all three write paths.
    ("tools.skill_manager_tool", "_create_skill", ["_attach_size_warning("]),
    ("tools.skill_manager_tool", "_edit_skill", ["_attach_size_warning("]),
    ("tools.skill_manager_tool", "_patch_skill", ["_attach_size_warning("]),
]

# (dotted module, function name, upstream forms that must be GONE)
ABSENT = [
    # Replaced by the (result, manifest) pair. Anchored on the STORE, not on the
    # `return cached` of the lookup: our `return cached_result` contains that
    # string, so an absence anchor there is green whether the seam survives or
    # not.
    ("agent.prompt_builder", "build_skills_system_prompt", ["_SKILLS_PROMPT_CACHE[cache_key] = result"]),
    # Replaced by should_kill_stale(); a 3-way can leave the old total-time
    # comparison shadowing ours in the same branch.
    ("agent.chat_completion_helpers", "interruptible_api_call", ["_elapsed > _stale_timeout"]),
    # Replaced by handoff_payload_excerpt(); the first-line cut is what chopped
    # the worker's report.
    ("hermes_cli.kanban_db", "complete_task", ["_ev_lines[0][:400]"]),
    ("hermes_cli.kanban_db", "edit_completed_task_result", ["_ev_lines[0][:400]"]),
    # Same excerpt bug on the gateway side: four hard character cuts replaced by
    # handoff_excerpt(). The module-level fallback def still slices, but it is a
    # different function, so it cannot satisfy these.
    (
        "gateway.kanban_watchers",
        "_kanban_notifier_watcher",
        ["payload_summary[:200]", "task.result[:160]", "str(ev.payload['reason'])[:160]"],
    ),
]


@pytest.mark.parametrize(
    "dotted,func,symbols",
    PRESENT,
    ids=[f"{d.split('.')[-1]}:{f}:{'+'.join(s)}" for d, f, s in PRESENT],
)
def test_seam_is_present(dotted: str, func: str, symbols: list[str]) -> None:
    body = _code_only(_one_function(_module_source(dotted), func))
    for sym in symbols:
        assert sym in body, (
            f"overlay seam {sym!r} is missing from {dotted}.{func} — the patch "
            f"applied only partially, or upstream restructured around it"
        )


@pytest.mark.parametrize(
    "dotted,func,forms",
    ABSENT,
    ids=[f"{d.split('.')[-1]}:{f}" for d, f, _ in ABSENT],
)
def test_replaced_upstream_form_is_gone(dotted: str, func: str, forms: list[str]) -> None:
    body = _code_only(_one_function(_module_source(dotted), func))
    for form in forms:
        assert form not in body, (
            f"upstream form {form!r} is still in {dotted}.{func} — a 3-way merge "
            f"can add our line and leave theirs beside it, which reads as success"
        )


def test_acp_provider_label_is_ours() -> None:
    """The `/model` header reads from here; the button reads from
    ``agent.acp_model_choice``. They must not drift apart again."""
    from agent.acp_model_choice import ACP_PICKER_NAME
    from hermes_cli.providers import _LABEL_OVERRIDES

    assert _LABEL_OVERRIDES["copilot-acp"] == ACP_PICKER_NAME
