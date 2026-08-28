"""Handling one ACP ``session/update`` from the engine.

Local overlay module (not upstream). Three concerns ride on this event and
none of them are upstream's: which text is the answer versus progress
narration, the ledger of what the engine actually ran, and the live progress
pulses the gateway renders.

Keeping the body here means the insertion in ``copilot_acp_client`` is a
single call. It is still an insertion — upstream owns the surrounding
dispatch — so a conflict there remains visible rather than silently drifting.
"""

from __future__ import annotations

from typing import Any

from agent.acp_background_work import note_background_tool_call
from agent.acp_native_tools import (
    _acp_tool_activity_label,
    _acp_tool_started_fields,
)

# Flattened narration length handed to the 💬 progress pulse. The gateway and
# ProgressCollapser cap it again for their surface; this only stops a runaway
# text run from travelling through the callback chain whole.
_PULSE_CAP = 1000

# A finished sentence may trail a closer ("…локально)"), so strip those before
# judging whether the utterance ended.
_UTTERANCE_CLOSERS = "»”’\"')]}"
_UTTERANCE_END = ".!?…:;"


def _reset_text_run(client: Any, carry: str = "", carry_slot: Any = None) -> None:
    client._acp_text_run_mid = ""
    client._acp_boundary_pending = False
    client._acp_narration_slot = None
    # Prefix of an utterance the engine cut mid-sentence, and the rail slot it
    # already occupies — see _carry_after_split.
    client._acp_narration_carry = carry
    client._acp_narration_carry_slot = carry_slot
    client._acp_pulsed_len = len(carry)


def _narration_so_far(client: Any, flushed: str) -> str:
    """This run's text with the carried prefix of the same utterance, if any."""
    carry = str(getattr(client, "_acp_narration_carry", "") or "")
    return carry + flushed if carry else flushed


def _pulse_narration(client: Any, flushed: str) -> None:
    """Fire the live 💬 pulse — once per length the utterance has reached.

    A run can hit several tool boundaries (a permission-driven ``tool_call``,
    then ``tool_progress`` keep-alives) without growing; re-pulsing identical
    text would repeat the line in the bubble.
    """
    full = _narration_so_far(client, flushed)
    if len(full) <= int(getattr(client, "_acp_pulsed_len", 0) or 0):
        return
    client._acp_pulsed_len = len(full)
    line = " ".join(full.split())[:_PULSE_CAP]
    if line:
        client._fire_engine_activity("_thinking", line)


def _carry_after_split(client: Any, raw: str) -> str:
    """What the NEXT run must re-state so the utterance stays one entry.

    A model can close its API message mid-sentence at a tool-call boundary and
    continue the same sentence in the next one, under a new ``messageId``. The
    split is real — this layer commits the run — but the utterance is not over,
    and the gateway's ``ProgressCollapser`` recognises a continuation only as a
    prefix superset (``b.startswith(a)``); a bare suffix ("ет Airflow…") lands
    as a second 💬 line, i.e. one sentence read as two entries.

    So the layer that KNOWS about the split states the accumulated text and the
    display layer needs no new heuristic (skill hermes-acp-bridge §6.1: the
    support that knows tells the one that guesses). Nothing is carried when the
    run ended on a sentence boundary — there the next narration is a genuinely
    new step and must get its own line.
    """
    body = raw.strip()
    tail = body.rstrip(_UTTERANCE_CLOSERS)
    if not tail or tail[-1] in _UTTERANCE_END:
        return ""
    carry = _narration_so_far(client, body)
    if raw[-1:].isspace():
        # The split fell on whitespace — keep it, or the halves glue into one
        # word once the pulse is flattened.
        carry += " "
    # Past the cap both sides of the collapser's comparison are truncated and
    # stop matching; a fresh line is the honest fallback.
    return carry if len(carry) <= _PULSE_CAP else ""


def _slot_index(slot: Any, reasoning_parts: list[str] | None) -> int | None:
    """Index of a reserved rail slot, if it still belongs to this transcript."""
    if not slot or reasoning_parts is None:
        return None
    holder, idx = slot
    if holder is reasoning_parts and 0 <= idx < len(holder):
        return idx
    return None


def _place_narration(
    client: Any,
    reasoning_parts: list[str] | None,
    payload: str,
) -> Any:
    """Write the committed narration onto the reasoning rail; return its slot.

    Preference order: the slot of the utterance this run CONTINUES (so the
    whole sentence replaces its own prefix, in the prefix's original place),
    then the slot reserved at the boundary (keeps narration ahead of the ⚙️
    lines that follow it), then a plain append.
    """
    if reasoning_parts is None:
        return None
    carry_idx = _slot_index(getattr(client, "_acp_narration_carry_slot", None), reasoning_parts)
    own_idx = _slot_index(getattr(client, "_acp_narration_slot", None), reasoning_parts)
    idx = carry_idx if carry_idx is not None else own_idx
    if idx is None:
        reasoning_parts.append(payload)
        return (reasoning_parts, len(reasoning_parts) - 1)
    if own_idx is not None and own_idx != idx:
        reasoning_parts[own_idx] = ""  # the prefix already holds the place
    reasoning_parts[idx] = payload
    return (reasoning_parts, idx)


def _mark_text_boundary(
    client: Any,
    text_parts: list[str],
    reasoning_parts: list[str] | None,
) -> None:
    """A tool event arrived while text is buffered — remember, do not flush.

    The flush itself is deferred to ``_commit_text_run`` because a tool event
    is NOT proof that the text block ended: ``claude-agent-acp`` surfaces a
    ``tool_call`` from the permission path "before the assistant message's
    tool_use block streams to us" (its own comment in ``acp-agent.js``), and
    emits async ``tool_call_update``s (``tool_progress`` keep-alives, terminal
    output/exit) that can land between two text deltas. Flushing on those cut
    a single narration in half mid-word and, when the late update followed the
    final answer, amputated the reply itself.

    A slot is reserved in the reasoning transcript now and filled on commit,
    so narration keeps its position ahead of the ⚙️ lines that follow it.
    """
    client._acp_boundary_pending = True
    if reasoning_parts is not None and getattr(client, "_acp_narration_slot", None) is None:
        client._acp_narration_slot = (reasoning_parts, len(reasoning_parts))
        reasoning_parts.append("")
    _pulse_narration(client, "".join(text_parts).strip())


def _commit_text_run(
    client: Any,
    text_parts: list[str],
    reasoning_parts: list[str] | None,
) -> tuple[str, Any]:
    """The text run really ended before a tool call — it is narration.

    Only text after the LAST tool call is the user-facing reply, so a committed
    run moves to the reasoning rail (hidden when ``show_reasoning=false``).

    Returns ``(carry, carry_slot)`` — what the next run has to re-state when the
    engine cut the utterance mid-sentence (see ``_carry_after_split``).
    """
    raw = "".join(text_parts)
    flushed = raw.strip()
    text_parts.clear()
    carry: str = ""
    carry_slot: Any = None
    if flushed:
        _pulse_narration(client, flushed)
        payload = _narration_so_far(client, flushed) + "\n\n"
        slot = _place_narration(client, reasoning_parts, payload)
        carry = _carry_after_split(client, raw)
        carry_slot = slot if carry else None
    return carry, carry_slot


def handle_session_update(
    client: Any,
    update: dict[str, Any],
    text_parts: list[str] | None,
    reasoning_parts: list[str] | None,
) -> None:
    """Route one session update into text, reasoning, ledger and pulses."""
    kind = str(update.get("sessionUpdate") or "").strip()
    content = update.get("content") or {}
    chunk_text = ""
    if isinstance(content, dict):
        chunk_text = str(content.get("text") or "")
    if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
        # One assistant message = one text run, identified by ACP's messageId
        # (the engine stamps it on message/thought chunks only). A tool event
        # merely SUSPECTS the run ended; the next chunk decides:
        #   same messageId  → the block was still streaming, the boundary was
        #                     premature → keep accumulating (no split);
        #   new/absent id   → the run really ended → commit it as narration.
        mid = str(update.get("messageId") or "")
        carry: str = ""
        carry_slot: Any = None
        if getattr(client, "_acp_boundary_pending", False):
            prev_mid = str(getattr(client, "_acp_text_run_mid", "") or "")
            if mid and prev_mid and mid == prev_mid:
                client._acp_boundary_pending = False
            else:
                carry, carry_slot = _commit_text_run(client, text_parts, reasoning_parts)
        if not text_parts:
            _reset_text_run(client, carry, carry_slot)
            client._acp_text_run_mid = mid
        text_parts.append(chunk_text)
    elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
        reasoning_parts.append(chunk_text)
    elif kind in ("tool_call", "tool_call_update"):
        client._record_native_tool_call(update, kind)
        # Same event, different question: the ledger asks "what ran?", this
        # asks "what was parked outside the call and will die with the
        # bridge?". Read from the raw update because the ledger truncates
        # arguments at 2000 chars and an Agent prompt is far longer than that.
        note_background_tool_call(client, update)
        # Message text emitted BEFORE a tool call is the engine's
        # progress narration, not the final answer. Reclassify it as
        # reasoning so show_reasoning=false hides it; only the text
        # after the last tool call remains the user-facing reply.
        # Deferred on purpose — see _mark_text_boundary.
        if text_parts:
            _mark_text_boundary(client, text_parts, reasoning_parts)
        if kind == "tool_call" and reasoning_parts is not None:
            label = _acp_tool_activity_label(update)
            if label:
                reasoning_parts.append(f"\n⚙️ {label}\n")
        # Live tool pulse (⚙️ tool: …, gated on tool_progress).
        # The engine's initial tool_call often arrives WITHOUT
        # rawInput (the command/path lands later via
        # tool_call_update), so an eager pulse renders as a
        # preview-less "⚙️ Terminal..." line. Defer the pulse until
        # arguments are available and fire it exactly once per
        # toolCallId; id-less updates keep the eager behavior.
        fired = getattr(client, "_acp_fired_tool_ids", None)
        if fired is None:
            fired = client._acp_fired_tool_ids = set()
        pending = getattr(client, "_acp_pending_tools", None)
        if pending is None:
            pending = client._acp_pending_tools = {}
        tool_id = str(
            update.get("toolCallId") or update.get("tool_call_id") or ""
        ) or None
        _tname, _tprev, _targs = _acp_tool_started_fields(update)
        _has_details = bool(_tprev or _targs)
        if kind == "tool_call":
            if tool_id is None:
                client._fire_engine_activity("tool.started", _tname, _tprev, _targs)
            elif _has_details:
                fired.add(tool_id)
                client._fire_engine_activity("tool.started", _tname, _tprev, _targs)
            else:
                pending[tool_id] = _tname
        elif (
            tool_id is not None
            and tool_id in pending
            and tool_id not in fired
            and _has_details
        ):
            orig_name = pending.pop(tool_id)
            fired.add(tool_id)
            if _tname == "tool" and orig_name:
                # The update usually carries no title — re-map with
                # the remembered one so shell tools still hit the
                # "terminal" fenced-block rendering.
                _tname, _tprev, _targs = _acp_tool_started_fields(
                    {**update, "title": orig_name}
                )
            client._fire_engine_activity("tool.started", _tname, _tprev, _targs)
