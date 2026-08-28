"""Hermes-specific behaviour bolted onto the upstream ACP client.

Local overlay module (not upstream).

These methods do not exist upstream at all — they are ours end to end — so
they live in a mixin the upstream class inherits rather than as insertions
into its body. The patch then carries one line (the base-class list) instead
of a hundred, and a `hermes update` has nothing here to merge.

Note what is deliberately NOT here: anything that edits the *inside* of an
upstream method. Copying such a method into a subclass would freeze
upstream's version of it — their later fixes would silently stop reaching us
(on 2026-07-25 upstream changed `_run_prompt` itself). Those stay as small
insertions in the patch, where a conflict is visible and forces a decision.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.acp_native_tools import (
    _acp_tool_started_fields,
    _normalize_native_tool_name,
)


class HermesACPMixin:
    """Engine-activity relay, native-call ledger and curator-session flag."""


    def _fire_engine_activity(
        self,
        event_type: str,
        tool_name: str | None = None,
        preview: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Relay engine-internal activity to the owning agent's progress rail.

        ``_activity_agent`` is attached at client construction (see
        ``create_openai_client``). The gateway re-binds
        ``agent.tool_progress_callback`` per turn, so we resolve it at call
        time; without an attached agent (aux clients, tests) this is a no-op.
        Runs on the ACP stdout-reader thread — the gateway callback is
        thread-safe (queue-backed) by design.
        """
        agent = getattr(self, "_activity_agent", None)
        if agent is not None:
            # Liveness. A non-streaming ACP call produces no bytes while the
            # engine works, so every watchdog that measures "time since
            # something happened" sees a dead call and kills a turn that is
            # perfectly healthy. There are two such watchdogs and they read
            # DIFFERENT clocks, so both have to be fed:
            #   * the stale-call detector reads _acp_last_event_ts
            #     (agent/acp_stale.py);
            #   * the cron inactivity watchdog — and the gateway's "Working…"
            #     status — read the standard _last_activity_ts, updated via
            #     _touch_activity. Missing this one killed a wiki-ingest cron
            #     at exactly 600s with "idle for 600s — last activity: waiting
            #     for non-streaming API response" (2026-07-25).
            try:
                agent._acp_last_event_ts = time.time()
            except Exception:
                pass
            try:
                # Doubles as the human-readable status the heartbeat shows.
                desc = str(preview or tool_name or "engine working").strip()
                agent._touch_activity(f"engine: {desc[:80]}")
            except Exception:
                pass
        cb = getattr(agent, "tool_progress_callback", None) if agent is not None else None
        if cb is None:
            return
        try:
            cb(event_type, tool_name, preview, args)
        except Exception:
            pass

    def _curator_session(self) -> bool:
        """True when this client serves the background skill-curator fork."""
        agent = getattr(self, "_activity_agent", None)
        platform = getattr(agent, "platform", "") if agent is not None else ""
        return str(platform or "").strip().lower() == "curator"

    _NATIVE_CALL_CAP = 800
    _NATIVE_CALL_ARGS_CAP = 2000

    def _record_native_tool_call(self, update: dict[str, Any], kind: str) -> None:
        """Track engine tool calls ({name, arguments}) for post-run reports.

        The initial tool_call often arrives without rawInput (arguments land
        via tool_call_update), so entries are indexed by toolCallId and
        backfilled. Never raises.
        """
        try:
            calls = self._native_tool_calls
            index = self._native_call_index
        except AttributeError:
            calls = self._native_tool_calls = []
            index = self._native_call_index = {}
        # Mirror onto the owning agent. Per-REQUEST clients are created with
        # shared=False and discarded when the call returns (run_agent
        # _create_request_openai_client), so a ledger kept only on the client
        # dies with it — curator then reads agent.client._native_tool_calls and
        # sees [] even though the engine ran tools (verified 2026-07-25: a
        # background skill review wrote a skill via the MCP bridge while the
        # ledger read 0). The agent outlives every request, so it owns the
        # authoritative list; the client-local one stays for compatibility.
        agent = getattr(self, "_activity_agent", None)
        if agent is not None:
            try:
                shared = getattr(agent, "_acp_native_tool_calls", None)
                if shared is None:
                    shared = agent._acp_native_tool_calls = []
                if shared is not calls:
                    calls = self._native_tool_calls = shared
            except Exception:
                pass
        try:
            tool_id = str(
                update.get("toolCallId") or update.get("tool_call_id") or ""
            ) or None
            name, _preview, args = _acp_tool_started_fields(update)
            norm = _normalize_native_tool_name(name)
            args_json = ""
            if isinstance(args, dict):
                try:
                    args_json = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args_json = str(args)
                if len(args_json) > self._NATIVE_CALL_ARGS_CAP:
                    args_json = args_json[: self._NATIVE_CALL_ARGS_CAP] + "…"
            if tool_id is not None and tool_id in index:
                # Known call: backfill arguments/name from the update.
                entry = calls[index[tool_id]]
                if args_json and not entry.get("arguments"):
                    entry["arguments"] = args_json
                if entry.get("name") in ("", "tool") and norm not in ("", "tool"):
                    entry["name"] = norm
                return
            if kind == "tool_call" or tool_id is not None:
                if len(calls) >= self._NATIVE_CALL_CAP:
                    return
                calls.append({"name": norm, "arguments": args_json})
                if tool_id is not None:
                    index[tool_id] = len(calls) - 1
        except Exception:
            pass
