#!/usr/bin/env python3
"""Prove the native-MCP tool path end to end, at two layers.

Layer 1 (default): spawn the exact `hermes-tools` MCP server the ACP client
hands the engine (session/new.mcpServers) and run a real MCP handshake over
stdio — initialize, tools/list, and a read-only tools/call. Proves the server
starts, advertises Hermes tools, and DISPATCHES them (regression guard for the
FastMCP `**kwargs` schema bug: a broken schema makes every call fail with
"kwargs Field required").

Layer 2 (--acp): drive the real claude-agent-acp engine with those mcpServers
attached and a prompt that forces a tool call, then assert the turn completed
on claude-fable-5 without a reasoning_extraction refusal / opus fallback.
Requires the bridge token + network; skipped unless --acp is passed.

Usage:
  venv/bin/python3 scripts/acp_native_tools_probe.py
  venv/bin/python3 scripts/acp_native_tools_probe.py --acp
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def probe_mcp(call_tool: str | None) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from agent.copilot_acp_client import _hermes_tools_mcp_servers

    spec = _hermes_tools_mcp_servers()[0]
    env = dict(os.environ)
    for pair in spec.get("env", []):
        env[pair["name"]] = pair["value"]

    params = StdioServerParameters(command=spec["command"], args=spec["args"], env=env)
    print(f"[mcp] spawning: {spec['command']} {' '.join(spec['args'])}")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = [t.name for t in listed.tools]
            print(f"[mcp] initialize OK — {len(names)} tools advertised")
            print(f"[mcp] tools: {', '.join(names)}")
            if not names:
                print("[mcp] FAIL: server advertised zero tools")
                return 1

            target = call_tool or ("skills_list" if "skills_list" in names else None)
            if target and target in names:
                print(f"[mcp] calling read-only tool: {target}")
                res = await session.call_tool(target, {})
                text = "".join(getattr(b, "text", "") or "" for b in (res.content or []))
                preview = text.strip().replace("\n", " ")[:160]
                print(f"[mcp] call isError={res.isError} len={len(text)} preview={preview!r}")
                if res.isError:
                    print("[mcp] FAIL: read-only tool call errored (dispatch broken)")
                    return 1
            print("[mcp] PASS")
            return 0


def probe_acp() -> int:
    os.environ["HERMES_ACP_NATIVE_TOOLS"] = "1"
    os.environ.setdefault(
        "HERMES_COPILOT_ACP_COMMAND",
        os.path.expanduser("~/.hermes/claude-acp-run.sh"),
    )
    from agent.copilot_acp_client import CopilotACPClient

    client = CopilotACPClient()
    if not client._native_tools or not client._mcp_servers:
        print("[acp] FAIL: native tools not configured on client")
        return 1
    print(f"[acp] native_tools=True mcpServers={[s['name'] for s in client._mcp_servers]}")

    prompt = (
        "Call the hermes-tools `skills_list` tool and reply with exactly "
        "'COUNT=<number of skills>'. Use the tool; do not guess."
    )
    resp = client.chat.completions.create(
        model=os.getenv("HERMES_ACP_MODEL", "claude-fable-5"),
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    model_used = str(getattr(resp, "model", "?"))
    low = text.lower()
    refused = "reasoning_extraction" in low or "declined this request" in low
    fell_back = "opus" in model_used.lower()
    print(f"[acp] model_used={model_used} refused={refused} fallback={fell_back}")
    print(f"[acp] response[:200]={text.strip()[:200]!r}")
    if refused or fell_back:
        print("[acp] FAIL: refusal / opus fallback still present")
        return 1
    if "COUNT=" not in text:
        print("[acp] WARN: model did not return tool-derived COUNT= (may not have called tool)")
    print("[acp] PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acp", action="store_true", help="also drive the real engine")
    ap.add_argument("--call-tool", default=None, help="read-only tool to call in the MCP layer")
    args = ap.parse_args()

    rc = asyncio.run(probe_mcp(args.call_tool))
    if rc != 0:
        return rc
    if args.acp:
        rc = probe_acp()
    return rc


if __name__ == "__main__":
    sys.exit(main())
