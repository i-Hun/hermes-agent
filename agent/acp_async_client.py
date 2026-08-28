"""Async facade over the (synchronous) Copilot ACP client.

Local overlay module (not upstream).

``_to_async_client`` converts a sync auxiliary client into its async
counterpart for every provider that has one — Codex, Anthropic, Bedrock,
Gemini — and returned the ACP client *unchanged*::

    if isinstance(sync_client, CopilotACPClient):
        return sync_client, model

That is fine for anything that calls ``create()`` directly, and wrong for
every caller that awaits it. ``_ACPChatCompletions.create`` is a plain
method returning a ``SimpleNamespace`` completion, so the async auxiliary
path (``async_call_llm`` → ``_acreate`` → ``await client.chat.completions
.create(...)``) died with

    TypeError: object types.SimpleNamespace can't be used in 'await' expression

The visible casualty was image analysis: the gateway pre-analyses inbound
photos through ``vision_analyze`` on the async path, so on an ACP-backed
profile *every* image the user sent came back as "I could not make it out"
(founder, 2026-08-05 08:44 and 2026-08-10 22:48). Nothing about the image or
the engine was at fault — the coroutine never existed.

The bridge is a subprocess speaking JSON-RPC over pipes, i.e. blocking I/O
with no async transport to graft onto. The honest conversion is therefore a
thread hand-off rather than a native async client: ``asyncio.to_thread``
keeps the event loop free for the minutes a prompt can take, which is the
property the caller actually needs.

Attribute access falls through to the wrapped client (``__getattr__``) so the
ledger, the cancellation flag and the native-tools switch keep working on the
async instance — those are read off the client object by name elsewhere.
"""

from __future__ import annotations

import asyncio
from typing import Any


class _AsyncACPCompletions:
    """``chat.completions`` with an awaitable ``create``."""

    def __init__(self, sync_client: Any):
        self._sync_client = sync_client

    async def create(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(
            self._sync_client.chat.completions.create, **kwargs
        )


class _AsyncACPChat:
    def __init__(self, sync_client: Any):
        self.completions = _AsyncACPCompletions(sync_client)


class AsyncCopilotACPClient:
    """Async-compatible wrapper matching ``AsyncOpenAI.chat.completions.create()``."""

    def __init__(self, sync_client: Any):
        self._sync_client = sync_client
        self.chat = _AsyncACPChat(sync_client)
        self.api_key = getattr(sync_client, "api_key", "copilot-acp")
        self.base_url = getattr(sync_client, "base_url", "")

    def __getattr__(self, name: str) -> Any:
        # Only consulted for attributes this wrapper does not define, so the
        # three set in __init__ keep their own values.
        return getattr(self._sync_client, name)
