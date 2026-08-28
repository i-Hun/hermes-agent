"""Images over the ACP bridge: awaiting the client, and not dropping the image.

Two defects, one visible symptom (`founder`, 2026-08-05 and 2026-08-10: every
inbound photo came back as "could not make it out"):

* ``_to_async_client`` handed the SYNC ACP client to awaiting callers, so the
  gateway's async image pre-analysis died with "SimpleNamespace can't be used
  in 'await' expression";
* behind that crash, ``_format_messages_as_prompt`` silently dropped the
  image part, so a fixed ``await`` alone would have produced a confident
  description of an image the engine never saw.
"""

from __future__ import annotations

import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.acp_async_client import AsyncCopilotACPClient
from agent.acp_image_blocks import (
    ACPImagesUnsupported,
    build_prompt_blocks,
    engine_accepts_images,
    extract_image_blocks,
)
from agent.copilot_acp_client import CopilotACPClient

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n-not-a-real-png-").decode()
_DATA_URL = f"data:image/png;base64,{_PNG}"


def _image_message(url: str = _DATA_URL) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


class AsyncACPClientTests(unittest.TestCase):
    def test_create_is_awaitable_and_returns_the_completion(self) -> None:
        completion = SimpleNamespace(choices=[], usage=None, model="claude-opus-5")
        sync_client = SimpleNamespace(
            api_key="k",
            base_url="acp://copilot",
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: completion)
            ),
        )

        async_client = AsyncCopilotACPClient(sync_client)
        got = asyncio.run(async_client.chat.completions.create(model="claude-opus-5"))

        self.assertIs(got, completion)

    def test_create_runs_off_the_event_loop(self) -> None:
        """A prompt blocks for minutes; the loop must stay responsive."""
        seen: dict = {}

        def _create(**kwargs):
            seen["loop_running"] = _loop_is_running()
            return SimpleNamespace(choices=[])

        def _loop_is_running() -> bool:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return False
            return True

        sync_client = SimpleNamespace(
            api_key="k",
            base_url="acp://copilot",
            chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
        )

        asyncio.run(AsyncCopilotACPClient(sync_client).chat.completions.create())

        self.assertFalse(seen["loop_running"], "create() ran on the event loop thread")

    def test_unknown_attributes_fall_through_to_the_sync_client(self) -> None:
        sync_client = SimpleNamespace(
            api_key="k", base_url="acp://copilot", chat=None, _native_tools=True
        )

        self.assertTrue(AsyncCopilotACPClient(sync_client)._native_tools)

    def test_to_async_client_wraps_the_acp_client(self) -> None:
        from agent.auxiliary_client import _to_async_client

        client, model = _to_async_client(CopilotACPClient(acp_cwd="/tmp"), "claude-opus-5")

        self.assertIsInstance(client, AsyncCopilotACPClient)
        self.assertEqual(model, "claude-opus-5")
        self.assertTrue(asyncio.iscoroutinefunction(client.chat.completions.create))


class ImageBlockExtractionTests(unittest.TestCase):
    def test_data_url_becomes_an_acp_image_block(self) -> None:
        blocks = extract_image_blocks([_image_message()])

        self.assertEqual(
            blocks, [{"type": "image", "mimeType": "image/png", "data": _PNG}]
        )

    def test_order_of_several_images_is_preserved(self) -> None:
        second = base64.b64encode(b"second").decode()
        messages = [
            _image_message(),
            _image_message(f"data:image/jpeg;base64,{second}"),
        ]

        blocks = extract_image_blocks(messages)

        self.assertEqual([b["data"] for b in blocks], [_PNG, second])
        self.assertEqual([b["mimeType"] for b in blocks], ["image/png", "image/jpeg"])

    def test_text_only_conversation_yields_no_blocks(self) -> None:
        self.assertEqual(
            extract_image_blocks([{"role": "user", "content": "plain text"}]), []
        )

    def test_remote_url_is_not_fetched(self) -> None:
        self.assertEqual(
            extract_image_blocks([_image_message("https://example.com/cat.png")]), []
        )

    def test_non_image_and_malformed_data_urls_are_skipped(self) -> None:
        self.assertEqual(
            extract_image_blocks([_image_message("data:application/pdf;base64,AAAA")]),
            [],
        )
        self.assertEqual(
            extract_image_blocks([_image_message("data:image/png;base64,!!!not-b64")]),
            [],
        )


class PromptBlockAssemblyTests(unittest.TestCase):
    def test_capability_is_read_off_the_initialize_result(self) -> None:
        live = {
            "agentCapabilities": {
                "promptCapabilities": {"image": True, "embeddedContext": True}
            }
        }

        self.assertTrue(engine_accepts_images(live))
        self.assertFalse(engine_accepts_images({"agentCapabilities": {}}))
        self.assertFalse(engine_accepts_images(None))

    def test_text_first_then_images(self) -> None:
        images = extract_image_blocks([_image_message()])

        blocks = build_prompt_blocks("describe it", images, accepts_images=True)

        self.assertEqual(blocks[0], {"type": "text", "text": "describe it"})
        self.assertEqual(blocks[1]["type"], "image")

    def test_engine_without_image_capability_refuses_rather_than_strips(self) -> None:
        images = extract_image_blocks([_image_message()])

        with self.assertRaises(ACPImagesUnsupported):
            build_prompt_blocks("describe it", images, accepts_images=False)

    def test_no_images_means_the_plain_text_prompt(self) -> None:
        self.assertEqual(
            build_prompt_blocks("hello", [], accepts_images=False),
            [{"type": "text", "text": "hello"}],
        )


class EndToEndPromptShapeTests(unittest.TestCase):
    """The image survives _create_chat_completion → the session/prompt payload."""

    def test_image_reaches_the_wire(self) -> None:
        client = CopilotACPClient(acp_cwd="/tmp")
        sent: dict = {}

        def _fake_run_prompt(prompt_text, **kwargs):
            init_result = {
                "agentCapabilities": {"promptCapabilities": {"image": True}}
            }
            from agent.acp_image_blocks import build_prompt_blocks as _build

            sent["blocks"] = _build(
                prompt_text,
                getattr(client, "_acp_prompt_images", None),
                accepts_images=engine_accepts_images(init_result),
            )
            return "a red circle", ""

        with patch.object(client, "_run_prompt", side_effect=_fake_run_prompt):
            completion = client._create_chat_completion(
                model="claude-opus-5", messages=[_image_message()]
            )

        self.assertEqual(completion.choices[0].message.content, "a red circle")
        self.assertEqual([b["type"] for b in sent["blocks"]], ["text", "image"])
        self.assertEqual(sent["blocks"][1]["data"], _PNG)

    def test_text_only_call_still_sends_one_text_block(self) -> None:
        client = CopilotACPClient(acp_cwd="/tmp")

        with patch.object(client, "_run_prompt", return_value=("ok", "")):
            client._create_chat_completion(
                model="claude-opus-5",
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(client._acp_prompt_images, [])


if __name__ == "__main__":
    unittest.main()
