"""Carrying images across the ACP wire instead of dropping them.

Local overlay module (not upstream).

``_format_messages_as_prompt`` renders the OpenAI-shaped conversation into
one text blob, and ``_render_message_content`` keeps only the ``text`` parts
of a multimodal message. Every ``{"type": "image_url", ...}`` part was
therefore discarded without a word, and the engine received

    User:
    Describe everything visible in this image in thorough detail. …

with no image attached — an instruction to describe something that is not
there. That is a worse failure than the ``await`` crash it hid behind
(agent/acp_async_client): a crash is reported, a stripped image is answered
confidently from nothing.

ACP carries images natively. ``claude-agent-acp`` 0.58.1 advertises

    "promptCapabilities": {"image": true, "embeddedContext": true}

in its ``initialize`` result, and the prompt array takes
``{"type": "image", "data": <base64>, "mimeType": <mime>}`` blocks alongside
the text one. So the fix is to stop flattening the image away and put it on
the wire in the shape the protocol already defines.

Two deliberate choices:

* **Capability-gated, and loud when absent.** If an engine does not claim
  ``promptCapabilities.image``, this raises instead of sending the text
  alone. A refusal ("this engine cannot accept images") is recoverable
  information; a hallucinated description of an image the model never saw is
  not.
* **Remote URLs are left as text.** Only ``data:`` URLs become blocks. An
  ``http(s)`` image URL would require this module to fetch it — a network
  read on behalf of whatever produced the message — so it stays in the text
  prompt, where the engine can decide to fetch it with its own tools.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any

_log = logging.getLogger(__name__)

# data:image/png;base64,iVBORw0K…
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<params>;[^,]*)?,(?P<payload>.*)$",
    re.DOTALL,
)

_DEFAULT_MIME = "image/png"


class ACPImagesUnsupported(RuntimeError):
    """The ACP engine did not advertise ``promptCapabilities.image``."""


def _parse_data_url(url: str) -> tuple[str, str] | None:
    """Return ``(mime, base64_payload)`` for a base64 ``data:`` image URL."""
    match = _DATA_URL_RE.match(url.strip())
    if not match:
        return None
    params = match.group("params") or ""
    if "base64" not in params.lower():
        # Percent-encoded data URLs are legal but never produced by Hermes'
        # vision path; re-encoding one here would be guesswork about charset.
        return None
    mime = (match.group("mime") or _DEFAULT_MIME).lower()
    if not mime.startswith("image/"):
        return None
    payload = re.sub(r"\s+", "", match.group("payload") or "")
    if not payload:
        return None
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    return mime, payload


def _image_url_of(part: dict[str, Any]) -> str | None:
    """Pull the URL out of an OpenAI image content part."""
    if part.get("type") not in {"image_url", "input_image", "image"}:
        return None
    raw = part.get("image_url") or part.get("image") or part.get("url")
    if isinstance(raw, dict):
        raw = raw.get("url")
    if isinstance(raw, str) and raw.strip():
        return raw
    return None


def extract_image_blocks(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Collect ACP image blocks from an OpenAI-shaped message list.

    Order is preserved: an engine that receives several images needs the same
    sequence the caller wrote them in for "the first screenshot" to mean
    anything.
    """
    blocks: list[dict[str, str]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = _image_url_of(part)
            if not url:
                continue
            parsed = _parse_data_url(url)
            if parsed is None:
                # Remote URL or an unparseable payload — the text prompt still
                # mentions it, so say why it is not attached rather than
                # letting the count of images silently differ from the count
                # the caller sent.
                _log.debug("ACP image block skipped: not an inline base64 data URL")
                continue
            mime, payload = parsed
            blocks.append({"type": "image", "mimeType": mime, "data": payload})
    return blocks


def engine_accepts_images(init_result: Any) -> bool:
    """Read ``promptCapabilities.image`` off an ACP ``initialize`` result."""
    if not isinstance(init_result, dict):
        return False
    caps = init_result.get("agentCapabilities")
    if not isinstance(caps, dict):
        return False
    prompt_caps = caps.get("promptCapabilities")
    if not isinstance(prompt_caps, dict):
        return False
    return bool(prompt_caps.get("image"))


def build_prompt_blocks(
    prompt_text: str,
    images: list[dict[str, str]] | None,
    *,
    accepts_images: bool,
) -> list[dict[str, str]]:
    """Assemble the ``session/prompt`` array: text first, then images.

    Raises :class:`ACPImagesUnsupported` when images are present but the
    engine cannot take them — see the module docstring.
    """
    blocks: list[dict[str, str]] = [{"type": "text", "text": prompt_text}]
    if not images:
        return blocks
    if not accepts_images:
        raise ACPImagesUnsupported(
            f"The ACP engine did not advertise promptCapabilities.image, so "
            f"{len(images)} image(s) cannot be delivered. Refusing to send the "
            f"prompt text alone — the answer would describe an image the model "
            f"never received."
        )
    blocks.extend(images)
    _log.info("ACP prompt carries %d image block(s)", len(images))
    return blocks
