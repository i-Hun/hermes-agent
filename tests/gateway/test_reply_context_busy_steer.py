"""A reply must reach the addressee of the answer — through the inbound turn AND through a mid-turn steer.

Incident 06.11.2026: the secretary profile was mid-turn when Олег replied to its progress message with
«Что это значит». The message arrived as an out-of-band steer, and the steer renderer emitted only
the text plus the origin JSON: the reply pointer and the quoted text were dropped even though the
adapter had resolved them, so the agent saw a bare question and had to ask which message was meant.
The normal inbound path (``_prepend_inbound_reply_context``) already injected it; the steer path never did.

Cases below use the real group/topic ids of that incident (chat -1003948593463, topic 22).
"""

import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

GROUP_CHAT = "-1003948593463"
TOPIC = "22"
HUMAN_QUOTE = "в magnit-mcp есть места, привязанные к нашему серверу"
BOT_QUOTE = "iteration 42/500, running: terminal"


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source(thread_id: str = TOPIC) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=GROUP_CHAT,
        chat_name="Агенты",
        chat_type="group",
        user_id="176320333",
        user_name="Олег Нагорный",
        thread_id=thread_id,
    )


def _event(text: str, **reply) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="58426", **reply)


def _origin(payload: str) -> dict:
    return json.loads(payload.splitlines()[1])


# -- mid-turn steering (the path that lost the quote) -------------------------------------------


def test_steer_carries_quoted_reply_of_a_human_message():
    """Case 1, topic 22: reply to another person's message → the quote reaches the busy agent."""
    ev = _event("Что это значит", reply_to_message_id="58415", reply_to_text=HUMAN_QUOTE)
    payload = _runner()._steer_text_with_origin(ev.text, ev)

    assert "[Replying to" in payload
    assert HUMAN_QUOTE in payload
    assert payload.endswith("\n\nЧто это значит")
    assert _origin(payload)["reply_to_message_id"] == "58415"


def test_steer_carries_quoted_reply_of_the_bots_own_message():
    """Case 2, topic 22: reply to OUR OWN progress message → same quote, flagged as ours."""
    ev = _event("Что это значит", reply_to_message_id="58448",
                reply_to_text=BOT_QUOTE, reply_to_is_own_message=True)
    payload = _runner()._steer_text_with_origin(ev.text, ev)

    assert BOT_QUOTE in payload
    assert "your previous message" in payload
    assert _origin(payload)["reply_to_message_id"] == "58448"
    assert _origin(payload)["reply_to_is_own_message"] is True


def test_steer_keeps_the_reply_pointer_when_the_quoted_text_is_unavailable():
    """A rich/echo-less reply still tells the agent WHICH message was quoted (id + resolve hint)."""
    ev = _event("Что это значит", reply_to_message_id="58448",
                reply_to_text=None, reply_to_is_own_message=True)
    payload = _runner()._steer_text_with_origin(ev.text, ev)

    assert "[Replying to" in payload
    assert "58448" in payload
    assert "resolve it from your earlier messages" in payload
    assert _origin(payload)["reply_to_message_id"] == "58448"


def test_steer_without_a_reply_adds_no_note():
    """A plain message must not grow a fake quote line."""
    ev = _event("Это сделал?")
    payload = _runner()._steer_text_with_origin(ev.text, ev)

    assert "[Replying to" not in payload
    assert payload.endswith("\n\nЭто сделал?")
    assert "reply_to_message_id" not in _origin(payload)


def test_busy_steer_entry_point_delivers_the_quote_to_the_running_agent():
    """The real incident path: steer mode on a live agent, not the renderer in isolation."""
    runner = _runner()
    source = _source()
    ev = _event("Что это значит", reply_to_message_id="58448",
                reply_to_text=BOT_QUOTE, reply_to_is_own_message=True)

    class Receiver:
        payload = None

        def steer(self, text):
            self.payload = text
            return True

    receiver = Receiver()
    runner._session_state("key").turn.agent = receiver
    import asyncio

    asyncio.get_event_loop_policy()  # keep pytest-asyncio-free: this test is a plain sync driver
    ok = asyncio.run(runner._resolve_busy_steer_or_redirect(ev, "key", "steer", receiver))
    assert ok.steered is True
    assert BOT_QUOTE in receiver.payload
    assert "your previous message" in receiver.payload


# -- inbound turn path (regression guards) -------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_turn_keeps_the_quote_for_a_human_reply():
    runner = _runner()
    source = _source()
    ev = MessageEvent(text="Что это значит", source=source, message_id="58426",
                      reply_to_message_id="58415", reply_to_text=HUMAN_QUOTE)
    text = await runner._prepare_inbound_message_text(event=ev, source=source, history=[])
    assert text.startswith("[Replying to")
    assert HUMAN_QUOTE in text


@pytest.mark.asyncio
async def test_inbound_turn_keeps_the_pointer_when_text_is_unavailable():
    runner = _runner()
    source = _source()
    ev = MessageEvent(text="Что это значит", source=source, message_id="58426",
                      reply_to_message_id="58448", reply_to_text=None, reply_to_is_own_message=True)
    text = await runner._prepare_inbound_message_text(event=ev, source=source, history=[])
    assert text.startswith("[Replying to")
    assert "58448" in text


# -- telegram adapter: which inbound shapes count as a reply -------------------------------------
# The adapter resolves the pointer. In a forum topic Telegram also anchors every plain message on
# the topic-root service message; that anchor is NOT a reply and must not become quote noise.


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***",
                                    extra={"allowed_topics": [], "allowed_chats": []})
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._message_handler = None
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def _topic_message(text, *, reply_to=None, thread_id=TOPIC, is_topic_message=True):
    return SimpleNamespace(
        message_id=58426, text=text, caption=None, entities=[], caption_entities=[],
        message_thread_id=int(thread_id), is_topic_message=is_topic_message,
        chat=SimpleNamespace(id=int(GROUP_CHAT), type="supergroup", title="Агенты", is_forum=True),
        from_user=SimpleNamespace(id=176320333, full_name="Олег Нагорный", first_name="Олег"),
        reply_to_message=reply_to, quote=None, date=None, forum_topic_created=None,
    )


def _reply(from_user_id, message_id, text=None, caption=None, forum_topic_created=None):
    return SimpleNamespace(
        message_id=message_id, text=text, caption=caption,
        from_user=SimpleNamespace(id=from_user_id, full_name="Кто-то") if from_user_id else None,
        forum_topic_created=forum_topic_created,
    )


def test_topic_root_anchor_is_not_a_reply():
    """A plain message in topic 22 carries the topic-root anchor (id == thread id) — no quote."""
    adapter = _adapter()
    msg = _topic_message("Это сделал?",
                         reply_to=_reply(None, int(TOPIC), forum_topic_created=object()))
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_message_id is None
    assert event.reply_to_text is None


def test_reply_to_a_human_in_a_topic_resolves_text():
    adapter = _adapter()
    msg = _topic_message("Что это значит", reply_to=_reply(176320333, 58415, text=HUMAN_QUOTE))
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_message_id == "58415"
    assert event.reply_to_text == HUMAN_QUOTE
    assert event.reply_to_is_own_message is False


def test_reply_to_our_own_message_is_flagged_and_quote_survives_to_the_busy_agent():
    adapter = _adapter()
    msg = _topic_message("Что это значит", reply_to=_reply(999, 58448, text=BOT_QUOTE))
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_message_id == "58448"
    assert event.reply_to_is_own_message is True
    payload = _runner()._steer_text_with_origin(event.text, event)
    assert BOT_QUOTE in payload
    assert "your previous message" in payload


def test_reply_to_an_echoless_message_keeps_the_pointer():
    """Rich messages come back with no text/caption; the pointer must survive anyway."""
    adapter = _adapter()
    msg = _topic_message("Что это значит", reply_to=_reply(999, 58448))
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_message_id == "58448"
    assert event.reply_to_is_own_message is True


# -- batching: a reply merged behind a plain lead-in must not lose its quote ---------------------


def test_batch_merge_adopts_reply_context_of_a_later_message():
    seed = _event("конец сообщения")
    seed.absorb_reply_context(_event("Что это значит", reply_to_message_id="58415",
                                     reply_to_text=HUMAN_QUOTE))

    assert seed.reply_to_message_id == "58415"
    assert seed.reply_to_text == HUMAN_QUOTE


def test_batch_merge_keeps_the_seed_quote_when_both_are_replies():
    seed = _event("first", reply_to_message_id="58415", reply_to_text="первый")
    seed.absorb_reply_context(_event("second", reply_to_message_id="58420", reply_to_text="второй"))

    assert seed.reply_to_message_id == "58415"
    assert seed.reply_to_text == "первый"


def test_batch_merge_does_not_invent_a_reply():
    seed = _event("plain")
    seed.absorb_reply_context(_event("also plain"))
    assert seed.reply_to_message_id is None
    assert seed.reply_to_text is None
