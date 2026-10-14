"""Sync provider facades must not see Relay's private asyncio loop."""
import asyncio
import contextvars
import threading
from types import SimpleNamespace

import pytest

from agent import relay_llm, relay_runtime
from agent.turn_recovery import describe_invalid_response

scope = contextvars.ContextVar("provider_scope", default="outside")


def attempt():
    token = scope.set("provider")
    try:
        result = object.__new__(relay_llm._ManagedAttempt)
        result.context = contextvars.copy_context()
        result.raw_response = {}
        result.provider_request = lambda next_request: next_request
        return result
    finally:
        scope.reset(token)


def guarded():
    assert scope.get() == "provider"
    assert relay_runtime._MANAGED_CALLBACK_DEPTH.get() > 0


class DualModeClient:
    """Same public sync/async selection contract as DirectSDK Client.create."""

    def __init__(self, provider):
        self.provider = provider

    def create(self, request):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.provider(request)
        return self.acreate(request)

    async def acreate(self, request):
        return await asyncio.to_thread(self.provider, request)


@pytest.mark.parametrize("inside_loop", [False, True])
def test_sync_facade_returns_completion_not_coroutine(inside_loop):
    owner = attempt()
    value = SimpleNamespace(content="ok")
    calls = []

    def provider(request):
        guarded()
        calls.append(request)
        return value

    client = DualModeClient(provider)

    def check(result):
        assert result == {"content": "ok"}
        assert owner.raw_response["value"] is value
        assert calls == [{}]
        assert scope.get() == "outside"

    async def under_loop():
        check(await owner.invoke(client.create, {}))

    if inside_loop:
        asyncio.run(under_loop())
    else:
        check(owner.invoke(client.create, {}))


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_sync_errors_keep_identity_and_are_recorded(error_type):
    owner = attempt()
    error = error_type("provider error")

    def fail(request):
        guarded()
        raise error

    async def run():
        with pytest.raises(error_type) as caught:
            await owner.invoke(DualModeClient(fail).create, {})
        assert caught.value is error
        assert owner.raw_response == {"error": error}

    asyncio.run(run())


def test_async_cancellation_reaches_provider():
    owner = attempt()

    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def provider(request):
            guarded()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(owner.invoke_async(provider, {}))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert isinstance(owner.raw_response["error"], asyncio.CancelledError)
        assert "value" not in owner.raw_response

    asyncio.run(run())


def test_sync_provider_external_interrupt_unblocks():
    owner = attempt()
    entered = threading.Event()
    cancel = threading.Event()

    def interrupt():
        if entered.wait(2):
            cancel.set()

    worker = threading.Thread(target=interrupt, daemon=True)
    worker.start()

    def provider(request):
        guarded()
        entered.set()
        assert cancel.wait(2), "interrupt did not reach provider"
        raise InterruptedError("cancelled")

    async def run():
        with pytest.raises(InterruptedError, match="cancelled"):
            await owner.invoke(DualModeClient(provider).create, {})

    try:
        asyncio.run(run())
    finally:
        worker.join(3)
    assert not worker.is_alive()
    assert "value" not in owner.raw_response


@pytest.mark.parametrize("early_close", [False, True])
def test_stream_facade_sync_iteration_and_cleanup(early_close):
    owner = attempt()
    closed = []

    class Stream:
        def __iter__(self):
            guarded()
            return iter([{"delta": "ok"}, {"delta": "more"}])

        def close(self):
            guarded()
            closed.append(True)

    def factory(request):
        guarded()
        return Stream()

    stream = object.__new__(relay_llm.ManagedLlmStream)
    stream._closed = True  # test owns the provider generator, not a Relay lease
    stream._stream_factory = DualModeClient(factory).create
    stream._completed_response_predicate = None
    stream._on_stream_created = None
    stream._accept_chunk = None
    stream._raw_chunks = []

    async def run():
        generator = stream._provider_stream(owner, {})
        try:
            assert await anext(generator) == {"delta": "ok"}
            if not early_close:
                assert [item async for item in generator] == [{"delta": "more"}]
        finally:
            await generator.aclose()

    asyncio.run(run())
    assert closed == [True]
    assert stream._provider_completed is (not early_close)


@pytest.mark.parametrize("response", ["bad", 42, ["bad"], object(), "", 0])
@pytest.mark.parametrize("verbose", [False, True])
def test_malformed_diagnostics_without_dict(response, verbose):
    message, provider, hint = describe_invalid_response(
        SimpleNamespace(verbose_logging=verbose), response, 0.0,
    )
    assert type(response).__name__ in message
    assert provider == "Unknown"
    assert hint


def test_provider_error_diagnostics_are_preserved():
    response = SimpleNamespace(error=SimpleNamespace(message="rate limit", code=429), model="fixture")
    message, provider, hint = describe_invalid_response(SimpleNamespace(verbose_logging=False), response, 0)
    assert "rate limit" in message
    assert provider == "model=fixture"
    assert "429" in hint


@pytest.fixture()
def native_turn(tmp_path, monkeypatch):
    pytest.importorskip("nemo_relay")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="dual-mode", platform="cron",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn", task_id="test")
    lease.host.retain_managed_execution("test.dual")
    token = scope.set("provider")
    try:
        yield
    finally:
        scope.reset(token)
        lease.host.release_managed_execution("test.dual")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "error"])
def test_native_relay_dual_mode_facade(native_turn, mode):
    value = SimpleNamespace(content="done")
    calls = []

    def provider(request):
        guarded()
        calls.append(request)
        if mode == "error":
            raise ValueError("provider failure")
        return iter([{"delta": "done"}]) if mode == "stream" else value

    client = DualModeClient(provider)
    kwargs = dict(session_id="dual-mode", name="fixture", model_name="fixture")
    if mode == "sync":
        assert relay_llm.execute({}, client.create, **kwargs) is value
    elif mode == "async":
        assert asyncio.run(relay_llm.execute_async({}, client.create, **kwargs)) is value
    elif mode == "stream":
        stream = relay_llm.stream({}, client.create, finalizer=lambda: {"content": "done"}, **kwargs)
        try:
            assert list(stream) == [{"delta": "done"}]
        finally:
            stream.close()
    else:
        with pytest.raises(ValueError, match="provider failure"):
            relay_llm.execute({}, client.create, **kwargs)
    assert calls == [{}]


def provider_stream(factory):
    stream = object.__new__(relay_llm.ManagedLlmStream)
    stream._closed = True
    stream._stream_factory = factory
    stream._completed_response_predicate = None
    stream._on_stream_created = None
    stream._accept_chunk = None
    stream._raw_chunks = []
    return stream


@pytest.mark.parametrize("early_close", [False, True])
def test_async_iterator_keeps_task_context_and_closes_distinct_iterator(early_close):
    owner = attempt()
    closed = []
    local = contextvars.ContextVar("stream-local", default=None)

    class Iterable:
        def __aiter__(self):
            async def generate():
                guarded()
                task = asyncio.current_task()
                token = local.set("owned")
                try:
                    yield {"delta": "first"}
                    assert asyncio.current_task() is task
                    assert local.get() == "owned"
                    yield {"delta": "second"}
                finally:
                    assert asyncio.current_task() is task
                    local.reset(token)
                    closed.append("iterator")
            return generate()

        async def aclose(self):
            guarded()
            closed.append("resource")

    async def factory(request):
        guarded()
        return Iterable()

    stream = provider_stream(factory)

    async def run():
        generator = stream._provider_stream(owner, {})
        try:
            assert await anext(generator) == {"delta": "first"}
            if not early_close:
                assert [item async for item in generator] == [{"delta": "second"}]
        finally:
            await generator.aclose()
        assert local.get() is None

    asyncio.run(run())
    assert closed == ["iterator", "resource"]


def test_cancel_waiting_stream_factory_joins_provider_cleanup():
    owner = attempt()
    closed = []

    async def run():
        entered = asyncio.Event()

        async def factory(request):
            guarded()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                closed.append(True)

        stream = provider_stream(factory)
        generator = stream._provider_stream(owner, {})
        pull = asyncio.create_task(anext(generator))
        await asyncio.wait_for(entered.wait(), timeout=2)
        pull.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pull
        await generator.aclose()
        assert closed == [True]
        assert not stream._provider_completed

    asyncio.run(run())


@pytest.mark.parametrize("external", [False, True])
def test_native_process_format_does_not_imply_http_headers(native_turn, monkeypatch, external):
    from hermes_cli import runtime_provider_backends
    monkeypatch.setattr(runtime_provider_backends, "_is_external_process_provider", lambda name: external)
    request = {"model": "fixture", "messages": []}
    owner = relay_llm._ManagedAttempt.resolve(
        "dual-mode", request, {"api_mode": "chat_completions"}, name="fixture", model_name="fixture",
    )
    assert owner is not None
    intercepted = SimpleNamespace(content=request, headers={"traceparent": "trace"})
    final = owner.provider_request(intercepted)
    assert ("extra_headers" in final) is (not external)
    intercepted.headers["x-middleware"] = "keep"
    final = owner.provider_request(intercepted)
    assert final["extra_headers"]["x-middleware"] == "keep"
    assert ("traceparent" in final["extra_headers"]) is (not external)
