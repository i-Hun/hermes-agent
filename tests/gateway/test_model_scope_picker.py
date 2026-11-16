"""Behaviour contract for the ``/model`` scope screen (bare ``/model`` on a chat surface).

The picker's tap used to persist however the command's flags resolved — on a profile with
``model.persist_switch_by_default: true`` that silently rewrote ``config.yaml``, and tapping a
button gave no way to say "this session only" (only typing ``--session`` did). A bare ``/model``
now asks for the scope first; the tap builds a copy of the switch context with the chosen
``persist_global`` and then opens the usual provider → model picker.

The adapter is fake but capability-shaped (``send_choice_picker`` + ``send_model_picker`` on the
class, because the handler gates on the type), and the real ``_handle_model_command`` runs against a
temp ``HERMES_HOME``: the assertions are about ``config.yaml`` on disk, not about the closures.
"""

import types

import hermes_yaml as yaml
import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakeScopeAdapter:
    """Captures both pickers; ``sent`` records the order, which is the feature under test."""

    def __init__(self, scope_success=True):
        self.scope_callback = None
        self.scope_title = None
        self.scope_choices = None
        self.model_callback = None
        self.sent = []
        self.scope_success = scope_success

    async def send_choice_picker(self, *, title, choices, on_choice_selected, **kwargs):
        self.scope_title, self.scope_choices = title, choices
        self.scope_callback = on_choice_selected
        self.sent.append("scope")
        return types.SimpleNamespace(success=self.scope_success)

    async def send_model_picker(self, *, on_model_selected, providers, **kwargs):
        self.model_callback = on_model_selected
        self.sent.append("model")
        return types.SimpleNamespace(success=True)


class _FakePlainAdapter:
    """A surface with no picker at all (plain text chat): only the listing can answer."""

    name = "plain"


class _FakeModelOnlyAdapter:
    """A Slack-shaped surface: a model picker but no flat choice picker."""

    def __init__(self):
        self.model_callback = None
        self.sent = []

    async def send_model_picker(self, *, on_model_selected, providers, **kwargs):
        self.model_callback = on_model_selected
        self.sent.append("model")
        return types.SimpleNamespace(success=True)


def _make_runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openrouter",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
        is_global=True,
    )


def _setup_isolated_home(tmp_path, monkeypatch, *, persist_by_default, providers=None):
    """Temp ``HERMES_HOME`` with a configured route; ``providers`` seeds the picker's catalog."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "model": {
                "default": "old-model",
                "provider": "custom",
                "base_url": "https://api.custom.example/v1",
                "persist_switch_by_default": persist_by_default,
            },
            "providers": {},
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **kw: providers if providers is not None
        else [{"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-5.5"]}],
    )
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: _fake_switch_result())
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: 272000)
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [])
    return cfg_path


def _written_model(cfg_path):
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_by_default", [True, False], ids=["persist-default", "session-default"])
async def test_bare_model_asks_scope_before_listing_models(tmp_path, monkeypatch, persist_by_default):
    """Bare ``/model`` sends the scope screen only — no provider list yet — and marks the scope the
    command would have used, so the screen never contradicts the flags it was opened with."""
    adapter = _FakeScopeAdapter()
    _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=persist_by_default)

    sent = await _make_runner(adapter)._handle_model_command(_make_event("/model"))

    assert sent is None, "the scope picker owns the reply; no text should be returned"
    assert adapter.sent == ["scope"]
    assert adapter.model_callback is None, "models must not be listed before the scope is chosen"
    assert [choice["value"] for choice in adapter.scope_choices] == ["session", "global"]
    assert [choice["is_current"] for choice in adapter.scope_choices] == [not persist_by_default, persist_by_default]


@pytest.mark.asyncio
async def test_scope_session_tap_keeps_persist_default_out_of_config(tmp_path, monkeypatch):
    """The profile opts into persist-by-default; tapping "this session" must still leave
    ``config.yaml`` untouched — that is the whole point of the screen."""
    adapter = _FakeScopeAdapter()
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=True)
    runner = _make_runner(adapter)
    await runner._handle_model_command(_make_event("/model"))

    confirmation = await adapter.scope_callback("12345", "session")

    assert adapter.sent == ["scope", "model"], "the scope tap must open the provider→model picker"
    assert adapter.model_callback is not None
    assert "This session only" in confirmation
    switch_reply = await adapter.model_callback("12345", "gpt-5.5", "openrouter")
    assert "gpt-5.5" in switch_reply
    assert _written_model(cfg_path)["default"] == "old-model"


@pytest.mark.asyncio
async def test_scope_global_tap_writes_config(tmp_path, monkeypatch):
    """No persist-by-default opt-in; choosing "globally" is the only thing that makes the tap write."""
    adapter = _FakeScopeAdapter()
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=False)
    runner = _make_runner(adapter)
    await runner._handle_model_command(_make_event("/model"))

    confirmation = await adapter.scope_callback("12345", "global")
    switch_reply = await adapter.model_callback("12345", "gpt-5.5", "openrouter")

    assert "gpt-5.5" in switch_reply
    assert "Globally" in confirmation
    written = _written_model(cfg_path)
    assert written["default"] == "gpt-5.5"
    assert written["provider"] == "openrouter"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "persist_by_default"),
    [("/model --global", False), ("/model --session", True)],
    ids=["explicit-global", "explicit-session"],
)
async def test_explicit_scope_flag_skips_the_scope_screen(tmp_path, monkeypatch, command, persist_by_default):
    """A typed flag already answered the question — asking again would be noise."""
    adapter = _FakeScopeAdapter()
    _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=persist_by_default)

    sent = await _make_runner(adapter)._handle_model_command(_make_event(command))

    assert sent is None
    assert adapter.sent == ["model"]
    assert adapter.scope_callback is None


@pytest.mark.asyncio
async def test_scope_picker_send_failure_falls_through_to_the_direct_picker(tmp_path, monkeypatch):
    """A scope screen the platform refused to send must not cost the user the model list."""
    adapter = _FakeScopeAdapter(scope_success=False)
    _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=True)

    sent = await _make_runner(adapter)._handle_model_command(_make_event("/model"))

    assert sent is None
    assert adapter.sent == ["scope", "model"]
    assert adapter.model_callback is not None


@pytest.mark.asyncio
async def test_surface_without_any_picker_returns_the_text_listing(tmp_path, monkeypatch):
    adapter = _FakePlainAdapter()
    _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=True)

    reply = await _make_runner(adapter)._handle_model_command(_make_event("/model"))

    assert "Current:" in reply
    assert "/model <name> --global" in reply


@pytest.mark.asyncio
async def test_unknown_scope_value_never_writes_config(tmp_path, monkeypatch):
    """Fail-closed: only an explicit "global" tap may rewrite config.yaml, so a value the screen
    never offers (a stale button from an older message) keeps the write out."""
    adapter = _FakeScopeAdapter()
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=True)
    runner = _make_runner(adapter)
    await runner._handle_model_command(_make_event("/model"))

    await adapter.scope_callback("12345", "bogus")
    switch_reply = await adapter.model_callback("12345", "gpt-5.5", "openrouter")

    assert "gpt-5.5" in switch_reply
    assert _written_model(cfg_path)["default"] == "old-model"


@pytest.mark.asyncio
async def test_surface_without_choice_picker_keeps_the_direct_picker(tmp_path, monkeypatch):
    """Slack has a model picker but no flat choice picker: same tap-to-pick flow as before."""
    adapter = _FakeModelOnlyAdapter()
    _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=True)

    sent = await _make_runner(adapter)._handle_model_command(_make_event("/model"))

    assert sent is None
    assert adapter.sent == ["model"]


@pytest.mark.asyncio
async def test_empty_catalog_falls_back_to_the_text_listing(tmp_path, monkeypatch):
    """No provider can be listed after the scope tap: the screen's reply becomes the text list
    instead of a confirmation pointing at a picker that never arrived."""
    adapter = _FakeScopeAdapter()
    _setup_isolated_home(tmp_path, monkeypatch, persist_by_default=True, providers=[])
    runner = _make_runner(adapter)
    await runner._handle_model_command(_make_event("/model"))

    reply = await adapter.scope_callback("12345", "session")

    assert "Current:" in reply
    assert adapter.sent == ["scope"], "an empty catalog must not pretend a picker was sent"
