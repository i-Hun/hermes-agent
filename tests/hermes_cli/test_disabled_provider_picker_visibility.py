"""Disabled picker visibility is static and never authorizes a runtime route."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import model_switch_providers as picker


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(picker, "_build_curated_lists", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {})
    monkeypatch.setattr("hermes_cli.models.CANONICAL_PROVIDERS", [])
    profile = SimpleNamespace(display_name="Old Claude", fallback_models=("opus", "sonnet"))
    monkeypatch.setattr("providers.get_provider_profile", lambda slug: profile)
    probe = Mock(side_effect=AssertionError("disabled route must not probe/authenticate"))
    monkeypatch.setattr(picker, "_entry_credentials", probe)
    monkeypatch.setattr(picker, "_overlay_has_creds", probe)
    monkeypatch.setattr(picker, "_overlay_has_env_creds", probe)
    monkeypatch.setattr(picker, "_pool_usable", probe)
    monkeypatch.setattr(picker, "_prefetch_provider_models_parallel", probe)
    return probe


@pytest.mark.parametrize("non_blocking", [False, True])
def test_disabled_visible_is_static_and_not_current(offline, non_blocking):
    rows = picker.list_picker_providers(
        current_provider="old", current_model="not-a-static-model", max_models=1,
        user_providers={"old": {"enabled": False, "picker_visible": True,
                                "base_url": "https://must-not-probe.invalid"}},
        non_blocking_catalogs=non_blocking)
    assert len(rows) == 1
    row = rows[0]
    assert row["slug"] == "old"
    assert row["name"] == "Old Claude — отключён"
    assert row["disabled"] is True
    assert row["is_current"] is False
    assert row["models"] == ["opus"]
    assert row["total_models"] == 2
    offline.assert_not_called()


@pytest.mark.parametrize("visible", [False, None, "true"])
def test_disabled_hidden_by_default_and_bool_only(offline, visible):
    assert picker.list_picker_providers(
        user_providers={"old": {"enabled": False, "picker_visible": visible}}) == []
    offline.assert_not_called()


def test_exclusion_still_wins(offline):
    assert picker.list_picker_providers(
        user_providers={"old": {"enabled": False, "picker_visible": True}},
        excluded_providers=["old"]) == []


def test_disabled_config_models_override_fallback(offline):
    rows = picker.list_picker_providers(user_providers={
        "old": {"enabled": False, "picker_visible": True, "models": ["pinned"]}})
    assert rows[0]["models"] == ["pinned"]


def test_disabled_overlay_skips_auth_and_prefetch(offline, monkeypatch):
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {
        "old": SimpleNamespace(auth_type="external_process")})
    rows = picker.list_picker_providers(
        user_providers={"old": {"enabled": False, "picker_visible": True}})
    assert rows[0]["disabled"] is True
    offline.assert_not_called()


def test_visibility_does_not_bypass_runtime_guard(monkeypatch):
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider._config_mod, "load_config", lambda: {
        "providers": {"old": {"enabled": False, "picker_visible": True}}})
    with pytest.raises(ValueError, match="disabled"):
        runtime_provider.resolve_runtime_provider(requested="old")
