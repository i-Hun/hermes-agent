"""The agent's memory config must bound the external provider's actual join."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from agent.agent_init import _init_memory
from agent.memory_manager import MemoryManager
from tests.agent.test_memory_provider import FakeMemoryProvider


@pytest.mark.parametrize("setting, expected", [("", 8.0), ("  external_prefetch_timeout: 25\n", 25.0)])
@pytest.mark.parametrize("stuck", [False, True])
def test_config_reaches_prefetch_join(tmp_path, monkeypatch, caplog, setting, expected, stuck):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory:\n  provider: timeout-test\n" + setting)
    config = yaml.safe_load(config_path.read_text())
    provider = FakeMemoryProvider("timeout-test")
    provider._prefetch_result = "recalled context"
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: provider)
    # Only unrelated agent identity and facade logging are stubbed. Keep the real
    # config normalization, constructor, registration and prefetch dispatch.
    monkeypatch.setattr("agent.agent_init._memory_provider_init_kwargs", lambda *args: {"session_id": "test"})
    monkeypatch.setattr("agent.agent_init._ra", lambda: SimpleNamespace(logger=logging.getLogger(__name__)))
    agent = SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[], tools=[])
    _init_memory(agent, config, False, "telegram")
    manager = agent._memory_manager
    assert isinstance(manager, MemoryManager)
    assert provider.initialized

    thread = Mock()
    thread.is_alive.return_value = stuck

    def spawn(target, **kwargs):
        if not stuck:
            thread.start.side_effect = target
        return thread

    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", spawn)
    with caplog.at_level(logging.WARNING, logger="agent.memory_manager"):
        result = manager.prefetch_all("query", session_id="test")
    thread.start.assert_called_once_with()
    thread.join.assert_called_once_with(expected)
    if stuck:
        assert result == ""
        assert f"prefetch timed out after {expected:.1f}s" in caplog.text
        assert "skipping it until the stuck call returns" in caplog.text
        assert manager.prefetch_all("next query", session_id="test") == ""
        thread.start.assert_called_once_with()
        thread.join.assert_called_once_with(expected)
    else:
        assert result == "recalled context"
        assert provider.prefetch_queries == ["query"]
        assert "timed out" not in caplog.text


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), float("-inf"), "nan", "inf"])
def test_invalid_prefetch_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="external_prefetch_timeout must be positive"):
        MemoryManager(external_prefetch_timeout=timeout)
