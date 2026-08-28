from types import SimpleNamespace

from agent.agent_init import apply_custom_provider_extra_body




def test_custom_provider_extra_body_preserves_caller_override():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={
            "extra_body": {
                "reasoning_effort": "low",
                "caller_only": True,
            }
        },
    )

    apply_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "caller_only": True,
    }




def test_named_custom_provider_extra_body_matches_provider_key():
    agent = SimpleNamespace(
        provider="custom:zai-coding-plan",
        model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4",
        request_overrides={},
    )

    apply_custom_provider_extra_body(
        agent,
        [
            {
                "provider_key": "other-provider",
                "name": "Other Provider",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": True},
            },
            {
                "provider_key": "zai-coding-plan",
                "name": "Z.AI Coding Plan",
                "base_url": "https://api.z.ai/api/coding/paas/v4/",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": False},
            },
        ],
    )

    assert agent.request_overrides == {"extra_body": {"enable_thinking": False}}


_ROUTES = [
    {
        "provider_key": "litellm-proxy",
        "name": "LiteLLM Proxy",
        "base_url": "https://proxy.test/api",
        "model": "ProxyModel",
        "extra_body": {"drop_params": True},
    },
    {
        "provider_key": "vllm-box",
        "name": "vLLM Box",
        "base_url": "https://vllm.test/v1",
        "model": "glm-5.2",
        "extra_body": {"enable_thinking": True},
    },
]


def test_extra_body_follows_the_route_on_switch():
    """A re-route must send the destination's extra_body, not the origin's."""
    agent = SimpleNamespace(
        provider="anthropic",
        model="claude-sonnet-5",
        base_url="https://api.anthropic.com",
        request_overrides={},
    )

    apply_custom_provider_extra_body(agent, _ROUTES)
    assert agent.request_overrides == {}

    # Switch onto the litellm-backed endpoint: without drop_params it 400s on
    # reasoning_effort and the session silently falls back (the bug this guards).
    agent.provider, agent.model = "custom", "ProxyModel"
    agent.base_url = "https://proxy.test/api"
    apply_custom_provider_extra_body(agent, _ROUTES)
    assert agent.request_overrides == {"extra_body": {"drop_params": True}}

    # Switch to a different custom endpoint: previous keys must not survive.
    agent.provider, agent.model = "custom", "glm-5.2"
    agent.base_url = "https://vllm.test/v1"
    apply_custom_provider_extra_body(agent, _ROUTES)
    assert agent.request_overrides == {"extra_body": {"enable_thinking": True}}

    # Switch back to a non-custom route: nothing left over to leak upstream.
    agent.provider, agent.model = "anthropic", "claude-sonnet-5"
    agent.base_url = "https://api.anthropic.com"
    apply_custom_provider_extra_body(agent, _ROUTES)
    assert agent.request_overrides == {}


def test_caller_extra_body_survives_every_reroute():
    """Caller-supplied keys are the user's, not the route's — they persist."""
    agent = SimpleNamespace(
        provider="custom",
        model="ProxyModel",
        base_url="https://proxy.test/api",
        request_overrides={"extra_body": {"caller_only": True}},
    )

    apply_custom_provider_extra_body(agent, _ROUTES)
    assert agent.request_overrides["extra_body"] == {
        "drop_params": True,
        "caller_only": True,
    }

    agent.provider, agent.model = "anthropic", "claude-sonnet-5"
    agent.base_url = "https://api.anthropic.com"
    apply_custom_provider_extra_body(agent, _ROUTES)
    assert agent.request_overrides["extra_body"] == {"caller_only": True}
