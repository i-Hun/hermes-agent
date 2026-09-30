"""Sol 6.1 opt-in context must obey the account catalog ceiling."""
from agent import model_metadata as mm
from hermes_cli.codex_models import _finalize_codex_models


def test_sol61_variant_is_picker_only():
    base = 'gpt-6.1-sol'
    variant = base + mm.CODEX_CONTEXT_VARIANT_SUFFIX
    assert variant in _finalize_codex_models([base])
    assert mm.strip_codex_context_variant_suffix(variant) == base
    assert mm.strip_codex_context_variant_suffix(base) == base


def test_sol61_context_respects_live_ceiling(monkeypatch):
    token = 'test-only-token'
    fp = mm._codex_oauth_token_fingerprint(token)
    monkeypatch.setattr(mm, '_fetch_codex_oauth_context_lengths_with_source',
                        lambda _: ({'gpt-6.1-sol': 272000}, True))
    monkeypatch.setattr(mm, '_codex_oauth_max_context_cache',
                        {fp: {'gpt-6.1-sol': 872000}})
    assert mm._resolve_codex_oauth_context_length_with_source('gpt-6.1-sol', token)[0] == 272000
    assert mm._resolve_codex_oauth_context_length_with_source('gpt-6.1-sol-900k', token)[0] == 872000
    mm._codex_oauth_max_context_cache[fp]['gpt-6.1-sol'] = 700000
    assert mm._resolve_codex_oauth_context_length_with_source('gpt-6.1-sol-900k', token)[0] == 700000
