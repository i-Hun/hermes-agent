"""``/model`` on a profile whose engine is the ACP bridge (local overlay).

Covers agent/acp_model_choice.py and the two call-outs it owns in
``hermes_cli.model_switch.switch_model``.

The defect these guard against, measured 2026-08-09 in a live topic: ``/model
claude-fable-5`` resolved ``claude-*`` to the direct Anthropic API, every turn
answered ``400 Third-party apps now draw from your extra usage``, slid back onto
the bridge, and answered on the model ``.env`` pinned — while the ack said the
switch was done.

Since 2026-08-10 the model has one source, ``model.default`` in ``config.yaml``,
which a ``--global`` switch already persists — so the ack seam writes nothing and
was cut down to ``filter_switch_warnings``: drop one warning that fires on the
healthy case, and say when a switch leaves the bridge. Everything it used to
narrate (which scope moved, a leftover ``.env`` pin) had a second, better
reporter — the gateway's own localized scope line, and ``acp_config.enforce``,
which refuses to start over that pin.
"""

from unittest.mock import patch

import pytest

from agent import acp_config, acp_model_choice


@pytest.fixture(autouse=True)
def _isolated_profile(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    # A migrated .env: secrets and gates, no model. That is the whole point.
    (home / ".env").write_text(
        "# comment\nTELEGRAM_BOT_TOKEN=shhh\n", encoding="utf-8",
    )
    (home / ".env").chmod(0o600)
    (home / "config.yaml").write_text(
        "model:\n  default: claude-opus-5\n  provider: copilot-acp\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (acp_config.MODEL_VAR, acp_config.FALLBACK_VAR):
        monkeypatch.delenv(var, raising=False)
    acp_config._DOTENV_CACHE.clear()
    acp_config._CONFIG_CACHE.clear()
    acp_config._WARNED.clear()
    yield home
    acp_config._DOTENV_CACHE.clear()
    acp_config._CONFIG_CACHE.clear()
    acp_config._WARNED.clear()


def _config_text(home):
    return (home / "config.yaml").read_text()


def _dotenv_text(home):
    return (home / ".env").read_text()


def _leftover(home, value, var=acp_config.MODEL_VAR):
    (home / ".env").write_text(f"{var}={value}\n", encoding="utf-8")
    acp_config._DOTENV_CACHE.clear()


def _set_model(value):
    acp_config.set_model(value)
    acp_config._CONFIG_CACHE.clear()


# --- the route ---------------------------------------------------------------

@pytest.mark.parametrize("raw,explicit,current,expected", [
    # The defect: an engine model on an ACP profile must stay on the bridge.
    ("claude-fable-5", "", "copilot-acp", "copilot-acp"),
    ("anthropic/claude-opus-5", "", "copilot-acp", "copilot-acp"),
    # An explicit --provider always wins: leaving the bridge stays possible,
    # it just has to be said out loud.
    ("claude-fable-5", "anthropic", "copilot-acp", ""),
    # Not our business.
    ("claude-fable-5", "", "openrouter", ""),
    ("gpt-5", "", "copilot-acp", ""),
    ("", "", "copilot-acp", ""),
    (None, "", "copilot-acp", ""),
])
def test_coerce_switch(raw, explicit, current, expected):
    assert acp_model_choice.coerce_switch(raw, explicit, current) == (expected, "")


# --- the ack -----------------------------------------------------------------
#
# Cut 2026-08-10 to one job. What it used to add on every switch — "model.default
# moves to X" / "this session only" — duplicated the gateway's own
# ``saved_global`` / ``session_only_hint`` line, which is printed right below it
# and is localized, while the ack was English-only. The leftover-pin note went
# with it: ``acp_config.enforce`` refuses to start a gateway over that pin.

def test_the_copilot_catalog_note_is_dropped_for_engine_models(_isolated_profile):
    """The curated catalog is GitHub Copilot's; the bridge runs claude-agent-acp.

    Measured 2026-08-10, ``/model claude-fable-5`` came back "``claude-fable-5``
    was not found in the GitHub Copilot ACP curated catalog … the model may still
    work" — three lines, on a switch that worked. A warning that fires on the
    healthy case is one people learn to scroll past.
    """
    note = "Note: `claude-opus-5` was not found in the GitHub Copilot ACP curated catalog."
    out = acp_model_choice.filter_switch_warnings(
        new_model="claude-opus-5", target_provider="copilot-acp",
        current_provider="copilot-acp",
        warnings=["something else", note], catalog_note=note,
    )
    assert note not in out
    assert "something else" in out


def test_a_healthy_switch_adds_nothing(_isolated_profile):
    """The guard against re-growing the ack: on the working path this seam is
    silent, because the gateway already says what happened, in the user's
    language."""
    assert acp_model_choice.filter_switch_warnings(
        new_model="claude-fable-5", target_provider="copilot-acp",
        current_provider="copilot-acp",
    ) == []


def test_a_leftover_pin_is_not_the_acks_business(_isolated_profile):
    """It is a start-time defect with a start-time reporter (acp_config.enforce),
    not a line in a reply about something else."""
    _leftover(_isolated_profile, "claude-opus-5")
    assert acp_model_choice.filter_switch_warnings(
        new_model="claude-fable-5", target_provider="copilot-acp",
        current_provider="copilot-acp",
    ) == []


def test_the_ack_writes_nothing_at_all(_isolated_profile):
    """The caller persists ``model.default``; a second writer here is the
    duplicate that was removed on 2026-08-10, and one that can fail silently
    after the ack has already said "saved"."""
    before_cfg, before_env = _config_text(_isolated_profile), _dotenv_text(_isolated_profile)
    _leftover(_isolated_profile, "claude-opus-5")
    after_leftover = _dotenv_text(_isolated_profile)
    acp_model_choice.filter_switch_warnings(
        new_model="claude-fable-5", target_provider="copilot-acp",
        current_provider="copilot-acp",
    )
    assert _config_text(_isolated_profile) == before_cfg
    assert _dotenv_text(_isolated_profile) == after_leftover
    assert before_env  # the fixture's .env really did have content to protect


def test_other_warnings_are_never_dropped(_isolated_profile):
    note = "catalog says no"
    out = acp_model_choice.filter_switch_warnings(
        new_model="claude-opus-5", target_provider="copilot-acp",
        current_provider="copilot-acp",
        warnings=["deprecation notice", note], catalog_note=note,
    )
    assert "deprecation notice" in out


def test_the_catalog_note_survives_off_the_bridge(_isolated_profile):
    note = "catalog says no"
    out = acp_model_choice.filter_switch_warnings(
        new_model="gpt-5", target_provider="openrouter",
        current_provider="openrouter",
        warnings=[note], catalog_note=note,
    )
    assert note in out


def test_leaving_the_bridge_is_announced(_isolated_profile):
    """The one line worth keeping: the direct API answers 400 on a subscription
    login, and nothing else in the ack says the route changed."""
    notes = acp_model_choice.filter_switch_warnings(
        new_model="gpt-5", target_provider="openrouter",
        current_provider="copilot-acp",
    )
    assert any("leaves it" in n and "subscription login" in n for n in notes)
    assert _config_text(_isolated_profile).count("claude-opus-5") == 1


def test_non_acp_profiles_get_nothing(_isolated_profile):
    warnings = ["upstream said something"]
    assert acp_model_choice.filter_switch_warnings(
        new_model="gpt-5", target_provider="openrouter",
        current_provider="openrouter", warnings=warnings,
    ) == warnings


# --- through the real switch_model -------------------------------------------

@pytest.fixture()
def _acp_command(tmp_path, monkeypatch):
    cmd = tmp_path / "claude-acp-run.sh"
    cmd.write_text("#!/bin/sh\nexit 0\n")
    cmd.chmod(0o755)
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(cmd))
    return cmd


@pytest.fixture(autouse=True)
def _no_catalog_network():
    """models.dev / provider /models are HTTP. Not this test's subject."""
    import hermes_cli.model_switch as ms
    import hermes_cli.models as models

    accepted = {"accepted": True, "persist": True, "recognized": True, "message": ""}
    with patch.object(models, "validate_requested_model", return_value=accepted), \
         patch.object(ms, "get_model_info", return_value=None), \
         patch.object(ms, "get_model_capabilities", return_value=None):
        yield


def _switch(**kw):
    from hermes_cli.model_switch import switch_model

    args = dict(
        raw_input="claude-fable-5",
        current_provider="copilot-acp",
        current_model="claude-opus-5",
        current_base_url="acp://copilot",
        current_api_key="copilot-acp",
        is_global=False,
    )
    args.update(kw)
    return switch_model(**args)


def test_switch_model_keeps_an_engine_model_on_the_bridge(_acp_command):
    result = _switch()
    assert result.success, result.error_message
    assert result.target_provider == "copilot-acp"
    # The regression in one assertion: this used to be https://api.anthropic.com,
    # which answers 400 on a subscription login.
    assert "anthropic.com" not in (result.base_url or "")
    assert (result.base_url or "").startswith("acp://")


@pytest.mark.parametrize("is_global", [True, False])
def test_switch_model_says_nothing_on_the_healthy_path(
    _acp_command, _isolated_profile, is_global
):
    """Both scopes, one expectation: an empty ack.

    switch_model itself persists nothing — its caller writes ``model.default``,
    and the gateway prints the scope line. Asserting the files are untouched is
    the guard against re-growing a second writer inside the overlay: two writers
    is the failure being undone.
    """
    before = _config_text(_isolated_profile)
    result = _switch(is_global=is_global)
    assert result.success, result.error_message
    assert result.warning_message == ""
    assert _config_text(_isolated_profile) == before
    assert _dotenv_text(_isolated_profile) == "# comment\nTELEGRAM_BOT_TOKEN=shhh\n"


def test_switch_model_explicit_provider_still_leaves_the_bridge(_acp_command):
    """Not a lock-in: --provider is how you say you meant it."""
    result = _switch(explicit_provider="anthropic")
    assert result.target_provider == "anthropic"


def test_a_non_acp_profile_is_untouched(_acp_command):
    result = _switch(
        current_provider="openrouter", current_model="x-ai/grok-4",
        current_base_url="https://openrouter.ai/api/v1", current_api_key="k",
    )
    assert result.target_provider != "copilot-acp"


# --- the buttons -------------------------------------------------------------
#
# 2026-08-09, second half of the same defect: fixing the typed command left the
# picker offering moa / copilot / anthropic and no bridge row at all, so the
# only claude button led to the direct API — the 400 again. There was no correct
# button to press.

ANTHROPIC_ROW = {
    "slug": "anthropic",
    "name": "Anthropic",
    "is_current": False,
    "is_user_defined": False,
    "models": ["claude-fable-5", "claude-sonnet-5", "claude-opus-4-7"],
    "total_models": 12,
}
COPILOT_ROW = {
    "slug": "copilot",
    "name": "GitHub Copilot",
    "models": ["gpt-5.4", "gpt-4.1"],
    "total_models": 17,
}


def _rows(providers=None, **kw):
    args = dict(current_provider="copilot-acp", current_model="claude-opus-5")
    args.update(kw)
    return acp_model_choice.picker_rows(
        [dict(ANTHROPIC_ROW), dict(COPILOT_ROW)] if providers is None else providers,
        **args,
    )


def _row(rows, slug):
    return next((r for r in rows if r.get("slug") == slug), None)


def test_the_bridge_gets_a_row_and_it_is_first(_isolated_profile):
    rows = _rows()
    assert rows[0]["slug"] == "copilot-acp"
    # The slug is not decoration: it is what a press hands back as
    # explicit_provider, and switch_model resolves that to acp://copilot.
    assert rows[0]["models"][0] == "claude-opus-5"
    assert "claude-sonnet-5" in rows[0]["models"]


def test_the_row_leads_with_what_this_profile_actually_runs(_isolated_profile):
    """The profile's model is model.default — the source, not a session override."""
    _set_model("claude-haiku-4-5-20251001")
    rows = _rows(current_model="")
    assert rows[0]["models"][0] == "claude-haiku-4-5-20251001"


def test_a_config_model_outside_the_known_list_still_gets_a_button(_isolated_profile):
    # KNOWN_ENGINE_MODELS is a display list, not a gate (_ENGINE_MODEL_RE is the
    # gate). A profile configured ahead of it must not lose its own model.
    assert "claude-opus-9" not in acp_config.KNOWN_ENGINE_MODELS
    _set_model("claude-opus-9")
    assert "claude-opus-9" in _rows(current_model="")[0]["models"]


def test_a_dead_dotenv_pin_no_longer_decides_the_first_button(_isolated_profile):
    """The picker must show what answers, and .env is not what answers."""
    _leftover(_isolated_profile, "claude-haiku-4-5-20251001")
    assert _rows(current_model="")[0]["models"][0] == "claude-opus-5"


def test_the_direct_api_row_loses_the_ids_the_bridge_serves(_isolated_profile):
    row = _row(_rows(), "anthropic")
    assert row["models"] == ["claude-opus-4-7"]
    # Upstream's own aggregator dedup reports the visible count (#45954).
    assert row["total_models"] == 1


def test_dedup_uses_the_whole_bridge_list_not_the_visible_buttons(_isolated_profile):
    # max_models truncates what is shown. If the dedup read the truncated list,
    # every id below the cut would keep its broken second button.
    rows = _rows(max_models=1)
    assert len(rows[0]["models"]) == 1
    assert _row(rows, "anthropic")["models"] == ["claude-opus-4-7"]


def test_a_prefixed_id_is_a_different_route_and_survives(_isolated_profile):
    # OpenRouter's anthropic/claude-opus-5 genuinely works on an ACP profile.
    # Matching by exact id is what keeps it: it is not the same id.
    aggregator = {
        "slug": "openrouter",
        "models": ["anthropic/claude-opus-5"],
        "total_models": 1,
    }
    rows = _rows([dict(ANTHROPIC_ROW), aggregator])
    assert _row(rows, "openrouter")["models"] == ["anthropic/claude-opus-5"]


def test_a_users_own_endpoint_is_never_gutted(_isolated_profile):
    # #47077: the specific provider is not a duplicate of anything.
    mine = {
        "slug": "custom:mine",
        "is_user_defined": True,
        "api_url": "http://x",
        "models": ["claude-opus-5"],
        "total_models": 1,
    }
    rows = _rows([mine])
    assert _row(rows, "custom:mine")["models"] == ["claude-opus-5"]


def test_untouched_rows_are_not_copied(_isolated_profile):
    original = dict(COPILOT_ROW)
    rows = _rows([original])
    assert _row(rows, "copilot") is original


def test_a_session_stuck_on_the_direct_api_still_sees_the_way_back(_isolated_profile):
    # The gateway reads the session override before opening the picker, so a
    # session that already fell onto anthropic reports it as current. Asking
    # only that question would withhold the row from the session that needs it.
    rows = _rows(current_provider="anthropic", current_model="claude-fable-5")
    assert rows[0]["slug"] == "copilot-acp"
    assert rows[0]["models"][0] == "claude-fable-5"


def test_the_profile_level_question_is_model_provider(_isolated_profile):
    """``is_acp_profile`` asks config.yaml, not "does a pin exist".

    The pin is gone from migrated profiles, so keying the row off one would
    have withheld it from every profile that did the migration correctly.
    """
    assert acp_model_choice.is_acp_profile("anthropic") is True
    (_isolated_profile / "config.yaml").write_text(
        "model:\n  default: claude-opus-5\n  provider: anthropic\n", encoding="utf-8"
    )
    acp_config._CONFIG_CACHE.clear()
    assert acp_model_choice.is_acp_profile("anthropic") is False
    # …unless this very session is already routed through the bridge.
    assert acp_model_choice.is_acp_profile("copilot-acp") is True


def test_a_profile_that_is_not_on_the_bridge_gets_nothing(_isolated_profile):
    (_isolated_profile / "config.yaml").write_text(
        "model:\n  default: qwen3\n  provider: custom:local-llama\n", encoding="utf-8"
    )
    acp_config._CONFIG_CACHE.clear()
    providers = [dict(ANTHROPIC_ROW)]
    rows = acp_model_choice.picker_rows(
        providers, current_provider="custom:local-llama", current_model="qwen3"
    )
    assert rows is providers


# --- the profile that only FALLS BACK to the bridge --------------------------
#
# 2026-08-19: `secretary` moved onto a corporate HTTP endpoint and kept
# `fallback_model: {provider: copilot-acp, model: claude-sonnet-5}`. Every turn
# that endpoint refused was answered by the bridge — and the picker had no
# bridge row at all, because "is this the bridge" was asked only of
# `model.provider`. Worse than a missing row: the direct-API row got the
# claude-* ids back, so the only visible claude button led to the 400 again.

def _fallback_profile(home, *, provider="copilot-acp", model="claude-sonnet-5"):
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: MagnitGPT\n"
        "  provider: magnit-gpt\n"
        f"{acp_config.ROUTE_FALLBACK_KEY}:\n"
        f"  provider: {provider}\n"
        f"  model: {model}\n",
        encoding="utf-8",
    )
    acp_config._CONFIG_CACHE.clear()


def _fallback_rows(home, **kw):
    _fallback_profile(home, **kw)
    return acp_model_choice.picker_rows(
        [dict(ANTHROPIC_ROW), dict(COPILOT_ROW)],
        current_provider="magnit-gpt",
        current_model="MagnitGPT",
    )


def test_a_failover_to_the_bridge_earns_the_row(_isolated_profile):
    rows = _fallback_rows(_isolated_profile)
    assert rows[0]["slug"] == "copilot-acp"


def test_the_row_leads_with_the_model_that_actually_answers(_isolated_profile):
    # Neither the session's model nor model.default is an engine model here, so
    # the failover model is the only one that has ever answered on this profile.
    rows = _fallback_rows(_isolated_profile)
    assert rows[0]["models"][0] == "claude-sonnet-5"


def test_offered_is_not_current(_isolated_profile):
    # The picker prints a "current" tag off this flag. This profile runs on
    # magnit-gpt; saying otherwise would be a second, wrong answer to "what
    # answers here".
    assert _fallback_rows(_isolated_profile)[0]["is_current"] is False
    assert _rows()[0]["is_current"] is True


def test_the_direct_api_row_still_loses_the_ids_the_bridge_serves(_isolated_profile):
    # The reason the missing row was worse than cosmetic.
    assert _row(_fallback_rows(_isolated_profile), "anthropic")["models"] == [
        "claude-opus-4-7"
    ]


def test_a_failover_that_is_not_the_bridge_changes_nothing(_isolated_profile):
    """The case this detector must stay silent about.

    `fallback_model` is an ordinary Hermes key — most profiles that carry one
    do not point it at the bridge, and a row offered to them would be a button
    to a backend they have not configured.
    """
    _fallback_profile(_isolated_profile, provider="anthropic", model="claude-opus-5")
    providers = [dict(ANTHROPIC_ROW)]
    rows = acp_model_choice.picker_rows(
        providers, current_provider="magnit-gpt", current_model="MagnitGPT"
    )
    assert rows is providers


def test_an_unreadable_failover_block_is_not_an_invitation(_isolated_profile):
    (_isolated_profile / "config.yaml").write_text(
        "model:\n  default: MagnitGPT\n  provider: magnit-gpt\n"
        f"{acp_config.ROUTE_FALLBACK_KEY}: not-a-mapping\n",
        encoding="utf-8",
    )
    acp_config._CONFIG_CACHE.clear()
    assert acp_model_choice.is_acp_profile("magnit-gpt") is False


def test_the_two_fallback_keys_are_not_the_same_fact(_isolated_profile):
    """`model.acp_fallback` picks another engine model; `fallback_model` picks
    another backend. Reading one for the other is how this bug got written."""
    (_isolated_profile / "config.yaml").write_text(
        "model:\n"
        "  default: MagnitGPT\n"
        "  provider: magnit-gpt\n"
        f"  {acp_config.FALLBACK_KEY}: claude-haiku-4-5-20251001\n",
        encoding="utf-8",
    )
    acp_config._CONFIG_CACHE.clear()
    assert acp_config.route_fallback_provider() == ""
    assert acp_model_choice.is_acp_profile("magnit-gpt") is False


def test_never_two_bridge_rows(_isolated_profile):
    stale = {"slug": "copilot-acp", "name": "stale", "models": ["copilot-acp"]}
    rows = _rows([stale, dict(ANTHROPIC_ROW)])
    assert [r["slug"] for r in rows].count("copilot-acp") == 1
    assert rows[0]["name"] == acp_model_choice.ACP_PICKER_NAME


@pytest.mark.parametrize("providers", [None, [], ["junk"], [{"slug": "x"}]])
def test_a_broken_payload_never_breaks_the_picker(_isolated_profile, providers):
    acp_model_choice.picker_rows(providers, current_provider="copilot-acp")


def test_engine_models_orders_current_then_profile_then_known(_isolated_profile):
    models = acp_model_choice.engine_models("claude-sonnet-5")
    assert models[0] == "claude-sonnet-5"      # this session
    assert models[1] == "claude-opus-5"        # model.default
    assert models[2:] == [
        m for m in acp_config.KNOWN_ENGINE_MODELS
        if m not in ("claude-sonnet-5", "claude-opus-5")
    ]


def test_the_known_list_has_one_owner():
    # A second copy is a second truth: the picker's buttons and bridge-doctor's
    # "no such model" check must not be able to disagree.
    assert acp_config.KNOWN_ENGINE_MODELS[0] == "claude-opus-5"
    assert all(
        acp_config.normalize_model(m) == m for m in acp_config.KNOWN_ENGINE_MODELS
    )


# --- through the real list_picker_providers ----------------------------------

def test_list_picker_providers_offers_the_bridge(_isolated_profile):
    """The seam: the overlay must actually be wired into the upstream function."""
    from hermes_cli import model_switch as ms

    with patch.object(
        ms, "list_authenticated_providers",
        return_value=[dict(ANTHROPIC_ROW), dict(COPILOT_ROW)],
    ):
        rows = ms.list_picker_providers(
            current_provider="copilot-acp", current_model="claude-opus-5"
        )
    assert rows[0]["slug"] == "copilot-acp"
    assert "claude-opus-5" in rows[0]["models"]


def test_a_row_the_dedup_empties_is_dropped_downstream(_isolated_profile):
    """Emptied rows must not survive as dead buttons — upstream's own filter."""
    from hermes_cli import model_switch as ms

    only_bridge_models = dict(
        ANTHROPIC_ROW, models=["claude-fable-5", "claude-sonnet-5"]
    )
    with patch.object(
        ms, "list_authenticated_providers", return_value=[only_bridge_models]
    ):
        rows = ms.list_picker_providers(current_provider="copilot-acp")
    assert [r["slug"] for r in rows] == ["copilot-acp"]


# --- the button the bridge's row actually gets --------------------------------
#
# A correct row that upstream then folds under someone else's name is not a fix:
# group_providers() treats copilot-acp as the GitHub Copilot family (on a stock
# install it *is* `copilot --acp`), so the fleet's own models ended up two taps
# deep behind "GitHub Copilot ▸" while the button that reads like the right one
# — Anthropic — was the direct API.

BRIDGE_PROVS = [{"slug": "copilot-acp"}, {"slug": "moa"}, {"slug": "copilot"},
                {"slug": "anthropic"}]


def _folded(providers=None):
    from hermes_cli.models import group_providers

    provs = BRIDGE_PROVS if providers is None else providers
    fold = acp_model_choice.ungrouped_fold(group_providers, provs)
    return fold([p["slug"] for p in provs])


def test_upstream_really_does_fold_the_bridge_away():
    """The premise. If upstream stops folding, the wrapper is dead weight."""
    from hermes_cli.models import group_providers

    rows = group_providers([p["slug"] for p in BRIDGE_PROVS])
    groups = [r for r in rows if r["kind"] == "group"]
    assert any("copilot-acp" in g["members"] for g in groups)


def test_the_bridge_gets_its_own_top_level_button():
    rows = _folded()
    assert rows[0] == {"kind": "single", "slug": "copilot-acp"}
    assert not any(
        "copilot-acp" in r.get("members", []) for r in rows if r["kind"] == "group"
    )


def test_the_family_it_left_is_not_broken():
    rows = _folded()
    # Copilot alone is no longer a family: a one-member group renders as a
    # "▸" drill-down onto a single button.
    assert {"kind": "single", "slug": "copilot"} in rows
    assert {"kind": "single", "slug": "anthropic"} in rows
    assert {"kind": "single", "slug": "moa"} in rows


def test_a_larger_family_keeps_its_remaining_members():
    def fold(_slugs):
        return [{"kind": "group", "group_id": "copilot", "label": "GitHub Copilot",
                 "members": ["copilot", "copilot-acp", "copilot-x"]}]

    wrapped = acp_model_choice.ungrouped_fold(fold, BRIDGE_PROVS)
    rows = wrapped(["copilot-acp"])
    assert rows[0]["slug"] == "copilot-acp"
    assert rows[1]["members"] == ["copilot", "copilot-x"]


def test_a_profile_without_the_bridge_is_not_wrapped_at_all():
    from hermes_cli.models import group_providers

    assert acp_model_choice.ungrouped_fold(
        group_providers, [{"slug": "anthropic"}]
    ) is group_providers


def test_a_fold_that_never_grouped_us_is_passed_through_untouched():
    sentinel = [{"kind": "single", "slug": "copilot-acp"}]
    wrapped = acp_model_choice.ungrouped_fold(lambda _s: sentinel, BRIDGE_PROVS)
    assert wrapped(["copilot-acp"]) is sentinel


def test_a_broken_fold_never_costs_the_user_the_keyboard():
    wrapped = acp_model_choice.ungrouped_fold(lambda _s: ["junk"], BRIDGE_PROVS)
    assert wrapped(["copilot-acp"]) == ["junk"]


def test_the_telegram_keyboard_offers_the_bridge_first():
    """The seam: the real keyboard builder, not a source-text assertion.

    Needs the real ``python-telegram-bot``: ``tests/gateway/conftest.py``
    installs a mock ``telegram`` package into ``sys.modules`` for the whole
    session, and ``InlineKeyboardMarkup`` built from that mock carries no
    rows — the assertions below would then fail on the mock rather than on
    the overlay. Collected alone this file gets the real package; collected
    with any gateway test it does not, so ask rather than assume.
    """
    import sys

    telegram_mod = sys.modules.get("telegram")
    if telegram_mod is not None and not hasattr(telegram_mod, "__file__"):
        pytest.skip("mocked telegram package (tests/gateway/conftest.py)")

    from plugins.platforms.telegram.adapter import TelegramAdapter

    rows = [
        dict(p, name=n, total_models=c, is_current=(p["slug"] == "copilot-acp"))
        for p, n, c in zip(
            BRIDGE_PROVS,
            ["Claude (ACP bridge)", "Mixture of Agents", "GitHub Copilot", "Anthropic"],
            [5, 1, 17, 3],
        )
    ]
    keyboard, _info = TelegramAdapter._build_provider_keyboard(
        TelegramAdapter, rows, 0
    )
    labels = [b.text for row in keyboard.inline_keyboard for b in row]
    data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert data[0] == "mp:copilot-acp", labels
    assert labels[0] == "✓ Claude (ACP bridge) (5)"
    # And the wrong vendor's drill-down is gone from the top level.
    assert not any(str(d).startswith("mpg:copilot") for d in data), data


def test_provider_label_matches_picker_row_name():
    """The /model header and the button that leads there must read the same.

    The header comes from ``provider_label("copilot-acp")`` (upstream label
    table); the button comes from ``ACP_PICKER_NAME``.  When they disagreed,
    the header still said "GitHub Copilot ACP" while the row said
    "Claude (ACP bridge)" — and the header was read as a separate, older
    button that had supposedly always existed.
    """
    from hermes_cli.models import provider_label

    assert provider_label("copilot-acp") == acp_model_choice.ACP_PICKER_NAME
