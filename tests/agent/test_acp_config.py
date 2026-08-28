"""Single-source-of-truth resolution for the ACP bridge (local overlay).

Covers agent/acp_config.py plus the call sites that used to resolve
HERMES_ACP_MODEL / HERMES_ACP_NATIVE_TOOLS on their own.

The inversion these guard (2026-08-10): the engine model comes from
``model.default`` in the profile's ``config.yaml`` and nowhere else.
``HERMES_ACP_MODEL`` is an *output* of that resolution — the spawn-time channel
into ``claude-acp-run.sh`` — never an input to it. A surviving ``.env`` line is
a defect to report, not a config to obey.
"""

import os
from unittest.mock import patch

import pytest

from agent import acp_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in (
        acp_config.MODEL_VAR,
        acp_config.FALLBACK_VAR,
        acp_config.NATIVE_VAR,
        acp_config.OVERRIDE_VAR,
    ):
        monkeypatch.delenv(var, raising=False)
    # HERMES_HOME must be isolated, and it matters more than it used to:
    # config.yaml is now the source, so without this the suite would resolve
    # against the operator's live profile and change verdict the moment they
    # press /model. Every helper below writes inside this tempdir.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    acp_config._DOTENV_CACHE.clear()
    acp_config._CONFIG_CACHE.clear()
    acp_config._WARNED.clear()
    yield
    acp_config._DOTENV_CACHE.clear()
    acp_config._CONFIG_CACHE.clear()
    acp_config._WARNED.clear()


def _dotenv(tmp_path, body):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    acp_config._DOTENV_CACHE.clear()
    return path


def _config(tmp_path, body):
    """Write the isolated profile's config.yaml verbatim."""
    path = tmp_path / "home" / "config.yaml"
    path.write_text(body, encoding="utf-8")
    acp_config._CONFIG_CACHE.clear()
    return path


def _model(tmp_path, value, *, provider=None, key="default"):
    """Write ``model.<key>`` (and optionally ``model.provider``) into config.yaml."""
    body = "model:\n"
    if value is not None:
        body += f"  {key}: {value}\n"
    if provider is not None:
        body += f"  provider: {provider}\n"
    return _config(tmp_path, body)


# --- normalize_model ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("claude-opus-5", "claude-opus-5"),
    ("  claude-opus-4-8  ", "claude-opus-4-8"),
    ("anthropic/claude-opus-5", "claude-opus-5"),
    ("copilot-acp/claude-fable-5", "claude-fable-5"),
    ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    # Provider labels are NOT models. auxiliary_client hands these over as if
    # they were (ledger rows with requested_model="copilot-acp"), and spawning
    # the engine on one is worse than falling back.
    ("copilot-acp", None),
    ("gpt-5", None),
    ("", None),
    (None, None),
    ("anthropic/", None),
])
def test_normalize_model(raw, expected):
    assert acp_config.normalize_model(raw) == expected


# --- resolve_model -----------------------------------------------------------

def test_config_default_decides(tmp_path):
    """The whole inversion in one line: model.default is what answers."""
    _model(tmp_path, "claude-opus-5")
    assert acp_config.resolve_model() == "claude-opus-5"


def test_a_request_wins_over_the_config_default(tmp_path):
    """A per-run choice: session /model, -m, cron model=, a subagent's model=."""
    _model(tmp_path, "claude-opus-5")
    assert acp_config.resolve_model("claude-fable-5") == "claude-fable-5"


def test_a_provider_qualified_request_is_normalised(tmp_path):
    """The wrapper wants the bare engine name; the catalog hands over anthropic/…"""
    _model(tmp_path, "claude-opus-5")
    assert acp_config.resolve_model("anthropic/claude-sonnet-5") == "claude-sonnet-5"


def test_a_request_that_is_just_the_config_default_resolves_the_same(tmp_path):
    """The common turn: the agent's model IS model.default. Nothing special."""
    _model(tmp_path, "claude-opus-5")
    assert acp_config.resolve_model("claude-opus-5") == "claude-opus-5"
    assert acp_config.resolve_model("anthropic/claude-opus-5") == "claude-opus-5"
    assert acp_config.resolve_model("CLAUDE-OPUS-5") == "CLAUDE-OPUS-5"


def test_a_leftover_env_pin_no_longer_decides(tmp_path, monkeypatch):
    """THE inversion, and the incident it undoes.

    2026-08-09 researcher / 2026-08-10 secretary: config.yaml said one model,
    ``HERMES_ACP_MODEL`` in .env said another, and the .env won — so /model,
    /status and the picker all agreed with each other and disagreed with the
    process that answered. config.yaml wins now; .env is not consulted.
    """
    _model(tmp_path, "claude-fable-5")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-opus-5")
    _dotenv(tmp_path / "home", "HERMES_ACP_MODEL=claude-opus-5\n")
    assert acp_config.resolve_model() == "claude-fable-5"


def test_a_leftover_env_pin_does_not_beat_a_request_either(tmp_path, monkeypatch):
    _model(tmp_path, "claude-opus-5")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-opus-5")
    assert acp_config.resolve_model("claude-sonnet-5") == "claude-sonnet-5"


def test_the_legacy_pin_is_the_last_resort_before_the_default(tmp_path, monkeypatch):
    """Deprecated, but not a downgrade: an un-migrated profile keeps its model.

    Only reachable when config.yaml names no engine model at all — otherwise
    the pin is dead. Verbatim, because claude-acp-run.sh uses the same string.
    """
    _config(tmp_path, "model: {}\n")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-sonnet-5")
    assert acp_config.resolve_model() == "claude-sonnet-5"


def test_the_legacy_pin_fallback_says_it_is_deprecated(tmp_path, monkeypatch, caplog):
    _config(tmp_path, "model: {}\n")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-sonnet-5")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        acp_config.resolve_model()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "claude-sonnet-5" in msg and "config.yaml" in msg


def test_nothing_configured_anywhere_uses_the_legacy_default():
    assert acp_config.resolve_model() == acp_config.LEGACY_DEFAULT_MODEL


def test_a_provider_label_is_not_a_request(tmp_path):
    """The aux path hands over "copilot-acp" as if it were a model."""
    assert acp_config.resolve_model("copilot-acp") == acp_config.LEGACY_DEFAULT_MODEL
    _model(tmp_path, "claude-opus-5")
    assert acp_config.resolve_model("copilot-acp") == "claude-opus-5"


def test_a_blank_request_is_not_a_value(tmp_path):
    _model(tmp_path, "claude-opus-5")
    assert acp_config.resolve_model("   ") == "claude-opus-5"


# --- and the substitution must be SAID out loud ------------------------------
# 2026-08-06: researcher edited model.default to claude-fable-5, probed with
# "answer ok" (which proves something is alive, not which model), and reported
# the switch done. Something else kept answering for three hours; the only
# detector was the next day's ledger diff.

def test_an_explicit_choice_is_said_out_loud(tmp_path, caplog):
    _model(tmp_path, "claude-opus-5")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        acp_config.resolve_model("claude-fable-5")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    # Both models named, and the source named, or the operator who reaches this
    # line cannot tell which of the two files to go and edit.
    assert "claude-fable-5" in msg and "claude-opus-5" in msg
    assert "config.yaml" in msg


def test_a_request_equal_to_the_config_is_silent(tmp_path, caplog):
    _model(tmp_path, "claude-opus-5")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        acp_config.resolve_model("anthropic/claude-opus-5")
    assert caplog.records == []


def test_a_plain_config_resolve_is_silent(tmp_path, caplog):
    """The healthy turn must produce no line at all, or the log teaches scrolling."""
    _model(tmp_path, "claude-opus-5")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        acp_config.resolve_model()
    assert caplog.records == []


def test_a_model_outside_the_known_list_warns_but_still_runs(tmp_path, caplog):
    """KNOWN_ENGINE_MODELS is a display list, not a gate — a newer profile runs."""
    assert "claude-opus-9" not in acp_config.KNOWN_ENGINE_MODELS
    _model(tmp_path, "claude-opus-9")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        assert acp_config.resolve_model("claude-opus-9") == "claude-opus-9"
    assert any("claude-opus-9" in r.getMessage() for r in caplog.records)


def test_warns_once_per_choice_not_once_per_turn(tmp_path, caplog):
    """A gateway resolves on every spawn; one line per substitution, not per turn.

    Keyed by the chosen model, so a *second*, different substitution is still
    reported instead of being swallowed by the first one's key.
    """
    _model(tmp_path, "claude-opus-5")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        for _ in range(3):
            acp_config.resolve_model("claude-fable-5")
        assert len(caplog.records) == 1
        acp_config.resolve_model("claude-sonnet-5")
    assert len(caplog.records) == 2


# --- the source itself: config_model / config_provider / is_acp_config -------

def test_config_model_reads_model_default(tmp_path):
    _model(tmp_path, "claude-opus-5")
    assert acp_config.config_model() == "claude-opus-5"


def test_config_model_normalises_the_provider_qualified_form(tmp_path):
    """researcher's config.yaml carries anthropic/… — same engine, two spellings."""
    _model(tmp_path, "anthropic/claude-opus-5")
    assert acp_config.config_model() == "claude-opus-5"


def test_config_model_accepts_the_scalar_shape(tmp_path):
    """``model: claude-x`` with no sub-keys is a shape real profiles use."""
    _config(tmp_path, "model: claude-sonnet-5\n")
    assert acp_config.config_model() == "claude-sonnet-5"
    # A scalar names no provider, and inventing one would put a profile on the
    # bridge that never asked for it.
    assert acp_config.config_provider() == ""
    assert acp_config.is_acp_config() is False


def test_config_model_falls_back_to_the_model_key(tmp_path):
    _model(tmp_path, "claude-sonnet-5", key="model")
    assert acp_config.config_model() == "claude-sonnet-5"


@pytest.mark.parametrize("body", [
    "",                        # empty file
    "model: {}\n",             # no default
    "model:\n  default: 7\n",  # not a model name
    "model:\n  default: gpt-5\n",  # not an *engine* model
    "provider: openrouter\n",
    ": : not yaml : :\n[",     # unparseable
])
def test_config_model_is_none_when_it_cannot_tell(tmp_path, body):
    _config(tmp_path, body)
    assert acp_config.config_model() is None


def test_config_model_is_none_when_there_is_no_file(tmp_path):
    (tmp_path / "home" / "config.yaml").unlink(missing_ok=True)
    assert acp_config.config_model() is None
    assert acp_config.config_provider() == ""


def test_config_cache_follows_edits(tmp_path):
    """A long-lived gateway must see a /model write on the very next turn.

    The cache key is the stat, not the path — a value cache would serve the
    pre-switch model until someone restarted the service, which is precisely
    the "it says it switched and it did not" shape being removed.
    """
    path = _model(tmp_path, "claude-opus-5")
    assert acp_config.config_model() == "claude-opus-5"
    os.utime(path, (0, 0))
    path.write_text("model:\n  default: claude-sonnet-5\n", encoding="utf-8")
    assert acp_config.config_model() == "claude-sonnet-5"   # no manual clear


def test_config_provider_and_is_acp_config(tmp_path):
    _model(tmp_path, "claude-opus-5", provider="copilot-acp")
    assert acp_config.config_provider() == "copilot-acp"
    assert acp_config.is_acp_config() is True


def test_is_acp_config_is_case_insensitive(tmp_path):
    _model(tmp_path, "claude-opus-5", provider="Copilot-ACP")
    assert acp_config.is_acp_config() is True


def test_is_acp_config_is_false_off_the_bridge(tmp_path):
    _model(tmp_path, "claude-opus-5", provider="anthropic")
    assert acp_config.config_provider() == "anthropic"
    assert acp_config.is_acp_config() is False


def test_config_label_is_still_the_same_function():
    """Back-compat alias: the old name decided nothing, this one decides all."""
    assert acp_config.config_label is acp_config.config_model


def test_acp_providers_is_the_slug_switch_model_resolves():
    assert "copilot-acp" in acp_config.ACP_PROVIDERS


# --- explicit_choice: reporting only, but it must still be right -------------

@pytest.mark.parametrize("configured,requested,expected", [
    # Something other than config.yaml picked this: /model --session, -m,
    # a cron job's model=, a subagent's model=.
    ("claude-opus-5", "claude-fable-5", "claude-fable-5"),
    ("claude-opus-5", "anthropic/claude-sonnet-5", "claude-sonnet-5"),
    # The agent's model IS model.default — nobody chose, so nothing to report.
    ("claude-opus-5", "claude-opus-5", None),
    ("claude-opus-5", "anthropic/claude-opus-5", None),
    ("claude-opus-5", "CLAUDE-OPUS-5", None),
    # Nothing usable requested (aux path hands over provider labels).
    ("claude-opus-5", "copilot-acp", None),
    ("claude-opus-5", None, None),
    # No config to compare against -> cannot tell -> not reported as a choice.
    (None, "claude-fable-5", None),
])
def test_explicit_choice(tmp_path, configured, requested, expected):
    if configured is not None:
        _model(tmp_path, configured)
    assert acp_config.explicit_choice(requested) == expected


# --- legacy_pin --------------------------------------------------------------

def test_legacy_pin_found_in_the_dotenv_file(tmp_path):
    """A line somebody has to delete."""
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-opus-5\n")
    assert acp_config.legacy_pin(path=path) == "claude-opus-5"


def test_legacy_pin_found_only_in_the_process_env(tmp_path, monkeypatch):
    """An export or a systemd Environment= — a different thing to go and fix."""
    path = _dotenv(tmp_path, "TELEGRAM_BOT_TOKEN=shhh\n")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-fable-5")
    assert acp_config.legacy_pin(path=path) == "claude-fable-5"


def test_legacy_pin_prefers_the_file(tmp_path, monkeypatch):
    """Both set: name the one that is a line, because that is the one to delete."""
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-opus-5\n")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-fable-5")
    assert acp_config.legacy_pin(path=path) == "claude-opus-5"


def test_legacy_pin_is_empty_when_nothing_is_left(tmp_path):
    assert acp_config.legacy_pin(path=_dotenv(tmp_path, "")) == ""


def test_a_blank_legacy_pin_is_not_a_value(tmp_path, monkeypatch):
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=   \n")
    monkeypatch.setenv(acp_config.MODEL_VAR, "  ")
    assert acp_config.legacy_pin(path=path) == ""


def test_legacy_model_vars_names_both_deprecated_keys():
    # bridge-doctor and these tests must name the same set, or one of them
    # cleans a .env the other still trusts.
    assert acp_config.LEGACY_MODEL_VARS == (
        acp_config.MODEL_VAR, acp_config.FALLBACK_VAR,
    )


# --- set_model: the only writer left -----------------------------------------

CONFIG_WITH_COMMENTS = """\
# Hermes profile — hand-maintained, comments are load-bearing.
model:
  # the engine that answers
  default: claude-opus-5
  provider: copilot-acp
max_concurrent_sessions: 5
agent:
  max_turns: 150
"""


def test_set_model_rewrites_the_default_in_place(tmp_path):
    """Line-based on purpose: a yaml round-trip drops every comment in the file."""
    path = _config(tmp_path, CONFIG_WITH_COMMENTS)
    assert acp_config.set_model("claude-sonnet-5") == "claude-sonnet-5"
    assert path.read_text() == CONFIG_WITH_COMMENTS.replace(
        "default: claude-opus-5", "default: claude-sonnet-5"
    )


def test_set_model_leaves_everything_else_byte_for_byte(tmp_path):
    path = _config(tmp_path, CONFIG_WITH_COMMENTS)
    acp_config.set_model("claude-sonnet-5")
    text = path.read_text()
    for survivor in (
        "# Hermes profile — hand-maintained, comments are load-bearing.",
        "  # the engine that answers",
        "  provider: copilot-acp",
        "max_concurrent_sessions: 5",
        "  max_turns: 150",
    ):
        assert survivor in text
    # The provider must survive as *data* too, not just as text: dropping it
    # would silently take the profile off the bridge.
    assert acp_config.config_provider() == "copilot-acp"


def test_set_model_normalises_a_provider_qualified_name(tmp_path):
    path = _config(tmp_path, CONFIG_WITH_COMMENTS)
    assert acp_config.set_model("anthropic/claude-fable-5") == "claude-fable-5"
    assert "default: claude-fable-5\n" in path.read_text()


def test_set_model_appends_the_key_when_the_block_lacks_it(tmp_path):
    path = _config(tmp_path, "model:\n  provider: copilot-acp\nother: 1\n")
    acp_config.set_model("claude-sonnet-5")
    assert path.read_text() == (
        "model:\n  default: claude-sonnet-5\n  provider: copilot-acp\nother: 1\n"
    )


def test_set_model_creates_the_block_when_there_is_none(tmp_path):
    path = _config(tmp_path, "max_concurrent_sessions: 5")  # no trailing newline
    acp_config.set_model("claude-sonnet-5")
    assert path.read_text() == (
        "max_concurrent_sessions: 5\nmodel:\n  default: claude-sonnet-5\n"
    )
    assert acp_config.config_model() == "claude-sonnet-5"


def test_set_model_does_not_walk_out_of_the_model_block(tmp_path):
    """A ``default:`` under a later top-level key is somebody else's setting."""
    path = _config(
        tmp_path,
        "model:\n  provider: copilot-acp\nvoice:\n  default: nova\n",
    )
    acp_config.set_model("claude-sonnet-5")
    assert path.read_text() == (
        "model:\n  default: claude-sonnet-5\n  provider: copilot-acp\n"
        "voice:\n  default: nova\n"
    )


@pytest.mark.parametrize("bad", ["gpt-5", "copilot-acp", "", None, "anthropic/"])
def test_set_model_refuses_anything_that_is_not_an_engine_model(tmp_path, bad):
    """The aux path hands over provider labels; spawning on one is worse than
    falling back, and writing one into the source makes it permanent."""
    path = _config(tmp_path, CONFIG_WITH_COMMENTS)
    with pytest.raises(ValueError):
        acp_config.set_model(bad)
    # Refusal means untouched, not half-written.
    assert path.read_text() == CONFIG_WITH_COMMENTS


def test_set_model_is_atomic_and_keeps_the_mode(tmp_path):
    path = _config(tmp_path, CONFIG_WITH_COMMENTS)
    # 0o640, not 0o600: mkstemp already creates 0o600, so asserting 0o600 here
    # cannot tell "we copied the original's mode" from "we got lucky" — and a
    # mutation that deletes the chmod passed a 0o600 test unnoticed.
    path.chmod(0o640)
    acp_config.set_model("claude-sonnet-5")
    assert path.stat().st_mode & 0o777 == 0o640
    # No temp files left behind by the replace.
    assert [p.name for p in path.parent.glob(".config.*")] == []


def test_set_model_takes_effect_without_a_restart(tmp_path):
    """One write, one source: the next resolve already answers on it."""
    _config(tmp_path, CONFIG_WITH_COMMENTS)
    acp_config.set_model("claude-sonnet-5")
    assert acp_config.resolve_model() == "claude-sonnet-5"


def test_set_model_writes_the_path_it_was_given(tmp_path):
    other = tmp_path / "elsewhere.yaml"
    other.write_text("model:\n  default: claude-opus-5\n", encoding="utf-8")
    _config(tmp_path, CONFIG_WITH_COMMENTS)
    assert acp_config.set_model("claude-fable-5", path=other) == "claude-fable-5"
    assert "claude-fable-5" in other.read_text()
    assert acp_config.config_model() == "claude-opus-5"  # the profile untouched


# --- native_tools_enabled ----------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("maybe", False),
])
def test_native_tools_gate(monkeypatch, raw, expected):
    monkeypatch.setenv(acp_config.NATIVE_VAR, raw)
    assert acp_config.native_tools_enabled() is expected


def test_native_tools_defaults_off():
    assert acp_config.native_tools_enabled() is False


# --- read_dotenv -------------------------------------------------------------

def test_read_dotenv_parses_real_shapes(tmp_path):
    path = _dotenv(tmp_path, "\n".join([
        "# comment",
        "",
        "HERMES_ACP_NATIVE_TOOLS=1",
        'HERMES_ACP_MODEL="claude-opus-5"',
        "export OTHER=x",
        "#HERMES_ACP_MODEL=claude-fable-5",
    ]))
    assert acp_config.read_dotenv(acp_config.MODEL_VAR, path=path) == "claude-opus-5"
    assert acp_config.read_dotenv(acp_config.NATIVE_VAR, path=path) == "1"
    assert acp_config.read_dotenv("OTHER", path=path) == "x"
    assert acp_config.read_dotenv("NOPE", path=path) is None


def test_read_dotenv_missing_file(tmp_path):
    assert acp_config.read_dotenv(acp_config.MODEL_VAR, path=tmp_path / "absent") is None


def test_read_dotenv_cache_follows_edits(tmp_path):
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-opus-5\n")
    assert acp_config.read_dotenv(acp_config.MODEL_VAR, path=path) == "claude-opus-5"
    # A long-lived gateway must see a hot-reloaded .env: cache key is the stat,
    # not the path.
    os.utime(path, (0, 0))
    path.write_text("HERMES_ACP_MODEL=claude-fable-5\n", encoding="utf-8")
    assert acp_config.read_dotenv(acp_config.MODEL_VAR, path=path) == "claude-fable-5"


# --- divergences -------------------------------------------------------------

def test_a_clean_dotenv_diverges_from_nothing(tmp_path, monkeypatch):
    _model(tmp_path, "claude-opus-5", provider="copilot-acp")
    path = _dotenv(tmp_path, "HERMES_ACP_NATIVE_TOOLS=1\nTELEGRAM_BOT_TOKEN=shhh\n")
    monkeypatch.setenv(acp_config.NATIVE_VAR, "1")
    assert acp_config.divergences(path=path) == []


def test_no_divergence_when_nothing_is_set_anywhere(tmp_path):
    assert acp_config.divergences(path=_dotenv(tmp_path, "")) == []


def test_a_leftover_pin_naming_another_model_is_fatal(tmp_path):
    """A genuine second source — the exact confusion the inversion removes."""
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-fable-5\n")
    found = acp_config.divergences(path=path)
    assert [(d["var"], d["kind"], d["severity"]) for d in found] == [
        (acp_config.MODEL_VAR, "legacy_pin", "fatal"),
    ]
    detail = found[0]["detail"]
    # Both models and the file to edit, or the operator is guessing.
    assert "claude-fable-5" in detail and "claude-opus-5" in detail
    assert str(path) in detail and "config.yaml" in detail


def test_a_leftover_pin_that_merely_duplicates_is_a_warning(tmp_path):
    """A dead line that agrees today is the one that disagrees after /model."""
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-opus-5\n")
    found = acp_config.divergences(path=path)
    assert [(d["kind"], d["severity"]) for d in found] == [("legacy_pin", "warn")]
    assert "delete the line." in found[0]["detail"]
    assert "model.default" in found[0]["detail"]


def test_a_duplicate_written_the_other_way_round_is_still_only_a_warning(tmp_path):
    """anthropic/claude-opus-5 and claude-opus-5 are one engine, two spellings."""
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=anthropic/claude-opus-5\n")
    assert [d["severity"] for d in acp_config.divergences(path=path)] == ["warn"]


def test_a_leftover_pin_only_in_the_process_env_is_reported(tmp_path, monkeypatch):
    """An export or a systemd drop-in: reported, and named as such."""
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-fable-5")
    found = acp_config.divergences(path=path)
    assert [(d["kind"], d["severity"]) for d in found] == [("legacy_pin", "fatal")]
    assert found[0]["env"] == "claude-fable-5" and found[0]["dotenv"] == ""
    assert "the process environment" in found[0]["detail"]


def test_a_leftover_fallback_pin_is_reported_but_does_not_stop_a_gateway(tmp_path):
    """It is deprecated, not overruled — the failover path still reads it.

    ``fallback_model()`` honours a ``.env`` fallback while ``config.yaml`` names
    none, so a profile that never migrated keeps the downgrade it chose. Calling
    that fatal would have refused to start a gateway over a variable the engine
    was still obeying — a guard punishing the state it created.
    """
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_FALLBACK_MODEL=claude-haiku-4-5-20251001\n")
    found = acp_config.divergences(path=path)
    assert [(d["var"], d["kind"], d["severity"]) for d in found] == [
        (acp_config.FALLBACK_VAR, "legacy_pin", "warn"),
    ]
    assert f"model.{acp_config.FALLBACK_KEY}" in found[0]["detail"]
    acp_config.enforce("gateway", path=path)  # must not raise


def test_a_config_fallback_that_disagrees_with_the_leftover_is_fatal(tmp_path):
    """Once config names one, the .env line is a genuine second source."""
    _config(
        tmp_path,
        "model:\n  default: claude-opus-5\n  acp_fallback: claude-sonnet-5\n",
    )
    path = _dotenv(tmp_path, "HERMES_ACP_FALLBACK_MODEL=claude-haiku-4-5-20251001\n")
    found = acp_config.divergences(path=path)
    assert [(d["var"], d["severity"]) for d in found] == [
        (acp_config.FALLBACK_VAR, "fatal"),
    ]


def test_the_config_fallback_wins_over_the_deprecated_variable(tmp_path):
    _config(
        tmp_path,
        "model:\n  default: claude-opus-5\n  acp_fallback: claude-sonnet-5\n",
    )
    _dotenv(tmp_path, "HERMES_ACP_FALLBACK_MODEL=claude-haiku-4-5-20251001\n")
    assert acp_config.fallback_model() == "claude-sonnet-5"


def test_no_fallback_anywhere_means_no_distinct_fallback(tmp_path):
    """Empty, so _acp_failover_model answers with the primary and declines."""
    _model(tmp_path, "claude-opus-5")
    _dotenv(tmp_path, "")
    assert acp_config.fallback_model() == ""


def test_a_leftover_pin_with_no_config_to_compare_is_only_a_warning(tmp_path):
    """"Cannot tell" must not become "refuse to start"."""
    _config(tmp_path, "model: {}\n")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-fable-5\n")
    assert [d["severity"] for d in acp_config.divergences(path=path)] == ["warn"]


def test_the_native_tools_gate_still_has_its_own_checks(tmp_path):
    """HERMES_ACP_NATIVE_TOOLS is a gate, not a model: .env is still its source."""
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_NATIVE_TOOLS=0\n")
    with patch.dict(os.environ, {acp_config.NATIVE_VAR: "1"}):
        found = acp_config.divergences(path=path)
    assert [(d["var"], d["kind"], d["severity"]) for d in found] == [
        (acp_config.NATIVE_VAR, "conflict", "fatal"),
    ]


def test_an_unsourced_gate_is_fatal(tmp_path, monkeypatch):
    """The shape that actually happens: .env silent, something else decides.

    load_hermes_dotenv applies .env with override=True, so a *conflict* never
    survives into a running process — .env has already won. The survivable
    failure is .env saying nothing, and then a gateway and a `hermes -p … -z`
    of the same profile resolve different answers.
    """
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "")
    monkeypatch.setenv(acp_config.NATIVE_VAR, "1")
    found = acp_config.divergences(path=path)
    assert [(d["var"], d["kind"], d["severity"]) for d in found] == [
        (acp_config.NATIVE_VAR, "unsourced", "fatal"),
    ]


# --- enforce -----------------------------------------------------------------

def test_gateway_refuses_to_start_on_a_disagreeing_leftover(tmp_path):
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-fable-5\n")
    with pytest.raises(acp_config.ACPConfigError) as exc:
        acp_config.enforce("gateway", path=path)
    # The message must name both sources and the way out, or the operator is
    # debugging a dead fleet with no agent to ask.
    assert str(path) in str(exc.value)
    assert "config.yaml" in str(exc.value)
    assert acp_config.OVERRIDE_VAR in str(exc.value)


def test_gateway_refuses_to_start_on_an_unsourced_gate(tmp_path, monkeypatch):
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "")
    monkeypatch.setenv(acp_config.NATIVE_VAR, "1")
    with pytest.raises(acp_config.ACPConfigError):
        acp_config.enforce("gateway", path=path)


def test_a_merely_duplicating_leftover_does_not_stop_a_gateway(tmp_path, caplog):
    """Dead but harmless today: warn, do not take the fleet down for it."""
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-opus-5\n")
    with caplog.at_level("WARNING", logger=acp_config.log.name):
        found = acp_config.enforce("gateway", path=path)  # must not raise
    assert [d["severity"] for d in found] == ["warn"]
    assert any("HERMES_ACP_MODEL" in r.getMessage() for r in caplog.records)


def test_cli_override_is_allowed(tmp_path):
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-fable-5\n")
    found = acp_config.enforce("cli", path=path)  # must not raise
    assert [d["severity"] for d in found] == ["fatal"]


def test_explicit_override_flag_downgrades_the_gateway(tmp_path, monkeypatch):
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-fable-5\n")
    monkeypatch.setenv(acp_config.OVERRIDE_VAR, "1")
    acp_config.enforce("gateway", path=path)  # must not raise


def test_clean_gateway_start_is_silent(tmp_path, monkeypatch):
    _model(tmp_path, "claude-opus-5", provider="copilot-acp")
    path = _dotenv(tmp_path, "HERMES_ACP_NATIVE_TOOLS=1\n")
    monkeypatch.setenv(acp_config.NATIVE_VAR, "1")
    assert acp_config.enforce("gateway", path=path) == []


# --- snapshot ----------------------------------------------------------------

def test_snapshot_shape(tmp_path, monkeypatch):
    _model(tmp_path, "claude-opus-5", provider="copilot-acp")
    path = _dotenv(tmp_path, "HERMES_ACP_NATIVE_TOOLS=1\n")
    monkeypatch.setenv(acp_config.NATIVE_VAR, "1")
    snap = acp_config.snapshot(path=path)
    assert snap["model"] == {
        "config": "claude-opus-5",
        "provider": "copilot-acp",
        "legacy_pin": "",
        "resolved": "claude-opus-5",
    }
    assert snap["native_tools"]["resolved"] is True
    assert snap["divergences"] == []
    assert snap["config"].endswith("config.yaml")


def test_snapshot_shows_the_leftover_bridge_doctor_must_report(tmp_path):
    _model(tmp_path, "claude-opus-5")
    path = _dotenv(tmp_path, "HERMES_ACP_MODEL=claude-fable-5\n")
    snap = acp_config.snapshot(path=path)
    assert snap["model"]["legacy_pin"] == "claude-fable-5"
    assert snap["model"]["resolved"] == "claude-opus-5"
    assert [d["severity"] for d in snap["divergences"]] == ["fatal"]


# --- the call sites now delegate ---------------------------------------------

def test_client_effective_model_delegates(tmp_path):
    from agent.copilot_acp_client import _acp_effective_model

    _model(tmp_path, "claude-opus-5")
    assert _acp_effective_model() == "claude-opus-5"
    assert _acp_effective_model("claude-sonnet-5") == "claude-sonnet-5"


def test_native_tools_gate_delegates(monkeypatch):
    from agent.acp_native_tools import _acp_native_tools_enabled

    monkeypatch.setenv(acp_config.NATIVE_VAR, "1")
    assert _acp_native_tools_enabled() is True
    monkeypatch.setenv(acp_config.NATIVE_VAR, "0")
    assert _acp_native_tools_enabled() is False


def test_subprocess_env_always_pins_the_model(tmp_path):
    """MODEL_VAR's one remaining job: the wrapper must never guess its own default."""
    from agent import copilot_acp_client as cac

    _model(tmp_path, "claude-opus-5")
    with patch.object(cac, "_build_subprocess_env", return_value={}):
        assert cac._acp_env_with_model(None)[acp_config.MODEL_VAR] == "claude-opus-5"
        # per-call failover still forces its own model
        assert cac._acp_env_with_model("claude-opus-4-8")[acp_config.MODEL_VAR] == "claude-opus-4-8"


def test_subprocess_env_pins_requested_when_nothing_configured(tmp_path):
    from agent import copilot_acp_client as cac

    with patch.object(cac, "_build_subprocess_env", return_value={}):
        env = cac._acp_env_with_model(None, requested="claude-opus-5")
    assert env[acp_config.MODEL_VAR] == "claude-opus-5"


def test_an_explicit_choice_reaches_the_wrapper_env(tmp_path):
    """End of the chain: what resolve_model picks is what the bridge spawns on."""
    from agent import copilot_acp_client as cac

    _model(tmp_path, "claude-opus-5")
    with patch.object(cac, "_build_subprocess_env", return_value={}):
        env = cac._acp_env_with_model(None, requested="claude-fable-5")
    assert env[acp_config.MODEL_VAR] == "claude-fable-5"


def test_a_leftover_pin_never_reaches_the_wrapper(tmp_path, monkeypatch):
    """The child env is rewritten from config.yaml on EVERY spawn.

    So even an un-deleted export cannot survive into the engine: the value the
    wrapper sees is the one Python resolved, not the one it inherited.
    """
    from agent import copilot_acp_client as cac

    _model(tmp_path, "claude-opus-5")
    monkeypatch.setenv(acp_config.MODEL_VAR, "claude-fable-5")
    with patch.object(
        cac, "_build_subprocess_env",
        return_value={acp_config.MODEL_VAR: "claude-fable-5"},
    ):
        env = cac._acp_env_with_model(None)
    assert env[acp_config.MODEL_VAR] == "claude-opus-5"


def test_aux_call_reaches_the_wrapper_on_the_configured_model(tmp_path):
    """The third resolver: auxiliary_client had its own model logic.

    Utility calls (title generation) reach the bridge through the same
    ``create()`` and used to hand it the PROVIDER label as the model; the
    engine then ran whatever ``claude-acp-run.sh`` defaulted to. Assert the
    whole chain — create() → _run_prompt → subprocess env — now pins the model
    the single source names, for that exact input.
    """
    from agent import copilot_acp_client as cac

    _model(tmp_path, "claude-opus-5")
    client = cac.CopilotACPClient.__new__(cac.CopilotACPClient)
    client._native_tools = False
    captured = {}

    def _fake_run_prompt(prompt_text, *, timeout_seconds, model_override=None):
        with patch.object(cac, "_build_subprocess_env", return_value={}):
            captured["env"] = cac._acp_env_with_model(
                model_override, requested=getattr(client, "_acp_requested_model", None),
            )
        return "hi", ""

    client._run_prompt = _fake_run_prompt
    with patch.object(cac, "_format_messages_as_prompt", return_value="p"):
        completion = client._create_chat_completion(
            model="copilot-acp", messages=[{"role": "user", "content": "t"}],
        )
    assert captured["env"][acp_config.MODEL_VAR] == "claude-opus-5"
    assert completion.model == "claude-opus-5"


def test_failover_uses_the_requested_model_as_primary_label(monkeypatch):
    """Ledger honesty: requested/used must not differ just because nothing is set."""
    from agent import acp_failover

    monkeypatch.delenv(acp_config.MODEL_VAR, raising=False)

    class _Client:
        _activity_agent = None

        def _run_prompt(self, prompt_text, *, timeout_seconds, model_override=None):
            return "ok", ""

    with patch.object(acp_failover, "_arm_for_call", lambda _c: None):
        _, _, used, fired = acp_failover.run_prompt_with_failover(
            _Client(), "hi", 30.0, "claude-opus-5",
        )
    assert (used, fired) == ("claude-opus-5", False)
