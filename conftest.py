"""Repo-root conftest owned by the ACP-bridge overlay.

Why this file exists at all
---------------------------
The overlay's doctrine is "new files never conflict; only thin hooks into
upstream files can". Measured against v2026.8.3 (2839 upstream commits after
our pinned base) that held perfectly for the 25 overlay modules — and failed
for the small edits we had made *inside upstream test files and upstream audio
code*: 8 of 10 conflicts came from there, and the biggest hook we own
(``copilot_acp_client.py``, +454/-41) applied cleanly.

pytest loads ``<rootdir>/conftest.py`` for every collected test, whatever
directory it lives in. Upstream has no root conftest, so this is a new file
under the same doctrine — and it can do everything those edits did:

1. blank the overlay's own env flags (was: an insertion into
   ``tests/conftest.py::_HERMES_BEHAVIORAL_VARS``);
2. fasten the test-audio seat belt (was: hooks inside ``tools/tts_tool.py``
   and ``tools/voice_mode.py``, plus four opt-outs edited into
   ``tests/tools/*``);
3. record the upstream tests whose premise the overlay deliberately inverts
   (was: deleting/rewriting them in place).

If upstream ever ships its own root ``conftest.py``, ``git reset --hard``
during ``hermes update`` will refuse to overwrite this untracked file and say
so — loud, not silent. ``install.sh --check`` reports it too.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# 1. Env hermeticity for the overlay's own flags.
#
# They live in the host's real ``.env``, and any test that triggers an env load
# leaks them into ``os.environ`` for every test that follows — that is how
# ``test_copilot_acp_client``'s text tool-call shim assertions started failing
# only when run after ``tests/gateway/test_usage_limit_reply.py`` or
# ``tests/cron/`` (the shim is disabled while native tools are on). Blank them
# like any other behavioral var so a suite's result never depends on this
# machine's .env.
# ---------------------------------------------------------------------------

_ACP_BEHAVIORAL_VARS = (
    "HERMES_ACP_NATIVE_TOOLS",
    "HERMES_ACP_MODEL",
    "HERMES_ACP_FALLBACK_MODEL",
)


@pytest.fixture(autouse=True)
def _acp_overlay_env_hermeticity(monkeypatch):
    for name in _ACP_BEHAVIORAL_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 2. Test-audio seat belt — see tools/test_audio_guard.py for the incident and
#    for what is (and is not) muted.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _acp_audio_seat_belt(monkeypatch):
    from tools.test_audio_guard import audio_output_blocked_for_tests, install_seat_belt

    if not audio_output_blocked_for_tests():
        # HERMES_ALLOW_TEST_AUDIO=1 — the test's subject IS the player and it
        # mocks the device itself.
        return
    install_seat_belt(monkeypatch)


# ---------------------------------------------------------------------------
# 3. Upstream tests whose premise the overlay deliberately inverts.
#
# Two modes, and the difference is not cosmetic:
#
#   "xfail"    — the test genuinely FAILS against the overlay. strict=True, so
#                if it ever starts passing (the overlay regressed to upstream
#                behaviour) the run goes red and we find out.
#   "deselect" — the test still PASSES, because it asserts against a hand-copied
#                simulation rather than the code we changed. It is not a
#                regression signal, it is a statement about the gateway that is
#                no longer true. Dropping it is honest; xfail(strict) would be
#                wrong (it would fail on a green test).
#
# Rot guard below: if the file is collected but the nodeid is not, the run
# fails. A stale entry can therefore never rot into silence.
# ---------------------------------------------------------------------------

_OVERLAY_INVALIDATED = (
    # NOTE, 2026-08-04: the overlay used to DELETE
    # tests/agent/test_copilot_acp_client.py::test_request_permission_is_not_auto_allowed
    # outright. Re-checked when this table was built: it passes against the
    # overlay (a request_permission with no toolCall still resolves to
    # "cancelled"), so the deletion had been carrying a conflict for nothing.
    # It is not listed here — it simply runs.
    #
    # NOTE, 2026-08-04 (later, during the v2026.8.3 port): the table's single
    # entry —
    #   tests/gateway/test_telegram_progress_edit_transient.py
    #       ::test_flood_control_sets_can_edit_false
    # — is GONE. Not by our hand: upstream deleted that test in v2026.8.3, and
    # the rot guard below caught it on the first run against the new base
    # (pytest.UsageError, not a silent no-op — which is the whole reason the
    # guard exists). Removing the entry is therefore correct on BOTH bases:
    # measured on the pinned base with the deselect lifted, the file is
    # 22 passed — it was never a red test, only a statement about a gateway we
    # no longer run, and upstream has now made the same statement by deleting
    # it. The flood rule is proven by tests/gateway/test_progress_flood.py.
    #
    # The table being empty is the healthy state: an overlay that has to
    # invalidate upstream tests is an overlay that disagrees with upstream.
)


def _nodeid_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def pytest_collection_modifyitems(config, items):
    collected_files = {item.nodeid.split("::", 1)[0] for item in items}
    deselected = []

    for entry in _OVERLAY_INVALIDATED:
        nodeid, mode, reason = entry["nodeid"], entry["mode"], entry["reason"]
        target_file = _nodeid_file(nodeid)
        if target_file not in collected_files:
            continue  # this run does not touch that file — nothing to say

        matched = [item for item in items if item.nodeid == nodeid]
        if not matched:
            raise pytest.UsageError(
                f"ACP overlay: {target_file} was collected but "
                f"{nodeid!r} no longer exists. The overlay's list of "
                f"deliberately-invalidated upstream tests is stale — update "
                f"conftest.py::_OVERLAY_INVALIDATED (this check exists so the "
                f"list cannot rot into silence)."
            )

        for item in matched:
            if mode == "xfail":
                item.add_marker(
                    pytest.mark.xfail(reason=reason, strict=True, run=True)
                )
            else:
                deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        for item in deselected:
            items.remove(item)


def pytest_report_header(config):
    del config
    return (
        "acp-overlay: root conftest active "
        f"(env-blank {len(_ACP_BEHAVIORAL_VARS)}, audio seat belt on, "
        f"{len(_OVERLAY_INVALIDATED)} upstream tests overridden)"
    )
