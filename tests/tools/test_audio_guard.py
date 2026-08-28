"""The test-audio seat belt: pytest must never reach the machine's speakers.

Regression cover for the 2026-07-30 incident where ``pytest tests/gateway``
spoke "This is the first sentence." out loud through the user's desktop.

Since 2026-08-04 the belt is fastened by the repo-root ``conftest.py`` instead
of by hooks inside ``tools/tts_tool.py`` / ``tools/voice_mode.py``, and it acts
on the two *device* seams (sounddevice, the system players) rather than on the
playback functions — see ``tools/test_audio_guard.py`` for why.

Note the self-referential trap these tests sit in: the root conftest has
ALREADY fastened the same belt for every test in this file. Asserting "the
speakers stayed silent" would therefore pass even if
:func:`install_seat_belt` did nothing. Every test below either lifts the belt
first (restoring the real seams) or asserts on what ``install_seat_belt``
returns — never on ambient silence.
"""

from __future__ import annotations

import queue
import shutil
import sys
import threading

import pytest

from tools import test_audio_guard as guard


@pytest.fixture
def unbelted(monkeypatch):
    """Undo the conftest's belt for this test, and hand back the real seams.

    Restoring ``shutil.which`` and dropping the poisoned ``sounddevice`` entry
    is what lets a test observe the difference the belt makes.
    """
    current = shutil.which
    monkeypatch.setattr(shutil, "which", getattr(current, "__wrapped__", current))
    if sys.modules.get("sounddevice", "absent") is None:
        monkeypatch.delitem(sys.modules, "sounddevice")
    return monkeypatch


# ---------------------------------------------------------------------------
# audio_output_blocked_for_tests
# ---------------------------------------------------------------------------

class TestGuardPredicate:
    def test_blocked_inside_pytest_by_default(self, monkeypatch):
        monkeypatch.delenv(guard.ALLOW_ENV, raising=False)
        # PYTEST_CURRENT_TEST is set by pytest itself for this very test.
        assert guard.audio_output_blocked_for_tests() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_opt_in_unblocks(self, monkeypatch, value):
        monkeypatch.setenv(guard.ALLOW_ENV, value)
        assert guard.audio_output_blocked_for_tests() is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
    def test_non_truthy_opt_in_stays_blocked(self, monkeypatch, value):
        monkeypatch.setenv(guard.ALLOW_ENV, value)
        assert guard.audio_output_blocked_for_tests() is True

    def test_production_is_never_blocked(self, monkeypatch):
        """No PYTEST_CURRENT_TEST → the guard is inert (one dict lookup)."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv(guard.ALLOW_ENV, raising=False)
        assert guard.audio_output_blocked_for_tests() is False


# ---------------------------------------------------------------------------
# install_seat_belt — what it actually fastens
# ---------------------------------------------------------------------------

class TestInstallSeatBelt:
    def test_reports_every_seam_it_blocked(self):
        """An empty list is the failure mode worth seeing, so it is asserted."""
        mp = pytest.MonkeyPatch()
        try:
            blocked = guard.install_seat_belt(mp)
        finally:
            mp.undo()

        assert "sounddevice" in blocked
        assert any("shutil.which" in item for item in blocked)

    def test_sounddevice_import_is_made_to_fail(self):
        """The streaming path writes PCM straight to an OutputStream; the only
        way to keep it off the device is to deny it the module."""
        mp = pytest.MonkeyPatch()
        try:
            guard.install_seat_belt(mp)
            with pytest.raises(ImportError):
                import sounddevice  # noqa: F401
            assert sys.modules["sounddevice"] is None
        finally:
            mp.undo()

    def test_system_players_are_denied_but_other_binaries_are_not(self):
        mp = pytest.MonkeyPatch()
        try:
            guard.install_seat_belt(mp)
            for player in ("ffplay", "aplay", "afplay", "/usr/bin/ffplay"):
                assert shutil.which(player) is None, player
            # Everything else resolves normally — the belt is a scalpel, and a
            # blanket which()→None would break unrelated tooling probes.
            assert shutil.which("sh")
        finally:
            mp.undo()

    def test_undo_restores_the_real_seams(self, unbelted):
        real_which = shutil.which
        mp = pytest.MonkeyPatch()
        guard.install_seat_belt(mp)
        assert shutil.which is not real_which
        mp.undo()
        assert shutil.which is real_which


# ---------------------------------------------------------------------------
# tools.voice_mode playback entry points, driven for real
# ---------------------------------------------------------------------------

class TestVoiceModePlaybackIsMuted:
    def test_play_audio_file_finds_no_route_to_a_device(self, unbelted, tmp_path):
        """No sounddevice, no player → False, and no process was spawned."""
        import subprocess

        import tools.voice_mode as vm

        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"RIFF0000WAVEfmt ")

        def _boom(*_a, **_k):  # pragma: no cover - must not be reached
            raise AssertionError("audio device acquired during a test run")

        unbelted.setattr(subprocess, "Popen", _boom)
        guard.install_seat_belt(unbelted)

        assert vm.play_audio_file(str(wav)) is False

    def test_play_beep_refuses(self, unbelted):
        """play_beep has no fallback player: denying sounddevice is enough."""
        import tools.voice_mode as vm

        guard.install_seat_belt(unbelted)
        vm.play_beep()  # returns quietly instead of raising

    def test_a_test_that_mocks_the_device_is_untouched(self, unbelted):
        """The reason no opt-out table is needed any more.

        ``tests/tools/test_voice_mode.py`` mocks ``_import_audio``; the belt
        never enters that path, so those tests keep passing unmodified.
        """
        np = pytest.importorskip("numpy")

        import tools.voice_mode as vm

        played = []

        class _FakeSd:
            def play(self, *a, **k):
                played.append(a)

            def get_stream(self):
                return None

            def wait(self, *a, **k):
                return None

            def stop(self):
                pass

        unbelted.setattr(vm, "_import_audio", lambda: (_FakeSd(), np))
        guard.install_seat_belt(unbelted)

        vm.play_beep()
        assert played, "a mocked device must still be reachable"


# ---------------------------------------------------------------------------
# The actual incident: tools.tts_tool.stream_tts_to_speaker
# ---------------------------------------------------------------------------

class TestStreamTtsToSpeakerIsSilentUnderPytest:
    """Replays TestStreamTtsToSpeaker::test_sentence_splitting — the test that
    made the desktop talk — and asserts the pipeline still works but stays mute.
    """

    def _run(self, monkeypatch, *, deltas, fasten):
        import subprocess

        import tools.tts_tool as tts

        spawned: list = []
        synthesized: list[str] = []

        # Stand in for the synthesizer so this test needs no TTS provider; the
        # subject here is whether the RESULT reaches a device.
        def _fake_tts(**kw):
            synthesized.append(kw.get("text"))
            path = kw.get("output_path")
            if path:
                with open(path, "wb") as fh:
                    fh.write(b"ID3fake-mp3-payload")
            return {"success": True}

        monkeypatch.setattr(tts, "text_to_speech_tool", _fake_tts)

        class _FakeProc:
            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def _fake_popen(cmd, *a, **k):
            spawned.append(cmd)
            return _FakeProc()

        monkeypatch.setattr(subprocess, "Popen", _fake_popen)

        if fasten:
            guard.install_seat_belt(monkeypatch)

        q: queue.Queue = queue.Queue()
        for delta in deltas:
            q.put(delta)
        q.put(None)

        spoken: list[str] = []
        done = threading.Event()
        tts.stream_tts_to_speaker(
            q, threading.Event(), done, display_callback=spoken.append
        )
        return spoken, synthesized, spawned, done

    def test_display_still_works_but_no_player_is_spawned(self, unbelted):
        spoken, synthesized, spawned, done = self._run(
            unbelted,
            deltas=["This is the first sentence. ", "This is the second sentence. "],
            fasten=True,
        )

        # The original assertions of test_sentence_splitting still hold …
        assert done.is_set()
        assert len(spoken) >= 2
        # … synthesis is deliberately still reached (see the module docstring) …
        assert synthesized
        # … while nothing was ever handed to a player.
        assert spawned == []

    def test_without_the_belt_the_audio_path_is_reached(self, unbelted):
        """Proof the belt — not a broken pipeline — is what silences this.

        Without it, the very same run hands an MP3 to a system player. This is
        the 2026-07-30 incident, reproduced. Skipped where the host has no
        player installed at all, because then there is nothing to prove.
        """
        if not any(shutil.which(p) for p in ("ffplay", "aplay", "afplay")):
            pytest.skip("host has no system audio player — nothing to silence")

        _, synthesized, spawned, done = self._run(
            unbelted, deltas=["This is the first sentence. "], fasten=False
        )

        assert done.is_set()
        assert synthesized and "first sentence" in synthesized[0]
        assert spawned, "the unbelted path must reach a player"
