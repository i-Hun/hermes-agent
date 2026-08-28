"""Seat belt: never drive the machine's real audio output from a test run.

Real incident, 2026-07-30: ``pytest tests/gateway`` made the desktop say
"This is the first sentence." out loud while its owner was working.
``TestStreamTtsToSpeaker`` calls ``stream_tts_to_speaker`` with no provider or
device patched; the sync path synthesized real MP3s and handed them to
``play_audio_file``.

Where the belt is fastened
--------------------------
Until 2026-08-04 the belt was a hook *inside* ``tools/tts_tool.py`` and
``tools/voice_mode.py`` — two upstream files, so every ``hermes update`` risked
a conflict there (and 2026.8.3, whose headline feature is streaming TTS with
barge-in, delivered exactly that: +1094/-148 in ``tts_tool.py`` alone).

It is now applied from the repo-root ``conftest.py`` instead: an autouse
fixture calls :func:`install_seat_belt`, which replaces the *device-facing*
callables for the duration of each test. Upstream source is untouched, so this
mechanism cannot conflict with anything.

Where the belt is fastened *within* a call — and why that changed
-----------------------------------------------------------------
The in-tree hook sat at the top of ``play_beep`` / ``play_audio_file`` /
``stream_tts_to_speaker`` and read ``HERMES_ALLOW_TEST_AUDIO`` at CALL time, so
a test could opt out from its own body — which four upstream tests did, via
edits we had to carry.

An autouse fixture cannot read the env at call time; it decides at SETUP, before
the test body runs. Porting the opt-out that way would have needed a table of
upstream nodeids in our conftest — a list that rots.

So the belt moved down to the two seams where a sound actually leaves the
process, and the opt-outs stopped being necessary at all:

* ``import sounddevice`` is made to fail (``sys.modules[...] = None``), which is
  the ordinary state of a headless box without PortAudio and which every caller
  here already handles;
* ``shutil.which`` denies the *system players* (``afplay``/``ffplay``/``aplay``
  …) that ``play_audio_file`` falls back to.

A test that mocks the device itself — ``_import_audio``, ``mock_sd``, its own
``shutil.which`` — is untouched, because the belt never enters its path. That
is precisely the set of tests that used to need the escape hatch.

NOT muted: synthesis. The in-tree hook returned before ``text_to_speech_tool``
ran, which also saved the CPU of a piper render. Muting synthesis from outside
would mean stubbing ``text_to_speech_tool`` globally, and 20+ upstream test
files assert on it — the cure would be worse than the cost. Synthesis is
expensive, not audible; the belt's job is silence.

``HERMES_ALLOW_TEST_AUDIO=1`` still works, now as a whole-run operator escape
(``HERMES_ALLOW_TEST_AUDIO=1 pytest …``) rather than a per-test one.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

#: Escape hatch for tests whose subject *is* the playback plumbing.
ALLOW_ENV = "HERMES_ALLOW_TEST_AUDIO"


def audio_output_blocked_for_tests() -> bool:
    """True when the caller is inside pytest and has not opted into audio.

    In production ``PYTEST_CURRENT_TEST`` is absent, so this is one dict lookup.
    """
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return os.environ.get(ALLOW_ENV, "").strip().lower() not in _TRUTHY


def note_suppressed(what: str) -> None:
    """Log the suppression so a silent test run is still an explained one."""
    logger.info(
        "Audio output suppressed during test run (%s). Set %s=1 to allow.",
        what,
        ALLOW_ENV,
    )


#: Command-line players ``play_audio_file`` falls back to when sounddevice is
#: unavailable. Denying them via ``shutil.which`` is what keeps the fallback
#: path silent; the list is the union of every branch in that function plus the
#: usual suspects, so a new branch upstream does not quietly re-open the device.
BLOCKED_PLAYERS = frozenset({
    "afplay", "ffplay", "aplay", "paplay", "play", "mpv", "mpg123", "cvlc",
    "vlc", "pw-play", "gst-play-1.0", "powershell", "powershell.exe",
    # ffmpeg is deliberately NOT here: it is a TRANSCODER that upstream
    # calls for Opus conversion and (since v2026.8.13) silence trimming
    # for cloud STT. Blocking it by name muted a feature that never
    # touches the speakers — the guard was fastened too high. The one
    # playback path that used ffmpeg is the WSL fallback, and it needs
    # powershell.exe as well; that basename is now blocked directly,
    # which is where the device actually is.
})


def install_seat_belt(monkeypatch) -> List[str]:
    """Deny the process every route to the speakers, for ``monkeypatch``'s life.

    Returns the list of seams actually blocked, so a caller (and the tests) can
    assert the belt was fastened rather than trust that it was. An empty list
    means nothing was blocked — the failure mode worth seeing.

    Deliberately does NOT replace ``play_audio_file``/``play_beep``: a test that
    mocks the device itself must keep exercising the real function bodies, and
    that is exactly the set of tests which used to need an opt-out.

    ``monkeypatch`` is pytest's fixture (or a bare ``pytest.MonkeyPatch``); the
    undo happens through it, so production imports are never affected.
    """
    blocked: List[str] = []

    # 1. The WAV path and every OutputStream go through sounddevice. Making the
    #    import fail routes them into the "no audio device" branch that every
    #    caller here already handles — ``sys.modules[name] = None`` is the
    #    documented way to do that.
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    blocked.append("sounddevice")

    # 2. …which lands in the system-player fallback. Deny it the binaries.
    #    A test that patches ``shutil.which`` itself does so from its body,
    #    i.e. after this fixture, so its patch wins — as it should.
    import shutil

    _real_which = shutil.which

    def _which_without_players(cmd, *args, **kwargs):
        name = os.path.basename(str(cmd))
        if name in BLOCKED_PLAYERS:
            note_suppressed(f"system player {name}")
            return None
        return _real_which(cmd, *args, **kwargs)

    # Carried so a test can reach the unblocked lookup without knowing whether
    # the belt is one or several layers deep (``__wrapped__`` is the stdlib
    # convention functools.wraps uses for exactly this).
    _which_without_players.__wrapped__ = getattr(_real_which, "__wrapped__", _real_which)

    monkeypatch.setattr(shutil, "which", _which_without_players)
    blocked.append("shutil.which(system players)")

    return blocked
