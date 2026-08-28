"""Tests for ``hermes kanban db-path``.

The command exists to kill a specific footgun: the default board's DB
lives at ``<root>/kanban.db`` while the sibling ``<root>/kanban/``
directory holds workspaces/logs/attachments. An agent that guesses
``<root>/kanban/kanban.db`` and opens it with plain ``sqlite3`` gets a
silently-created 0-byte file and concludes the board is empty.

So the two properties worth locking are: it prints the *canonical* path,
and it creates nothing while doing so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """A HERMES_HOME with no kanban DB and no kanban/ directory yet."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_db_path_prints_canonical_path(fresh_home):
    out = kc.run_slash("db-path").strip()
    assert out == str(fresh_home / "kanban.db")
    # One line exactly — callers use it as `sqlite3 -readonly "$(...)"`.
    assert "\n" not in out


def test_db_path_creates_nothing(fresh_home):
    """The whole point: asking where the DB is must not make one."""
    kc.run_slash("db-path")
    kc.run_slash("db-path --json")
    assert not (fresh_home / "kanban.db").exists()
    # And no stray DB in the sibling directory, which is the path agents
    # actually guessed wrong twice.
    assert not (fresh_home / "kanban" / "kanban.db").exists()


def test_db_path_does_not_point_into_the_sibling_directory(fresh_home):
    """Regression guard for the guessed path itself."""
    out = kc.run_slash("db-path").strip()
    assert out != str(fresh_home / "kanban" / "kanban.db")
    assert Path(out).parent == fresh_home


def test_db_path_json_reports_existence_and_roots(fresh_home):
    payload = json.loads(kc.run_slash("db-path --json"))
    assert payload["board"] == kb.DEFAULT_BOARD
    assert payload["db_path"] == str(fresh_home / "kanban.db")
    assert payload["exists"] is False
    assert payload["size_bytes"] == 0
    assert payload["source"] == "board"
    # The sibling roots are what makes <root>/kanban/ look DB-shaped.
    assert payload["workspaces_root"] == str(fresh_home / "kanban" / "workspaces")
    assert payload["logs_dir"] == str(fresh_home / "kanban" / "logs")
    assert payload["attachments_root"] == str(fresh_home / "kanban" / "attachments")


def test_db_path_json_reports_a_live_db(fresh_home):
    kb.init_db()
    payload = json.loads(kc.run_slash("db-path --json"))
    assert payload["exists"] is True
    assert payload["size_bytes"] > 0


def test_db_path_honours_env_pin(fresh_home, tmp_path, monkeypatch):
    """The dispatcher pins HERMES_KANBAN_DB into worker env; workers that
    ask for the path must get the pinned one, not the board default."""
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    payload = json.loads(kc.run_slash("db-path --json"))
    assert payload["db_path"] == str(pinned)
    assert payload["source"] == "HERMES_KANBAN_DB"
    assert not pinned.exists()


def test_db_path_resolves_per_board(fresh_home):
    kb.create_board("beta")
    out = kc.run_slash("--board beta db-path").strip()
    assert out == str(kb.board_dir("beta") / "kanban.db")
    # ...and the default board is unaffected by the override.
    assert kc.run_slash("db-path").strip() == str(fresh_home / "kanban.db")


def test_paths_alias_matches_db_path(fresh_home):
    assert kc.run_slash("paths").strip() == kc.run_slash("db-path").strip()
