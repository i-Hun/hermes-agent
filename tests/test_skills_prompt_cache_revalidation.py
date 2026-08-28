"""The skills index must not be frozen for the life of the process.

Overlay test (19.08.2026). A skill written by ANOTHER process — ``skill_manage``
running inside an ACP backend, a hand edit, ``git pull``,
``hermes-patch-update`` — used to stay invisible in the ``available_skills``
block of the system prompt until the gateway was restarted: the in-process LRU
in ``build_skills_system_prompt`` was keyed on configuration only, and the disk
snapshot that DOES carry an mtime/size manifest is consulted solely on an LRU
miss, which never happened. Four ``corp-*`` skills were lost that way and Oleg
asked for one of them by name.

The fix stores ``(result, manifest)`` in the LRU and revalidates the manifest on
every hit. These tests pin both halves: staleness is detected, and an unchanged
directory is still served from cache instead of rescanned.
"""

from __future__ import annotations

import pytest

import agent.prompt_builder as pb


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    """Point the builder at a throwaway skills dir (never rmtree'd here)."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr(pb, "get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr(pb, "get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr(pb, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_a, **_kw: set())
    pb.clear_skills_system_prompt_cache()
    yield skills_dir
    pb.clear_skills_system_prompt_cache()


def _write_skill(skills_dir, name: str) -> None:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: overlay revalidation probe {name}\n---\n\nbody\n",
        encoding="utf-8",
    )


def test_skill_written_by_another_process_is_visible_without_restart(skills_home):
    _write_skill(skills_home, "alpha-skill")
    first = pb.build_skills_system_prompt()
    assert "alpha-skill" in first

    # Same process, no cache invalidation call: exactly what the ACP backend
    # cannot do to the gateway's memory.
    _write_skill(skills_home, "beta-skill")
    second = pb.build_skills_system_prompt()
    assert "beta-skill" in second, (
        "the LRU served a stale index — a skill on disk is invisible to the model"
    )


def test_edited_description_is_picked_up(skills_home):
    _write_skill(skills_home, "gamma-skill")
    assert "overlay revalidation probe gamma-skill" in pb.build_skills_system_prompt()

    (skills_home / "gamma-skill" / "SKILL.md").write_text(
        "---\nname: gamma-skill\ndescription: rewritten description\n---\n\nbody\n",
        encoding="utf-8",
    )
    assert "rewritten description" in pb.build_skills_system_prompt()


def test_removed_skill_disappears(skills_home):
    _write_skill(skills_home, "delta-skill")
    assert "delta-skill" in pb.build_skills_system_prompt()

    (skills_home / "delta-skill" / "SKILL.md").unlink()
    assert "delta-skill" not in pb.build_skills_system_prompt()


def test_unchanged_directory_is_served_from_the_lru(skills_home, monkeypatch):
    """Revalidation must not turn every call into a full rebuild."""
    _write_skill(skills_home, "epsilon-skill")
    cold = {"n": 0}
    original = pb._load_skills_snapshot

    def counting(*a, **kw):
        cold["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(pb, "_load_skills_snapshot", counting)

    pb.build_skills_system_prompt()
    assert cold["n"] == 1
    for _ in range(3):
        pb.build_skills_system_prompt()
    assert cold["n"] == 1, "an unchanged skills dir was rescanned — the LRU stopped working"


def test_lru_entry_carries_a_manifest(skills_home):
    _write_skill(skills_home, "zeta-skill")
    pb.build_skills_system_prompt()
    (_result, manifest), = pb._SKILLS_PROMPT_CACHE.values()
    assert "zeta-skill/SKILL.md" in manifest
