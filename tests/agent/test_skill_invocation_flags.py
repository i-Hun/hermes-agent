"""``disable-model-invocation`` (Agent Skills / Claude Code frontmatter key).

A skill carrying ``disable-model-invocation: true`` is no longer OFFERED to the model — it
leaves the system-prompt ``<available_skills>`` index, ``skills_list()`` and the not-found
hint — while every explicit load keeps working: ``skill_view(name)``, ``/skill:name`` and
``--skills``. Name resolution is untouched, so a hidden copy still owns its name.

The gate is offer-time only and fails open: only a boolean ``true`` (or the quoted word)
disables. That is what lets a dangerous skill (rollback, decommission, freeze) be reachable
by a human's explicit request and never by the model's own description-matching.
"""

import json
from unittest.mock import patch

import pytest

from agent.skill_commands import scan_skill_commands
from agent.skill_utils import TIER_EXTERNAL, skill_model_invocable
from tools.skills_tool import find_all_skills, skill_view, skills_list


def _write_skill(root, name, extra=""):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n{extra}---\n\n# {name}\n\nStep 1.\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.mark.parametrize("frontmatter, expected", [
    ({}, True),
    ({"disable-model-invocation": True}, False),
    ({"disable-model-invocation": False}, True),
    ({"disable-model-invocation": "true"}, False),
    ({"disable-model-invocation": " TRUE "}, False),
    ({"disable-model-invocation": " False "}, True),
    ({"disable-model-invocation": "yes"}, True),  # not a boolean: the default applies
    ({"disable-model-invocation": None}, True),
])
def test_model_invocable_parsing(frontmatter, expected):
    assert skill_model_invocable(frontmatter) is expected


@pytest.mark.parametrize("yaml_line, expected", [
    ("", True),
    ("disable-model-invocation: true\n", False),
    ("disable-model-invocation: false\n", True),
    ('disable-model-invocation: "true"\n', False),
    ("disable-model-invocation: yes\n", False),  # YAML 1.1: an unquoted `yes` IS boolean true
    ("disable-model-invocation: maybe\n", True),  # garbage value: fail open
])
def test_frontmatter_round_trip(tmp_path, yaml_line, expected):
    """The key is read through the REAL frontmatter parser, not a dict the test built."""
    from agent.skill_utils import parse_frontmatter
    skill_file = _write_skill(tmp_path, "probe", yaml_line) / "SKILL.md"
    frontmatter, _body = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
    assert skill_model_invocable(frontmatter) is expected


@pytest.fixture
def prompt_builder(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield pb
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_hidden_skill_stays_out_of_the_prompt_index_on_scan_and_snapshot(tmp_path, prompt_builder):
    pb = prompt_builder
    skills = tmp_path / "skills"
    _write_skill(skills, "wrapper")
    _write_skill(skills, "helper", "disable-model-invocation: true\n")
    _write_skill(skills, "quoted", 'disable-model-invocation: "true"\n')

    def build():
        return pb._build_skills_system_prompt_inner(skills, [], None, None, None)

    cold = build()
    assert "wrapper: Description for wrapper." in cold
    assert "helper" not in cold and "quoted" not in cold
    snapshot = pb._load_skills_snapshot(skills)
    assert snapshot is not None
    assert {e["skill_name"]: e["model_invocable"] for e in snapshot["skills"]} == {
        "wrapper": True, "helper": False, "quoted": False}
    pb.clear_skills_system_prompt_cache()  # in-process cache only: the next build uses the disk snapshot
    assert build() == cold


def test_snapshot_from_an_older_version_is_rebuilt(tmp_path, prompt_builder):
    """v5 snapshots carry no ``model_invocable``; a stale one must not leak the hidden row."""
    pb = prompt_builder
    skills = tmp_path / "skills"
    _write_skill(skills, "helper", "disable-model-invocation: true\n")
    pb._build_skills_system_prompt_inner(skills, [], None, None, None)
    snapshot_path = pb._skills_prompt_snapshot_path()
    stale = json.loads(snapshot_path.read_text(encoding="utf-8"))
    stale["version"] = 5
    for entry in stale["skills"]:
        entry.pop("model_invocable")
    snapshot_path.write_text(json.dumps(stale), encoding="utf-8")
    pb.clear_skills_system_prompt_cache()

    assert "helper" not in pb._build_skills_system_prompt_inner(skills, [], None, None, None)


def test_hidden_external_skill_is_not_indexed(tmp_path, prompt_builder):
    """``external_dirs`` skills take the same gate as the profile's own tree."""
    pb = prompt_builder
    skills, external = tmp_path / "skills", tmp_path / "external"
    skills.mkdir()
    _write_skill(external, "ext-helper", "disable-model-invocation: true\n")
    _write_skill(external, "ext-visible")

    index = pb._build_skills_system_prompt_inner(skills, [(TIER_EXTERNAL, external)], None, None, None)
    assert "ext-visible" in index
    assert "ext-helper" not in index


def test_hidden_plugin_skill_is_not_indexed(tmp_path, prompt_builder, monkeypatch):
    """Plugin skills reach ``<available_skills>`` through their own registry path, not the disk scan."""
    pb = prompt_builder
    skills = tmp_path / "skills"
    skills.mkdir()
    from hermes_cli import plugins

    class _Manager:
        def list_plugin_skill_metadata(self):
            return [{"name": f"acme:{n}", "description": f"Description for {n}.", "category": "plugin",
                     "frontmatter": fm} for n, fm in (("visible", {}), ("helper", {"disable-model-invocation": True}))]

    monkeypatch.setattr(plugins, "discover_plugins", lambda *a, **k: None)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: _Manager())
    index = pb._build_skills_system_prompt_inner(skills, [], None, None, None)
    assert "acme:visible" in index
    assert "acme:helper" not in index


def test_hidden_skill_still_owns_its_name(tmp_path, prompt_builder):
    """Precedence is unchanged: a hidden local skill shadows a same-named external one."""
    pb = prompt_builder
    skills, external = tmp_path / "skills", tmp_path / "external"
    _write_skill(skills, "deploy", "disable-model-invocation: true\n")
    _write_skill(external, "deploy")

    assert "deploy" not in pb._build_skills_system_prompt_inner(
        skills, [(TIER_EXTERNAL, external)], None, None, None)


def test_skills_list_and_not_found_hint_omit_hidden_skills_but_skill_view_loads_them(tmp_path):
    import tools.skills_tool as st

    _write_skill(tmp_path, "wrapper")
    _write_skill(tmp_path, "helper", "disable-model-invocation: true\n")

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
            patch("agent.skill_utils.get_disabled_skill_names", lambda *_: set()):
        st.clear_skills_cache()
        listed = json.loads(skills_list())
        missing = json.loads(skill_view("no-such-skill"))
        loaded = json.loads(skill_view("helper"))
        everything = {s["name"] for s in find_all_skills()}

    assert [s["name"] for s in listed["skills"]] == ["wrapper"]
    assert missing["available_skills"] == ["wrapper"]
    assert loaded["success"] is True and "Step 1." in loaded["content"]
    # User-facing listings (banner, /skills, web API) keep showing it.
    assert everything == {"wrapper", "helper"}


def test_hidden_skill_keeps_its_slash_command(tmp_path):
    """``/skill:name`` is the whole point of the flag: the human must still be able to ask."""
    import tools.skills_tool as st

    _write_skill(tmp_path, "user-only", "disable-model-invocation: true\n")

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
            patch("agent.skill_utils.get_disabled_skill_names", lambda *_: set()):
        st.clear_skills_cache()
        commands = scan_skill_commands()

    assert "/user-only" in commands

def _plugin_stub(**methods):
    class _Manager:
        _plugins: dict = {}  # _legacy_portable_skill_name probes it for agent-plugin- namespaces only

    for name, fn in methods.items():
        setattr(_Manager, name, staticmethod(fn))
    return _Manager()


def test_skills_list_plugin_rows_apply_the_gate(tmp_path, monkeypatch):
    """Plugin rows reach ``skills_list()`` through the registry loop, not the disk catalog."""
    import tools.skills_tool as st
    from hermes_cli import plugins

    manager = _plugin_stub(list_plugin_skill_metadata=lambda: [
        {"name": f"acme:{n}", "description": f"Description for {n}.", "category": "plugin", "frontmatter": fm}
        for n, fm in (("visible", {}), ("helper", {"disable-model-invocation": True}))
    ])
    monkeypatch.setattr(plugins, "discover_plugins", lambda *a, **k: None)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
            patch("agent.skill_utils.get_disabled_skill_names", lambda *_: set()):
        st.clear_skills_cache()
        listed = json.loads(skills_list())

    assert [s["name"] for s in listed["skills"]] == ["acme:visible"]


def test_plugin_not_found_hint_omits_hidden_plugin_skills(tmp_path, monkeypatch):
    """A miss inside a plugin namespace must not advertise that plugin's hidden skills."""
    import tools.skills_tool as st
    from hermes_cli import plugins

    manager = _plugin_stub(
        find_plugin_skill=lambda name: None,
        list_plugin_skills=lambda namespace: ["visible", "helper"],
        list_plugin_skill_metadata=lambda: [
            {"name": "acme:visible", "description": "Description for visible.", "category": "plugin", "frontmatter": {}},
            {"name": "acme:helper", "description": "Description for helper.", "category": "plugin",
             "frontmatter": {"disable-model-invocation": True}},
        ],
    )
    monkeypatch.setattr(plugins, "discover_plugins", lambda *a, **k: None)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        missing = json.loads(skill_view("acme:nope"))

    assert missing["success"] is False
    assert missing["available_skills"] == ["acme:visible"]
