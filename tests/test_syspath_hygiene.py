"""Guard: the ``tests/`` directory must never land on ``sys.path``.

Every subpackage under ``tests/`` is named after the production package it
covers (``tests/agent``, ``tests/tools``, ``tests/gateway``, …) and
``tests/acp`` collides with the installed ``acp`` distribution. So the moment
``tests/`` is prepended to ``sys.path``, a plain ``import acp`` resolves to
``tests/acp/__init__.py`` instead of the real one and every later import of
``acp.schema`` dies with ``ModuleNotFoundError``.

That failure is order-dependent: it only bites the tests collected *after* the
offending module, which makes it look like flakiness rather than a one-line
off-by-one. It was real — ``test_web_server_host_header.py`` used
``parents[1]`` where every sibling uses ``parents[2]``, and it took down
``test_kanban_review_surfaces.py`` in any batch run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# Only ``parents[N]`` anchored on the test module's own ``__file__`` — a
# ``parents[N]`` hanging off some other module's path means something else.
_PARENTS_RE = re.compile(r"Path\(__file__\)(?:\.resolve\(\))?\.parents\[(\d+)\]")
_INSERTS_PATH = ("sys.path.insert", "sys.path.append")


def _test_sources() -> list[Path]:
    return sorted(p for p in TESTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_tests_dir_is_not_on_syspath() -> None:
    """Runtime check — catches a polluter regardless of how it got there."""
    assert str(TESTS_DIR) not in sys.path, (
        f"{TESTS_DIR} is on sys.path; every tests/<pkg> now shadows the "
        "production package of the same name"
    )


def test_import_acp_resolves_outside_the_tests_tree() -> None:
    """The concrete collision that shipped the bug."""
    acp = pytest.importorskip("acp")
    origin = Path(getattr(acp, "__file__", "") or "").resolve()
    assert TESTS_DIR not in origin.parents, (
        f"`import acp` resolved to {origin} — tests/acp is shadowing the real "
        "acp package"
    )
    pytest.importorskip("acp.schema")


def test_no_test_module_prepends_the_tests_dir() -> None:
    """Static check — order-independent, so it fails on the guilty file itself.

    A ``sys.path`` insert in ``tests/<pkg>/test_x.py`` wants the repo root,
    which is ``parents[2]``. ``parents[1]`` is ``tests/`` — the bug.
    """
    offenders: list[str] = []
    for source in _test_sources():
        text = source.read_text(encoding="utf-8", errors="replace")
        if not any(marker in text for marker in _INSERTS_PATH):
            continue
        # parents[N] that resolves to TESTS_DIR for a file at this depth.
        tests_dir_index = len(source.relative_to(TESTS_DIR).parts) - 1
        for match in _PARENTS_RE.finditer(text):
            if int(match.group(1)) == tests_dir_index:
                offenders.append(
                    f"{source.relative_to(TESTS_DIR.parent)}: parents[{tests_dir_index}] "
                    f"is the tests/ dir (did you mean parents[{tests_dir_index + 1}]?)"
                )
        if re.search(r"sys\.path\.(insert|append)\([^)]*['\"][^'\"]*/tests['\"]", text):
            offenders.append(
                f"{source.relative_to(TESTS_DIR.parent)}: inserts a literal .../tests path"
            )
    assert not offenders, "sys.path pollution:\n  " + "\n  ".join(offenders)
