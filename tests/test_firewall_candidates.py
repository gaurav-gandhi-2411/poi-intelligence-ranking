"""Candidate-generation import firewall (spec.md section 7/14):
`src/poi_rank/candidates/**` must never import from `models/` or `scoring/` (or their
submodules) -- candidates is upstream of those phases, same rule shape as
`tests/test_firewall.py`/`tests/test_firewall_data.py`/`tests/test_firewall_features.py`.
`candidates/` MAY import from `poi_rank.features` and `poi_rank.data` (it consumes
their output -- that is the expected, legitimate direction). `candidates/` must also
never reference `_oracle` at all -- it has no writer exemption, same as `features/`.

Parses every `.py` file under `src/poi_rank/candidates/` with `ast` (not string/regex
matching) and inspects `import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = (
    "poi_rank.models",
    "poi_rank.scoring",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_DIR = REPO_ROOT / "src" / "poi_rank" / "candidates"


def _imported_module_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == root or module_name.startswith(root + ".") for root in FORBIDDEN_ROOTS
    )


def test_candidates_dir_exists() -> None:
    assert CANDIDATES_DIR.is_dir(), f"expected {CANDIDATES_DIR} to exist"


def test_candidates_never_imports_downstream_phases() -> None:
    py_files = sorted(CANDIDATES_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under candidates/"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if _is_forbidden(module_name):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden module '{module_name}'"
                )

    assert not violations, "candidates/ firewall violated:\n" + "\n".join(violations)


def test_candidates_never_references_oracle_dir() -> None:
    """`candidates/**` must never reference `_oracle` at all -- only
    `eval/oracle.py` (reader) and `datagen/oracle_export.py` (writer) may, per
    `tests/test_oracle_isolation.py`'s allowlist."""
    py_files = sorted(CANDIDATES_DIR.rglob("*.py"))
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in py_files
        if "_oracle" in path.read_text(encoding="utf-8")
    ]
    assert not violations, "candidates/ referenced '_oracle' (firewall violation):\n" + "\n".join(
        violations
    )


def test_candidates_may_import_features_and_data() -> None:
    """Sanity check that the firewall test itself isn't accidentally over-broad:
    `candidates/` genuinely does (and is allowed to) import from `poi_rank.features`
    and `poi_rank.data` -- consuming upstream phases' output is the expected
    direction, not a violation."""
    py_files = sorted(CANDIDATES_DIR.rglob("*.py"))
    imported: set[str] = set()
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported.update(_imported_module_names(tree))
    assert any(m.startswith("poi_rank.features") for m in imported)
    assert any(m.startswith("poi_rank.data") for m in imported)
