"""Ranking-model import firewall (spec.md section 8/14): `src/poi_rank/models/**`
must never import from `scoring/` (or its submodules) -- models is upstream of
scoring, same rule shape as `tests/test_firewall_candidates.py`. `models/` MAY import
from `poi_rank.data`, `poi_rank.features`, and `poi_rank.candidates` (it consumes
their output/machinery -- the expected, legitimate direction). `models/` must also
never reference the oracle-only export directory at all -- no writer exemption, same
as `features/`/`candidates/`.

Parses every `.py` file under `src/poi_rank/models/` with `ast` (not string/regex
matching) and inspects `import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = ("poi_rank.scoring",)
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "src" / "poi_rank" / "models"


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


def test_models_dir_exists() -> None:
    assert MODELS_DIR.is_dir(), f"expected {MODELS_DIR} to exist"


def test_models_never_imports_scoring() -> None:
    py_files = sorted(MODELS_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under models/"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if _is_forbidden(module_name):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden module '{module_name}'"
                )

    assert not violations, "models/ firewall violated:\n" + "\n".join(violations)


def test_models_never_references_oracle_dir() -> None:
    """`models/**` must never reference `_oracle` at all -- only `eval/oracle.py`
    (reader) and `datagen/oracle_export.py` (writer) may, per
    `tests/test_oracle_isolation.py`'s allowlist."""
    py_files = sorted(MODELS_DIR.rglob("*.py"))
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in py_files
        if "_oracle" in path.read_text(encoding="utf-8")
    ]
    assert not violations, "models/ referenced '_oracle' (firewall violation):\n" + "\n".join(
        violations
    )


def test_models_may_import_data_features_candidates() -> None:
    """Sanity check that the firewall test itself isn't accidentally over-broad:
    `models/` genuinely does (and is allowed to) import from `poi_rank.data`,
    `poi_rank.features`, and `poi_rank.candidates` -- consuming upstream phases'
    output/machinery is the expected direction, not a violation."""
    py_files = sorted(MODELS_DIR.rglob("*.py"))
    imported: set[str] = set()
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported.update(_imported_module_names(tree))
    assert any(m.startswith("poi_rank.data") for m in imported)
    assert any(m.startswith("poi_rank.features") for m in imported)
    assert any(m.startswith("poi_rank.candidates") for m in imported)
