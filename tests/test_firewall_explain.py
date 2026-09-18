"""Explainability import firewall (spec.md section 10/14): `src/poi_rank/explain/**`
must never reference the oracle-only export directory -- same `_oracle` isolation
rule shape as every other non-`datagen`/`eval` module (`tests/
test_firewall_scoring.py`/`tests/test_firewall_models.py`). `explain/` is the LAST
stage before output, so unlike `scoring/`/`models/` it has no "never imports
downstream-package X" half of the firewall (nothing is downstream of it) -- only the
oracle-isolation half applies here, per the task's explicit instruction.

Parses every `.py` file under `src/poi_rank/explain/` with `ast` (not string/regex
matching) and inspects `import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPLAIN_DIR = REPO_ROOT / "src" / "poi_rank" / "explain"


def _imported_module_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_explain_dir_exists() -> None:
    assert EXPLAIN_DIR.is_dir(), f"expected {EXPLAIN_DIR} to exist"


def test_explain_never_references_oracle_dir() -> None:
    """`explain/**` must never reference `_oracle` at all -- only `eval/oracle.py`
    (reader) and `datagen/oracle_export.py` (writer) may, per
    `tests/test_oracle_isolation.py`'s allowlist."""
    py_files = sorted(EXPLAIN_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under explain/"

    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in py_files
        if "_oracle" in path.read_text(encoding="utf-8")
    ]
    assert not violations, "explain/ referenced '_oracle' (firewall violation):\n" + "\n".join(
        violations
    )


def test_explain_may_import_data_features_candidates_models_scoring() -> None:
    """Sanity check that the firewall test itself isn't accidentally over-broad:
    `explain/` genuinely does (and is allowed to) import from `poi_rank.data` and
    `poi_rank.scoring` -- consuming upstream phases' output/machinery is the
    expected direction, not a violation. (`poi_rank.models` is imported
    transitively via `poi_rank.scoring.compatibility`/`poi_rank.models.baselines`,
    not necessarily by name in every single file, so only `data` and `scoring` are
    asserted directly here -- both are genuinely imported by at least one file.)
    """
    py_files = sorted(EXPLAIN_DIR.rglob("*.py"))
    imported: set[str] = set()
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported.update(_imported_module_names(tree))
    assert any(m.startswith("poi_rank.data") for m in imported)
    assert any(m.startswith("poi_rank.scoring") for m in imported)
    assert any(m.startswith("poi_rank.models") for m in imported)


def test_explain_never_imported_by_scoring() -> None:
    """The inverse half of the firewall, already enforced (and independently owned)
    by `tests/test_firewall_scoring.py
    ::test_scoring_never_imports_explain` -- restated here as a cross-reference so a
    reader of `test_firewall_explain.py` alone isn't misled into thinking `explain/`
    has no firewall constraint pointing INTO it, just none pointing OUT to a
    downstream package (module docstring)."""
    scoring_dir = REPO_ROOT / "src" / "poi_rank" / "scoring"
    py_files = sorted(scoring_dir.rglob("*.py"))
    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if module_name == "poi_rank.explain" or module_name.startswith("poi_rank.explain."):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports '{module_name}'")
    assert not violations, "scoring/ imported explain/ (firewall violation):\n" + "\n".join(
        violations
    )
