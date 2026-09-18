"""Scoring-layer import firewall (spec.md section 9/14): `src/poi_rank/scoring/**`
must never import from `explain/` (or its submodules) -- scoring is upstream of
explain, same rule shape as `tests/test_firewall_models.py`/
`tests/test_firewall_candidates.py`. `scoring/` MAY import from `poi_rank.data`,
`poi_rank.features`, `poi_rank.candidates`, and `poi_rank.models` (it consumes their
output/machinery -- the expected, legitimate direction). `scoring/` must also never
reference the oracle-only export directory at all -- no writer exemption, same as
every other non-`datagen`/`eval` module.

Parses every `.py` file under `src/poi_rank/scoring/` with `ast` (not string/regex
matching) and inspects `import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = ("poi_rank.explain",)
REPO_ROOT = Path(__file__).resolve().parents[1]
SCORING_DIR = REPO_ROOT / "src" / "poi_rank" / "scoring"


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


def test_scoring_dir_exists() -> None:
    assert SCORING_DIR.is_dir(), f"expected {SCORING_DIR} to exist"


def test_scoring_never_imports_explain() -> None:
    py_files = sorted(SCORING_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under scoring/"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if _is_forbidden(module_name):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden module '{module_name}'"
                )

    assert not violations, "scoring/ firewall violated:\n" + "\n".join(violations)


def test_scoring_never_references_oracle_dir() -> None:
    """`scoring/**` must never reference `_oracle` at all -- only `eval/oracle.py`
    (reader) and `datagen/oracle_export.py` (writer) may, per
    `tests/test_oracle_isolation.py`'s allowlist."""
    py_files = sorted(SCORING_DIR.rglob("*.py"))
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in py_files
        if "_oracle" in path.read_text(encoding="utf-8")
    ]
    assert not violations, "scoring/ referenced '_oracle' (firewall violation):\n" + "\n".join(
        violations
    )


def test_scoring_may_import_data_features_candidates_models() -> None:
    """Sanity check that the firewall test itself isn't accidentally over-broad:
    `scoring/` genuinely does (and is allowed to) import from `poi_rank.data`,
    `poi_rank.features`, `poi_rank.candidates`, and `poi_rank.models` -- consuming
    upstream phases' output/machinery is the expected direction, not a violation."""
    py_files = sorted(SCORING_DIR.rglob("*.py"))
    imported: set[str] = set()
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported.update(_imported_module_names(tree))
    assert any(m.startswith("poi_rank.data") for m in imported)
    assert any(m.startswith("poi_rank.features") for m in imported)
    assert any(m.startswith("poi_rank.candidates") for m in imported)
    assert any(m.startswith("poi_rank.models") for m in imported)


def test_scoring_never_imports_eval() -> None:
    """Not required by spec.md's own firewall wording, but a deliberate design
    choice documented in `scoring/_ranking_metrics.py`'s module docstring: `eval/`
    naturally consumes `scoring/`'s output (mirroring how it already consumes
    `models/`'s), so an import the other way would invert that boundary. Verified
    here so the choice cannot silently regress."""
    py_files = sorted(SCORING_DIR.rglob("*.py"))
    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if module_name == "poi_rank.eval" or module_name.startswith("poi_rank.eval."):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports '{module_name}'")
    msg = "scoring/ imported from eval/ (design-boundary violation):\n" + "\n".join(violations)
    assert not violations, msg
