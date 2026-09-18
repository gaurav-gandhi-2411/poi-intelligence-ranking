"""Feature-store import firewall (spec.md section 5/14): `src/poi_rank/features/**`
must never import from `candidates/`, `models/`, or `scoring/` (or their submodules)
-- features is upstream of those phases, same rule shape as
`tests/test_firewall.py`/`tests/test_firewall_data.py`. features/ must also never
reference `_oracle` at all (unlike `datagen/**`, it has no writer exemption).

Parses every `.py` file under `src/poi_rank/features/` with `ast` (not string/regex
matching) and inspects `import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = (
    "poi_rank.models",
    "poi_rank.candidates",
    "poi_rank.scoring",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "src" / "poi_rank" / "features"


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


def test_features_dir_exists() -> None:
    assert FEATURES_DIR.is_dir(), f"expected {FEATURES_DIR} to exist"


def test_features_never_imports_downstream_phases() -> None:
    py_files = sorted(FEATURES_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under features/"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if _is_forbidden(module_name):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden module '{module_name}'"
                )

    assert not violations, "features/ firewall violated:\n" + "\n".join(violations)


def test_features_never_references_oracle_dir() -> None:
    """`features/**` must never reference `_oracle` at all -- only `eval/oracle.py`
    (reader) and `datagen/oracle_export.py` (writer) may, per
    tests/test_oracle_isolation.py's allowlist. Unlike `datagen/**`, features/ has no
    writer role and so has zero legitimate reasons to spell out that name."""
    py_files = sorted(FEATURES_DIR.rglob("*.py"))
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in py_files
        if "_oracle" in path.read_text(encoding="utf-8")
    ]
    assert not violations, "features/ referenced '_oracle' (firewall violation):\n" + "\n".join(
        violations
    )
