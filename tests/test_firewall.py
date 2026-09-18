"""DGP import firewall (spec.md section 1.1): `datagen/**` must never import from
`features/`, `models/`, `candidates/`, or `scoring/` (or their submodules).

Parses every `.py` file under `src/poi_rank/datagen/` with `ast` (not string/regex
matching) and inspects `import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = (
    "poi_rank.features",
    "poi_rank.models",
    "poi_rank.candidates",
    "poi_rank.scoring",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATAGEN_DIR = REPO_ROOT / "src" / "poi_rank" / "datagen"


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


def test_datagen_dir_exists() -> None:
    assert DATAGEN_DIR.is_dir(), f"expected {DATAGEN_DIR} to exist"


def test_datagen_never_imports_downstream_phases() -> None:
    py_files = sorted(DATAGEN_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under datagen/"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if _is_forbidden(module_name):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden module '{module_name}'"
                )

    assert not violations, "DGP firewall violated:\n" + "\n".join(violations)
