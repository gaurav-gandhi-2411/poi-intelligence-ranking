"""Eval-package import firewall (spec.md section 11/14, this phase's own explicit
instruction): `eval/` is the TERMINAL consumer of every upstream phase (`datagen`,
`data`, `features`, `candidates`, `models`, `scoring`, `explain`) -- nothing OUTSIDE
`eval/` may import from it. This is the mirror-image firewall to
`tests/test_firewall_scoring.py::test_scoring_never_imports_eval` (which already
covers the scoring-specific half of this) -- this test scans EVERY OTHER phase
package under `src/poi_rank/` at once (excluding `cli.py`, the composition root,
which legitimately imports every phase to wire the CLI), so a future new module in
any upstream package can never quietly reverse the dependency direction without a
test catching it.

Parses every `.py` file with `ast` (not string/regex matching) and inspects
`import` / `from ... import` statements directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "poi_rank"
UPSTREAM_PACKAGE_DIRS: tuple[str, ...] = (
    "datagen",
    "data",
    "features",
    "candidates",
    "models",
    "scoring",
    "explain",
)


def _imported_module_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_eval_dir_exists() -> None:
    assert (SRC_DIR / "eval").is_dir()


def test_no_upstream_package_imports_eval() -> None:
    violations: list[str] = []
    for pkg in UPSTREAM_PACKAGE_DIRS:
        pkg_dir = SRC_DIR / pkg
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module_name in _imported_module_names(tree):
                if module_name == "poi_rank.eval" or module_name.startswith("poi_rank.eval."):
                    violations.append(f"{path.relative_to(REPO_ROOT)} imports '{module_name}'")

    assert not violations, "eval/ firewall violated (upstream imports eval/):\n" + "\n".join(
        violations
    )


def test_eval_may_import_upstream_packages() -> None:
    """Sanity check that the firewall test itself isn't accidentally over-broad:
    `eval/` genuinely does (and is allowed to) import from every upstream package --
    it is the terminal consumer, by design (module docstring)."""
    eval_dir = SRC_DIR / "eval"
    imported: set[str] = set()
    for path in sorted(eval_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported.update(_imported_module_names(tree))
    assert any(m.startswith("poi_rank.models") for m in imported)
    assert any(m.startswith("poi_rank.scoring") for m in imported)
    assert any(m.startswith("poi_rank.candidates") for m in imported)
    assert any(m.startswith("poi_rank.features") for m in imported)
