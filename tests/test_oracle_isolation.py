"""Oracle isolation (spec.md section 1.1): `data/synthetic/_oracle/` must be read by
NOTHING except `src/poi_rank/eval/oracle.py`.

**Two documented exemptions, not one:**
  - `eval/oracle.py` — the designated future READER. Does not exist yet in this
    phase; the test must not fail merely because the file is absent (it passes
    vacuously on the reader side until that file lands).
  - `datagen/oracle_export.py` — the designated WRITER. spec.md section 8 requires
    datagen to populate `_oracle/`, so *something* in the codebase must legitimately
    spell out that subdirectory name. Confining that knowledge to exactly one module
    (see `oracle_export.ORACLE_SUBDIR_NAME` / `oracle_dir_from_output`) keeps the
    blast radius of this exemption to one file, mirroring the one-file exemption on
    the reader side. No other file should ever be added to this allowlist without
    updating spec.md's isolation contract.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "poi_rank"
ALLOWED_PATHS = {
    SRC_DIR / "eval" / "oracle.py",
    SRC_DIR / "datagen" / "oracle_export.py",
}


def test_no_module_outside_allowlist_references_oracle_dir() -> None:
    py_files = sorted(SRC_DIR.rglob("*.py"))
    assert py_files, "expected at least one .py file under src/poi_rank/"

    allowed_resolved = {p.resolve() for p in ALLOWED_PATHS}
    violations: list[str] = []
    for path in py_files:
        if path.resolve() in allowed_resolved:
            continue
        text = path.read_text(encoding="utf-8")
        if "_oracle" in text:
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert not violations, (
        "Oracle isolation violated — only eval/oracle.py (reader) and "
        "datagen/oracle_export.py (writer) may reference '_oracle':\n" + "\n".join(violations)
    )
