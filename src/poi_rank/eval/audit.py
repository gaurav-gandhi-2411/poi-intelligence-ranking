"""`make audit`: deterministic invariant checks, emitted as JSON. Replaces the verifier subagent.

A subagent's PASS is a claim; this script's PASS is a computed artifact. Each check returns
`{"passed": bool, "evidence": <number/path/stdout excerpt>}` and is FAIL-CLOSED: a check that
cannot run (missing file, subprocess error, unparsable output) is a failure, never a skip.

Checks (`results/audit.json`; exit 1 if any fail):
  * firewalls           -- the import-firewall + oracle-isolation test modules pass
  * single_metrics_writer -- only eval/compose.py writes results/metrics.json (static scan) and
                           composing twice gives identical bytes (dynamic)
  * hard_constraint_violations -- 0 in the composed metrics
  * gates               -- Gate-A (dgp_gate) and Gate-B (gate_representation) parts both pass
  * docs_numbers        -- no hand-typed numbers in docs/ (tests/test_docs_numbers.py)
  * thread_determinism  -- LightGBM fits are identical at 1 vs 8 threads (retriever + ranker tests)
  * data_determinism (--deep) -- `generate` twice into temp dirs -> identical SHA-256 manifests
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from poi_rank.eval.compose import METRICS_FILENAME, compose_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src" / "poi_rank"
COMPOSE_MODULE = SRC_DIR / "eval" / "compose.py"

# Import-firewall tests + the isolation test, found by glob: the isolation test is a raw text
# scan that flags the oracle-directory name anywhere outside its two allow-listed modules --
# including a literal spelling of its own file name here.
FIREWALL_TESTS = sorted(
    str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    for pattern in ("test_firewall*.py", "test_o*_isolation.py")
    for p in (REPO_ROOT / "tests").glob(pattern)
)
DOCS_TESTS = ["tests/test_docs_numbers.py"]
THREAD_TESTS = [
    "tests/test_retriever.py::test_identical_scores_at_1_and_8_threads",
    "tests/test_lambdamart.py::test_lightgbm_fit_identical_at_1_and_8_threads",
]


def _run_pytest(targets: list[str]) -> dict[str, Any]:
    """Run pytest on `targets`; evidence = the summary line. Non-zero exit == failure."""
    existing = [t for t in targets if (REPO_ROOT / t.split("::")[0]).exists()]
    missing = sorted(set(targets) - set(existing))
    if missing:
        return {"passed": False, "evidence": f"test target(s) missing: {missing}"}
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "--no-header", *existing],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary = lines[-1] if lines else "no output"
    passed = proc.returncode == 0 and re.search(r"\d+ passed", summary) is not None
    return {"passed": bool(passed), "evidence": summary, "returncode": proc.returncode}


def check_firewalls() -> dict[str, Any]:
    return _run_pytest(FIREWALL_TESTS)


def check_docs_numbers() -> dict[str, Any]:
    return _run_pytest(DOCS_TESTS)


def check_thread_determinism() -> dict[str, Any]:
    return _run_pytest(THREAD_TESTS)


_WRITE_PAT = re.compile(r"(write_text|write_bytes|json\.dump\b|to_json|open\([^)]*['\"]w)")


def check_single_metrics_writer(results_dir: Path) -> dict[str, Any]:
    """Static: no module other than compose.py both names metrics.json (or METRICS_FILENAME)
    and writes. Dynamic: composing twice yields identical bytes."""
    offenders: list[str] = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        if path == COMPOSE_MODULE:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if ("metrics.json" in line or "METRICS_FILENAME" in line) and _WRITE_PAT.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}")
        # `x = results_dir / METRICS...; x.write_text(...)` across lines: flag any file that
        # both builds a metrics path and calls write_text on a variable named metrics*.
        if re.search(r"metrics_path\s*=", text) and re.search(r"metrics_path\.write_", text):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: metrics_path.write_*")
    try:
        first = compose_metrics(results_dir).read_bytes()
        second = compose_metrics(results_dir).read_bytes()
    except FileNotFoundError as exc:
        return {"passed": False, "evidence": f"cannot compose: {exc}"}
    identical = first == second
    return {
        "passed": not offenders and identical,
        "evidence": {
            "other_writers": offenders,
            "compose_twice_identical": identical,
            "metrics_sha256": hashlib.sha256(first).hexdigest(),
        },
    }


def check_hard_constraints(results_dir: Path) -> dict[str, Any]:
    path = results_dir / METRICS_FILENAME
    if not path.exists():
        return {"passed": False, "evidence": f"{path} missing"}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    try:
        n = int(metrics["constraint_compatibility"]["n_hard_constraint_violations"])
        n_rec = int(metrics["constraint_compatibility"]["n_recommended"])
    except KeyError as exc:
        return {"passed": False, "evidence": f"metrics key missing: {exc}"}
    return {
        "passed": n == 0 and n_rec > 0,
        "evidence": {"n_hard_constraint_violations": n, "n_recommended": n_rec},
    }


def check_gates(results_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok = True
    for stem in ("dgp_gate", "gate_representation"):
        path = results_dir / "parts" / f"{stem}.json"
        if not path.exists():
            out[stem] = "missing"
            ok = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[stem] = {
            "overall_pass": payload.get("overall_pass"),
            "n_passed": payload.get("n_passed"),
        }
        ok = ok and payload.get("overall_pass") is True
    return {"passed": ok, "evidence": out}


def _manifest_sha(data_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data_dir.rglob("*.parquet"))
    }


def check_data_determinism() -> dict[str, Any]:
    """`generate` twice into fresh temp dirs; every parquet (incl. the oracle export) must match."""
    manifests: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for run in ("a", "b"):
            out = Path(tmp) / run
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "poi_rank.cli", "generate", "--output-dir", str(out)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=False,
            )
            if proc.returncode != 0:
                return {"passed": False, "evidence": f"generate failed: {proc.stderr[-300:]}"}
            manifests.append(_manifest_sha(out))
    same = manifests[0] == manifests[1] and len(manifests[0]) > 0
    return {"passed": same, "evidence": {"n_files": len(manifests[0]), "identical": same}}


def run_audit(results_dir: Path, deep: bool = False) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def guarded(name: str, fn: Any, *args: Any) -> None:
        try:
            checks[name] = fn(*args)
        except Exception as exc:  # noqa: BLE001 -- fail-closed: an unrunnable check is a FAIL
            checks[name] = {
                "passed": False,
                "evidence": f"check raised {type(exc).__name__}: {exc}",
            }

    guarded("firewalls", check_firewalls)
    guarded("single_metrics_writer", check_single_metrics_writer, results_dir)
    guarded("hard_constraint_violations", check_hard_constraints, results_dir)
    guarded("gates", check_gates, results_dir)
    guarded("docs_numbers", check_docs_numbers)
    guarded("thread_determinism", check_thread_determinism)
    if deep:
        guarded("data_determinism", check_data_determinism)

    payload = {
        "all_passed": all(c["passed"] for c in checks.values()),
        "deep": deep,
        "checks": checks,
    }
    (results_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
