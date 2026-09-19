"""Run `make reproduce` / `make reproduce-full` stage by stage (make itself is not installed on
the dev box) and record per-stage wall-clock to results/parts/timings.json.

    uv run python scripts/time_reproduce.py            # reproduce
    uv run python scripts/time_reproduce.py --full     # reproduce-full
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = [
    "generate",
    "prepare",
    "features",
    "candidates",
    "gate-dgp",
    "train",
    "evaluate",
    "representation",
    "gate-representation",
    "compose",
]
FULL_EXTRA = ["recommend", "scenarios", "lodo"]

full = "--full" in sys.argv
stages = REPRODUCE + (FULL_EXTRA if full else [])
env = {**os.environ, "PYTHONHASHSEED": "0"}
rows: list[dict[str, object]] = []
t_all = time.perf_counter()
for stage in stages:
    t0 = time.perf_counter()
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "poi_rank.cli", stage],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    secs = time.perf_counter() - t0
    rows.append({"stage": stage, "seconds": round(secs, 1), "returncode": proc.returncode})
    print(f"{stage:22s} {secs:7.1f}s rc={proc.returncode}", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-1500:], proc.stderr[-1500:])
        break
if full:
    t0 = time.perf_counter()
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "poi_rank.eval.report"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    rows.append({"stage": "docs", "seconds": round(time.perf_counter() - t0, 1), "returncode": proc.returncode})
total = round(time.perf_counter() - t_all, 1)
print(f"TOTAL {total}s")
parts = ROOT / "results" / "parts"
out = {"target": "reproduce-full" if full else "reproduce", "total_seconds": total, "stages": rows}
(parts / ("timings_full.json" if full else "timings.json")).write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8"
)
if full:
    # `reproduce` is the first len(REPRODUCE) stages of the same run; record it too so both
    # targets are timed from one execution.
    base = rows[: len(REPRODUCE)]
    reproduce_out = {
        "target": "reproduce",
        "total_seconds": round(sum(float(r["seconds"]) for r in base), 1),  # type: ignore[arg-type]
        "stages": base,
    }
    (parts / "timings.json").write_text(json.dumps(reproduce_out, indent=2) + "\n", encoding="utf-8")
# Fold the fresh timings into metrics.json and the docs.
for cmd in (["-m", "poi_rank.cli", "compose"], ["-m", "poi_rank.eval.report"]):
    subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=False)  # noqa: S603
