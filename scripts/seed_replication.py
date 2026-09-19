"""5-seed replication of the full pipeline (generate -> ... -> evaluate) into isolated sandboxes.

Seed 42 is the committed run (results/parts/evaluate.json); seeds 43-46 are regenerated end to
end (a new synthetic dataset, new retriever, new boosters) with the top-level `seed` of
datagen/features/model/eval/scoring configs replaced. Everything runs in
`<scratch>/seed_<n>/`; the repository is never written to except the final summary
`results/parts/seed_replication.json`. Reports mean, sd and range of the headline metrics so the
single-seed CIs in RESULTS.md are accompanied by a seed-variance table.

    uv run python scripts/seed_replication.py [--scratch DIR] [--seeds 43 44 45 46]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
STAGES = ["generate", "prepare", "features", "candidates", "train", "evaluate"]


def _with_seed(name: str, seed: int, dest: Path) -> Path:
    text = (CONFIGS / name).read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^seed:\s*\d+", f"seed: {seed}", text, count=1)
    assert n == 1, f"no top-level seed in {name}"
    out = dest / name
    out.write_text(text, encoding="utf-8")
    return out


def run_seed(seed: int, scratch: Path) -> dict[str, Any]:
    base = scratch / f"seed_{seed}"
    cfg_dir, data, art, res = base / "configs", base / "data", base / "artifacts", base / "results"
    for d in (cfg_dir, data, art, res):
        d.mkdir(parents=True, exist_ok=True)
    cfg = {n: _with_seed(n, seed, cfg_dir) for n in
           ("datagen.yaml", "features.yaml", "model.yaml", "eval.yaml", "scoring.yaml")}
    cmds = {
        "generate": ["generate", "--config-path", cfg["datagen.yaml"], "--output-dir", data],
        "prepare": ["prepare", "--config-path", cfg["features.yaml"], "--data-dir", data],
        "features": ["features", "--config-path", cfg["features.yaml"], "--data-dir", data,
                     "--artifacts-dir", art],
        "candidates": ["candidates", "--config-path", cfg["features.yaml"], "--data-dir", data,
                       "--artifacts-dir", art],
        "train": ["train", "--features-config-path", cfg["features.yaml"],
                  "--model-config-path", cfg["model.yaml"], "--scoring-config-path",
                  cfg["scoring.yaml"], "--data-dir", data, "--artifacts-dir", art,
                  "--results-dir", res],
        "evaluate": ["evaluate", "--features-config-path", cfg["features.yaml"],
                     "--model-config-path", cfg["model.yaml"], "--eval-config-path",
                     cfg["eval.yaml"], "--datagen-config-path", cfg["datagen.yaml"],
                     "--scoring-config-path", cfg["scoring.yaml"], "--data-dir", data,
                     "--artifacts-dir", art, "--results-dir", res],
    }
    env = {"PYTHONHASHSEED": "0", **__import__("os").environ}
    t0 = time.perf_counter()
    for stage in STAGES:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "poi_rank.cli", *map(str, cmds[stage])],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False,
        )
        print(f"seed {seed} {stage:11s} rc={proc.returncode} {time.perf_counter() - t0:6.0f}s", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"seed {seed} stage {stage} failed:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    return json.loads((res / "parts" / "evaluate.json").read_text(encoding="utf-8"))


def headline(m: dict[str, Any]) -> dict[str, float]:
    s = m["systems"]
    return {
        "ndcg10_lambdamart_ips": s["lambdamart_ips"]["metrics"]["ndcg@10"]["mean"],
        "ndcg10_popularity": s["popularity"]["metrics"]["ndcg@10"]["mean"],
        "ndcg10_content_cosine": s["content_cosine"]["metrics"]["ndcg@10"]["mean"],
        "ndcg10_logistic_regression": s["logistic_regression"]["metrics"]["ndcg@10"]["mean"],
        "pct_of_oracle_ceiling": s["lambdamart_ips"]["pct_of_ceiling_ndcg10"],
        "candidate_recall_overall": m["candidate_recall"]["overall"]["recall_mean"],
        "candidate_recall_long_tail": m["candidate_recall"]["long_tail"]["recall_mean"],
        "longtail_share": m["longtail"]["share"],
        "longtail_precision": m["longtail"]["precision"],
        "coverage_at_10": m["coverage"]["primary_system"]["catalog_coverage_at_10"],
        "bias_gap_popularity": m["bias_gap"]["popularity"]["gap"],
        "bias_gap_primary": m["bias_gap"]["lambdamart_ips"]["gap"],
        "ece_after": m["calibration"]["ece_after"],
        "within_cross_ratio_true_labels": m["personalization"]["archetype"]["within_cross_ratio"] or float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=Path("C:/Users/gaura/seed_replication_scratch"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46])
    args = ap.parse_args()
    committed = json.loads((ROOT / "results" / "parts" / "evaluate.json").read_text(encoding="utf-8"))
    runs = {42: headline(committed)}
    for seed in args.seeds:
        runs[seed] = headline(run_seed(seed, args.scratch))
    names = list(runs[42])
    summary = {
        n: {
            "per_seed": {str(k): v[n] for k, v in runs.items()},
            "mean": mean(v[n] for v in runs.values()),
            "sd": pstdev([v[n] for v in runs.values()]),
            "min": min(v[n] for v in runs.values()),
            "max": max(v[n] for v in runs.values()),
        }
        for n in names
    }
    out = {
        "seeds": sorted(runs),
        "note": "Seed 42 is the committed run; the others regenerate the dataset and refit "
        "everything with the top-level seed replaced (retriever, boosters, calibration).",
        "metrics": summary,
    }
    (ROOT / "results" / "parts" / "seed_replication.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
