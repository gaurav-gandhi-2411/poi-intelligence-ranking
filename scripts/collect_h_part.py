"""Aggregate experiment H's evidence (`results/experiments/h/*.json`) plus the measured
train/holdout feature skew (pre-fix vs current `traveler_features.parquet`) into
`results/parts/h_experiments.json`, which `compose` merges into `results/metrics.json` so the docs
can cite every number by path.

    uv run python scripts/collect_h_part.py

The pre-fix skew statistics are recomputed from the traveler-feature table as committed at commit
`ec8ac4a` (the submission tag before the fix) via `git show`, not typed in.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "results" / "experiments" / "h"
PRE_FIX_COMMIT = "ec8ac4a"


def _skew(tf: pd.DataFrame, trips: pd.DataFrame) -> dict[str, dict[str, float]]:
    d = tf.merge(trips[["trip_id", "is_holdout"]], on="trip_id")
    out: dict[str, dict[str, float]] = {}
    for name, g in (("train_trips", d[~d["is_holdout"]]), ("holdout_trips", d[d["is_holdout"]])):
        out[name] = {
            "n_trips": float(len(g)),
            "days_since_last_interaction_median": float(
                g["implicit_days_since_last_interaction"].median()
            ),
            "interaction_count_mean": float(g["implicit_interaction_count"].mean()),
        }
    return out


def _pre_fix_headline() -> dict[str, float]:
    """Headline numbers of the submission as first tagged (commit `PRE_FIX_COMMIT`)."""
    blob = subprocess.run(  # noqa: S603
        ["git", "show", f"{PRE_FIX_COMMIT}:results/metrics.json"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    m = json.loads(blob)
    sysm = m["systems"]
    seeds = m["seed_replication"]["metrics"]
    return {
        "ndcg10_primary": sysm["lambdamart_ips"]["metrics"]["ndcg@10"]["mean"],
        "ndcg10_content_cosine": sysm["content_cosine"]["metrics"]["ndcg@10"]["mean"],
        "ndcg10_popularity": sysm["popularity"]["metrics"]["ndcg@10"]["mean"],
        "wilcoxon_primary_vs_content_cosine_p": m["wilcoxon"]["lambdamart_ips_vs_content_cosine"][
            "p_value"
        ],
        "pct_of_oracle_ceiling": sysm["lambdamart_ips"]["pct_of_ceiling_ndcg10"],
        "confidence_decile_spearman": m["confidence_decile_validation"]["spearman_rho"],
        "longtail_share": m["longtail"]["share"],
        "longtail_precision": m["longtail"]["precision"],
        "within_cross_ratio": m["personalization"]["archetype"]["within_cross_ratio"],
        "five_seed_mean_primary": seeds["ndcg10_lambdamart_ips"]["mean"],
        "five_seed_mean_content_cosine": seeds["ndcg10_content_cosine"]["mean"],
        "scenario4_overlap": json.loads(
            subprocess.run(  # noqa: S603
                ["git", "show", f"{PRE_FIX_COMMIT}:results/scenarios/overlap_matrix.json"],  # noqa: S607
                cwd=ROOT,
                capture_output=True,
                check=True,
            ).stdout
        )["diagnostic_vs_base_jaccard"],
    }


def _load(name: str) -> dict:
    return json.loads((H / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    trips = pd.read_parquet(ROOT / "data" / "synthetic" / "trips.parquet")
    blob = subprocess.run(  # noqa: S603
        ["git", "show", f"{PRE_FIX_COMMIT}:data/synthetic/traveler_features.parquet"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    pre = pd.read_parquet(io.BytesIO(blob))
    post = pd.read_parquet(ROOT / "data" / "synthetic" / "traveler_features.parquet")
    out = {
        "pre_fix_commit": PRE_FIX_COMMIT,
        "pre_fix_headline": _pre_fix_headline(),
        "feature_skew_pre_fix": _skew(pre, trips),
        "feature_skew_post_fix": _skew(post, trips),
        "leakfix_sandbox_seed42_k195": _load("leakfix_sandbox_seed42_K195"),
        "h0a_shap_shares_final_model": _load("h0a_shap_groups")["share"],
        "h0a_shap_shares_pre_fix_model": _load("h0a_shap_groups_leaky_features")["share"],
        "h0b_pre_fix_frames": _load("h0b_probe_leaky_features"),
        "h0b_corrected_frames": _load("h0b_probe"),
        "baseline_corrected": _load("baseline_corrected"),
        "h1": _load("h1_cross_features"),
        "h2": _load("h2_init_score"),
        "h3": _load("h3_group_tuning"),
        "h3b": _load("h3b_combinations"),
        "decision": _load("decision"),
        "holdout_ablation_no_h_seed42": _load("holdout_ablation_no_h_seed42"),
    }
    path = ROOT / "results" / "parts" / "h_experiments.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
