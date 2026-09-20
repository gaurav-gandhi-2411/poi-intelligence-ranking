"""J2(b): why did the confidence-decile Spearman fall from 0.879 to 0.358 after the skew fix?

Decomposes the confidence score into its five terms (traveler evidence, POI impressions, reviews,
ensemble agreement, calibration stability) and reports, per tree (pre-fix commit ec8ac4a and the
current tree): the decile Spearman (recomputed with the same code as the scorecard), the trip-level
Spearman of each term with per-trip NDCG@10, and how the confidence inputs are distributed on the
holdout. Nothing is re-tuned. Run it in a checkout of each tree:

    PYTHONPATH=<tree>/src python <tree>/scripts/diagnose_confidence.py --out <file.json>
"""

from __future__ import annotations

import torch  # noqa: F401, I001  (must import before pandas on this machine)

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import ModelConfig
from poi_rank.scoring import calibration as cal
from poi_rank.scoring import confidence as conf
from poi_rank.scoring._ranking_metrics import ndcg_at_k
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import confidence_decile_validation, run_scoring_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
ART = ROOT / "artifacts"
CFG = ROOT / "configs"


def _spearman(a: Any, b: Any) -> float | None:
    r = stats.spearmanr(a, b).statistic
    return None if np.isnan(r) else float(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print("poi_rank from", __import__("poi_rank").__file__)
    feature_cfg = FeatureBuildConfig.from_yaml(CFG / "features.yaml")
    model_cfg = ModelConfig.from_yaml(CFG / "model.yaml")
    scoring_cfg = ScoringConfig.from_yaml(CFG / "scoring.yaml")
    cand_cfg = CandidatesConfig.from_yaml(CFG / "features.yaml")
    result = run_scoring_pipeline(
        DATA,
        ART,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        cand_cfg.geo,
        cand_cfg.longtail.pop_pct_cutoff,
    )
    full: pd.DataFrame = result["full_frame"].reset_index(drop=True)
    ccfg = scoring_cfg.confidence
    num, cat = numeric_feature_columns(full), categorical_feature_columns(full)
    ens = conf.ensemble_std_scores(conf.load_ensemble(ccfg.ensemble_seeds, ART), full, num, cat)
    raw = result["raw_score"]
    bin_w = cal.calibration_bin_width(result["calibrator"], raw.to_numpy(dtype=np.float64))
    terms = {
        "traveler_evidence": conf.evidence_shrinkage(
            full["implicit_interaction_count"].to_numpy(dtype=np.float64), ccfg.traveler_evidence_k
        ),
        "poi_impressions": conf.evidence_shrinkage(
            full["behav_impressions"].to_numpy(dtype=np.float64), ccfg.poi_impression_k
        ),
        "reviews": conf.evidence_shrinkage(
            np.log1p(full["review_count"].to_numpy(dtype=np.float64)), ccfg.review_count_k
        ),
        "ensemble_agreement": conf.ensemble_agreement(ens, ccfg.ensemble_std_tau),
        "calibration_stability": conf.calibration_stability(bin_w, ccfg.calibration_bin_width_tau),
    }
    for k, v in terms.items():
        full[f"term_{k}"] = v
    full["ensemble_std"] = ens
    top_k = scoring_cfg.output.top_k

    rows = []
    for trip_id, g in full.groupby("trip_id", sort=True):
        top = g.sort_values(["utility", "poi_id"], ascending=[False, True]).head(top_k)
        ndcg = ndcg_at_k(
            g["label"].to_numpy(dtype=np.int64),
            g["utility"].to_numpy(dtype=np.float64),
            g["poi_id"].to_numpy(dtype=object),
            top_k,
        )
        row: dict[str, Any] = {
            "trip_id": trip_id,
            "confidence": float(top["confidence"].mean()),
            "ndcg": ndcg,
            "interaction_count": float(g["implicit_interaction_count"].iloc[0]),
            "ensemble_std_top": float(top["ensemble_std"].mean()),
        }
        for k in terms:
            row[k] = float(top[f"term_{k}"].mean())
        rows.append(row)
    trips = pd.DataFrame(rows).dropna(subset=["ndcg"])
    decile = confidence_decile_validation(full, top_k)
    out: dict[str, Any] = {
        "decile_spearman_recomputed": decile["spearman_rho"],
        "n_trips": len(trips),
        "trip_level_spearman_confidence_vs_ndcg": _spearman(trips["confidence"], trips["ndcg"]),
        "trip_level_spearman_by_term_vs_ndcg": {
            k: _spearman(trips[k], trips["ndcg"]) for k in terms
        },
        "trip_level_between_trip_sd_by_term": {k: float(trips[k].std()) for k in terms},
        "trip_level_sd_confidence": float(trips["confidence"].std()),
        "mean_ndcg": float(trips["ndcg"].mean()),
        "sd_ndcg": float(trips["ndcg"].std()),
        "holdout_interaction_count": {
            "mean": float(trips["interaction_count"].mean()),
            "median": float(trips["interaction_count"].median()),
            "share_zero": float((trips["interaction_count"] == 0).mean()),
        },
        "ensemble_std_top10": {
            "mean": float(trips["ensemble_std_top"].mean()),
            "sd": float(trips["ensemble_std_top"].std()),
        },
        "spearman_interaction_count_vs_ndcg": _spearman(trips["interaction_count"], trips["ndcg"]),
        "spearman_ensemble_std_vs_ndcg": _spearman(trips["ensemble_std_top"], trips["ndcg"]),
        "confidence_weights": {
            "traveler": ccfg.weight_traveler,
            "poi": ccfg.weight_poi,
            "reviews": ccfg.weight_reviews,
            "ensemble": ccfg.weight_ensemble,
            "calibration": ccfg.weight_calibration,
        },
    }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
