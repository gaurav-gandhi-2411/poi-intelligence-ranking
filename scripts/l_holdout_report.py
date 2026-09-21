"""L4: the holdout, read ONCE, after the L1/L2/L3 selections were frozen.

Reports, for the shipped scoring configuration (gamma 0, lambda 0.8) and the selected one (from
`configs/scoring.yaml`): (a) K1's gradient-reproduction statistic, for the raw ranker score and for the
served utility, split by cold-start / with-history trips; (b) the served-list flip-overlap on the real
holdout trips, split the same way; (c) served NDCG@10, long-tail share/precision and category entropy.
Nothing here selects anything. Writes `results/parts/l_holdout_report.json`.

    uv run python scripts/l_holdout_report.py
"""

from __future__ import annotations

import torch  # noqa: F401, I001  (must import before pandas on this machine)

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.eval import l_select as ls
from poi_rank.eval import longtail as lt
from poi_rank.eval.decision_register import per_trip_ndcg10
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.ranking_data import load_holdout_evaluation_frame
from poi_rank.scoring import calibration as cal
from poi_rank.scoring import compatibility as compat
from poi_rank.scoring.config import ScoringConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
ART = ROOT / "artifacts"
CONF = ROOT / "configs"
OUT = ROOT / "results" / "parts" / "l_holdout_report.json"


def main() -> None:
    feature_cfg = FeatureBuildConfig.from_yaml(CONF / "features.yaml")
    cand_cfg = CandidatesConfig.from_yaml(CONF / "features.yaml")
    scoring_cfg = ScoringConfig.from_yaml(CONF / "scoring.yaml")
    budget = feature_cfg.traveler_features.budget_target_price_level
    u = scoring_cfg.utility
    selected = {"gamma": u.gamma, "lambda": scoring_cfg.diversity.lambda_default}
    shipped = {"gamma": ls.SHIPPED_GAMMA, "lambda": ls.SHIPPED_LAMBDA}

    frame = load_holdout_evaluation_frame(DATA, budget)
    booster = lm.load_boosters(ART)["lambdamart_ips"]
    numeric, categorical = numeric_feature_columns(frame), categorical_feature_columns(frame)
    with (ART / "calibrator.pkl").open("rb") as f:
        calibrator = pickle.load(f)  # noqa: S301 (our own artifact)
    pois_df = pd.read_parquet(DATA / "pois_prepared.parquet")
    trips_df = pd.read_parquet(DATA / "trips.parquet")
    travelers_df = pd.read_parquet(DATA / "travelers.parquet")
    candidates_df = pd.read_parquet(DATA / "candidates.parquet")
    compat_frame = compat.compute_compatibility_frame(
        candidates_df,
        set(frame["trip_id"].unique()),
        trips_df,
        travelers_df,
        pois_df,
        budget,
        cand_cfg.geo,
        scoring_cfg.compatibility,
    )

    def variant(fr: pd.DataFrame) -> pd.DataFrame:
        work = fr.copy()
        raw = lm.score_booster(booster, work, numeric, categorical)
        work["raw_score"] = raw
        work["relevance"] = cal.apply_calibrator(calibrator, raw)
        full = work.merge(compat_frame, on=["trip_id", "poi_id"], how="inner")
        return full.sort_values(["trip_id", "poi_id"]).reset_index(drop=True)

    base = variant(frame)
    flipped = variant(ls.flip_touristiness(frame, DATA, budget))

    trip_cold = {
        str(t): bool(g["implicit_interaction_count"].iloc[0] == 0)
        for t, g in base.groupby("trip_id", sort=True)
    }
    pop_pct = dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True))
    category = dict(zip(pois_df["poi_id"], pois_df["category"], strict=True))
    labels = {
        (str(t), str(p)): int(v)
        for t, p, v in zip(base["trip_id"], base["poi_id"], base["label"], strict=True)
    }
    k = scoring_cfg.output.top_k
    div_cfg = scoring_cfg.diversity
    args_common = (u.alpha, u.beta, u.pref_align_center, u.pref_align_scale)

    def serve(cfg: dict[str, float]) -> dict[str, Any]:
        util_b = ls.utilities(base, cfg["gamma"], *args_common)
        util_f = ls.utilities(flipped, cfg["gamma"], *args_common)
        surv_b, pools_b = ls.served_pools(base, util_b, div_cfg)
        surv_f, pools_f = ls.served_pools(flipped, util_f, div_cfg)
        lists_b = ls.lists_at_lambda(surv_b, pools_b, div_cfg, cfg["lambda"], k)
        lists_f = ls.lists_at_lambda(surv_f, pools_f, div_cfg, cfg["lambda"], k)
        overlap = {t: ls.jaccard(lists_b[t], lists_f[t]) for t in lists_b}
        cold = [v for t, v in overlap.items() if trip_cold[t]]
        warm = [v for t, v in overlap.items() if not trip_cold[t]]
        score = pd.Series(ls.list_score_column(base, lists_b), index=base.index)
        ndcg = per_trip_ndcg10(base, score)
        rep = lt.longtail_share_and_precision(lists_b, pop_pct, cand_cfg.longtail.pop_pct_cutoff, labels)
        surv_mask = (base["hard_gate"] == 1.0).to_numpy()
        grad = ls.gradient_reproduction(base.loc[surv_mask], util_b[surv_mask])
        return {
            "config": cfg,
            "served_ndcg10_mean": float(ndcg.mean()),
            "n_trips_ndcg": int(len(ndcg)),
            "longtail_share": rep.share,
            "longtail_precision": rep.precision,
            "category_entropy_bits": ls.category_entropy(lists_b, category),
            "flip_overlap": float(np.mean(list(overlap.values()))),
            "flip_overlap_cold_start": float(np.mean(cold)),
            "flip_overlap_with_history": float(np.mean(warm)),
            "n_cold_start_trips": len(cold),
            "gradient_reproduction_of_utility": grad,
        }

    surv_mask = (base["hard_gate"] == 1.0).to_numpy()
    raw_grad = ls.gradient_reproduction(
        base.loc[surv_mask], base.loc[surv_mask, "raw_score"].to_numpy(dtype=np.float64)
    )
    result = {
        "method": "scripts/l_holdout_report.py; holdout read once after L1/L2/L3 were frozen",
        "n_holdout_trips": int(base["trip_id"].nunique()),
        "raw_ranker_gradient_reproduction": raw_grad,
        "shipped_config": serve(shipped),
        "selected_config": serve(selected),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
