"""L1+L2 selection: gamma (preference alignment) x MMR lambda on the train-carved VALIDATION split.

Pre-registered rule: `docs/experiments/L-final.md`. Re-scoring only (the shipped booster is not refit);
the holdout is not touched here. Writes `results/parts/l12_selection.json`.

    uv run python scripts/l12_select.py
"""

from __future__ import annotations

import torch  # noqa: F401, I001  (must import before pandas on this machine)

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.eval import l_select as ls
from poi_rank.eval.config import EvalConfig
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_train_ranking_frame
from poi_rank.scoring import calibration as cal
from poi_rank.scoring import compatibility as compat
from poi_rank.scoring.config import ScoringConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
ART = ROOT / "artifacts"
CONF = ROOT / "configs"
OUT = ROOT / "results" / "parts" / "l12_selection.json"


def main() -> None:
    t0 = time.time()
    feature_cfg = FeatureBuildConfig.from_yaml(CONF / "features.yaml")
    cand_cfg = CandidatesConfig.from_yaml(CONF / "features.yaml")
    model_cfg = ModelConfig.from_yaml(CONF / "model.yaml")
    scoring_cfg = ScoringConfig.from_yaml(CONF / "scoring.yaml")
    eval_cfg = EvalConfig.from_yaml(CONF / "eval.yaml")
    budget = feature_cfg.traveler_features.budget_target_price_level

    train_frame = load_train_ranking_frame(DATA, budget)
    consts = ls.pref_align_constants(train_frame)
    print("pref_align constants", consts, flush=True)
    _fit, val = lm.train_val_split_by_trip(
        train_frame, model_cfg.lambdamart.val_fraction, model_cfg.lambdamart.val_split_seed
    )
    booster = lm.load_boosters(ART)["lambdamart_ips"]
    numeric, categorical = numeric_feature_columns(val), categorical_feature_columns(val)
    with (ART / "calibrator.pkl").open("rb") as f:
        calibrator = pickle.load(f)  # noqa: S301 (our own artifact)

    pois_df = pd.read_parquet(DATA / "pois_prepared.parquet")
    trips_df = pd.read_parquet(DATA / "trips.parquet")
    travelers_df = pd.read_parquet(DATA / "travelers.parquet")
    candidates_df = pd.read_parquet(DATA / "candidates.parquet")
    interactions_train = pd.read_parquet(DATA / "interactions_train.parquet")
    compat_frame = compat.compute_compatibility_frame(
        candidates_df,
        set(val["trip_id"].unique()),
        trips_df,
        travelers_df,
        pois_df,
        budget,
        cand_cfg.geo,
        scoring_cfg.compatibility,
    )

    def variant(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        raw = lm.score_booster(booster, work, numeric, categorical)
        work["relevance"] = cal.apply_calibrator(calibrator, raw)
        full = work.merge(compat_frame, on=["trip_id", "poi_id"], how="inner")
        return full.sort_values(["trip_id", "poi_id"]).reset_index(drop=True)

    base = variant(val)
    flipped = variant(ls.flip_touristiness(val, DATA, budget))
    assert (
        base[["trip_id", "poi_id"]].to_numpy() == flipped[["trip_id", "poi_id"]].to_numpy()
    ).all()
    print(f"validation trips {base['trip_id'].nunique()} rows {len(base)}", flush=True)

    trip_cold = {
        str(t): bool(g["implicit_interaction_count"].iloc[0] == 0)
        for t, g in base.groupby("trip_id", sort=True)
    }
    ctx: dict[str, Any] = {
        "k": scoring_cfg.output.top_k,
        "div_cfg": scoring_cfg.diversity,
        "alpha": scoring_cfg.utility.alpha,
        "beta": scoring_cfg.utility.beta,
        "center": consts["center"],
        "scale": consts["scale"],
        "trip_cold": trip_cold,
        "p_expose": lm.train_frame_p_expose(base, interactions_train, pois_df),
        "pop_pct_by_poi": dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True)),
        # the SCORECARD long-tail definition (eval.yaml, bottom 50%), not the candidate
        # channel's 0.40 floor
        "cutoff": eval_cfg.long_tail_pop_pct_cutoff,
        "label_by_trip_poi": {
            (str(t), str(p)): int(v)
            for t, p, v in zip(base["trip_id"], base["poi_id"], base["label"], strict=True)
        },
        "category_by_poi": dict(zip(pois_df["poi_id"], pois_df["category"], strict=True)),
    }

    rows: list[dict[str, Any]] = []
    for gamma in ls.GAMMAS:
        rows.extend(ls.evaluate_config(base, flipped, gamma, ls.LAMBDAS, ctx))
        r = rows[-len(ls.LAMBDAS) :][ls.LAMBDAS.index(0.8)]
        print(
            f"gamma={gamma:<5} lam=0.8 ndcg={r['v_ndcg10_ips']:.4f} flip={r['flip_overlap']:.3f} "
            f"cold={r['flip_overlap_cold_start']:.3f} warm={r['flip_overlap_with_history']:.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    selection = ls.select_config(rows)
    result = {
        "method": "scripts/l12_select.py; rule pre-registered in docs/experiments/L-final.md",
        "constants": consts,
        "n_validation_trips": int(base["trip_id"].nunique()),
        "ndcg_tolerance": ls.NDCG_TOLERANCE,
        "entropy_floor_ratio": ls.ENTROPY_FLOOR_RATIO,
        "grid": rows,
        "selection": selection,
    }
    OUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
