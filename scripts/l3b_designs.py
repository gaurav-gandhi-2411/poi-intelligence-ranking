"""L3b: candidate-set design (A learned K=240, B legacy six-channel, C = A union B) on VALIDATION.

Pre-registered rule: `docs/experiments/L-final.md`. The ranker is retrained per design (system-8 recipe,
seeds 42/7/11/13) and scored on the same train-carved validation trips with a candidate-set-independent
NDCG denominator. The holdout is not touched. Writes `results/parts/l3b_designs.json`.

    uv run python scripts/l3b_designs.py
"""

from __future__ import annotations

import torch  # noqa: F401, I001  (must import before pandas on this machine)

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.eval import l3_designs as l3
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.decision_register import Lab, fit_lgb, score_lgb
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.pair_frame import build_ranking_frame, label_by_trip_poi
from poi_rank.features.reconcile import build_poi_id_canonical_map
from poi_rank.models import lambdamart as lm
from poi_rank.models.config import ModelConfig
from poi_rank.models.cross_ranking import attach_cross_features
from poi_rank.models.ranking_data import _load_common

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
CONF = ROOT / "configs"
OUT = ROOT / "results" / "parts" / "l3b_designs.json"
SEEDS = (42, 7, 11, 13)
CHANNELS = [
    "channel_geo",
    "channel_interest",
    "channel_semantic",
    "channel_cf",
    "channel_longtail",
    "channel_archetype",
    "channel_learned",
]


def union_design(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """A union B per trip, deduplicated on `(trip_id, poi_id)` with channel flags OR-ed."""
    both = pd.concat([a, b], ignore_index=True)
    merged = both.groupby(["trip_id", "poi_id"], as_index=False)[CHANNELS].max()
    merged[CHANNELS] = merged[CHANNELS].astype(bool)
    return merged


def main() -> None:
    t0 = time.time()
    feature_cfg = FeatureBuildConfig.from_yaml(CONF / "features.yaml")
    cand_cfg = CandidatesConfig.from_yaml(CONF / "features.yaml")
    model_cfg = ModelConfig.from_yaml(CONF / "model.yaml")
    eval_cfg = EvalConfig.from_yaml(CONF / "eval.yaml")
    budget = feature_cfg.traveler_features.budget_target_price_level
    d = _load_common(DATA)
    trips, travelers, pois = d["trips_df"], d["travelers_df"], d["pois_df"]
    pf, tf = d["poi_features_df"], d["traveler_features_df"]
    interactions = pd.read_parquet(DATA / "interactions_train.parquet")
    train_ids = set(trips.loc[~trips["is_holdout"], "trip_id"])

    cand_a = pd.read_parquet(DATA / "candidates.parquet")
    cand_b = pd.read_parquet(DATA / "candidates_legacy6.parquet")
    designs = {"A": cand_a, "B": cand_b, "C": union_design(cand_a, cand_b)}

    cmap = build_poi_id_canonical_map(pois)
    exposed = (
        pd.concat(
            [
                label_by_trip_poi(interactions, cmap).rename("label"),
                lm.p_expose_by_trip_poi(interactions, cmap).rename("p_expose"),
            ],
            axis=1,
        )
        .reset_index()
        .dropna(subset=["label"])
    )
    pop_pct = dict(zip(pois["poi_id"], pois["pop_pct"], strict=True))
    dest_pois = {k: set(g["poi_id"]) for k, g in pois.groupby("destination")}
    dest_of_trip = dict(zip(trips["trip_id"], trips["destination"], strict=True))

    results: dict[str, dict[str, Any]] = {}
    val_trip_sets: dict[str, set[str]] = {}
    for name, cand in designs.items():
        frame = attach_cross_features(
            build_ranking_frame(
                cand, train_ids, interactions, trips, travelers, pois, pf, tf, budget
            ),
            DATA,
            budget,
        )
        print(f"design {name}: frame {frame.shape} ({time.time() - t0:.0f}s)", flush=True)
        lab = Lab(
            train_frame=frame,
            holdout_frame=frame.iloc[:0],
            interactions_train=interactions,
            pois_df=pois,
            model_cfg=model_cfg,
            eval_cfg=eval_cfg,
        )
        per_seed: list[float] = []
        val_frame = None
        for seed in SEEDS:
            booster, num, cat, _fit, val = fit_lgb(lab, seed=seed)
            val_frame = val
            val_trips = {str(t) for t in val["trip_id"].unique()}
            ideal = l3.ideal_dcg_by_trip(exposed.loc[exposed["trip_id"].isin(val_trips)])
            p_val = lm.train_frame_p_expose(val, interactions, pois)
            score = score_lgb(booster, num, cat, val).to_numpy(dtype=np.float64)
            per_seed.append(l3.fixed_denominator_ndcg10(val, score, p_val, ideal))
            print(
                f"  seed {seed}: v_ndcg10 {per_seed[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True
            )
        assert val_frame is not None
        val_trips = {str(t) for t in val_frame["trip_id"].unique()}
        val_trip_sets[name] = val_trips
        sets = {
            str(t): set(g["poi_id"])
            for t, g in cand.loc[cand["trip_id"].isin(val_trips)].groupby("trip_id")
        }
        rec = l3.candidate_recall(
            sets,
            exposed,
            val_trips,
            pop_pct,
            cand_cfg.longtail.pop_pct_cutoff,
            dest_pois,
            dest_of_trip,
        )
        results[name] = {
            "v_ndcg10_per_seed": per_seed,
            "v_ndcg10": float(np.mean(per_seed)),
            "v_ndcg10_seed_sd": float(np.std(per_seed, ddof=1)),
            "effective_k": float(np.mean([len(s) for s in sets.values()])),
            "n_validation_trips": len(val_trips),
            **rec,
        }
        del frame, lab, val_frame
        gc.collect()

    assert val_trip_sets["A"] == val_trip_sets["B"] == val_trip_sets["C"]
    decision = l3.select_design(results)
    payload = {
        "method": "scripts/l3b_designs.py; rule pre-registered in docs/experiments/L-final.md",
        "seeds": list(SEEDS),
        "designs": results,
        "decision": decision,
        "gate_amendment": {
            "long_tail_recall_floor": l3.LONG_TAIL_RECALL_FLOOR,
            "effective_k_cap": l3.EFFECTIVE_K_CAP,
            "adoption_bar": l3.ADOPTION_BAR,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"designs": results, "decision": decision}, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
