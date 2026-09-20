"""J2(a): why did the touristiness-flip overlap (scenario 4 vs 1) rise after the skew fix?

Measured, not hypothesised. Writes `results/parts/scenario4_diagnosis.json` (composed into
metrics.json; the docs cite it by path). Nothing is re-tuned; the final booster is only explained.

  A. Grouped/typed SHAP on the scenario-1 and scenario-4 candidate rows (synthetic, cold-start
     travelers): how much attribution the stated-preference and cross-feature groups carry versus
     POI-side groups, and which groups actually MOVE when only touristiness_pref is flipped.
  B. The same flip applied to REAL holdout trips (counterfactual): raw-score top-10 overlap for trips
     with history vs pure cold-start trips.

    uv run python scripts/diagnose_scenario4.py
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
from poi_rank.candidates.retriever import (
    Retriever,
    RetrieverInputs,
    score_full_catalog,
    top_k_by_trip,
)
from poi_rank.eval.scenarios import (
    ALL_SCENARIO_PROFILES,
    SCENARIO_DESTINATION,
    build_scenario_holdout_frame,
    build_scenario_travelers_and_trips,
    compute_scenario_traveler_features,
    generate_scenario_candidates,
    scenario_trip_id,
)
from poi_rank.explain.shap_groups import (
    FEATURE_GROUPS,
    build_group_membership,
    compute_grouped_shap,
)
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.traveler_features import localness_preference_gap
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns

try:  # absent in the pre-fix tree (commit ec8ac4a), which this script also runs against
    from poi_rank.models.cross_ranking import attach_cross_features
except ImportError:
    attach_cross_features = None
from poi_rank.models.ranking_data import load_holdout_evaluation_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
ART = ROOT / "artifacts"
PREF_COLUMNS = (
    "explicit_touristiness_pref",
    "interact_localness_gap",
    "xf_loc_align",
    "xf_loc_gap",
    "xf_loc_x_pref",
    "xf_pop_x_pref",
)


def _type_of(column: str) -> str:
    if column.startswith("xf_"):
        return "cross_features_xf"
    if column.startswith("interact_"):
        return "pair_features_interact"
    if column.startswith("explicit_"):
        return "stated_traveler_explicit"
    if column.startswith("implicit_"):
        return "traveler_history_implicit"
    return "poi_side"  # num_/cat_/geo_/behav_/text_emb_


def _shares(mean_abs: dict[str, float]) -> dict[str, float]:
    total = sum(mean_abs.values())
    return {k: v / total for k, v in mean_abs.items()}


def _mean_abs_by(raw: np.ndarray, cols: list[str], keyfn: Any) -> dict[str, float]:
    out: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        out.setdefault(keyfn(c), []).append(i)
    return {k: float(np.abs(raw[:, idx]).sum(axis=1).mean()) for k, idx in out.items()}


def _top10(scores: pd.Series, ids: pd.Series) -> set[str]:
    order = np.lexsort((ids.to_numpy(dtype=object), -scores.to_numpy()))
    return set(ids.to_numpy(dtype=object)[order][:10])


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b)


def scenario_rows() -> dict[str, Any]:
    c = ROOT / "configs"
    feature_cfg = FeatureBuildConfig.from_yaml(c / "features.yaml")
    candidates_cfg = CandidatesConfig.from_yaml(c / "features.yaml")
    pois = pd.read_parquet(DATA / "pois_prepared.parquet")
    pf = pd.read_parquet(DATA / "poi_features.parquet")
    real_travelers = pd.read_parquet(DATA / "travelers.parquet")
    real_trips = pd.read_parquet(DATA / "trips.parquet")
    interactions_train = pd.read_parquet(DATA / "interactions_train.parquet")
    travelers, trips = build_scenario_travelers_and_trips(pois, real_trips, SCENARIO_DESTINATION)
    feats = compute_scenario_traveler_features(
        real_travelers, real_trips, travelers, trips, interactions_train, pois, feature_cfg, ART
    )
    budget = feature_cfg.traveler_features.budget_target_price_level
    learned = candidates_cfg.learned
    assert learned is not None
    scores = score_full_catalog(
        RetrieverInputs(pois, travelers, trips, pf, feats, interactions_train, budget),
        Retriever.load(ART),
        sorted(trips["trip_id"]),
        learned.num_threads,
    )
    cands = generate_scenario_candidates(
        pois,
        real_travelers,
        travelers,
        trips,
        pf,
        feats,
        interactions_train,
        candidates_cfg,
        top_k_by_trip(scores, learned.quota),
    )
    try:
        frame = build_scenario_holdout_frame(cands, trips, travelers, pois, pf, feats, budget, DATA)
    except TypeError:  # pre-fix tree: no data_dir parameter
        frame = build_scenario_holdout_frame(cands, trips, travelers, pois, pf, feats, budget)
    booster = lm.load_boosters(ART)["lambdamart_ips"]
    num, cat = numeric_feature_columns(frame), categorical_feature_columns(frame)
    f1 = frame.loc[frame["trip_id"] == scenario_trip_id(1)].reset_index(drop=True)
    f4 = frame.loc[frame["trip_id"] == scenario_trip_id(4)].reset_index(drop=True)
    r1, r4 = (
        compute_grouped_shap(booster, f1, num, cat),
        compute_grouped_shap(booster, f4, num, cat),
    )
    s1 = lm.score_booster(booster, f1, num, cat)
    s4 = lm.score_booster(booster, f4, num, cat)
    cols = r1.feature_columns

    out: dict[str, Any] = {"n_candidates_s1": len(f1), "n_candidates_s4": len(f4)}
    membership = build_group_membership(num, cat)
    group_of = {c_: g for g, cs in membership.items() for c_ in cs}
    for name, res in (("scenario1_base", r1), ("scenario4_flip", r4)):
        raw = res.raw_shap_values
        out[f"{name}_attribution_share_by_group"] = _shares(
            _mean_abs_by(raw, cols, lambda c_: group_of[c_])
        )
        out[f"{name}_attribution_share_by_feature_type"] = _shares(
            _mean_abs_by(raw, cols, _type_of)
        )
    # the profile-side terms only: stated preference + cross features that read touristiness_pref
    pref_idx = [cols.index(c_) for c_ in PREF_COLUMNS if c_ in cols]
    out["pref_dependent_features"] = [cols[i] for i in pref_idx]
    out["scenario1_share_carried_by_pref_dependent_features"] = float(
        np.abs(r1.raw_shap_values[:, pref_idx]).sum(axis=1).mean()
        / np.abs(r1.raw_shap_values).sum(axis=1).mean()
    )
    # flip decomposition on the POIs present in BOTH candidate pools
    common = sorted(set(f1["poi_id"]) & set(f4["poi_id"]))
    i1 = f1.reset_index().set_index("poi_id").loc[common, "index"].to_numpy()
    i4 = f4.reset_index().set_index("poi_id").loc[common, "index"].to_numpy()
    d_raw = r4.raw_shap_values[i4] - r1.raw_shap_values[i1]
    move_by_group = _mean_abs_by(d_raw, cols, lambda c_: group_of[c_])
    move_by_type = _mean_abs_by(d_raw, cols, _type_of)
    out["n_common_pool"] = len(common)
    out["flip_mean_abs_shap_change_by_group"] = move_by_group
    out["flip_mean_abs_shap_change_by_feature_type"] = move_by_type
    out["flip_share_of_total_change_by_group"] = _shares(move_by_group)
    ds = s4.to_numpy()[i4] - s1.to_numpy()[i1]
    out["raw_score_sd_across_candidates_s1"] = float(s1.std())
    out["flip_mean_abs_raw_score_change"] = float(np.abs(ds).mean())
    out["flip_change_over_score_sd"] = float(np.abs(ds).mean() / s1.std())
    out["raw_score_spearman_s1_vs_s4_common_pool"] = float(
        stats.spearmanr(s1.to_numpy()[i1], s4.to_numpy()[i4]).statistic
    )
    a = _top10(s1.iloc[i1].reset_index(drop=True), f1["poi_id"].iloc[i1].reset_index(drop=True))
    b = _top10(s4.iloc[i4].reset_index(drop=True), f4["poi_id"].iloc[i4].reset_index(drop=True))
    out["raw_score_top10_jaccard_s1_vs_s4_common_pool"] = _jaccard(a, b)
    out["candidate_pool_jaccard"] = len(common) / len(set(f1["poi_id"]) | set(f4["poi_id"]))
    out["scenario_implicit_interaction_count"] = float(f1["implicit_interaction_count"].iloc[0])
    return out


def holdout_counterfactual() -> dict[str, Any]:
    c = ROOT / "configs"
    feature_cfg = FeatureBuildConfig.from_yaml(c / "features.yaml")
    budget = feature_cfg.traveler_features.budget_target_price_level
    frame = load_holdout_evaluation_frame(DATA, budget)
    booster = lm.load_boosters(ART)["lambdamart_ips"]
    num, cat = numeric_feature_columns(frame), categorical_feature_columns(frame)
    base = lm.score_booster(booster, frame, num, cat)
    flipped = frame.copy()
    flipped["explicit_touristiness_pref"] = -flipped["explicit_touristiness_pref"]
    flipped["interact_localness_gap"] = localness_preference_gap(
        flipped["num_localness"].to_numpy(dtype=np.float64),
        flipped["explicit_touristiness_pref"].to_numpy(dtype=np.float64),
    )
    if attach_cross_features is not None:
        flipped = attach_cross_features(flipped, DATA, budget)  # recomputes every xf_ column
    alt = lm.score_booster(booster, flipped, num, cat)
    rows = []
    for trip_id, idx in frame.groupby("trip_id").indices.items():
        ids = frame["poi_id"].iloc[idx]
        rows.append(
            {
                "trip_id": trip_id,
                "cold": float(frame["implicit_interaction_count"].iloc[idx[0]]) == 0.0,
                "jaccard": _jaccard(_top10(base.iloc[idx], ids), _top10(alt.iloc[idx], ids)),
                "abs_pref": float(abs(frame["explicit_touristiness_pref"].iloc[idx[0]])),
            }
        )
    df = pd.DataFrame(rows)
    strong = df["abs_pref"] >= 0.5
    return {
        "n_trips": len(df),
        "n_cold_start_trips": int(df["cold"].sum()),
        "flip_top10_jaccard_all_trips": float(df["jaccard"].mean()),
        "flip_top10_jaccard_trips_with_history": float(df.loc[~df["cold"], "jaccard"].mean()),
        "flip_top10_jaccard_cold_start_trips": float(df.loc[df["cold"], "jaccard"].mean()),
        "flip_top10_jaccard_strong_pref_trips": float(df.loc[strong, "jaccard"].mean()),
        "n_strong_pref_trips": int(strong.sum()),
        "note": "raw ranker score top-10 among the trip candidates, before the hard gate, utility and "
        "MMR layers; touristiness_pref negated and every dependent feature recomputed",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path, default=ROOT / "results" / "parts" / "scenario4_diagnosis.json"
    )
    args = ap.parse_args()
    print("poi_rank from", __import__("poi_rank").__file__)
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    p = metrics["personalization"]
    out = {
        "scenario_rows": scenario_rows(),
        "holdout_counterfactual_flip": holdout_counterfactual(),
        "holdout_personalization_reference": {
            "cross_archetype_jaccard_true_labels": p["archetype"]["cross_archetype_jaccard_mean"],
            "within_archetype_jaccard_true_labels": p["archetype"]["within_archetype_jaccard_mean"],
            "within_cross_ratio_true_labels": p["archetype"]["within_cross_ratio"],
            "perfect_ranker_within_cross_ratio": p["archetype_ideal_ranker_reference"][
                "within_cross_ratio"
            ],
            "mean_pairwise_jaccard_at_10_same_destination": p["mean_pairwise_jaccard_at_10"],
        },
        "feature_groups": list(FEATURE_GROUPS),
    }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
