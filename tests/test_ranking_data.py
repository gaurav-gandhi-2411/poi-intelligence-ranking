"""`models/ranking_data.py` tests: a small, fully hand-verified scenario (label join +
canonical poi_id remap + max-across-duplicate-interactions + the 4 engineered
interaction features), following the same hand-verification discipline
`tests/test_recall_metrics.py` established for Phase 4a. Plus a light real-data
integration check on `load_holdout_evaluation_frame`/`load_train_ranking_frame`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.models.ranking_data import (
    build_ranking_frame,
    label_by_trip_poi,
    load_holdout_biased_evaluation_frame,
    load_holdout_evaluation_frame,
    load_train_ranking_frame,
)

BUDGET_TARGET = BudgetTargetPriceLevel(low=1.5, medium=2.5, high=3.5)


@pytest.fixture
def small_scenario() -> dict[str, pd.DataFrame]:
    # P2 was merged away from a raw duplicate "P2_DUP" during Phase 2 dedup --
    # interactions below reference the RAW id, exactly like the real committed
    # dataset (mirrors tests/test_recall_metrics.py's small_scenario fixture).
    pois_df = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3"],
            "destination": ["testville"] * 3,
            "category": ["food", "museum", "food"],
            "tags": [["local"], ["history"], ["budget"]],
            "merged_poi_ids": [["P1"], ["P2", "P2_DUP"], ["P3"]],
        }
    )
    poi_features_df = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3"],
            "destination": ["testville"] * 3,
            "text_emb_00": [1.0, 0.0, 0.5],
            "text_emb_01": [0.0, 1.0, 0.5],
            "num_localness": [0.2, 0.8, 0.5],
            "num_price_level": [1.0, 4.0, 2.0],
            "num_pop_pct": [0.9, 0.1, 0.5],
        }
    )
    travelers_df = pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "mobility": ["walk"],
            "budget": ["low"],
            "interests": [["food"]],
        }
    )
    trips_df = pd.DataFrame(
        {
            "trip_id": ["TA"],
            "traveler_id": ["U1"],
            "destination": ["testville"],
            "stay_lat": [0.0],
            "stay_lon": [0.0],
        }
    )
    traveler_features_df = pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "trip_id": ["TA"],
            "implicit_taste_00": [1.0],
            "implicit_taste_01": [0.0],
            "explicit_touristiness_pref": [0.3],
        }
    )
    candidates_df = pd.DataFrame({"trip_id": ["TA", "TA", "TA"], "poi_id": ["P1", "P2", "P3"]})
    # P1: two interaction rows -- max(1, 3) = 3. "P2_DUP" remaps to canonical P2 -> 2.
    # P3: no interaction row at all -> label 0 (module docstring's default).
    interactions = pd.DataFrame(
        {
            "trip_id": ["TA", "TA", "TA"],
            "poi_id": ["P1", "P1", "P2_DUP"],
            "label": [1, 3, 2],
        }
    )
    return {
        "pois_df": pois_df,
        "poi_features_df": poi_features_df,
        "travelers_df": travelers_df,
        "trips_df": trips_df,
        "traveler_features_df": traveler_features_df,
        "candidates_df": candidates_df,
        "interactions": interactions,
    }


def test_label_by_trip_poi_max_and_canonical_remap(small_scenario: dict[str, pd.DataFrame]) -> None:
    from poi_rank.features.reconcile import build_poi_id_canonical_map

    canonical_map = build_poi_id_canonical_map(small_scenario["pois_df"])
    labels = label_by_trip_poi(small_scenario["interactions"], canonical_map)
    assert labels[("TA", "P1")] == 3  # max(1, 3)
    assert labels[("TA", "P2")] == 2  # remapped from P2_DUP
    assert ("TA", "P3") not in labels.index  # no interaction row at all


def test_build_ranking_frame_hand_verified(small_scenario: dict[str, pd.DataFrame]) -> None:
    frame = build_ranking_frame(
        small_scenario["candidates_df"],
        {"TA"},
        small_scenario["interactions"],
        small_scenario["trips_df"],
        small_scenario["travelers_df"],
        small_scenario["pois_df"],
        small_scenario["poi_features_df"],
        small_scenario["traveler_features_df"],
        BUDGET_TARGET,
    )
    assert list(frame["poi_id"]) == ["P1", "P2", "P3"]  # sorted (trip_id, poi_id)

    by_poi = frame.set_index("poi_id")
    assert by_poi.loc["P1", "label"] == 3
    assert by_poi.loc["P2", "label"] == 2
    assert by_poi.loc["P3", "label"] == 0  # default for "no interaction row"

    # interact_interest_match: interests={"food"}, P1 terms={"food","local"} -> 1/1;
    # P2 terms={"museum","history"} -> 0/1; P3 terms={"food","budget"} -> 1/1.
    assert by_poi.loc["P1", "interact_interest_match"] == pytest.approx(1.0)
    assert by_poi.loc["P2", "interact_interest_match"] == pytest.approx(0.0)
    assert by_poi.loc["P3", "interact_interest_match"] == pytest.approx(1.0)

    # interact_price_gap: budget=low -> target=1.5. |1.5-1.0|=0.5, |1.5-4.0|=2.5,
    # |1.5-2.0|=0.5.
    assert by_poi.loc["P1", "interact_price_gap"] == pytest.approx(0.5)
    assert by_poi.loc["P2", "interact_price_gap"] == pytest.approx(2.5)
    assert by_poi.loc["P3", "interact_price_gap"] == pytest.approx(0.5)

    # interact_localness_gap: touristiness_pref=0.3. |0.2-0.3|=0.1, |0.8-0.3|=0.5,
    # |0.5-0.3|=0.2.
    assert by_poi.loc["P1", "interact_localness_gap"] == pytest.approx(0.1)
    assert by_poi.loc["P2", "interact_localness_gap"] == pytest.approx(0.5)
    assert by_poi.loc["P3", "interact_localness_gap"] == pytest.approx(0.2, abs=1e-9)

    # interact_cos_taste_poi: taste=[1,0]. P1 emb=[1,0] -> cos=1.0; P2 emb=[0,1] ->
    # cos=0.0; P3 emb=[0.5,0.5] (unnormalized) -> cos = 0.5 / (1*sqrt(0.5)) = 0.7071.
    assert by_poi.loc["P1", "interact_cos_taste_poi"] == pytest.approx(1.0)
    assert by_poi.loc["P2", "interact_cos_taste_poi"] == pytest.approx(0.0, abs=1e-9)
    assert by_poi.loc["P3", "interact_cos_taste_poi"] == pytest.approx(0.70710678, rel=1e-6)


def test_build_ranking_frame_restricts_to_requested_trip_ids(
    small_scenario: dict[str, pd.DataFrame],
) -> None:
    frame = build_ranking_frame(
        small_scenario["candidates_df"],
        set(),  # no trips requested
        small_scenario["interactions"],
        small_scenario["trips_df"],
        small_scenario["travelers_df"],
        small_scenario["pois_df"],
        small_scenario["poi_features_df"],
        small_scenario["traveler_features_df"],
        BUDGET_TARGET,
    )
    assert len(frame) == 0


# ---------------------------------------------------------------------------
# Real-data integration smoke tests (session-scoped fixtures from conftest.py)
# ---------------------------------------------------------------------------


def test_load_holdout_evaluation_frame_real_data(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    frame = load_holdout_evaluation_frame(evaluate_ready_data_dir, budget)

    assert len(frame) > 0
    assert frame["label"].isna().sum() == 0
    assert frame["label"].min() >= 0
    assert frame["label"].max() <= 3
    assert not frame[["interact_cos_taste_poi", "interact_price_gap"]].isna().any().any()

    trips_df = pd.read_parquet(evaluate_ready_data_dir / "trips.parquet")
    holdout_trip_ids = set(trips_df.loc[trips_df["is_holdout"], "trip_id"])
    assert set(frame["trip_id"]) <= holdout_trip_ids


def test_load_train_ranking_frame_real_data(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    frame = load_train_ranking_frame(evaluate_ready_data_dir, budget)

    assert len(frame) > 0
    assert frame["label"].isna().sum() == 0

    trips_df = pd.read_parquet(evaluate_ready_data_dir / "trips.parquet")
    train_trip_ids = set(trips_df.loc[~trips_df["is_holdout"], "trip_id"])
    assert set(frame["trip_id"]) <= train_trip_ids
    # Both label classes must be present for baseline 6's LogisticRegression to fit.
    assert (frame["label"] >= 1).any()
    assert (frame["label"] == 0).any()


def test_holdout_and_train_frames_are_disjoint_in_trips(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    holdout = load_holdout_evaluation_frame(evaluate_ready_data_dir, budget)
    train = load_train_ranking_frame(evaluate_ready_data_dir, budget)
    assert set(holdout["trip_id"]).isdisjoint(set(train["trip_id"]))


def test_load_holdout_biased_evaluation_frame_real_data(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    frame = load_holdout_biased_evaluation_frame(evaluate_ready_data_dir, budget)

    assert len(frame) > 0
    assert frame["label"].isna().sum() == 0
    assert frame["label"].min() >= 0
    assert frame["label"].max() <= 3

    trips_df = pd.read_parquet(evaluate_ready_data_dir / "trips.parquet")
    holdout_trip_ids = set(trips_df.loc[trips_df["is_holdout"], "trip_id"])
    assert set(frame["trip_id"]) <= holdout_trip_ids


def test_biased_and_unbiased_holdout_frames_are_row_order_aligned(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any
) -> None:
    """`eval/run.py`'s bias-gap table reuses an already-computed unbiased-frame
    score `pd.Series` directly against the biased frame's `label` column without
    rescoring -- only valid if the two frames are row-for-row `(trip_id, poi_id)`
    aligned (module docstring's own stated invariant), asserted here directly
    rather than only relied upon implicitly."""
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    unbiased = load_holdout_evaluation_frame(evaluate_ready_data_dir, budget)
    biased = load_holdout_biased_evaluation_frame(evaluate_ready_data_dir, budget)

    assert len(unbiased) == len(biased)
    assert list(unbiased["trip_id"]) == list(biased["trip_id"])
    assert list(unbiased["poi_id"]) == list(biased["poi_id"])
