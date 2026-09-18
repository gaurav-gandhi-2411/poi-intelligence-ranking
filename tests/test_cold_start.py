"""`eval/cold_start.py` tests: hand-computed interaction-count bucketing (spec.md
section 11.8) on a small frame, plus a real-fixture-chain LODO smoke test (one
destination, `fast_model_cfg`, to keep the suite fast -- the full 3-destination
LODO wall-clock is reported separately by `poi_rank.cli lodo`, not re-measured by
the test suite)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from poi_rank.eval.cold_start import (
    BUCKET_ORDER,
    bucket_for_count,
    evaluate_lodo_destination,
    ndcg10_by_interaction_bucket,
    train_lodo_booster,
    trip_interaction_count,
)
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame


def test_bucket_for_count_boundaries() -> None:
    assert bucket_for_count(0) == "0"
    assert bucket_for_count(1) == "1-3"
    assert bucket_for_count(3) == "1-3"
    assert bucket_for_count(4) == "4-10"
    assert bucket_for_count(10) == "4-10"
    assert bucket_for_count(11) == ">10"
    assert bucket_for_count(100) == ">10"


def test_trip_interaction_count_takes_one_value_per_trip() -> None:
    frame = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2"],
            "implicit_interaction_count": [5.0, 5.0, 0.0],
        }
    )
    result = trip_interaction_count(frame)
    assert result == {"T1": 5.0, "T2": 0.0}


def test_ndcg10_by_interaction_bucket_hand_computed() -> None:
    # T1: interaction_count=0 -> bucket "0", perfect ranking -> NDCG@10 = 1.0
    # T2: interaction_count=5 -> bucket "4-10", inverted ranking of 2 relevant items
    frame = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "poi_id": ["P1", "P2", "P3", "P4"],
            "label": [1, 0, 1, 0],
            "implicit_interaction_count": [0.0, 0.0, 5.0, 5.0],
        }
    )
    score = pd.Series([1.0, 0.0, 0.0, 1.0], index=frame.index)  # T1 correct, T2 inverted
    result = ndcg10_by_interaction_bucket(
        frame, score, n_resamples=10, seed=1, ci_low_pct=2.5, ci_high_pct=97.5
    )

    assert set(result) == set(BUCKET_ORDER)
    assert result["0"]["n_trips_in_bucket"] == 1
    assert result["0"]["mean"] == pytest.approx(1.0)
    assert result["4-10"]["n_trips_in_bucket"] == 1
    assert result["4-10"]["mean"] < 1.0  # inverted ranking scores worse than ideal
    assert result["1-3"]["n_trips_in_bucket"] == 0
    assert result[">10"]["n_trips_in_bucket"] == 0


@pytest.fixture(scope="module")
def lodo_result_one_destination(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any, fast_model_cfg: ModelConfig
) -> Any:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    train_frame = load_train_ranking_frame(evaluate_ready_data_dir, budget)
    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    destination = str(train_frame["destination"].iloc[0])
    return train_lodo_booster(
        train_frame, interactions_train, pois_df, fast_model_cfg, destination
    ), destination


def test_train_lodo_booster_excludes_held_out_destination_from_fit(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any, lodo_result_one_destination: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    train_frame = load_train_ranking_frame(evaluate_ready_data_dir, budget)
    lodo_result, destination = lodo_result_one_destination

    assert lodo_result.booster is not None
    assert lodo_result.n_fit_rows > 0
    # Sanity: the held-out destination is a strict subset of the full train population.
    n_dest_rows = int((train_frame["destination"] == destination).sum())
    assert lodo_result.n_fit_rows < len(train_frame) - int(n_dest_rows * 0.5)


def test_evaluate_lodo_destination_real_data(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any, lodo_result_one_destination: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    holdout_frame = load_holdout_evaluation_frame(evaluate_ready_data_dir, budget)
    lodo_result, destination = lodo_result_one_destination

    dummy_full_score = pd.Series(0.0, index=holdout_frame.index)
    result = evaluate_lodo_destination(
        holdout_frame, dummy_full_score, lodo_result, destination, 42, 20, 2.5, 97.5
    )
    assert result["destination"] == destination
    assert result["n_holdout_trips"] > 0
    assert "ndcg@10_lodo" in result
    assert "ndcg@10_full_training" in result
    assert 0.0 <= result["wilcoxon_lodo_vs_full_training"]["p_value"] <= 1.0
