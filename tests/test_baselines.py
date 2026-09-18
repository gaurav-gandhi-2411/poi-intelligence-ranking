"""Baselines 1-6 (`models/baselines.py`) tests: each baseline must produce a valid
ranking (every candidate scored, no NaN/crash) for both a normal trip and a
cold-start (zero as-of-safe history) trip, plus targeted checks of each baseline's
own documented behavior (fallback rates, blend weights, design-matrix alignment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poi_rank.candidates.config import GeoChannelConfig
from poi_rank.models.baselines import (
    BaselineResult,
    baseline_content_cosine,
    baseline_item_knn_cf,
    baseline_popularity,
    baseline_popularity_geo_filter,
    baseline_random,
    categorical_feature_columns,
    fit_logistic_regression,
    numeric_feature_columns,
    score_logistic_regression,
)
from poi_rank.models.config import ContentCosineConfig, LogisticRegressionConfig

GEO_CFG = GeoChannelConfig(
    quota=60,
    h3_resolution=8,
    radius_km_walk=2.0,
    radius_km_public_transport=8.0,
    radius_km_car=25.0,
    radius_km_mixed=25.0,
)


@pytest.fixture
def two_trip_frame() -> pd.DataFrame:
    """One "normal" trip (T_NORMAL, non-zero interest/price/localness signal) and
    one "cold-start" trip (T_COLD, empty interests, zero taste-derived interaction
    features) -- every baseline that doesn't special-case history (1/2/3/4) must
    still score both without crashing; baselines that DO special-case history (5/6)
    are covered by their own dedicated fixtures below."""
    return pd.DataFrame(
        {
            "trip_id": ["T_NORMAL", "T_NORMAL", "T_COLD", "T_COLD"],
            "poi_id": ["P1", "P2", "P3", "P4"],
            "num_pop_pct": [0.9, 0.2, 0.5, 0.5],
            "geo_lat": [0.001, 0.01, 0.001, 1.0],
            "geo_lon": [0.001, 0.01, 0.001, 1.0],
            "stay_lat": [0.0, 0.0, 0.0, 0.0],
            "stay_lon": [0.0, 0.0, 0.0, 0.0],
            "mobility": ["walk", "walk", "public_transport", "public_transport"],
            "interact_interest_match": [1.0, 0.0, 0.0, 0.0],
            "interact_price_gap": [0.5, 2.0, 0.0, 3.0],
        }
    )


def _assert_valid_ranking(result: BaselineResult, frame: pd.DataFrame) -> None:
    assert len(result.score) == len(frame)
    assert result.score.index.equals(frame.index)
    assert not result.score.isna().any()
    assert np.isfinite(result.score.replace([np.inf, -np.inf], np.nan).dropna()).all()


def test_baseline_random_valid_and_deterministic(two_trip_frame: pd.DataFrame) -> None:
    r1 = baseline_random(two_trip_frame, seed=42)
    r2 = baseline_random(two_trip_frame, seed=42)
    _assert_valid_ranking(r1, two_trip_frame)
    pd.testing.assert_series_equal(r1.score, r2.score)
    assert (r1.score >= 0.0).all() and (r1.score < 1.0).all()


def test_baseline_random_different_seed_differs(two_trip_frame: pd.DataFrame) -> None:
    r1 = baseline_random(two_trip_frame, seed=1)
    r2 = baseline_random(two_trip_frame, seed=2)
    assert not r1.score.equals(r2.score)


def test_baseline_popularity_matches_pop_pct_column(two_trip_frame: pd.DataFrame) -> None:
    result = baseline_popularity(two_trip_frame)
    _assert_valid_ranking(result, two_trip_frame)
    pd.testing.assert_series_equal(
        result.score, two_trip_frame["num_pop_pct"].rename("score_popularity")
    )


def test_baseline_popularity_geo_filter_excludes_out_of_radius(
    two_trip_frame: pd.DataFrame,
) -> None:
    result = baseline_popularity_geo_filter(two_trip_frame, GEO_CFG)
    _assert_valid_ranking(result, two_trip_frame)
    # T_COLD: P3 is in-radius (walk-scale distance well under 8km transit radius),
    # P4 is ~157km away -- excluded (score -inf), not dropped.
    p4_score = result.score[two_trip_frame["poi_id"] == "P4"].iloc[0]
    assert p4_score == -np.inf
    assert result.diagnostics["fallback_rate"] == pytest.approx(0.0)


def test_baseline_popularity_geo_filter_falls_back_when_all_excluded() -> None:
    frame = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "poi_id": ["P1", "P2"],
            "num_pop_pct": [0.9, 0.1],
            "geo_lat": [50.0, 51.0],  # both far outside any radius
            "geo_lon": [50.0, 51.0],
            "stay_lat": [0.0, 0.0],
            "stay_lon": [0.0, 0.0],
            "mobility": ["walk", "walk"],
        }
    )
    result = baseline_popularity_geo_filter(frame, GEO_CFG)
    _assert_valid_ranking(result, frame)
    assert result.diagnostics["fallback_rate"] == pytest.approx(1.0)
    # Fallback: plain popularity, not -inf for everyone.
    pd.testing.assert_series_equal(
        result.score.reset_index(drop=True),
        frame["num_pop_pct"].rename("score_popularity_geo").reset_index(drop=True),
    )


def test_baseline_content_cosine_blend(two_trip_frame: pd.DataFrame) -> None:
    cfg = ContentCosineConfig(interest_weight=0.7, price_fit_weight=0.3)
    result = baseline_content_cosine(two_trip_frame, cfg)
    _assert_valid_ranking(result, two_trip_frame)
    # P1: interest_match=1.0, price_gap=0.5 -> price_fit=1-0.5/3=0.8333.
    # score = 0.7*1.0 + 0.3*0.8333 = 0.95.
    p1_score = result.score[two_trip_frame["poi_id"] == "P1"].iloc[0]
    assert p1_score == pytest.approx(0.7 * 1.0 + 0.3 * (1.0 - 0.5 / 3.0))


# ---------------------------------------------------------------------------
# Baseline 5: item-kNN CF -- dedicated scenario with a real as-of-safe seed history
# ---------------------------------------------------------------------------


@pytest.fixture
def cf_scenario() -> dict[str, pd.DataFrame]:
    pois_df = pd.DataFrame(
        {"poi_id": ["P1", "P2", "P3"], "merged_poi_ids": [["P1"], ["P2"], ["P3"]]}
    )
    trips_df = pd.DataFrame(
        {
            "trip_id": ["T_TRAIN", "T_NORMAL", "T_COLD"],
            "traveler_id": ["U1", "U1", "U2"],
            "start_date": pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-11"]),
        }
    )
    # U1 engaged P1 (label>=1) on an earlier trip -> non-empty seed for T_NORMAL.
    # U2 has no train-window history at all -> cold start for T_COLD. U3 (a third,
    # otherwise-irrelevant traveler) co-engages P1 AND P2 in one train trip -- the
    # co-interaction signal that gives P1-P2 a genuine nonzero CF similarity, so
    # T_NORMAL's ranking (seeded on U1's own P1) is a real, non-degenerate check,
    # not just "didn't crash".
    interactions_train = pd.DataFrame(
        {
            "traveler_id": ["U1", "U1", "U3", "U3"],
            "trip_id": ["T_TRAIN", "T_TRAIN", "T_TRAIN2", "T_TRAIN2"],
            "poi_id": ["P1", "P2", "P1", "P2"],
            "label": [3, 0, 3, 2],
            "timestamp": pd.to_datetime(["2024-01-01"] * 4),
        }
    )
    frame = pd.DataFrame(
        {
            "trip_id": ["T_NORMAL", "T_NORMAL", "T_COLD", "T_COLD"],
            "poi_id": ["P2", "P3", "P2", "P3"],
            "traveler_id": ["U1", "U1", "U2", "U2"],
            "num_pop_pct": [0.4, 0.6, 0.4, 0.6],
        }
    )
    return {
        "pois_df": pois_df,
        "trips_df": trips_df,
        "interactions_train": interactions_train,
        "frame": frame,
    }


def test_baseline_item_knn_cf_normal_and_cold_start(cf_scenario: dict[str, pd.DataFrame]) -> None:
    result = baseline_item_knn_cf(
        cf_scenario["frame"],
        cf_scenario["pois_df"],
        cf_scenario["interactions_train"],
        cf_scenario["trips_df"],
    )
    _assert_valid_ranking(result, cf_scenario["frame"])
    assert result.diagnostics["cold_start_fallback_rate"] == pytest.approx(0.5)  # 1 of 2 trips

    # T_COLD falls back to plain popularity.
    cold_scores = result.score[cf_scenario["frame"]["trip_id"] == "T_COLD"]
    cold_pop = cf_scenario["frame"].loc[cold_scores.index, "num_pop_pct"]
    pd.testing.assert_series_equal(
        cold_scores.reset_index(drop=True), cold_pop.reset_index(drop=True), check_names=False
    )

    # T_NORMAL: seed={P1} (U1's own earlier engagement). P2 co-occurs with P1 via
    # U3's engagement (genuine similarity > 0); P3 has no co-occurrence with P1 at
    # all (similarity == 0) -- P2 must rank strictly above P3.
    full = cf_scenario["frame"]
    p2_score = result.score[(full["trip_id"] == "T_NORMAL") & (full["poi_id"] == "P2")].iloc[0]
    p3_score = result.score[(full["trip_id"] == "T_NORMAL") & (full["poi_id"] == "P3")].iloc[0]
    assert p2_score > 0.0
    assert p3_score == pytest.approx(0.0)
    assert p2_score > p3_score


# ---------------------------------------------------------------------------
# Baseline 6: logistic regression -- fit on a train frame, score normal + cold-start
# ---------------------------------------------------------------------------


@pytest.fixture
def lr_train_and_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "trip_id": ["T1"] * 4 + ["T2"] * 4,
            "poi_id": ["P1", "P2", "P3", "P4"] * 2,
            "label": [3, 0, 1, 0, 2, 0, 3, 0],
            "num_pop_pct": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            "cat_category": pd.Categorical(
                ["food", "museum", "food", "museum", "food", "museum", "food", "museum"]
            ),
        }
    )
    # T_COLD: zero-signal row (all pop_pct 0.0, an unseen category) -- exercises
    # both the "cold-start" case and the unseen-category dummy-alignment path.
    eval_frame = pd.DataFrame(
        {
            "trip_id": ["T_NORMAL", "T_NORMAL", "T_COLD"],
            "poi_id": ["P1", "P2", "P5"],
            "num_pop_pct": [0.85, 0.15, 0.0],
            "cat_category": pd.Categorical(["food", "museum", "shopping"]),
        }
    )
    return train, eval_frame


def test_baseline_logistic_regression_normal_and_cold_start(
    lr_train_and_eval_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_frame, eval_frame = lr_train_and_eval_frames
    cfg = LogisticRegressionConfig(max_iter=500, C=1.0)
    model = fit_logistic_regression(train_frame, cfg, seed=42)

    assert numeric_feature_columns(train_frame) == ["num_pop_pct"]
    assert categorical_feature_columns(train_frame) == ["cat_category"]

    result = score_logistic_regression(model, eval_frame)
    _assert_valid_ranking(result, eval_frame)
    assert (result.score >= 0.0).all() and (result.score <= 1.0).all()
    assert result.diagnostics["n_features"] == len(model.numeric_columns) + len(model.dummy_columns)


def test_logistic_regression_deterministic_given_seed(
    lr_train_and_eval_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_frame, eval_frame = lr_train_and_eval_frames
    cfg = LogisticRegressionConfig(max_iter=500, C=1.0)
    m1 = fit_logistic_regression(train_frame, cfg, seed=7)
    m2 = fit_logistic_regression(train_frame, cfg, seed=7)
    s1 = score_logistic_regression(m1, eval_frame).score
    s2 = score_logistic_regression(m2, eval_frame).score
    pd.testing.assert_series_equal(s1, s2)
