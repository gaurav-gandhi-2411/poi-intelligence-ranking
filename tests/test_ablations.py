"""`eval/ablations.py` tests: hand-built-frame checks for the cheap ablations
(`-calibration`, `-MMR`, leave-one-channel-out) and a real-fixture-chain smoke test
for one feature-block retrain (`fast_model_cfg`'s small LightGBM settings keep this
fast -- the full 4-block table is exercised end-to-end by `eval/run.py`'s own
integration test, not re-measured here)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from poi_rank.candidates.channels import CHANNEL_NAMES
from poi_rank.eval.ablations import (
    FEATURE_BLOCK_PREFIXES,
    _candidate_keys_excluding_channel,
    _compare_ndcg10,
    calibration_row,
    feature_block_row,
    ips_weighting_row,
    leave_one_channel_out_row,
    train_block_dropped_booster,
)
from poi_rank.eval.metrics import aggregate_metric
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame


@pytest.fixture
def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "poi_id": ["P1", "P2", "P3", "P4"],
            "label": [1, 0, 1, 0],
        }
    )


def test_compare_ndcg10_identical_scores_zero_delta(small_frame: pd.DataFrame) -> None:
    score = pd.Series([1.0, 0.0, 1.0, 0.0], index=small_frame.index)
    row = _compare_ndcg10(
        "-test",
        "note",
        small_frame,
        score,
        small_frame,
        score,
        n_resamples=10,
        seed=1,
        ci_low_pct=2.5,
        ci_high_pct=97.5,
    )
    assert row.delta_ndcg10 == pytest.approx(0.0, abs=1e-9)
    assert row.status == "measured"


def test_compare_ndcg10_worse_ablated_score_is_negative_delta(small_frame: pd.DataFrame) -> None:
    full_score = pd.Series([1.0, 0.0, 1.0, 0.0], index=small_frame.index)  # perfect
    ablated_score = pd.Series([0.0, 1.0, 0.0, 1.0], index=small_frame.index)  # inverted
    row = _compare_ndcg10(
        "-test",
        "note",
        small_frame,
        full_score,
        small_frame,
        ablated_score,
        n_resamples=10,
        seed=1,
        ci_low_pct=2.5,
        ci_high_pct=97.5,
    )
    assert row.delta_ndcg10 is not None
    assert row.delta_ndcg10 < 0.0


def test_calibration_row_monotonic_transform_gives_near_zero_delta(
    small_frame: pd.DataFrame,
) -> None:
    """Isotonic calibration is monotonic non-decreasing -- a monotonic transform of
    the raw score never changes within-trip ranking, so delta must be exactly 0
    when the transform strictly preserves order (module docstring's own claim,
    verified here on a hand-built case with no tie-break-affecting flats)."""
    raw = pd.Series([2.0, 1.0, 5.0, 3.0], index=small_frame.index)
    calibrated = raw * 2.0 + 1.0  # strictly monotonic increasing transform
    row = calibration_row(
        small_frame, raw, calibrated, n_resamples=10, seed=1, ci_low_pct=2.5, ci_high_pct=97.5
    )
    assert row.delta_ndcg10 == pytest.approx(0.0, abs=1e-9)


def test_ips_weighting_row_assembles_from_precomputed_aggregates() -> None:
    per_trip_full = pd.Series([1.0, 0.5], index=["T1", "T2"])
    per_trip_ablated = pd.Series([0.8, 0.4], index=["T1", "T2"])
    full_agg = aggregate_metric(per_trip_full, 10, 1, 2.5, 97.5)
    ablated_agg = aggregate_metric(per_trip_ablated, 10, 1, 2.5, 97.5)
    row = ips_weighting_row(per_trip_full, per_trip_ablated, full_agg, ablated_agg)
    assert row.ablation == "-IPS_weighting"
    assert row.delta_ndcg10 == pytest.approx(ablated_agg.mean - full_agg.mean)


def test_candidate_keys_excluding_channel_hand_computed() -> None:
    channel_a, channel_b = CHANNEL_NAMES[0], CHANNEL_NAMES[1]
    other_channels = {c: False for c in CHANNEL_NAMES}
    row1 = {"trip_id": "T1", "poi_id": "P1", **other_channels, channel_a: True}
    row2 = {"trip_id": "T1", "poi_id": "P2", **other_channels, channel_b: True}
    candidates_df = pd.DataFrame([row1, row2])

    kept = _candidate_keys_excluding_channel(candidates_df, channel_a)
    assert ("T1", "P1") not in kept  # only surfaced by the excluded channel
    assert ("T1", "P2") in kept  # surfaced by channel_b, unaffected


def test_leave_one_channel_out_row_real_data(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    holdout_frame = load_holdout_evaluation_frame(evaluate_ready_data_dir, budget)
    candidates_df = pd.read_parquet(evaluate_ready_data_dir / "candidates.parquet")
    score = pd.Series(
        0.0, index=holdout_frame.index
    )  # constant score is enough to exercise the path

    row = leave_one_channel_out_row(
        "-CF_channel", "channel_cf", holdout_frame, candidates_df, score, 10, 1, 2.5, 97.5
    )
    assert row.status == "measured"
    assert row.ndcg10_ablated is not None
    # Removing a channel can only shrink (or leave unchanged) each trip's candidate
    # pool, never grow it.
    assert row.ndcg10_ablated.n_included + row.ndcg10_ablated.n_excluded <= len(
        holdout_frame["trip_id"].unique()
    )


def test_feature_block_prefixes_match_known_blocks() -> None:
    assert set(FEATURE_BLOCK_PREFIXES.values()) == {
        "text_emb_",
        "implicit_",
        "explicit_",
        "behav_",
    }


def test_train_block_dropped_booster_real_data_smoke(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any, fast_model_cfg: ModelConfig
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    train_frame = load_train_ranking_frame(evaluate_ready_data_dir, budget)
    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")

    booster, numeric_columns, categorical_columns = train_block_dropped_booster(
        train_frame, interactions_train, pois_df, fast_model_cfg, "behav_"
    )
    assert booster is not None
    assert not any(c.startswith("behav_") for c in numeric_columns)
    assert len(numeric_columns) > 0


def test_feature_block_row_real_data_smoke(
    evaluate_ready_data_dir: Any, feature_build_cfg: Any, fast_model_cfg: ModelConfig
) -> None:
    budget = feature_build_cfg.traveler_features.budget_target_price_level
    train_frame = load_train_ranking_frame(evaluate_ready_data_dir, budget)
    holdout_frame = load_holdout_evaluation_frame(evaluate_ready_data_dir, budget)
    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    dummy_full_score = pd.Series(0.0, index=holdout_frame.index)

    row = feature_block_row(
        "-behavioral_block",
        "behav_",
        train_frame,
        interactions_train,
        pois_df,
        holdout_frame,
        dummy_full_score,
        fast_model_cfg,
        10,
        1,
        2.5,
        97.5,
    )
    assert row.status == "measured"
    assert row.ndcg10_ablated is not None
