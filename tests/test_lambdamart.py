"""`models/lambdamart.py` (systems 7/8, spec.md section 8) tests: IPS-weight
computation on a hand-verifiable synthetic example, behavioral-dropout masking
correctness, LightGBM training/scoring on the real fixture chain (no NaN/crash), and
booster-level determinism (two independent training runs -> byte-identical
`.txt`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.models.config import LambdaMartConfig
from poi_rank.models.lambdamart import (
    apply_behavioral_dropout,
    behavioral_feature_columns,
    compute_ips_weights,
    fit_lambdamart_booster,
    load_boosters,
    run_train_lambdamart,
    save_boosters,
    score_booster,
    train_lambdamart_systems,
    train_val_split_by_trip,
)
from poi_rank.models.ranking_data import load_train_ranking_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"


# -----------------------------------------------------------------------------------
# IPS weight computation (pure, hand-verifiable)
# -----------------------------------------------------------------------------------


def test_compute_ips_weights_clip_and_group_normalization() -> None:
    """2 trips: A has 3 rows (p_expose 0.5, 0.1, unexposed), B has 2 rows (p_expose
    0.2, unexposed). Hand-computed expected output (module docstring's formula):
    raw = clip(1/p_expose, 1, 20) for observed rows, 1.0 for unexposed; normalized =
    raw * group_size / group_sum (each group's normalized weights sum to its own
    row count)."""
    trip_ids = pd.Series(["A", "A", "A", "B", "B"])
    p_expose = pd.Series([0.5, 0.1, np.nan, 0.2, np.nan])

    weights = compute_ips_weights(trip_ids, p_expose, clip_low=1.0, clip_high=20.0)

    # raw = [2, 10, 1] for A (sum=13, size=3); [5, 1] for B (sum=6, size=2)
    expected_a = np.array([2.0, 10.0, 1.0]) * 3 / 13
    expected_b = np.array([5.0, 1.0]) * 2 / 6
    expected = np.concatenate([expected_a, expected_b])
    np.testing.assert_allclose(weights.to_numpy(), expected, rtol=1e-10)

    # Each group's weights sum to its own row count.
    assert weights.iloc[:3].sum() == pytest.approx(3.0)
    assert weights.iloc[3:].sum() == pytest.approx(2.0)


def test_compute_ips_weights_clips_extreme_propensities() -> None:
    """A p_expose of 0.01 would give raw weight 100 -- clipped to 20 (ips_clip_high).
    A p_expose of 1.0 gives raw weight 1 -- already within [1, 20], untouched."""
    trip_ids = pd.Series(["A", "A"])
    p_expose = pd.Series([0.01, 1.0])

    weights = compute_ips_weights(trip_ids, p_expose, clip_low=1.0, clip_high=20.0)

    raw = np.array([20.0, 1.0])  # clip(100, 1, 20) = 20; clip(1, 1, 20) = 1
    expected = raw * 2 / raw.sum()
    np.testing.assert_allclose(weights.to_numpy(), expected, rtol=1e-10)


def test_compute_ips_weights_unexposed_rows_get_neutral_weight_pre_normalization() -> None:
    """A single-row group with NO logged impression: raw weight is exactly 1.0
    (module docstring's "no logged impression" handling), and since it is the only
    row in its group, normalization leaves it at 1.0 unchanged."""
    trip_ids = pd.Series(["A"])
    p_expose = pd.Series([np.nan])

    weights = compute_ips_weights(trip_ids, p_expose, clip_low=1.0, clip_high=20.0)

    assert weights.iloc[0] == pytest.approx(1.0)


# -----------------------------------------------------------------------------------
# Behavioral-block dropout
# -----------------------------------------------------------------------------------


@pytest.fixture
def toy_behav_frame() -> pd.DataFrame:
    n = 2000
    return pd.DataFrame(
        {
            "trip_id": ["T0001"] * n,
            "poi_id": [f"P{i:05d}" for i in range(n)],
            "behav_impressions": np.arange(n, dtype=np.float64),
            "behav_ctr_smoothed": np.arange(n, dtype=np.float64) / 10.0,
            "num_pop_pct": np.arange(n, dtype=np.float64) / n,
        }
    )


def test_behavioral_feature_columns_selects_only_behav_prefix(
    toy_behav_frame: pd.DataFrame,
) -> None:
    assert behavioral_feature_columns(toy_behav_frame) == [
        "behav_ctr_smoothed",
        "behav_impressions",
    ]


def test_apply_behavioral_dropout_masks_correct_fraction_and_columns(
    toy_behav_frame: pd.DataFrame,
) -> None:
    out, mask = apply_behavioral_dropout(toy_behav_frame, dropout_rate=0.15, seed=44)

    # Actual masked fraction is close to the target rate (2000 independent Bernoulli
    # draws -- a wide but bounded tolerance, not an exact 0.15).
    assert abs(mask.mean() - 0.15) < 0.03

    behav_cols = behavioral_feature_columns(toy_behav_frame)
    assert out.loc[mask, behav_cols].isna().all().all()
    assert not out.loc[~mask, behav_cols].isna().any().any()
    # Non-behavioral columns are untouched, masked rows included.
    assert (out["num_pop_pct"] == toy_behav_frame["num_pop_pct"]).all()
    assert (out["poi_id"] == toy_behav_frame["poi_id"]).all()


def test_apply_behavioral_dropout_is_seeded_reproducible(toy_behav_frame: pd.DataFrame) -> None:
    _, mask1 = apply_behavioral_dropout(toy_behav_frame, dropout_rate=0.15, seed=44)
    _, mask2 = apply_behavioral_dropout(toy_behav_frame, dropout_rate=0.15, seed=44)
    _, mask3 = apply_behavioral_dropout(toy_behav_frame, dropout_rate=0.15, seed=99)

    np.testing.assert_array_equal(mask1, mask2)
    assert not np.array_equal(mask1, mask3)


# -----------------------------------------------------------------------------------
# Train/val split by trip
# -----------------------------------------------------------------------------------


def test_train_val_split_by_trip_no_overlap_and_covers_every_row() -> None:
    frame = pd.DataFrame(
        {
            "trip_id": [f"T{i:04d}" for i in range(50) for _ in range(3)],
            "poi_id": [f"P{j}" for _ in range(50) for j in range(3)],
        }
    )
    fit_frame, val_frame = train_val_split_by_trip(frame, val_fraction=0.2, seed=43)

    fit_trips = set(fit_frame["trip_id"])
    val_trips = set(val_frame["trip_id"])
    assert fit_trips.isdisjoint(val_trips)
    assert fit_trips | val_trips == set(frame["trip_id"])
    assert len(fit_frame) + len(val_frame) == len(frame)
    assert len(val_trips) == pytest.approx(10, abs=1)  # ~20% of 50 trips


# -----------------------------------------------------------------------------------
# Real fixture chain: training runs without error, produces a valid ranking
# -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_lambdamart_cfg() -> LambdaMartConfig:
    return LambdaMartConfig(
        num_leaves=15,
        learning_rate=0.1,
        n_estimators=30,
        early_stopping_rounds=10,
        lambdarank_truncation_level=20,
        eval_ndcg_at=(10,),
        label_gain=(0, 1, 3, 7),
        min_data_in_leaf=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        num_threads=1,
        val_fraction=0.15,
        val_split_seed=43,
        ips_clip_low=1.0,
        ips_clip_high=20.0,
        behavioral_dropout_rate=0.15,
        behavioral_dropout_seed=44,
    )


@pytest.fixture(scope="module")
def real_train_frame(evaluate_ready_data_dir: Path, feature_build_cfg: Any) -> pd.DataFrame:
    return load_train_ranking_frame(
        evaluate_ready_data_dir, feature_build_cfg.traveler_features.budget_target_price_level
    )


def test_train_lambdamart_systems_produces_valid_rankings(
    evaluate_ready_data_dir: Path,
    real_train_frame: pd.DataFrame,
    fast_lambdamart_cfg: LambdaMartConfig,
) -> None:
    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")

    artifacts = train_lambdamart_systems(
        real_train_frame, interactions_train, pois_df, fast_lambdamart_cfg, seed=42
    )

    score_no_ips = score_booster(
        artifacts.booster_lambdamart,
        real_train_frame,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
    )
    score_ips = score_booster(
        artifacts.booster_lambdamart_ips,
        real_train_frame,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
    )
    assert not score_no_ips.isna().any()
    assert not score_ips.isna().any()
    assert len(score_no_ips) == len(real_train_frame)
    assert artifacts.diagnostics["n_fit_rows"] + artifacts.diagnostics["n_val_rows"] == len(
        real_train_frame
    )
    assert 0.0 < artifacts.diagnostics["ips_exposure_rate"] < 1.0


def test_score_booster_no_nan_on_real_holdout_frame(
    evaluate_ready_data_dir: Path,
    real_train_frame: pd.DataFrame,
    fast_lambdamart_cfg: LambdaMartConfig,
    feature_build_cfg: Any,
) -> None:
    from poi_rank.models.ranking_data import load_holdout_evaluation_frame

    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    artifacts = train_lambdamart_systems(
        real_train_frame, interactions_train, pois_df, fast_lambdamart_cfg, seed=42
    )

    holdout_frame = load_holdout_evaluation_frame(
        evaluate_ready_data_dir, feature_build_cfg.traveler_features.budget_target_price_level
    )
    score = score_booster(
        artifacts.booster_lambdamart_ips,
        holdout_frame,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
    )
    assert not score.isna().any()
    assert len(score) == len(holdout_frame)


def test_fit_lambdamart_booster_raises_no_error_with_ips_weight(
    real_train_frame: pd.DataFrame, fast_lambdamart_cfg: LambdaMartConfig
) -> None:
    """`fit_lambdamart_booster` accepts a `sample_weight` Series (system 8's IPS
    weight) without crashing and produces a booster with `best_iteration > 0`."""
    from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns

    fit_frame, val_frame = train_val_split_by_trip(real_train_frame, 0.15, 43)
    numeric_columns = numeric_feature_columns(real_train_frame)
    categorical_columns = categorical_feature_columns(real_train_frame)
    uniform_weight = pd.Series(np.ones(len(fit_frame)), index=fit_frame.index)

    booster = fit_lambdamart_booster(
        fit_frame,
        val_frame,
        numeric_columns,
        categorical_columns,
        fast_lambdamart_cfg,
        seed=42,
        sample_weight=uniform_weight,
    )
    assert booster.best_iteration > 0


# -----------------------------------------------------------------------------------
# Booster-level determinism (two independent training runs -> byte-identical .txt)
# -----------------------------------------------------------------------------------


def test_run_train_lambdamart_is_deterministic(
    evaluate_ready_data_dir: Path,
    feature_build_cfg: Any,
    fast_lambdamart_cfg: LambdaMartConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from poi_rank.models.config import (
        BaselinesConfig,
        ContentCosineConfig,
        ItemKnnCfConfig,
        LogisticRegressionConfig,
        ModelConfig,
    )

    model_cfg = ModelConfig(
        seed=42,
        baselines=BaselinesConfig(
            content_cosine=ContentCosineConfig(interest_weight=0.7, price_fit_weight=0.3),
            item_knn_cf=ItemKnnCfConfig(cf_score_missing_fallback="popularity"),
            logistic_regression=LogisticRegressionConfig(max_iter=300, C=1.0),
        ),
        lambdamart=fast_lambdamart_cfg,
    )

    artifacts_dir1 = tmp_path_factory.mktemp("artifacts1")
    artifacts_dir2 = tmp_path_factory.mktemp("artifacts2")

    run_train_lambdamart(evaluate_ready_data_dir, artifacts_dir1, model_cfg, feature_build_cfg)
    run_train_lambdamart(evaluate_ready_data_dir, artifacts_dir2, model_cfg, feature_build_cfg)

    for filename in ("model.txt", "model_no_ips.txt", "model_ips_no_dropout.txt"):
        bytes1 = (artifacts_dir1 / filename).read_bytes()
        bytes2 = (artifacts_dir2 / filename).read_bytes()
        assert bytes1 == bytes2, f"{filename} differs across two training runs"


def test_save_and_load_boosters_roundtrip_scores_match(
    real_train_frame: pd.DataFrame,
    fast_lambdamart_cfg: LambdaMartConfig,
    evaluate_ready_data_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A booster's predictions survive a save-to-disk / fresh-load round trip
    unchanged -- the scenario `poi_rank.cli evaluate` relies on (loading a booster
    `poi_rank.cli train` persisted in an earlier process)."""
    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    artifacts = train_lambdamart_systems(
        real_train_frame, interactions_train, pois_df, fast_lambdamart_cfg, seed=42
    )
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    save_boosters(artifacts, artifacts_dir)

    score_before = score_booster(
        artifacts.booster_lambdamart_ips,
        real_train_frame,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
    )
    loaded = load_boosters(artifacts_dir)
    score_after = score_booster(
        loaded["lambdamart_ips"],
        real_train_frame,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
    )
    np.testing.assert_allclose(score_before.to_numpy(), score_after.to_numpy(), rtol=1e-10)
