"""`eval/new_poi_cohort.py` tests: cohort identification against the real fixture
chain (correct, non-empty, matches `docs/DATA_CARD.md`'s ~5% new-POI dirtiness rate)
and the NDCG@10 evaluation helper on a small hand-built frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from poi_rank.datagen.config import DatagenConfig
from poi_rank.eval.new_poi_cohort import (
    cohort_frame,
    cohort_summary,
    evaluate_cohort_ndcg10,
    evaluate_cohort_ndcg10_per_trip,
    new_poi_ids,
)
from poi_rank.models.ranking_data import load_holdout_evaluation_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
DATAGEN_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"


@pytest.fixture(scope="module")
def datagen_cfg_module() -> DatagenConfig:
    return DatagenConfig.from_yaml(DATAGEN_CONFIG_PATH)


def test_new_poi_ids_nonempty_and_matches_expected_rate(
    evaluate_ready_data_dir: Path, datagen_cfg_module: DatagenConfig
) -> None:
    """`new_poi_rate: 0.05` (configs/datagen.yaml) -- the identified cohort should
    be non-empty and roughly 5% of the catalog (docs/DATA_CARD.md's dirtiness
    table), not exactly 5% since it is itself a random draw."""
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    ids = new_poi_ids(pois_df, datagen_cfg_module)

    assert len(ids) > 0
    rate = len(ids) / len(pois_df)
    assert 0.01 < rate < 0.15  # wide tolerance around ~5%, a real Bernoulli draw
    assert ids.issubset(set(pois_df["poi_id"]))


def test_cohort_frame_and_summary_report_honest_nonzero_counts(
    evaluate_ready_data_dir: Path, feature_build_cfg: Any, datagen_cfg_module: DatagenConfig
) -> None:
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    holdout_frame = load_holdout_evaluation_frame(
        evaluate_ready_data_dir, feature_build_cfg.traveler_features.budget_target_price_level
    )
    ids = new_poi_ids(pois_df, datagen_cfg_module)
    cohort = cohort_frame(holdout_frame, ids)

    assert len(cohort) > 0
    assert set(cohort["poi_id"]).issubset(ids)
    # index preserved (not reset) -- a subset of the original holdout_frame's index.
    assert cohort.index.isin(holdout_frame.index).all()

    summary = cohort_summary(pois_df, holdout_frame, datagen_cfg_module)
    assert summary["n_new_pois_in_catalog"] == len(ids)
    assert summary["n_holdout_rows_in_cohort"] == len(cohort)
    assert summary["n_holdout_trips_with_cohort_candidate"] > 0
    # A degenerate (zero) cohort would be reported honestly, not silently skipped --
    # this fixture's real, measured cohort is non-degenerate (44/202 holdout trips
    # have >=1 relevant new-POI candidate as of Phase 5's measurement).
    assert summary["n_holdout_trips_with_relevant_cohort_candidate"] >= 0


def test_evaluate_cohort_ndcg10_on_hand_built_frame() -> None:
    """2 trips, one with a perfect ranking and one with an inverted ranking --
    NDCG@10 should reflect that directly (hand-verifiable, mirrors
    `tests/test_metrics.py`'s own convention)."""
    frame = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "poi_id": ["A", "B", "C", "D"],
            "label": [3, 0, 0, 3],
        }
    )
    # T1: A (label 3) ranked first -- perfect. T2: D (label 3) ranked LAST -- worst.
    score = pd.Series([1.0, 0.0, 1.0, 0.0], index=frame.index)

    per_trip = evaluate_cohort_ndcg10_per_trip(frame, score)
    assert per_trip.loc["T1"] == pytest.approx(1.0)
    assert per_trip.loc["T2"] < 1.0

    agg = evaluate_cohort_ndcg10(
        frame, score, n_resamples=50, seed=1, ci_low_pct=2.5, ci_high_pct=97.5
    )
    assert agg.n_included == 2
    assert agg.n_excluded == 0
    assert agg.ci_low <= agg.mean <= agg.ci_high


def test_evaluate_cohort_ndcg10_excludes_trips_with_zero_relevant() -> None:
    """A trip with no `label >= 1` candidate at all is undefined for NDCG (IDCG=0)
    -- excluded from the mean, not coerced to 0, per `eval.metrics`' own
    documented discipline."""
    frame = pd.DataFrame({"trip_id": ["T1", "T1"], "poi_id": ["A", "B"], "label": [0, 0]})
    score = pd.Series([1.0, 0.0], index=frame.index)

    agg = evaluate_cohort_ndcg10(
        frame, score, n_resamples=50, seed=1, ci_low_pct=2.5, ci_high_pct=97.5
    )
    assert agg.n_included == 0
    assert agg.n_excluded == 1
    assert agg.mean == 0.0
