"""`eval/metrics.py` tests: NDCG/Precision/Recall/MAP/MRR hand-verified against small
textbook-style examples (never just trusting the implementation against itself),
bootstrap-CI sanity (non-degenerate width, deterministic given a seed), and Wilcoxon
sanity (a known-significant and a known-null paired comparison).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from poi_rank.eval.metrics import (
    aggregate_metric,
    average_precision,
    bootstrap_ci,
    compute_all_trip_metrics,
    dcg_at_k,
    ndcg_at_k,
    paired_wilcoxon,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Hand-verified example: 3 candidates, achieved ranking order == input order
# (scores strictly descending in the given order), labels = [1, 3, 0].
# Ideal ranking (sorted by label desc) = [3, 1, 0].
# ---------------------------------------------------------------------------

LABELS = np.array([1, 3, 0], dtype=np.int64)
SCORES = np.array([30.0, 20.0, 10.0], dtype=np.float64)
POI_IDS = np.array(["A", "B", "C"], dtype=object)


def test_dcg_at_k_hand_computed() -> None:
    dcg = dcg_at_k(LABELS, k=3)
    expected = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3) + (2**0 - 1) / math.log2(4)
    assert dcg == pytest.approx(expected)


def test_ndcg_at_k_hand_verified() -> None:
    result = ndcg_at_k(LABELS, SCORES, POI_IDS, k=3)
    dcg = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3) + (2**0 - 1) / math.log2(4)
    idcg = (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3) + (2**0 - 1) / math.log2(4)
    assert result == pytest.approx(dcg / idcg)


def test_ndcg_at_1_hand_verified() -> None:
    result = ndcg_at_k(LABELS, SCORES, POI_IDS, k=1)
    # Achieved DCG@1 = gain(label=1); ideal DCG@1 = gain(label=3).
    assert result == pytest.approx(((2**1 - 1) / math.log2(2)) / ((2**3 - 1) / math.log2(2)))
    assert result == pytest.approx(1.0 / 7.0)


def test_ndcg_undefined_when_all_labels_zero() -> None:
    labels = np.array([0, 0, 0], dtype=np.int64)
    result = ndcg_at_k(labels, SCORES, POI_IDS, k=3)
    assert result is None


def test_precision_at_k_hand_verified() -> None:
    assert precision_at_k(LABELS, SCORES, POI_IDS, k=1) == pytest.approx(1.0)  # top1 label=1
    assert precision_at_k(LABELS, SCORES, POI_IDS, k=2) == pytest.approx(1.0)  # top2 = [1,3]
    assert precision_at_k(LABELS, SCORES, POI_IDS, k=3) == pytest.approx(2.0 / 3.0)


def test_precision_at_k_always_defined_even_with_zero_relevant() -> None:
    labels = np.array([0, 0, 0], dtype=np.int64)
    assert precision_at_k(labels, SCORES, POI_IDS, k=2) == pytest.approx(0.0)


def test_recall_at_k_hand_verified() -> None:
    # n_relevant = 2 (labels 1 and 3).
    assert recall_at_k(LABELS, SCORES, POI_IDS, k=1) == pytest.approx(0.5)
    assert recall_at_k(LABELS, SCORES, POI_IDS, k=3) == pytest.approx(1.0)


def test_recall_at_k_undefined_when_zero_relevant() -> None:
    labels = np.array([0, 0, 0], dtype=np.int64)
    assert recall_at_k(labels, SCORES, POI_IDS, k=2) is None


def test_average_precision_hand_verified() -> None:
    # Ranked labels = [1, 3, 0] -> hits at rank 1 and 2.
    # precision@1=1/1=1.0, precision@2=2/2=1.0 -> AP = (1.0+1.0)/2 = 1.0.
    assert average_precision(LABELS, SCORES, POI_IDS) == pytest.approx(1.0)


def test_average_precision_partial_hand_verified() -> None:
    labels = np.array([0, 1, 3], dtype=np.int64)  # ranked (by score desc) = [0,1,3]
    result = average_precision(labels, SCORES, POI_IDS)
    # hits at rank2 (precision=1/2=0.5) and rank3 (precision=2/3) -> AP=(0.5+2/3)/2
    expected = (0.5 + 2.0 / 3.0) / 2.0
    assert result == pytest.approx(expected)


def test_reciprocal_rank_hand_verified() -> None:
    assert reciprocal_rank(LABELS, SCORES, POI_IDS) == pytest.approx(1.0)  # hit at rank 1
    labels = np.array([0, 3, 1], dtype=np.int64)  # ranked = [0,3,1] -> first hit at rank 2
    assert reciprocal_rank(labels, SCORES, POI_IDS) == pytest.approx(0.5)


def test_reciprocal_rank_undefined_when_zero_relevant() -> None:
    labels = np.array([0, 0, 0], dtype=np.int64)
    assert reciprocal_rank(labels, SCORES, POI_IDS) is None


def test_tie_break_is_poi_id_ascending() -> None:
    # Two candidates with identical scores -- must break ties by poi_id ascending
    # (matches candidates/channels.py's np.lexsort convention).
    labels = np.array([3, 0], dtype=np.int64)
    scores = np.array([5.0, 5.0], dtype=np.float64)
    poi_ids = np.array(["Z", "A"], dtype=object)  # "A" < "Z"
    result = precision_at_k(labels, scores, poi_ids, k=1)
    # "A" (label=0) ranks first on the tie-break -> precision@1 = 0.0, not 1.0.
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_all_trip_metrics / aggregate_metric: exclusion bookkeeping
# ---------------------------------------------------------------------------


def test_compute_all_trip_metrics_and_aggregate_excludes_undefined_trips() -> None:
    frame = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "poi_id": ["A", "B", "C", "D"],
            "label": [1, 0, 0, 0],  # T2 has zero relevant candidates
        }
    )
    score = pd.Series([2.0, 1.0, 2.0, 1.0])
    per_trip = compute_all_trip_metrics(frame, score, ndcg_ks=(1,), precision_ks=(), recall_ks=())
    assert per_trip.loc["T1", "ndcg@1"] == pytest.approx(1.0)
    assert pd.isna(per_trip.loc["T2", "ndcg@1"])

    agg = aggregate_metric(
        per_trip["ndcg@1"], n_resamples=200, seed=1, ci_low_pct=2.5, ci_high_pct=97.5
    )
    assert agg.n_included == 1
    assert agg.n_excluded == 1
    assert agg.mean == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap CI: non-degenerate, deterministic given seed
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic_given_seed() -> None:
    values = pd.Series(np.arange(50, dtype=float))
    lo1, hi1 = bootstrap_ci(values, n_resamples=500, seed=42, ci_low_pct=2.5, ci_high_pct=97.5)
    lo2, hi2 = bootstrap_ci(values, n_resamples=500, seed=42, ci_low_pct=2.5, ci_high_pct=97.5)
    assert (lo1, hi1) == (lo2, hi2)


def test_bootstrap_ci_is_non_degenerate_and_brackets_the_mean() -> None:
    values = pd.Series(np.arange(50, dtype=float))
    lo, hi = bootstrap_ci(values, n_resamples=2000, seed=42, ci_low_pct=2.5, ci_high_pct=97.5)
    assert lo < hi
    assert lo < values.mean() < hi
    assert hi - lo < (values.max() - values.min())  # narrower than the full raw range


def test_bootstrap_ci_width_shrinks_with_more_data_same_resamples() -> None:
    small = pd.Series(np.random.default_rng(0).normal(0.5, 0.2, 10))
    large = pd.Series(np.random.default_rng(0).normal(0.5, 0.2, 1000))
    lo_s, hi_s = bootstrap_ci(small, n_resamples=2000, seed=1, ci_low_pct=2.5, ci_high_pct=97.5)
    lo_l, hi_l = bootstrap_ci(large, n_resamples=2000, seed=1, ci_low_pct=2.5, ci_high_pct=97.5)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_handles_nan_values_gracefully() -> None:
    # Only 1 of 5 slots is NaN -- P(an entire 5-draw resample is all-NaN) = 0.2^5
    # (~0.03%), negligible at 500 resamples, so every valid resample's mean is
    # exactly 1.0 and the CI collapses to a point at 1.0.
    values = pd.Series([1.0, 1.0, 1.0, 1.0, np.nan])
    lo, hi = bootstrap_ci(values, n_resamples=500, seed=1, ci_low_pct=2.5, ci_high_pct=97.5)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_empty_series_returns_zero() -> None:
    assert bootstrap_ci(pd.Series([], dtype=float), 100, 1, 2.5, 97.5) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Paired Wilcoxon: known-significant and known-null cases
# ---------------------------------------------------------------------------


def test_paired_wilcoxon_known_significant_difference() -> None:
    a = pd.Series([5.0, 6.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    b = pd.Series([1.0] * 10)
    result = paired_wilcoxon(a, b)
    assert result.p_value < 0.01
    assert result.n_pairs == 10


def test_paired_wilcoxon_known_null_case() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    b = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    result = paired_wilcoxon(a, b)
    assert result.p_value > 0.5


def test_paired_wilcoxon_all_tied_does_not_crash() -> None:
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([1.0, 2.0, 3.0])
    result = paired_wilcoxon(a, b)
    assert result.p_value == pytest.approx(1.0)
    assert result.statistic == pytest.approx(0.0)


def test_paired_wilcoxon_restricts_to_pairs_defined_in_both() -> None:
    a = pd.Series([1.0, 2.0, np.nan], index=["T1", "T2", "T3"])
    b = pd.Series([3.0, 1.0, 5.0], index=["T1", "T2", "T3"])
    result = paired_wilcoxon(a, b)
    assert result.n_pairs == 2
