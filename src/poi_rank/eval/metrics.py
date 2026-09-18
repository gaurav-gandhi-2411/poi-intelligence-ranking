"""Ranking-quality metrics + evaluation orchestration (spec.md sections 8 / 11.1):
NDCG@{5,10,20}, Precision@{5,10}, Recall@{10,20}, MAP, MRR -- standard graded-relevance
ranking metrics computed per-trip then aggregated, with a 2,000-sample bootstrap 95%
CI over trips and paired Wilcoxon signed-rank tests vs the popularity baseline.

**Undefined-metric discipline, mirroring `candidates/recall_metrics.py`'s
`RecallResult`**: a trip whose candidate set contains ZERO ground-truth-relevant POIs
(`label >= 1`) has no ranking quality to measure for NDCG/Recall/MAP/MRR (IDCG == 0,
or zero relevant items to find) -- such trips are EXCLUDED from that metric's mean,
never silently coerced to 0 or 1, and the exclusion count is reported alongside every
aggregate. Precision@k is always defined (an empty top-k is trivially precision 0).
This is a REAL, expected occurrence here: the ~0.44 `candidate_recall@250` already
documented in docs/DATA_CARD.md means a meaningful share of holdout trips have none of
their true-relevant POIs inside their own candidate set at all -- every metric in this
module's table is bounded by that same candidate-set recall ceiling, reported honestly
per spec.md's own instruction, never hidden.

**"Systems"**: any `(trip_id, poi_id)`-aligned score column merged onto the same
evaluation frame (baselines 1-6, the oracle ceiling; later phases add LambdaMART/
LambdaMART+IPS without reworking this module -- see `run_evaluate`'s `scores` dict
parameter).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

FloatArray = npt.NDArray[np.float64]


# -----------------------------------------------------------------------------------
# Per-trip metric primitives (hand-verifiable against small textbook examples)
# -----------------------------------------------------------------------------------


def _rank_order(poi_ids: npt.NDArray[np.object_], scores: FloatArray) -> npt.NDArray[np.intp]:
    """Descending-score order, poi_id-ascending tie-break -- same `np.lexsort`
    determinism convention `candidates/channels.py` uses throughout."""
    order: npt.NDArray[np.intp] = np.lexsort((poi_ids, -scores))
    return order


def dcg_at_k(labels_in_rank_order: npt.NDArray[np.int64], k: int) -> float:
    """`sum_{i=1}^k (2^rel_i - 1) / log2(i + 1)` -- standard graded-gain DCG (matches
    LightGBM's `lambdarank` default label_gain convention, spec.md section 8), over
    `labels_in_rank_order` (already sorted by the ranking under evaluation)."""
    kk = min(k, len(labels_in_rank_order))
    if kk <= 0:
        return 0.0
    ranks = np.arange(1, kk + 1, dtype=np.float64)
    gains = np.power(2.0, labels_in_rank_order[:kk].astype(np.float64)) - 1.0
    discounts = np.log2(ranks + 1.0)
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    labels: npt.NDArray[np.int64], scores: FloatArray, poi_ids: npt.NDArray[np.object_], k: int
) -> float | None:
    """NDCG@k for one trip. Returns `None` (undefined) when the ideal ranking's
    DCG is 0 -- i.e. the trip's candidate set contains no relevant POI at all -- per
    module docstring."""
    order = _rank_order(poi_ids, scores)
    dcg = dcg_at_k(labels[order], k)
    ideal_order = np.argsort(-labels, kind="stable")
    idcg = dcg_at_k(labels[ideal_order], k)
    if idcg == 0.0:
        return None
    return dcg / idcg


def precision_at_k(
    labels: npt.NDArray[np.int64], scores: FloatArray, poi_ids: npt.NDArray[np.object_], k: int
) -> float:
    """Precision@k: fraction of the top-k ranked candidates with `label >= 1`.
    Always defined (a candidate set is never empty for a trip present in the frame)."""
    order = _rank_order(poi_ids, scores)
    ranked = labels[order]
    kk = min(k, len(ranked))
    if kk == 0:
        return 0.0
    return float((ranked[:kk] >= 1).sum() / kk)


def recall_at_k(
    labels: npt.NDArray[np.int64], scores: FloatArray, poi_ids: npt.NDArray[np.object_], k: int
) -> float | None:
    """Recall@k: fraction of the candidate set's `label >= 1` POIs found in the
    top-k. `None` if the candidate set has zero relevant POIs (undefined, per module
    docstring -- distinct from `candidates/recall_metrics.py`'s `candidate_recall@250`,
    which measures relevant-POI coverage of the CANDIDATE SET against the full
    catalog, not ranking quality WITHIN an already-fixed candidate set)."""
    n_relevant = int((labels >= 1).sum())
    if n_relevant == 0:
        return None
    order = _rank_order(poi_ids, scores)
    ranked = labels[order]
    kk = min(k, len(ranked))
    hit = int((ranked[:kk] >= 1).sum())
    return hit / n_relevant


def average_precision(
    labels: npt.NDArray[np.int64], scores: FloatArray, poi_ids: npt.NDArray[np.object_]
) -> float | None:
    """AP for one trip (mean of precision@i at every rank `i` holding a relevant
    POI, divided by the total number of relevant POIs). `None` if zero relevant."""
    n_relevant = int((labels >= 1).sum())
    if n_relevant == 0:
        return None
    order = _rank_order(poi_ids, scores)
    ranked = labels[order]
    hits = (ranked >= 1).astype(np.float64)
    cum_hits = np.cumsum(hits)
    precision_at_each_rank = cum_hits / (np.arange(len(ranked), dtype=np.float64) + 1.0)
    return float(np.sum(precision_at_each_rank * hits) / n_relevant)


def reciprocal_rank(
    labels: npt.NDArray[np.int64], scores: FloatArray, poi_ids: npt.NDArray[np.object_]
) -> float | None:
    """`1 / rank_of_first_relevant_hit` for one trip. `None` if zero relevant."""
    n_relevant = int((labels >= 1).sum())
    if n_relevant == 0:
        return None
    order = _rank_order(poi_ids, scores)
    ranked = labels[order]
    hit_positions = np.nonzero(ranked >= 1)[0]
    if len(hit_positions) == 0:
        return 0.0
    return float(1.0 / (hit_positions[0] + 1))


# -----------------------------------------------------------------------------------
# Per-trip metric table + aggregation
# -----------------------------------------------------------------------------------


def compute_all_trip_metrics(
    frame: pd.DataFrame,
    score: pd.Series,
    ndcg_ks: tuple[int, ...],
    precision_ks: tuple[int, ...],
    recall_ks: tuple[int, ...],
) -> pd.DataFrame:
    """One row per `trip_id` with every per-trip metric value (`NaN` where
    undefined -- see module docstring); `score` must be aligned to `frame.index`
    (a `models.baselines.BaselineResult.score`, or the oracle-ceiling score)."""
    working = frame[["trip_id", "poi_id", "label"]].copy()
    working["score"] = score.to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for trip_id, group in working.groupby("trip_id", sort=True):
        labels = group["label"].to_numpy(dtype=np.int64)
        scores = group["score"].to_numpy(dtype=np.float64)
        poi_ids = group["poi_id"].to_numpy(dtype=object)
        row: dict[str, Any] = {"trip_id": trip_id}
        for k in ndcg_ks:
            row[f"ndcg@{k}"] = ndcg_at_k(labels, scores, poi_ids, k)
        for k in precision_ks:
            row[f"precision@{k}"] = precision_at_k(labels, scores, poi_ids, k)
        for k in recall_ks:
            row[f"recall@{k}"] = recall_at_k(labels, scores, poi_ids, k)
        row["map"] = average_precision(labels, scores, poi_ids)
        row["mrr"] = reciprocal_rank(labels, scores, poi_ids)
        rows.append(row)
    return pd.DataFrame(rows).set_index("trip_id")


@dataclass(frozen=True)
class MetricAggregate:
    """Mean + bootstrap 95% CI for one metric, one system -- plus how many trips were
    included/excluded from the mean (mirrors `candidates/recall_metrics.RecallResult`
    -- never silently drops the exclusion count)."""

    mean: float
    n_included: int
    n_excluded: int
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "n_included": self.n_included,
            "n_excluded": self.n_excluded,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def bootstrap_ci(
    per_trip_values: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> tuple[float, float]:
    """Bootstrap 95% CI over trips (spec.md section 8: "2,000-sample bootstrap"):
    resamples the SET OF TRIPS with replacement, recomputes the mean each resample
    (excluding undefined/`NaN` trips from that resample's mean, exactly mirroring the
    point estimate's own exclusion rule -- not resampling only from the already-valid
    subset, so resample-to-resample variation in which trips happen to be
    excluded/included is itself reflected in the CI width), and reports the
    `[ci_low_pct, ci_high_pct]` percentiles of the resampled-mean distribution.
    Deterministic given `seed`.
    """
    values = per_trip_values.to_numpy(dtype=np.float64)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = values[idx]
    # An all-NaN resample row (every drawn trip undefined for this metric) makes
    # `np.nanmean` both raise `RuntimeWarning: Mean of empty slice` (a warnings-module
    # emission, not a floating-point exception -- `np.errstate` alone does not
    # suppress it) and return NaN, which `np.nan_to_num` below intentionally maps to
    # 0.0 (the same fallback the point estimate uses for "zero valid trips").
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(samples, axis=1)
    means = np.nan_to_num(means, nan=0.0)
    lo, hi = np.percentile(means, [ci_low_pct, ci_high_pct])
    return float(lo), float(hi)


def aggregate_metric(
    per_trip_values: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> MetricAggregate:
    """Mean (excluding `NaN`/undefined trips) + bootstrap 95% CI for one metric."""
    valid = per_trip_values.dropna()
    n_excluded = int(per_trip_values.isna().sum())
    mean = float(valid.mean()) if len(valid) else 0.0
    ci_low, ci_high = bootstrap_ci(per_trip_values, n_resamples, seed, ci_low_pct, ci_high_pct)
    return MetricAggregate(
        mean=mean, n_included=len(valid), n_excluded=n_excluded, ci_low=ci_low, ci_high=ci_high
    )


# -----------------------------------------------------------------------------------
# Paired Wilcoxon signed-rank test
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_pairs: int


def paired_wilcoxon(values_a: pd.Series, values_b: pd.Series) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test between two systems' per-trip metric values
    (matched by the shared `trip_id` index), restricted to trips where BOTH systems'
    value is defined. `scipy.stats.wilcoxon` raises on an all-zero-difference input
    (every pair tied, e.g. two identical baselines on a toy fixture) -- surfaced here
    as `(statistic=0.0, p_value=1.0)` rather than propagating the exception, so the
    harness never crashes on a legitimate null-difference comparison.
    """
    aligned = pd.concat([values_a.rename("a"), values_b.rename("b")], axis=1).dropna()
    if len(aligned) == 0 or (aligned["a"] == aligned["b"]).all():
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_pairs=len(aligned))
    stat, p_value = stats.wilcoxon(aligned["a"], aligned["b"])
    return WilcoxonResult(statistic=float(stat), p_value=float(p_value), n_pairs=len(aligned))


# -----------------------------------------------------------------------------------
# Full-table orchestration
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemMetrics:
    """Full metric table for one system: every per-metric `MetricAggregate` plus
    that system's own diagnostics dict (e.g. `BaselineResult.diagnostics`)."""

    metrics: dict[str, MetricAggregate]
    diagnostics: dict[str, Any]
    per_trip: pd.DataFrame  # metric_name -> per-trip Series (used for Wilcoxon pairing)


def evaluate_system(
    frame: pd.DataFrame,
    score: pd.Series,
    ndcg_ks: tuple[int, ...],
    precision_ks: tuple[int, ...],
    recall_ks: tuple[int, ...],
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
    diagnostics: dict[str, Any] | None = None,
) -> SystemMetrics:
    """Compute the full metric table for one system (one score column) against one
    evaluation frame."""
    per_trip = compute_all_trip_metrics(frame, score, ndcg_ks, precision_ks, recall_ks)
    metrics = {
        col: aggregate_metric(per_trip[col], n_resamples, seed, ci_low_pct, ci_high_pct)
        for col in per_trip.columns
    }
    return SystemMetrics(metrics=metrics, diagnostics=diagnostics or {}, per_trip=per_trip)
