"""New-POI cohort identification + evaluation (spec.md section 8: "New-POI
robustness ... Measured on the new-POI cohort").

A "new POI" is any catalog POI whose `created_at` falls inside the holdout window
(`datagen/catalog.py`'s `new_poi_rate` dirtiness injection, ~5% of the catalog,
`docs/DATA_CARD.md`'s dirtiness table: "zero train interactions by construction"):
`created_at >= split`, the SAME global train/holdout timeline boundary
(`datagen/timeline.py::build_timeline`) every trip's `is_holdout` flag already
derives from -- recomputed here directly from `configs/datagen.yaml`, not re-derived
from any trip-level heuristic, since `build_timeline` is the single source of truth
for this boundary.

**Why reading `datagen/` here does not violate the `models/` firewall's purpose**
(spec.md section 1.1): the firewall exists to stop the ranking model from cheating
using latent DGP information (true utility, archetype identity) it should never
observe. `timeline.split`/`created_at` are neither -- they are the PUBLIC,
already-observable train/holdout boundary every downstream phase's `is_holdout`
column already encodes; `pois_prepared.parquet`'s `created_at` column is itself an
exported, observable field every phase already reads. `eval/run.py` already reads
`datagen.oracle_export` for the same documented, non-cheating reason (locating the
oracle-only export directory for evaluation purposes, not for training). This module
lives in `eval/`, not `models/`, precisely so `models/lambdamart.py`'s own firewall
scan never has to reason about this distinction at all.
"""

from __future__ import annotations

import pandas as pd

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.timeline import build_timeline
from poi_rank.eval.metrics import MetricAggregate, aggregate_metric, compute_all_trip_metrics


def new_poi_ids(pois_df: pd.DataFrame, datagen_cfg: DatagenConfig) -> set[str]:
    """The set of `poi_id`s whose `created_at` falls at/after the train/holdout
    timeline split (module docstring) -- the ~5% "new POI" cohort."""
    timeline = build_timeline(datagen_cfg)
    return set(pois_df.loc[pois_df["created_at"] >= timeline.split, "poi_id"])


def cohort_frame(holdout_frame: pd.DataFrame, new_poi_id_set: set[str]) -> pd.DataFrame:
    """The holdout evaluation frame restricted to new-POI candidate rows, index
    PRESERVED (not reset) so a `(trip_id, poi_id)`-aligned score `pd.Series` can be
    subset with the identical boolean mask and stay row-order-aligned with this
    frame for `eval.metrics.compute_all_trip_metrics`."""
    return holdout_frame.loc[holdout_frame["poi_id"].isin(new_poi_id_set)]


def evaluate_cohort_ndcg10_per_trip(cohort_frame_df: pd.DataFrame, score: pd.Series) -> pd.Series:
    """Per-trip NDCG@10 (`eval.metrics.compute_all_trip_metrics`, restricted to the
    cohort's own candidate rows) -- the shared building block for both the point
    estimate/CI (`evaluate_cohort_ndcg10`) and the dropout-ablation paired Wilcoxon
    test (`eval/run.py`)."""
    per_trip = compute_all_trip_metrics(
        cohort_frame_df, score, ndcg_ks=(10,), precision_ks=(), recall_ks=()
    )
    return per_trip["ndcg@10"]


def evaluate_cohort_ndcg10(
    cohort_frame_df: pd.DataFrame,
    score: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> MetricAggregate:
    """NDCG@10 (mean + bootstrap CI), computed with the EXACT SAME
    `eval.metrics` functions every other system uses, restricted to the new-POI
    cohort's own candidate rows per trip (a trip with zero new-POI candidates
    contributes nothing; a trip with new-POI candidates but none `label >= 1` is
    excluded from the mean per `eval.metrics`' own undefined-metric discipline --
    see that module's docstring). `score` must be index-aligned to
    `cohort_frame_df` (i.e. the SAME score Series used for the full holdout table,
    subset with `cohort_frame_df.index`, or a score computed directly against
    `cohort_frame_df`)."""
    per_trip = evaluate_cohort_ndcg10_per_trip(cohort_frame_df, score)
    return aggregate_metric(per_trip, n_resamples, seed, ci_low_pct, ci_high_pct)


def cohort_summary(
    pois_df: pd.DataFrame, holdout_frame: pd.DataFrame, datagen_cfg: DatagenConfig
) -> dict[str, int]:
    """Cohort-size diagnostics reported alongside the NDCG comparison (module
    docstring's "report that honestly rather than silently skipping" instruction) --
    computed independent of any score, so it is meaningful even if the cohort turns
    out to be empty or degenerately small."""
    ids = new_poi_ids(pois_df, datagen_cfg)
    cf = cohort_frame(holdout_frame, ids)
    n_relevant_by_trip = cf.groupby("trip_id")["label"].apply(lambda s: int((s >= 1).sum()))
    return {
        "n_new_pois_in_catalog": len(ids),
        "n_holdout_rows_in_cohort": int(len(cf)),
        "n_holdout_trips_with_cohort_candidate": int(cf["trip_id"].nunique()),
        "n_holdout_trips_with_relevant_cohort_candidate": int((n_relevant_by_trip >= 1).sum()),
    }
