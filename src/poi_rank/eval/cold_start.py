"""Cold-start cohort evaluation (spec.md sections 11.8 / 12): NDCG@10 by traveler
interaction-count bucket (0 / 1-3 / 4-10 / >10), plus leave-one-destination-out
(LODO) -- train on 2 destinations, evaluate on the 3rd, for all 3 destinations,
turning "new destination transfers" from prose into a measured result.

New-POI cohort NDCG@10 is ALREADY measured by `eval/new_poi_cohort.py` (Phase 5) --
this module does not duplicate it; `eval/run.py` reports both under the same
`"cold_start"` payload section.

**LODO is deliberately NOT part of `poi_rank.cli evaluate` / `make reproduce`'s
default chain** (spec.md section 16's own cut-order list: "cut first ... LODO" if
time-constrained; this phase keeps it, but isolated): 3 extra full LightGBM
retrains, one full training pass each, measurably expensive relative to the
project's <5 min full-pipeline budget. `poi_rank.cli lodo` is its own command, run
AFTER `poi_rank.cli evaluate` -- it reads the existing `results/metrics.json`,
merges in a `"lodo"` section, and rewrites the same file, so `eval/report.py`'s
"every number comes from `results/metrics.json`" rule still holds for LODO numbers
without forcing the expensive retrain into the default reproducibility chain. See
`docs/DATA_CARD.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import lightgbm as lgb
import pandas as pd

from poi_rank.eval.metrics import (
    aggregate_metric,
    compute_all_trip_metrics,
    paired_wilcoxon,
)
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import baselines as bl
from poi_rank.models import lambdamart as lm
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame

INTERACTIONS_TRAIN_FILENAME = "interactions_train.parquet"
POIS_PREPARED_FILENAME = "pois_prepared.parquet"

BUCKET_ORDER: tuple[str, ...] = ("0", "1-3", "4-10", ">10")


def bucket_for_count(n: float) -> str:
    """0 / 1-3 / 4-10 / >10 (spec.md section 11.8, verbatim)."""
    if n <= 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 10:
        return "4-10"
    return ">10"


def trip_interaction_count(frame: pd.DataFrame) -> dict[str, float]:
    """One `implicit_interaction_count` value per `trip_id` -- a traveler-level,
    as-of-safe feature (`features/traveler_features.py`) constant within a trip's
    candidate rows, so the first row per group is exact, not an approximation."""
    return dict(frame.groupby("trip_id")["implicit_interaction_count"].first())


def ndcg10_by_interaction_bucket(
    frame: pd.DataFrame,
    score: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> dict[str, dict[str, Any]]:
    """NDCG@10 (mean + bootstrap CI, `eval.metrics`'s own machinery) per interaction-
    count bucket. A bucket with zero trips reports a `MetricAggregate` of all-zero/
    empty fields (never silently omitted from the returned dict -- every bucket in
    `BUCKET_ORDER` is always a key)."""
    counts = trip_interaction_count(frame)
    bucket_by_trip = {tid: bucket_for_count(c) for tid, c in counts.items()}
    per_trip = compute_all_trip_metrics(frame, score, ndcg_ks=(10,), precision_ks=(), recall_ks=())

    out: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_ORDER:
        trip_ids_in_bucket = [t for t, b in bucket_by_trip.items() if b == bucket]
        subset = per_trip.loc[per_trip.index.isin(trip_ids_in_bucket), "ndcg@10"]
        agg = aggregate_metric(subset, n_resamples, seed, ci_low_pct, ci_high_pct)
        result = agg.to_dict()
        result["n_trips_in_bucket"] = len(trip_ids_in_bucket)
        out[bucket] = result
    return out


# -----------------------------------------------------------------------------------
# Leave-one-destination-out (LODO)
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LodoTrainResult:
    booster: lgb.Booster
    numeric_columns: list[str]
    categorical_columns: list[str]
    n_fit_rows: int
    n_fit_trips: int


def train_lodo_booster(
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    model_cfg: ModelConfig,
    held_out_destination: str,
) -> LodoTrainResult:
    """Train ONE LambdaMART+IPS booster (system 8's exact recipe: IPS weighting +
    15% behavioral dropout on the fit split, module docstring) on `train_frame`
    with every row belonging to `held_out_destination` excluded from FIT entirely
    -- the held-out destination's own trips never appear in `fit_frame`/`val_frame`
    at all, only later in the SEPARATE evaluation step (`eval/run.py`, restricting
    the PRIMARY holdout frame to that destination). This is the exact filtering
    spec.md section 11.8 asks for: "train on 2 destinations, evaluate on the 3rd."
    """
    lm_cfg = model_cfg.lambdamart
    filtered = train_frame.loc[train_frame["destination"] != held_out_destination].reset_index(
        drop=True
    )
    numeric_columns = bl.numeric_feature_columns(filtered)
    categorical_columns = bl.categorical_feature_columns(filtered)

    fit_frame, val_frame = lm.train_val_split_by_trip(
        filtered, lm_cfg.val_fraction, lm_cfg.val_split_seed
    )
    p_expose_fit = lm.train_frame_p_expose(fit_frame, interactions_train, pois_df)
    ips_weight_fit = lm.compute_ips_weights(
        fit_frame["trip_id"], p_expose_fit, lm_cfg.ips_clip_low, lm_cfg.ips_clip_high
    )
    fit_frame_dropout, _mask = lm.apply_behavioral_dropout(
        fit_frame, lm_cfg.behavioral_dropout_rate, lm_cfg.behavioral_dropout_seed
    )
    booster = lm.fit_lambdamart_booster(
        fit_frame_dropout,
        val_frame,
        numeric_columns,
        categorical_columns,
        lm_cfg,
        model_cfg.seed,
        sample_weight=ips_weight_fit,
    )
    return LodoTrainResult(
        booster=booster,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        n_fit_rows=int(len(fit_frame)),
        n_fit_trips=int(fit_frame["trip_id"].nunique()),
    )


def evaluate_lodo_destination(
    holdout_frame: pd.DataFrame,
    full_training_score: pd.Series,
    lodo_result: LodoTrainResult,
    destination: str,
    eval_seed: int,
    n_resamples: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> dict[str, Any]:
    """Score the LODO booster against `destination`'s own holdout rows only, compare
    NDCG@10 (mean + CI + paired Wilcoxon) against the full-training system 8 baseline
    restricted to the SAME rows."""
    dest_frame = holdout_frame.loc[holdout_frame["destination"] == destination]
    lodo_score = lm.score_booster(
        lodo_result.booster,
        dest_frame,
        lodo_result.numeric_columns,
        lodo_result.categorical_columns,
    )
    full_score = full_training_score.loc[dest_frame.index]

    lodo_per_trip = compute_all_trip_metrics(dest_frame, lodo_score, (10,), (), ())["ndcg@10"]
    full_per_trip = compute_all_trip_metrics(dest_frame, full_score, (10,), (), ())["ndcg@10"]

    wilcoxon = paired_wilcoxon(lodo_per_trip, full_per_trip)
    return {
        "destination": destination,
        "n_holdout_trips": int(dest_frame["trip_id"].nunique()),
        "n_fit_rows": lodo_result.n_fit_rows,
        "n_fit_trips": lodo_result.n_fit_trips,
        "ndcg@10_lodo": aggregate_metric(
            lodo_per_trip, n_resamples, eval_seed, ci_low_pct, ci_high_pct
        ).to_dict(),
        "ndcg@10_full_training": aggregate_metric(
            full_per_trip, n_resamples, eval_seed, ci_low_pct, ci_high_pct
        ).to_dict(),
        "wilcoxon_lodo_vs_full_training": {
            "statistic": wilcoxon.statistic,
            "p_value": wilcoxon.p_value,
            "n_pairs": wilcoxon.n_pairs,
        },
    }


def run_lodo(
    data_dir: Path,
    artifacts_dir: Path,
    model_cfg: ModelConfig,
    feature_cfg: FeatureBuildConfig,
    eval_seed: int,
    n_resamples: int,
    ci_low_pct: float,
    ci_high_pct: float,
    destinations: tuple[str, ...],
) -> dict[str, Any]:
    """`poi_rank.cli lodo` entry point (module docstring): trains 3 destination-
    held-out boosters (one full LightGBM training pass each -- genuinely expensive,
    wall-clock reported), evaluates each against its own destination's holdout
    trips, and compares against the already-trained, full-training system 8
    (`artifacts/model.txt`, LOADED, never retrained here)."""
    start = perf_counter()
    budget_target_price_level = feature_cfg.traveler_features.budget_target_price_level
    train_frame = load_train_ranking_frame(data_dir, budget_target_price_level)
    holdout_frame = load_holdout_evaluation_frame(data_dir, budget_target_price_level)
    interactions_train = pd.read_parquet(data_dir / INTERACTIONS_TRAIN_FILENAME)
    pois_df = pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)

    full_training_scores = lm.load_and_score_holdout(artifacts_dir, holdout_frame)
    full_training_score = full_training_scores["lambdamart_ips"]

    per_destination: list[dict[str, Any]] = []
    for destination in destinations:
        lodo_result = train_lodo_booster(
            train_frame, interactions_train, pois_df, model_cfg, destination
        )
        per_destination.append(
            evaluate_lodo_destination(
                holdout_frame,
                full_training_score,
                lodo_result,
                destination,
                eval_seed,
                n_resamples,
                ci_low_pct,
                ci_high_pct,
            )
        )
    wall_clock_seconds = perf_counter() - start

    return {
        "per_destination": per_destination,
        "wall_clock_seconds": wall_clock_seconds,
        "n_destinations": len(destinations),
        "note": (
            "LODO is a SEPARATE command (`poi_rank.cli lodo`), not part of `make "
            "reproduce`'s default chain -- 3 full LightGBM retrains (module "
            "docstring). Run after `poi_rank.cli evaluate`; merges a 'lodo' key "
            "into the existing results/metrics.json rather than writing a "
            "separate file, so eval/report.py's 'every number from metrics.json' "
            "rule still holds for these numbers."
        ),
    }
