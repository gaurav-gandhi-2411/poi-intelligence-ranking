"""Ranking-data assembly (spec.md section 8): builds the ONE reusable "evaluation
frame" -- a `(trip_id, poi_id)`-keyed DataFrame joining a trip's candidate set
(`candidates.parquet`) with its POI features (`poi_features.parquet`), the traveler's
features (`traveler_features.parquet`), trip/traveler context needed by specific
baselines (stay point, mobility, budget, stated interests), a handful of engineered
traveler x POI interaction features (spec.md section 6, reused directly from
`features.traveler_features`, never reimplemented), and a graded 0-3 relevance label.

**Two frames, same assembly function, different label source and trip population**
(spec.md section 8):
  - `load_holdout_evaluation_frame` -- PRIMARY evaluation frame (spec.md section
    11.1): holdout trips only (`trips_df.is_holdout == True`), labels sourced from
    `interactions_holdout_random.parquet` (unbiased random-exposure holdout).
  - `load_train_ranking_frame` -- train-side frame for baselines that need training
    (logistic regression, and later LambdaMART/IPS): train trips only
    (`trips_df.is_holdout == False`), labels sourced from `interactions_train.parquet`
    (popularity-biased logging policy).

Both reuse the exact same `build_ranking_frame` assembly and the exact same
temporal-leakage-safe machinery Phase 3/4a already established
(`features.reconcile.build_poi_id_canonical_map`/`remap_interaction_poi_ids` for the
label join; `features.traveler_features.traveler_history_before` indirectly via
`candidates.union.cf_seed_poi_ids` is NOT used here -- that is baseline 5's own
concern in `models/baselines.py`, kept out of the shared frame to avoid computing a
CF-specific artifact for every baseline that doesn't need it).

**A candidate POI with no matching holdout/train interaction row for its trip gets
label 0** -- for the PRIMARY (holdout) frame this is a true, not assumed, negative:
`interactions_holdout_random.parquet` is a uniform-random-exposure log, so "no
interaction logged" means the traveler was (with uniform probability across the
catalog) never shown this candidate at all, OR was shown it and only ever produced a
`view` row (already label 0) -- either way, label 0 is the correct, non-circular
reading, not a negative-sampling assumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.features.pair_frame import (
    TASTE_PREFIX,
    TEXT_EMB_PREFIX,
    add_interaction_features,
    build_ranking_frame,
    label_by_trip_poi,
)

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "TASTE_PREFIX",
    "TEXT_EMB_PREFIX",
    "add_interaction_features",
    "build_ranking_frame",
    "label_by_trip_poi",
]

CANDIDATES_FILENAME = "candidates.parquet"
POIS_PREPARED_FILENAME = "pois_prepared.parquet"
POI_FEATURES_FILENAME = "poi_features.parquet"
TRAVELER_FEATURES_FILENAME = "traveler_features.parquet"
TRAVELERS_FILENAME = "travelers.parquet"
TRIPS_FILENAME = "trips.parquet"
INTERACTIONS_TRAIN_FILENAME = "interactions_train.parquet"
INTERACTIONS_HOLDOUT_RANDOM_FILENAME = "interactions_holdout_random.parquet"
INTERACTIONS_HOLDOUT_LOGGED_FILENAME = "interactions_holdout_logged.parquet"


def _load_common(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "candidates_df": pd.read_parquet(data_dir / CANDIDATES_FILENAME),
        "pois_df": pd.read_parquet(data_dir / POIS_PREPARED_FILENAME),
        "poi_features_df": pd.read_parquet(data_dir / POI_FEATURES_FILENAME),
        "traveler_features_df": pd.read_parquet(data_dir / TRAVELER_FEATURES_FILENAME),
        "travelers_df": pd.read_parquet(data_dir / TRAVELERS_FILENAME),
        "trips_df": pd.read_parquet(data_dir / TRIPS_FILENAME),
    }


def load_holdout_evaluation_frame(
    data_dir: Path, budget_target_price_level: BudgetTargetPriceLevel
) -> pd.DataFrame:
    """PRIMARY evaluation frame (spec.md section 11.1): holdout trips only, labels
    from `interactions_holdout_random.parquet`."""
    d = _load_common(data_dir)
    holdout_random = pd.read_parquet(data_dir / INTERACTIONS_HOLDOUT_RANDOM_FILENAME)
    holdout_trip_ids = set(d["trips_df"].loc[d["trips_df"]["is_holdout"], "trip_id"])
    return build_ranking_frame(
        d["candidates_df"],
        holdout_trip_ids,
        holdout_random,
        d["trips_df"],
        d["travelers_df"],
        d["pois_df"],
        d["poi_features_df"],
        d["traveler_features_df"],
        budget_target_price_level,
    )


def load_holdout_biased_evaluation_frame(
    data_dir: Path, budget_target_price_level: BudgetTargetPriceLevel
) -> pd.DataFrame:
    """SECONDARY evaluation frame (spec.md section 11.1's bias-gap table): the SAME
    holdout trips, SAME candidate universe, SAME feature columns as
    `load_holdout_evaluation_frame` -- the only difference is the label source,
    `interactions_holdout_logged.parquet` (the popularity-biased logging policy's
    LATER time window, spec.md section 1.3) instead of
    `interactions_holdout_random.parquet`. Because `build_ranking_frame`'s row
    order/index depend only on `candidates_df`/`trip_ids`/the feature-table joins
    (never on `interactions`), this frame is row-for-row `(trip_id, poi_id)`-aligned
    with `load_holdout_evaluation_frame`'s own output -- an already-computed score
    `pd.Series` for the unbiased frame can be reused directly against this frame's
    `label` column without rescoring, which is exactly how `eval/run.py` computes
    the bias-gap table cheaply (`tests/test_ranking_data.py` asserts this row-order
    invariant directly, not just relied upon implicitly).
    """
    d = _load_common(data_dir)
    holdout_logged = pd.read_parquet(data_dir / INTERACTIONS_HOLDOUT_LOGGED_FILENAME)
    holdout_trip_ids = set(d["trips_df"].loc[d["trips_df"]["is_holdout"], "trip_id"])
    return build_ranking_frame(
        d["candidates_df"],
        holdout_trip_ids,
        holdout_logged,
        d["trips_df"],
        d["travelers_df"],
        d["pois_df"],
        d["poi_features_df"],
        d["traveler_features_df"],
        budget_target_price_level,
    )


def load_train_ranking_frame(
    data_dir: Path, budget_target_price_level: BudgetTargetPriceLevel
) -> pd.DataFrame:
    """Train-side ranking frame for baselines that need training (logistic
    regression; later phases' LambdaMART/IPS): train trips only, labels from
    `interactions_train.parquet`."""
    d = _load_common(data_dir)
    interactions_train = pd.read_parquet(data_dir / INTERACTIONS_TRAIN_FILENAME)
    train_trip_ids = set(d["trips_df"].loc[~d["trips_df"]["is_holdout"], "trip_id"])
    return build_ranking_frame(
        d["candidates_df"],
        train_trip_ids,
        interactions_train,
        d["trips_df"],
        d["travelers_df"],
        d["pois_df"],
        d["poi_features_df"],
        d["traveler_features_df"],
        budget_target_price_level,
    )
