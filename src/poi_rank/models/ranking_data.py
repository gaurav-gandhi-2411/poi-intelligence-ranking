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
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.features.traveler_features import (
    cosine_similarity_taste_poi,
    interest_match_score,
    localness_preference_gap,
    price_gap,
)

FloatArray = npt.NDArray[np.float64]

TASTE_PREFIX = "implicit_taste_"
TEXT_EMB_PREFIX = "text_emb_"

CANDIDATES_FILENAME = "candidates.parquet"
POIS_PREPARED_FILENAME = "pois_prepared.parquet"
POI_FEATURES_FILENAME = "poi_features.parquet"
TRAVELER_FEATURES_FILENAME = "traveler_features.parquet"
TRAVELERS_FILENAME = "travelers.parquet"
TRIPS_FILENAME = "trips.parquet"
INTERACTIONS_TRAIN_FILENAME = "interactions_train.parquet"
INTERACTIONS_HOLDOUT_RANDOM_FILENAME = "interactions_holdout_random.parquet"


def label_by_trip_poi(interactions: pd.DataFrame, canonical_map: dict[str, str]) -> pd.Series:
    """`{(trip_id, poi_id): label}` -- canonical-remaps `interactions` through
    `canonical_map` (see `features/reconcile.py`'s module docstring: interaction logs
    reference the raw, pre-dedup catalog) then takes the MAX label across every
    interaction row for that `(trip_id, poi_id)` pair (a candidate can be exposed and
    interacted with more than once within a trip -- e.g. `view` then later `click` --
    the strongest observed signal is the correct graded-relevance label, not the
    first or last row). Returned as a `pd.Series` indexed by a `(trip_id, poi_id)`
    `MultiIndex`.
    """
    remapped = remap_interaction_poi_ids(interactions, canonical_map)
    return remapped.groupby(["trip_id", "poi_id"])["label"].max()


def add_interaction_features(
    frame: pd.DataFrame, budget_target_price_level: BudgetTargetPriceLevel
) -> pd.DataFrame:
    """Add the 4 traveler x POI interaction features spec.md section 6 names
    explicitly, each reused directly from `features.traveler_features` (never
    reimplemented): `interact_cos_taste_poi`, `interact_localness_gap`,
    `interact_interest_match`, `interact_price_gap`.

    **`interact_localness_gap` reading**: spec.md's literal formula is
    `|implicit_localness - touristiness_pref|`. Read here as the CANDIDATE POI's own
    observable `localness` (`num_localness`, varies per candidate) against the
    traveler's stated `touristiness_pref` (constant per trip) -- the natural
    per-(trip, candidate) interaction reading for a feature meant to answer "does
    this specific POI's localness match what this traveler says they want", as
    opposed to the traveler-level `implicit_mean_localness` aggregate (which is
    already a traveler-only feature the traveler feature table exports directly,
    with no candidate-POI dependence and so nothing new to interact against). See
    docs/DATA_CARD.md.
    """
    out = frame.copy()

    taste_cols = sorted(c for c in out.columns if c.startswith(TASTE_PREFIX))
    emb_cols = sorted(c for c in out.columns if c.startswith(TEXT_EMB_PREFIX))
    taste = out[taste_cols].to_numpy(dtype=np.float64)
    embs = out[emb_cols].to_numpy(dtype=np.float64)
    out["interact_cos_taste_poi"] = cosine_similarity_taste_poi(taste, embs)

    out["interact_localness_gap"] = localness_preference_gap(
        out["num_localness"].to_numpy(dtype=np.float64),
        out["explicit_touristiness_pref"].to_numpy(dtype=np.float64),
    )

    out["interact_interest_match"] = interest_match_score(
        [set(x) for x in out["interests"]],
        out["poi_category_raw"].astype(str).tolist(),
        [list(x) for x in out["poi_tags_raw"]],
    )

    out["interact_price_gap"] = price_gap(
        out["budget"].astype(str).tolist(),
        out["num_price_level"].to_numpy(dtype=np.float64),
        budget_target_price_level,
    )
    return out


def build_ranking_frame(
    candidates_df: pd.DataFrame,
    trip_ids: set[str],
    interactions: pd.DataFrame,
    trips_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    poi_features_df: pd.DataFrame,
    traveler_features_df: pd.DataFrame,
    budget_target_price_level: BudgetTargetPriceLevel,
) -> pd.DataFrame:
    """Assemble one ranking frame: one row per `(trip_id, poi_id)` candidate pair for
    every trip in `trip_ids`, joined with POI features, traveler features, trip/
    traveler context, engineered interaction features, and a graded 0-3 `label`
    (0 for any candidate with no matching row in `interactions`, per module
    docstring).
    """
    cand = candidates_df.loc[candidates_df["trip_id"].isin(trip_ids)].reset_index(drop=True)

    trip_ctx = trips_df[["trip_id", "traveler_id", "destination", "stay_lat", "stay_lon"]]
    frame = cand.merge(trip_ctx, on="trip_id", how="left")

    traveler_ctx = travelers_df[["traveler_id", "mobility", "budget", "interests"]]
    frame = frame.merge(traveler_ctx, on="traveler_id", how="left")

    poi_ctx = pois_df[["poi_id", "category", "tags"]].rename(
        columns={"category": "poi_category_raw", "tags": "poi_tags_raw"}
    )
    frame = frame.merge(poi_ctx, on="poi_id", how="left")

    pf = poi_features_df.drop(columns=["destination"])
    frame = frame.merge(pf, on="poi_id", how="left")

    frame = frame.merge(traveler_features_df, on=["traveler_id", "trip_id"], how="left")

    canonical_map = build_poi_id_canonical_map(pois_df)
    labels = label_by_trip_poi(interactions, canonical_map)
    label_df = labels.rename("label").reset_index()
    frame = frame.merge(label_df, on=["trip_id", "poi_id"], how="left")
    frame["label"] = frame["label"].fillna(0).astype(np.int64)

    frame = add_interaction_features(frame, budget_target_price_level)

    return frame.sort_values(["trip_id", "poi_id"]).reset_index(drop=True)


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
