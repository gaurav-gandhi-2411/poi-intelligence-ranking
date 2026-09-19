"""Trip x POI pair-frame assembly (the joins + engineered interaction features + graded label
behind every ranking frame). Lives in `features/` -- not `models/` -- so `candidates/` (which
may import `features/` but never `models/`) can build the SAME frame for its learned retriever
without duplicating the join logic. `models/ranking_data.py` re-exports every name here, so
existing importers are unchanged.
"""

from __future__ import annotations

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
