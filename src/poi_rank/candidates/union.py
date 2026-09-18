"""Orchestrates candidate generation end to end (spec.md section 7): loads
`pois_prepared.parquet`, `travelers.parquet`, `trips.parquet`, `poi_features.parquet`,
`traveler_features.parquet`, `interactions_train.parquet`; builds the per-destination
`DestinationIndex` precomputation and the global item-item CF matrix once; runs all 6
channels for every trip (train AND holdout -- holdout trips need candidate sets too,
for `candidates/recall_metrics.py`'s evaluation); writes
`data/synthetic/candidates.parquet`.

**Output schema**: one row per `(trip_id, poi_id)` that appears in >= 1 channel's own
selection, plus one boolean membership column per channel
(`channels.CHANNEL_NAMES`) -- a POI landing in multiple channels' own quota-respecting
selections counts once in the union but keeps every channel's membership flag, so
downstream listwise-group construction AND the leave-one-channel-out marginal-recall
analysis (`candidates/recall_metrics.py`) can both be computed post-hoc from this one
artifact, never by regenerating 6 separate channel runs.

Deterministic: trips are processed in `trip_id`-sorted order, the output is
re-sorted by `(trip_id, poi_id)` before writing, and the only stochastic step (the
long-tail channel's epsilon-greedy draw) is seeded per-trip via
`channels.trip_seed` -- a SHA256 digest of `(seed, trip_id)`, independent of
`PYTHONHASHSEED` or iteration order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.candidates.channels import (
    CHANNEL_NAMES,
    build_destination_indices,
    build_item_item_cf,
    channel_archetype,
    channel_cf,
    channel_geo,
    channel_interest,
    channel_longtail,
    channel_semantic,
    compute_semantic_similarity,
    trip_seed,
)
from poi_rank.candidates.config import CandidatesConfig
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.features.traveler_features import (
    assign_traveler_segments,
    group_interactions_by_traveler,
    traveler_history_before,
)

CANDIDATES_FILENAME = "candidates.parquet"
TASTE_PREFIX = "implicit_taste_"


def cf_seed_poi_ids(
    interactions_by_traveler: dict[str, pd.DataFrame],
    traveler_id: str,
    trip_id: str,
    as_of: pd.Timestamp,
    template: pd.DataFrame,
) -> list[str]:
    """The collaborative-filtering channel's per-trip seed set: the traveler's own
    as-of-safe, positively-engaged (`label >= 1`) interaction history. Reuses
    `features.traveler_features.traveler_history_before` directly -- same
    temporal-leakage discipline as Phase 3 (only the traveler's own EARLIER trips,
    never the current trip or any future one), never reimplemented. See
    `channels.build_item_item_cf`'s docstring for why the GLOBAL similarity matrix is
    fit on the full train window while only this per-trip SEED needs the as-of
    cutoff."""
    history = traveler_history_before(
        interactions_by_traveler, traveler_id, trip_id, as_of, template
    )
    result: list[str] = history.loc[history["label"] >= 1, "poi_id"].astype(str).tolist()
    return result


def _taste_vector_lookup(
    traveler_features_df: pd.DataFrame,
) -> dict[tuple[str, str], npt.NDArray[np.float64]]:
    """`{(traveler_id, trip_id): taste_vector}` -- reuses Phase 3's already-computed,
    as-of-safe `implicit_taste_*` columns directly rather than recomputing the taste
    vector here."""
    taste_cols = sorted(c for c in traveler_features_df.columns if c.startswith(TASTE_PREFIX))
    mat = traveler_features_df[taste_cols].to_numpy(dtype=np.float64)
    keys = list(
        zip(traveler_features_df["traveler_id"], traveler_features_df["trip_id"], strict=True)
    )
    return {k: mat[i] for i, k in enumerate(keys)}


def generate_candidates(
    pois_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    poi_features_df: pd.DataFrame,
    traveler_features_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    cfg: CandidatesConfig,
) -> pd.DataFrame:
    """Run all 6 candidate channels for every trip in `trips_df` and return the
    union membership table (see module docstring)."""
    dest_indices = build_destination_indices(
        pois_df, poi_features_df, cfg.traveler_segment_clusters
    )
    cf = build_item_item_cf(pois_df, interactions_train)
    segments = assign_traveler_segments(
        travelers_df, n_clusters=cfg.traveler_segment_clusters, seed=cfg.seed
    )

    canonical_map = build_poi_id_canonical_map(pois_df)
    remapped = remap_interaction_poi_ids(interactions_train, canonical_map)
    interactions_by_traveler = group_interactions_by_traveler(remapped)

    taste_lookup = _taste_vector_lookup(traveler_features_df)

    merged = trips_df.merge(travelers_df, on="traveler_id", how="left").sort_values("trip_id")

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        trip_id = str(row.trip_id)
        traveler_id = str(row.traveler_id)
        destination = str(row.destination)
        idx = dest_indices[destination]

        taste_vec = taste_lookup[(traveler_id, trip_id)]
        sims = compute_semantic_similarity(taste_vec, idx.embeddings)
        is_cold_start = float(np.linalg.norm(taste_vec)) == 0.0

        seed_poi_ids = cf_seed_poi_ids(
            interactions_by_traveler, traveler_id, trip_id, row.start_date, remapped
        )

        interests = set(row.interests)
        seg_val = segments.get(traveler_id)
        segment = int(seg_val) if seg_val is not None else 0
        rng = np.random.default_rng(trip_seed(cfg.seed, trip_id))

        selections: dict[str, list[str]] = {
            "channel_geo": channel_geo(idx, row.stay_lat, row.stay_lon, row.mobility, cfg.geo),
            "channel_interest": channel_interest(idx, interests, cfg.interest.quota),
            "channel_semantic": channel_semantic(idx, sims, cfg.semantic.quota),
            "channel_cf": channel_cf(idx, cf, seed_poi_ids, cfg.collaborative),
            "channel_longtail": channel_longtail(idx, sims, is_cold_start, cfg.longtail, rng),
            "channel_archetype": channel_archetype(idx, segment, cfg.archetype.quota),
        }

        membership: dict[str, set[str]] = {ch: set(ids) for ch, ids in selections.items()}
        union_ids = sorted(set().union(*membership.values()))
        for poi_id in union_ids:
            record: dict[str, Any] = {"trip_id": trip_id, "poi_id": poi_id}
            for ch in CHANNEL_NAMES:
                record[ch] = poi_id in membership[ch]
            rows.append(record)

    result = pd.DataFrame(rows, columns=["trip_id", "poi_id", *CHANNEL_NAMES])
    result = result.sort_values(["trip_id", "poi_id"]).reset_index(drop=True)
    return result


def run_candidates(cfg: CandidatesConfig, data_dir: Path) -> dict[str, Any]:
    """Load Phase 2/3 parquet outputs, run `generate_candidates`, write
    `data/synthetic/candidates.parquet`, and return a CLI/test summary dict."""
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    poi_features_df = pd.read_parquet(data_dir / "poi_features.parquet")
    traveler_features_df = pd.read_parquet(data_dir / "traveler_features.parquet")
    interactions_train = pd.read_parquet(data_dir / "interactions_train.parquet")

    candidates_df = generate_candidates(
        pois_df,
        travelers_df,
        trips_df,
        poi_features_df,
        traveler_features_df,
        interactions_train,
        cfg,
    )

    output_path = data_dir / CANDIDATES_FILENAME
    candidates_df.to_parquet(output_path, index=False)

    per_trip_counts = candidates_df.groupby("trip_id").size()
    return {
        "output_path": output_path,
        "n_trips": len(trips_df),
        "n_rows": len(candidates_df),
        "mean_candidates_per_trip": float(per_trip_counts.mean()) if len(per_trip_counts) else 0.0,
        "median_candidates_per_trip": float(per_trip_counts.median())
        if len(per_trip_counts)
        else 0.0,
        "min_candidates_per_trip": int(per_trip_counts.min()) if len(per_trip_counts) else 0,
        "max_candidates_per_trip": int(per_trip_counts.max()) if len(per_trip_counts) else 0,
    }
