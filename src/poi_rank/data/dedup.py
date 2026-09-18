"""Near-duplicate POI merging (spec.md section 4).

Blocking on `(destination, H3 resolution-8 cell)`, then within each block a pairwise
union-find merge: two rows merge iff `rapidfuzz.fuzz.token_set_ratio(name_a, name_b)
>= name_threshold` AND haversine distance `< max_distance_m`. Union-find (not a single
pairwise pass) so a cluster of 3+ mutually-similar rows collapses into one merged
record rather than a chain of partial pairwise merges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from rapidfuzz import fuzz

from poi_rank.data.geo_prep import h3_cell, haversine_km


@dataclass(frozen=True)
class DedupReport:
    """Summary of one dedup run -- the checkable correctness signal against Phase 1's
    injected near-duplicate rate (docs/DATA_CARD.md: ~4% by construction)."""

    n_input_rows: int
    n_output_rows: int
    n_merged_away: int
    merge_rate: float
    n_clusters_with_merge: int


def _find_merge_clusters(
    names: list[str],
    lats: npt.NDArray[np.float64],
    lons: npt.NDArray[np.float64],
    name_threshold: float,
    max_distance_m: float,
) -> list[list[int]]:
    """Union-find over pairwise (name similarity >= threshold) AND (haversine distance
    < max_distance_m) within one blocking group. Returns clusters as lists of local
    (within-block) row indices."""
    n = len(names)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            name_sim = fuzz.token_set_ratio(names[i], names[j])
            if name_sim < name_threshold:
                continue
            dist_km = haversine_km(lats[i], lons[i], lats[j], lons[j])
            if float(dist_km) * 1000.0 < max_distance_m:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _merge_group(rows: pd.DataFrame) -> pd.Series:
    """Merge a cluster of near-duplicate rows into one canonical row.

    - `review_count`: summed across the group.
    - `rating`: review-count-weighted average of the group's original ratings.
    - `description`: the longest (richest) description in the group.
    - `tags`: union across the group, order-preserving, de-duplicated.
    - every other field: taken from the "primary" row (highest original
      `review_count`) -- Phase 1 clones near-duplicates from a random base POI with an
      independently *reduced* review_count (docs/DATA_CARD.md), so the highest-review
      row recovers the original base record.
    """
    total_reviews = int(rows["review_count"].sum())
    weights = rows["review_count"].to_numpy(dtype=float)
    if weights.sum() > 0:
        weighted_rating = float(np.average(rows["rating"].to_numpy(dtype=float), weights=weights))
    else:
        weighted_rating = float(rows["rating"].mean())

    primary_idx = rows["review_count"].idxmax()
    primary = rows.loc[primary_idx].copy()

    longest_desc = max(rows["description"].tolist(), key=len)
    union_tags: list[str] = []
    seen: set[str] = set()
    for tag_list in rows["tags"]:
        for tag in tag_list:
            if tag not in seen:
                seen.add(tag)
                union_tags.append(tag)

    primary["review_count"] = total_reviews
    primary["rating"] = round(weighted_rating, 2)
    primary["description"] = longest_desc
    primary["tags"] = union_tags
    primary["merged_poi_ids"] = sorted(rows["poi_id"].tolist())
    primary["n_merged"] = len(rows)
    return primary


def dedup_pois(
    pois_df: pd.DataFrame,
    h3_resolution: int = 8,
    name_threshold: float = 88.0,
    max_distance_m: float = 50.0,
) -> tuple[pd.DataFrame, DedupReport]:
    """Dedup the raw (dirty) POI catalog. Returns the deduped catalog (one row per
    surviving cluster) plus a `DedupReport`."""
    df = pois_df.reset_index(drop=True).copy()
    lat_arr = df["lat"].to_numpy(dtype=float)
    lon_arr = df["lon"].to_numpy(dtype=float)
    df["_h3_cell"] = h3_cell(lat_arr, lon_arr, h3_resolution)

    merged_rows: list[pd.Series] = []
    n_clusters_with_merge = 0
    for _, group in df.groupby(["destination", "_h3_cell"], sort=False):
        idx_local = group.index.to_numpy()
        names = group["name"].tolist()
        lats = group["lat"].to_numpy(dtype=float)
        lons = group["lon"].to_numpy(dtype=float)
        clusters = _find_merge_clusters(names, lats, lons, name_threshold, max_distance_m)
        for cluster in clusters:
            global_idx = idx_local[cluster]
            cluster_rows = df.loc[global_idx]
            if len(cluster_rows) > 1:
                n_clusters_with_merge += 1
                merged_rows.append(_merge_group(cluster_rows))
            else:
                row = cluster_rows.iloc[0].copy()
                row["merged_poi_ids"] = [row["poi_id"]]
                row["n_merged"] = 1
                merged_rows.append(row)

    out = pd.DataFrame(merged_rows).drop(columns=["_h3_cell"]).reset_index(drop=True)
    n_input = len(df)
    n_output = len(out)
    n_merged_away = n_input - n_output
    report = DedupReport(
        n_input_rows=n_input,
        n_output_rows=n_output,
        n_merged_away=n_merged_away,
        merge_rate=n_merged_away / n_input if n_input else 0.0,
        n_clusters_with_merge=n_clusters_with_merge,
    )
    return out, report
