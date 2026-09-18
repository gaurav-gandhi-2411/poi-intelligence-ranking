"""POI id reconciliation between `interactions_*.parquet` (referencing Phase 1's raw,
pre-dedup `pois.parquet` poi_ids) and `pois_prepared.parquet` (one row per post-dedup
merge cluster, keyed by its *primary* member's `poi_id`).

Phase 2's dedup (`data/dedup.py`) merges near-duplicate POI rows into one surviving
record per cluster, keyed by the highest-review-count member's `poi_id`; every
original member id (including the survivor's own) is recorded in that row's
`merged_poi_ids` list column. Interaction logs were generated against the *raw*,
pre-dedup catalog (`datagen/pipeline.py` samples slates from the full un-deduped POI
array), so a real, non-trivial slice of the poi_ids referenced in
`interactions_train.parquet` (measured: 40 of 1,386 distinct train poi_ids, ~2.9%, on
the committed dataset) no longer exist as standalone rows in `pois_prepared.parquet`
-- they were merged into a *different* row's `poi_id`.

Every feature-table join against interaction logs must remap through
`build_poi_id_canonical_map` first, or behavioral aggregates for merged-away POIs
would silently vanish (counted as if those interactions never happened) rather than
correctly rolling up onto the surviving canonical row. Discovered by direct
inspection while building this phase (not called out in spec.md's Phase 3 task
description) -- documented in docs/DATA_CARD.md.
"""

from __future__ import annotations

import pandas as pd


def build_poi_id_canonical_map(pois_prepared: pd.DataFrame) -> dict[str, str]:
    """Map every raw (pre-dedup) `poi_id` referenced in `merged_poi_ids` to its
    surviving canonical `poi_id` in `pois_prepared`. Every prepared row's own
    `poi_id` is included (it is always a member of its own `merged_poi_ids` list, per
    `data/dedup.py`), so this map is also a safe identity map for already-canonical
    ids.
    """
    mapping: dict[str, str] = {}
    for row in pois_prepared.itertuples(index=False):
        canonical_id = str(row.poi_id)
        for original_id in row.merged_poi_ids:
            mapping[str(original_id)] = canonical_id
    return mapping


def remap_interaction_poi_ids(
    interactions: pd.DataFrame, canonical_map: dict[str, str]
) -> pd.DataFrame:
    """Return a copy of `interactions` with `poi_id` remapped to its canonical
    (post-dedup) id.

    Rows whose `poi_id` has no entry in `canonical_map` are dropped -- defensive only;
    against the committed, reconciled dataset this should never fire (every
    interaction poi_id is covered by some `pois_prepared` row's `merged_poi_ids`).
    Callers reconciling a known-good dataset are expected to assert
    `len(result) == len(interactions)` if they want to enforce that invariant loudly
    rather than silently.
    """
    out = interactions.copy()
    out["poi_id"] = out["poi_id"].map(canonical_map)
    return out.dropna(subset=["poi_id"]).reset_index(drop=True)
