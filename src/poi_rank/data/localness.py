"""Composite localness index (spec.md section 4) -- NOT inverse popularity.

```
localness = z(-pop_pct)*0.35 + z(-foreign_review_ratio)*0.30
          + z(haversine_to_tourist_centroid)*0.20 + z(local_tag_hits)*0.15
```

`z()` is a within-destination z-score. Tourist centroid = review-count-weighted
centroid of the top decile of POIs by `pop_pct`, per destination. `local_tag_hits`
counts tags overlapping `LOCAL_SIGNAL_TAGS`, a small hand-picked subset of the *real*
tag vocabulary Phase 1 generates (`datagen/taxonomy.py::TAGS`), not invented tags.

**Validation happens in `eval/oracle.py`, never here.** This module computes the
observable index only and must never import from or read the oracle latent-value
directory (firewall, spec.md section 1.1) -- enforced by the oracle-isolation test
and `tests/test_firewall_data.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poi_rank.data.geo_prep import haversine_km

# Tags that plausibly signal a "local" (non-touristy) POI, hand-picked from the real
# tag vocabulary Phase 1 generates (datagen/taxonomy.py::TAGS = [..., "local",
# "hidden-gem", "authentic", "touristy", ...]) -- chosen as the subset that reads as a
# genuine local-experience signal, not the full 20-tag vocabulary.
LOCAL_SIGNAL_TAGS: tuple[str, ...] = ("local", "hidden-gem", "authentic")


def _zscore_within_destination(series: pd.Series, destination: pd.Series) -> pd.Series:
    """Within-destination z-score; destinations with zero variance (or a single row)
    contribute 0.0 rather than inf/NaN."""
    grouped = series.groupby(destination)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    z = (series - mean) / std
    return z.fillna(0.0)


def compute_tourist_centroid(
    df: pd.DataFrame, pop_pct_col: str = "pop_pct", top_decile: float = 0.10
) -> pd.DataFrame:
    """Review-count-weighted centroid of the top decile of POIs by `pop_pct`, per
    destination. Returns one row per destination: `destination`, `centroid_lat`,
    `centroid_lon`."""
    rows: list[dict[str, object]] = []
    for dest, group in df.groupby("destination", sort=True):
        threshold = group[pop_pct_col].quantile(1.0 - top_decile)
        top = group.loc[group[pop_pct_col] >= threshold]
        weights = top["review_count"].to_numpy(dtype=float)
        if len(top) == 0 or weights.sum() <= 0:
            centroid_lat = float(group["lat"].mean())
            centroid_lon = float(group["lon"].mean())
        else:
            centroid_lat = float(np.average(top["lat"].to_numpy(dtype=float), weights=weights))
            centroid_lon = float(np.average(top["lon"].to_numpy(dtype=float), weights=weights))
        rows.append(
            {"destination": dest, "centroid_lat": centroid_lat, "centroid_lon": centroid_lon}
        )
    return pd.DataFrame(rows)


def compute_local_tag_hits(
    tags_col: pd.Series, local_signal_tags: tuple[str, ...] = LOCAL_SIGNAL_TAGS
) -> pd.Series:
    """Count of a POI's own tags that overlap `local_signal_tags`."""
    local_set = set(local_signal_tags)
    hits = tags_col.apply(lambda tags: sum(1 for t in tags if t in local_set))
    hits.name = "local_tag_hits"
    return hits


def compute_localness(
    df: pd.DataFrame,
    weight_popularity: float = 0.35,
    weight_foreign: float = 0.30,
    weight_geo: float = 0.20,
    weight_tag: float = 0.15,
    top_decile: float = 0.10,
    local_signal_tags: tuple[str, ...] = LOCAL_SIGNAL_TAGS,
) -> pd.DataFrame:
    """Compute the observable composite localness index.

    Requires `pop_pct`, `foreign_review_ratio`, `tags`, `lat`, `lon`, `review_count`,
    `destination` columns already present on `df`. Returns a DataFrame (aligned to
    `df.index`) with `localness`, `dist_to_tourist_centroid_km`, and `local_tag_hits`.
    """
    destination = df["destination"]
    centroids = compute_tourist_centroid(df, top_decile=top_decile)
    centroid_by_dest = centroids.set_index("destination")
    centroid_lat = destination.map(centroid_by_dest["centroid_lat"])
    centroid_lon = destination.map(centroid_by_dest["centroid_lon"])
    dist_to_centroid = pd.Series(
        haversine_km(
            df["lat"].to_numpy(dtype=float),
            df["lon"].to_numpy(dtype=float),
            centroid_lat.to_numpy(dtype=float),
            centroid_lon.to_numpy(dtype=float),
        ),
        index=df.index,
        name="dist_to_tourist_centroid_km",
    )
    local_tag_hits = compute_local_tag_hits(df["tags"], local_signal_tags)

    z_pop = _zscore_within_destination(-df["pop_pct"], destination)
    z_foreign = _zscore_within_destination(-df["foreign_review_ratio"], destination)
    z_geo = _zscore_within_destination(dist_to_centroid, destination)
    z_tag = _zscore_within_destination(local_tag_hits.astype(float), destination)

    localness = (
        weight_popularity * z_pop
        + weight_foreign * z_foreign
        + weight_geo * z_geo
        + weight_tag * z_tag
    )
    localness.name = "localness"

    return pd.DataFrame(
        {
            "localness": localness,
            "dist_to_tourist_centroid_km": dist_to_centroid,
            "local_tag_hits": local_tag_hits,
        }
    )
