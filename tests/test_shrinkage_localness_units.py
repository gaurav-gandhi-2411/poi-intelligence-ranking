"""Focused unit tests for `data/shrinkage.py` and `data/localness.py` on small
hand-built DataFrames (independent of the full generated-data fixtures)."""

from __future__ import annotations

import pandas as pd
import pytest

from poi_rank.data.localness import compute_local_tag_hits, compute_tourist_centroid
from poi_rank.data.shrinkage import compute_shrunk_rating


def test_shrunk_rating_pulls_sparse_poi_toward_destination_mean() -> None:
    df = pd.DataFrame(
        {
            "destination": ["seoul", "seoul", "seoul"],
            "rating": [5.0, 3.0, 3.0],  # a 5.0-with-1-review outlier
            "review_count": [1, 100, 100],
        }
    )
    shrunk = compute_shrunk_rating(df)
    # The sparse POI's shrunk rating must move away from its raw 5.0 toward the
    # destination mean, and be strictly less than the raw rating.
    assert shrunk.iloc[0] < 5.0
    dest_mean = df["rating"].mean()
    assert abs(shrunk.iloc[0] - dest_mean) < abs(5.0 - dest_mean)


def test_shrunk_rating_barely_moves_high_review_count_poi() -> None:
    # Realistic destination: median review_count (m) is small relative to a genuinely
    # popular POI's review_count (v), so shrinkage toward the destination mean is weak.
    df = pd.DataFrame(
        {
            "destination": ["seoul"] * 6,
            "rating": [4.5, 3.0, 3.5, 4.0, 3.2, 3.8],
            "review_count": [5000, 80, 90, 100, 70, 110],
        }
    )
    shrunk = compute_shrunk_rating(df)
    assert shrunk.iloc[0] == pytest.approx(4.5, abs=0.1)


def test_compute_tourist_centroid_weights_by_review_count() -> None:
    df = pd.DataFrame(
        {
            "destination": ["seoul"] * 4,
            "pop_pct": [1.0, 0.9, 0.1, 0.05],  # top decile threshold with top_decile=0.5 -> top 2
            "review_count": [1000, 10, 5, 5],
            "lat": [37.60, 37.50, 37.40, 37.30],
            "lon": [127.00, 127.10, 127.20, 127.30],
        }
    )
    centroids = compute_tourist_centroid(df, top_decile=0.5)
    row = centroids.loc[centroids["destination"] == "seoul"].iloc[0]
    # Heavily weighted toward the lat=37.60 POI (review_count=1000 vs 10).
    assert row["centroid_lat"] == pytest.approx(37.60, abs=0.02)


def test_compute_local_tag_hits_counts_overlap_only() -> None:
    tags = pd.Series(
        [["local", "foodie"], ["touristy", "luxury"], ["hidden-gem", "authentic", "local"]]
    )
    hits = compute_local_tag_hits(tags, local_signal_tags=("local", "hidden-gem", "authentic"))
    assert hits.tolist() == [1, 0, 3]
