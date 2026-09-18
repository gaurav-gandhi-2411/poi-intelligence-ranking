"""Empirical-Bayes rating shrinkage (spec.md section 4).

Pulls low-review-count ratings toward the destination mean, preventing a "5.0 with 3
reviews" POI from dominating downstream popularity/localness signals. Parameter-free
by construction: `C` and `m` are read directly from the data (destination mean rating,
destination median review count), never fit. Applied post-dedup, since `review_count`
and `rating` are only correct after near-duplicate merging.
"""

from __future__ import annotations

import pandas as pd


def compute_shrunk_rating(
    df: pd.DataFrame, rating_col: str = "rating", review_count_col: str = "review_count"
) -> pd.Series:
    """`r_shrunk = (v*R + m*C) / (v+m)`, where `v=review_count`, `R=rating`,
    `C`=destination mean rating, `m`=destination median review_count."""
    dest = df["destination"]
    v = df[review_count_col].astype(float)
    rating = df[rating_col].astype(float)
    c_dest_mean_rating = rating.groupby(dest).transform("mean")
    m_dest_median_reviews = v.groupby(dest).transform("median")
    r_shrunk = (v * rating + m_dest_median_reviews * c_dest_mean_rating) / (
        v + m_dest_median_reviews
    )
    r_shrunk.name = "rating_shrunk"
    return r_shrunk
