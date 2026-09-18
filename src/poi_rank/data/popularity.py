"""Within-destination popularity percentile (spec.md section 4).

`pop_pct = percentile_rank(log1p(review_count) * r_shrunk)`, computed **within
destination**. All popularity-derived features must stay within-destination
percentiles -- this is what spec.md says enables new-destination transfer in a later
phase, so this must never be computed as a global rank across destinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_popularity_percentile(
    df: pd.DataFrame,
    review_count_col: str = "review_count",
    rating_shrunk_col: str = "rating_shrunk",
) -> pd.Series:
    """Percentile rank (convention: `[0, 1]`, average-rank tie-breaking) of
    `log1p(review_count) * r_shrunk`, ranked independently within each `destination`
    group."""
    score = np.log1p(df[review_count_col].astype(float)) * df[rating_shrunk_col].astype(float)
    pop_pct = score.groupby(df["destination"]).rank(pct=True, method="average")
    pop_pct.name = "pop_pct"
    return pop_pct
