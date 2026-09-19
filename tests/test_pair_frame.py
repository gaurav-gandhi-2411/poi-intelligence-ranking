"""Pair-frame helpers: the vectorised interest match must equal the row-wise reference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from poi_rank.features.pair_frame import interest_match_vectorized
from poi_rank.features.traveler_features import interest_match_score


def test_interest_match_vectorized_equals_rowwise_reference() -> None:
    rng = np.random.default_rng(3)
    terms = [f"t{i}" for i in range(12)]
    cats = ["restaurant", "museum", "park"]
    trips = {
        f"T{i}": list(rng.choice(terms + cats, size=int(rng.integers(0, 5)), replace=False))
        for i in range(30)
    }
    pois = {
        f"P{j}": (
            str(rng.choice(cats)),
            list(rng.choice(terms, size=int(rng.integers(0, 6)), replace=False)),
        )
        for j in range(40)
    }
    rows = [(t, p) for t in trips for p in pois if rng.random() < 0.5]
    frame = pd.DataFrame(
        {
            "trip_id": [t for t, _ in rows],
            "poi_id": [p for _, p in rows],
            "interests": [trips[t] for t, _ in rows],
            "poi_category_raw": [pois[p][0] for _, p in rows],
            "poi_tags_raw": [pois[p][1] for _, p in rows],
        }
    )
    expected = interest_match_score(
        [set(x) for x in frame["interests"]],
        frame["poi_category_raw"].tolist(),
        [list(x) for x in frame["poi_tags_raw"]],
    )
    np.testing.assert_array_equal(interest_match_vectorized(frame), expected)
