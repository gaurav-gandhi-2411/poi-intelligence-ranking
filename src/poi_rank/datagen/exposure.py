"""Exposure policies: which POIs get shown to a traveler in a browsing session.

Two policies (spec.md section 1.3):
  - popularity-biased: p(expose) ~ popularity_raw^exponent * geo_proximity, used for
    `interactions_train.parquet` and the secondary `interactions_holdout_logged.parquet`.
  - uniform-random: every eligible POI equally likely, used for the primary
    `interactions_holdout_random.parquet`.

Both return the sampled slate's poi row-indices (into the eligible-POI array, in
presentation order) plus the logged propensity `p_expose` for each shown POI — the
propensity IPS correction (a later `models/` phase) will need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.geo import haversine_km


@dataclass(frozen=True)
class Slate:
    """A sampled slate: eligible-array indices in presentation order + propensities."""

    indices: npt.NDArray[np.intp]
    p_expose: npt.NDArray[np.float64]


def build_biased_slate(
    rng: np.random.Generator,
    popularity_raw: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    stay_lat: float,
    stay_lon: float,
    slate_size: int,
    cfg: DatagenConfig,
) -> Slate:
    """Popularity x geo-proximity weighted sample of `slate_size` POIs, no replacement.

    `p_expose` uses the standard PPS-sampling approximation `min(1, k * w_i / sum(w))`
    for probability-of-inclusion under a size-k weighted draw without replacement
    (documented in docs/DATA_CARD.md as an approximation, not an exact combinatorial
    inclusion probability).
    """
    n = len(popularity_raw)
    k = min(slate_size, n)
    dist_km = haversine_km(stay_lat, stay_lon, lat, lon)
    geo_prox = np.exp(-dist_km / cfg.exposure.geo_decay_km)
    weights = np.power(popularity_raw, cfg.exposure.popularity_exponent) * np.power(
        geo_prox, cfg.exposure.geo_weight_exponent
    )
    weights = np.clip(weights, 1e-12, None)
    probs = weights / weights.sum()

    chosen = rng.choice(n, size=k, replace=False, p=probs)
    p_expose = np.minimum(1.0, k * probs[chosen])
    # Present higher-weight POIs first, mimicking a ranked logging policy.
    order = np.argsort(-weights[chosen])
    return Slate(indices=chosen[order], p_expose=p_expose[order])


def build_random_slate(rng: np.random.Generator, n_eligible: int, slate_size: int) -> Slate:
    """Uniform-random sample of `slate_size` POIs, no replacement.

    `p_expose` is exact here (not an approximation): uniform sampling without
    replacement of size k from n items gives every included item inclusion
    probability exactly k/n.
    """
    k = min(slate_size, n_eligible)
    chosen = rng.choice(n_eligible, size=k, replace=False)
    order = rng.permutation(k)
    p_expose = np.full(k, k / n_eligible)
    return Slate(indices=chosen[order], p_expose=p_expose[order])
