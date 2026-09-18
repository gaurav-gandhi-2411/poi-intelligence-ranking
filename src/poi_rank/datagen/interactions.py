"""Stochastic interaction simulation within an exposed slate (spec.md section 1.2, 2.4).

Choice is Plackett-Luce over softmax(u/tau) within the slate, not argmax: higher-
utility POIs are more likely to get stronger engagement, not deterministically ranked.
Every exposed POI gets at least a `view` row (label 0, a true negative) unless it
received a stronger interaction or a `dismiss` — this is what gives the logs real
observed negatives with no negative sampling required downstream.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from poi_rank.datagen.sampling import pl_sample_without_replacement

INTERACTION_LABELS: dict[str, int] = {
    "booking": 3,
    "visit": 3,
    "navigate": 2,
    "save": 2,
    "share": 2,
    "click": 1,
    "view": 0,
    "dismiss": 0,
}

# Interaction-type distribution for the strongest (rank 0) vs weakest PL draws;
# `sample_interaction_type` blends between them by draw rank.
STRONG_DIST: dict[str, float] = {
    "booking": 0.12,
    "visit": 0.28,
    "navigate": 0.20,
    "save": 0.20,
    "share": 0.08,
    "click": 0.12,
}
WEAK_DIST: dict[str, float] = {
    "booking": 0.01,
    "visit": 0.04,
    "navigate": 0.10,
    "save": 0.10,
    "share": 0.05,
    "click": 0.70,
}
_ENGAGEMENT_TYPES = list(STRONG_DIST)


def sample_interaction_type(rng: np.random.Generator, rank: int, rank_decay: float) -> str:
    """Sample an engagement type; earlier PL draw ranks skew toward stronger types."""
    strength = math.exp(-rank / rank_decay)
    probs = np.array(
        [strength * STRONG_DIST[t] + (1 - strength) * WEAK_DIST[t] for t in _ENGAGEMENT_TYPES]
    )
    probs = probs / probs.sum()
    return str(rng.choice(_ENGAGEMENT_TYPES, p=probs))


def simulate_slate_choices(
    rng: np.random.Generator,
    utilities: npt.NDArray[np.float64],
    tau: float,
    engage_lambda: float,
    dismiss_lambda: float,
    rank_decay: float,
) -> list[tuple[int, str]]:
    """Simulate engagement/dismissal outcomes for one slate.

    Returns a list of (local_slate_index, interaction_type) for every POI that got a
    non-`view` outcome. Everything not returned defaults to `view` by the caller.
    """
    n = len(utilities)
    engage_weights = np.exp((utilities - utilities.max()) / tau)
    num_engage = min(n, rng.poisson(engage_lambda))
    engaged_idx = pl_sample_without_replacement(rng, engage_weights, num_engage)

    remaining = [i for i in range(n) if i not in engaged_idx]
    outcomes: list[tuple[int, str]] = []
    if remaining:
        rem_u = utilities[remaining]
        dismiss_weights = np.exp((rem_u.min() - rem_u) / tau)
        num_dismiss = min(len(remaining), rng.poisson(dismiss_lambda))
        dismissed_local = pl_sample_without_replacement(rng, dismiss_weights, num_dismiss)
        dismissed_idx = [remaining[i] for i in dismissed_local]
    else:
        dismissed_idx = []

    for rank, idx in enumerate(engaged_idx):
        outcomes.append((idx, sample_interaction_type(rng, rank, rank_decay)))
    for idx in dismissed_idx:
        outcomes.append((idx, "dismiss"))
    return outcomes
