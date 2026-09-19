"""Traveler and trip generation (spec.md section 2.3).

Each traveler's true taste vector is a soft Dirichlet mixture over the 8 archetypes
plus per-traveler Gaussian noise (never a hard archetype label). `interests[]` is
generated as a genuinely lossy projection of that latent vector — this is the load-
bearing mechanism that keeps explicit features from reaching the oracle ceiling on
their own (spec.md section 1.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.datagen.archetypes import ARCHETYPE_NAMES, ARCHETYPES, PROTOTYPE_MATRIX
from poi_rank.datagen.catalog import DEST_CENTERS
from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.taxonomy import INTEREST_LABELS
from poi_rank.datagen.timeline import Timeline

HOME_MARKETS: list[str] = ["domestic", "regional", "international"]
HOME_MARKET_PROBS: list[float] = [0.20, 0.30, 0.50]

MOBILITY_PROBS: dict[str, float] = {
    "walk": 0.25,
    "public_transport": 0.40,
    "car": 0.15,
    "mixed": 0.20,
}
PACE_PROBS: dict[str, float] = {"relaxed": 0.30, "moderate": 0.45, "packed": 0.25}
DIETARY_OPTIONS: list[str] = ["vegetarian", "vegan", "halal", "gluten-free"]
DIETARY_ITEM_PROB = 0.08

ARCHETYPE_DIRICHLET_ALPHA = 0.7  # < 1 favors soft-but-peaked mixtures, not near-uniform blends
TASTE_NOISE_STD = 0.15
TOURISTINESS_NOISE_STD = 0.15


def _sample_mixture_weighted_category(
    rng: np.random.Generator, mixture: npt.NDArray[np.float64], prob_key: str
) -> str:
    combined: dict[str, float] = {}
    for w, arch in zip(mixture, ARCHETYPES, strict=True):
        probs = getattr(arch, prob_key)
        for k, p in probs.items():
            combined[k] = combined.get(k, 0.0) + w * p
    keys = list(combined)
    vals = np.array([combined[k] for k in keys])
    total = vals.sum()
    if total <= 0:
        return str(rng.choice(keys))
    vals = vals / total
    return str(rng.choice(keys, p=vals))


def project_stated_interests(
    rng: np.random.Generator,
    taste_vec: npt.NDArray[np.float64],
    top_k: int,
    random_rate: float,
    omission_rate: float,
) -> list[str]:
    """Lossy projection of the latent taste vector into a stated `interests[]` list.

    Two independent noise mechanisms, both required by spec.md section 1.1:
      1. omission: each of the top-k latent-weight labels is independently dropped
         with probability `omission_rate` ("genuinely high-latent interests omitted").
      2. randomization: each surviving label is independently replaced by an
         unrelated label with probability `random_rate` ("stated interests that are
         random draws, not from latent taste").
    """
    order = np.argsort(-taste_vec)
    top_labels = [INTEREST_LABELS[i] for i in order[:top_k]]
    survivors = [label for label in top_labels if rng.random() >= omission_rate]
    if not survivors:
        survivors = [top_labels[0]]

    unrelated_pool = [label for label in INTEREST_LABELS if label not in top_labels]
    final: list[str] = []
    for label in survivors:
        if rng.random() < random_rate and unrelated_pool:
            final.append(str(rng.choice(unrelated_pool)))
        else:
            final.append(label)
    # de-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for label in final:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped or [str(rng.choice(INTEREST_LABELS))]


def _make_explicit_preferences(
    rng: np.random.Generator, top_interest: str, touristiness_pref: float, budget: str
) -> str:
    style = "local, off-the-beaten-path" if touristiness_pref < 0 else "iconic, must-see"
    budget_phrase = {
        "low": "keeping costs low",
        "medium": "a moderate budget",
        "high": "no strict budget limit",
    }[budget]
    return f"Looking for {top_interest} experiences, prefers {style} spots, with {budget_phrase}."


def generate_travelers(
    rng: np.random.Generator, cfg: DatagenConfig
) -> tuple[pd.DataFrame, dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
    """Generate the traveler table plus two oracle-only dicts: true latent taste vectors and the
    true archetype mixture weights (the latter is only ever exported for eval-side
    personalization scoring; drawing it consumes no extra RNG)."""
    destinations = list(DEST_CENTERS)
    n_per_dest = cfg.scale.travelers_per_destination
    rows: list[dict[str, Any]] = []
    taste_vectors: dict[str, npt.NDArray[np.float64]] = {}
    mixtures: dict[str, npt.NDArray[np.float64]] = {}

    traveler_idx = 0
    for dest in destinations:
        for _ in range(n_per_dest):
            traveler_idx += 1
            traveler_id = f"U{traveler_idx:04d}"

            mixture = rng.dirichlet(np.full(len(ARCHETYPE_NAMES), ARCHETYPE_DIRICHLET_ALPHA))
            mixtures[traveler_id] = mixture
            blended = mixture @ PROTOTYPE_MATRIX
            true_taste = blended + rng.normal(0, TASTE_NOISE_STD, size=blended.shape)
            norm = np.linalg.norm(true_taste)
            true_taste = true_taste / norm if norm > 0 else true_taste
            taste_vectors[traveler_id] = true_taste

            touristiness_mean = float(mixture @ np.array([a.touristiness_mean for a in ARCHETYPES]))
            touristiness_std = float(mixture @ np.array([a.touristiness_std for a in ARCHETYPES]))
            touristiness_pref = float(
                np.clip(rng.normal(touristiness_mean, max(touristiness_std, 0.05)), -1.0, 1.0)
            )

            budget = _sample_mixture_weighted_category(rng, mixture, "budget_probs")
            party_type = _sample_mixture_weighted_category(rng, mixture, "party_type_probs")

            interests = project_stated_interests(
                rng,
                true_taste,
                cfg.interests.top_k_latent,
                cfg.interests.random_interest_rate,
                cfg.interests.omission_rate,
            )

            mobility = str(rng.choice(list(MOBILITY_PROBS), p=list(MOBILITY_PROBS.values())))
            pace = str(rng.choice(list(PACE_PROBS), p=list(PACE_PROBS.values())))
            home_market = str(rng.choice(HOME_MARKETS, p=HOME_MARKET_PROBS))

            dietary = [d for d in DIETARY_OPTIONS if rng.random() < DIETARY_ITEM_PROB]

            accessibility_needs: list[str] = []
            if party_type == "family_young_kids" and rng.random() < 0.5:
                accessibility_needs.append("stroller")
            if rng.random() < 0.03:
                accessibility_needs.append("wheelchair")

            explicit_preferences = _make_explicit_preferences(
                rng, interests[0], touristiness_pref, budget
            )

            rows.append(
                {
                    "traveler_id": traveler_id,
                    "home_market": home_market,
                    "home_destination": dest,
                    "party_type": party_type,
                    "budget": budget,
                    "mobility": mobility,
                    "interests": interests,
                    "touristiness_pref": touristiness_pref,
                    "pace": pace,
                    "accessibility_needs": accessibility_needs,
                    "explicit_preferences": explicit_preferences,
                    "dietary": dietary,
                }
            )

    return pd.DataFrame(rows), taste_vectors, mixtures


PARTY_SIZE_RANGE: dict[str, tuple[int, int]] = {
    "solo": (1, 1),
    "couple": (2, 2),
    "family_young_kids": (3, 5),
    "family_teens": (3, 5),
    "friends": (3, 6),
}


def _sample_party_size(rng: np.random.Generator, party_type: str) -> int:
    lo, hi = PARTY_SIZE_RANGE[party_type]
    return int(rng.integers(lo, hi + 1))


def generate_trips(
    rng: np.random.Generator, travelers_df: pd.DataFrame, cfg: DatagenConfig, timeline: Timeline
) -> pd.DataFrame:
    """Generate ~800 trips: every traveler gets one, a fixed subset gets a second."""
    n_travelers = len(travelers_df)
    n_two_trip = round(cfg.scale.two_trip_traveler_fraction * n_travelers)
    two_trip_positions = set(rng.choice(n_travelers, size=n_two_trip, replace=False).tolist())

    destinations = list(DEST_CENTERS)
    rows: list[dict[str, Any]] = []
    trip_idx = 0
    usable_span_days = max((timeline.end - timeline.start).days - 14, 1)

    for pos, traveler in enumerate(travelers_df.itertuples(index=False)):
        n_trips = 2 if pos in two_trip_positions else 1
        first_start = timeline.start + pd.Timedelta(days=int(rng.integers(0, usable_span_days)))
        trip_starts = [first_start]
        if n_trips == 2:
            gap_days = int(rng.integers(60, 400))
            second_start = min(
                first_start + pd.Timedelta(days=gap_days), timeline.end - pd.Timedelta(days=3)
            )
            trip_starts.append(second_start)

        for trip_num, start_date in enumerate(trip_starts):
            trip_idx += 1
            trip_id = f"T{trip_idx:04d}"
            duration = int(np.clip(rng.lognormal(mean=1.3, sigma=0.4), 2, 10))
            end_date = start_date + pd.Timedelta(days=duration)

            if trip_num == 0:
                destination = traveler.home_destination
            else:
                destination = str(rng.choice(destinations))

            center_lat, center_lon = DEST_CENTERS[destination]
            stay_lat = center_lat + rng.normal(0, 0.02)
            stay_lon = center_lon + rng.normal(0, 0.025)

            party_size = _sample_party_size(rng, traveler.party_type)

            rows.append(
                {
                    "trip_id": trip_id,
                    "traveler_id": traveler.traveler_id,
                    "destination": destination,
                    "start_date": start_date,
                    "end_date": end_date,
                    "trip_duration_days": duration,
                    "party_size": party_size,
                    "stay_lat": float(stay_lat),
                    "stay_lon": float(stay_lon),
                    "trip_sequence": trip_num + 1,
                }
            )

    return pd.DataFrame(rows)
