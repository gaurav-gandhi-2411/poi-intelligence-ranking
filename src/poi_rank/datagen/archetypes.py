"""8 traveler archetypes: prototype taste vectors + typical trip attribute mixtures.

Travelers are NOT hard-classified into one of these 8 — `travelers.py` draws a soft
Dirichlet mixture weight vector over them per traveler and blends the prototype taste
vectors, so archetypes are a generative prior, never a stored label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from poi_rank.datagen.taxonomy import CATEGORY_INDEX, TAG_INDEX, TASTE_DIM


@dataclass(frozen=True)
class Archetype:
    """One archetype's generative prior over taste, trip context, and party."""

    name: str
    category_weights: dict[str, float]
    tag_weights: dict[str, float]
    touristiness_mean: float
    touristiness_std: float
    budget_probs: dict[str, float]
    party_type_probs: dict[str, float] = field(default_factory=dict)


ARCHETYPES: list[Archetype] = [
    Archetype(
        name="local_food_explorer",
        category_weights={"restaurant": 1.0, "cafe": 0.6, "shopping": 0.3},
        tag_weights={
            "local": 1.0,
            "authentic": 0.9,
            "hidden-gem": 0.7,
            "foodie": 1.0,
            "budget-friendly": 0.4,
        },
        touristiness_mean=-0.6,
        touristiness_std=0.2,
        budget_probs={"low": 0.25, "medium": 0.55, "high": 0.20},
        party_type_probs={"solo": 0.35, "couple": 0.35, "friends": 0.30},
    ),
    Archetype(
        name="history_architecture",
        category_weights={"historic_site": 1.0, "museum": 0.8, "religious_site": 0.5},
        tag_weights={"historic": 1.0, "cultural": 0.9, "artsy": 0.5},
        touristiness_mean=0.3,
        touristiness_std=0.3,
        budget_probs={"low": 0.15, "medium": 0.50, "high": 0.35},
        party_type_probs={"solo": 0.30, "couple": 0.50, "friends": 0.20},
    ),
    Archetype(
        name="family_activities",
        category_weights={"family_activity": 1.0, "nature_park": 0.6, "entertainment": 0.4},
        tag_weights={"family-friendly": 1.0, "outdoor": 0.5, "relaxing": 0.4},
        touristiness_mean=0.1,
        touristiness_std=0.3,
        budget_probs={"low": 0.15, "medium": 0.60, "high": 0.25},
        party_type_probs={"family_young_kids": 0.55, "family_teens": 0.35, "friends": 0.10},
    ),
    Archetype(
        name="nightlife",
        category_weights={"nightlife": 1.0, "restaurant": 0.4, "entertainment": 0.5},
        tag_weights={"lively": 1.0, "trendy": 0.8, "touristy": 0.2},
        touristiness_mean=0.2,
        touristiness_std=0.3,
        budget_probs={"low": 0.20, "medium": 0.50, "high": 0.30},
        party_type_probs={"friends": 0.60, "couple": 0.30, "solo": 0.10},
    ),
    Archetype(
        name="nature_outdoor",
        category_weights={"nature_park": 1.0, "viewpoint": 0.6},
        tag_weights={"nature": 1.0, "outdoor": 1.0, "adventurous": 0.6, "relaxing": 0.5},
        touristiness_mean=-0.2,
        touristiness_std=0.3,
        budget_probs={"low": 0.35, "medium": 0.50, "high": 0.15},
        party_type_probs={"solo": 0.30, "couple": 0.35, "friends": 0.35},
    ),
    Archetype(
        name="shopping",
        category_weights={"shopping": 1.0, "cafe": 0.3},
        tag_weights={"shopping": 1.0, "trendy": 0.6, "luxury": 0.3},
        touristiness_mean=0.3,
        touristiness_std=0.3,
        budget_probs={"low": 0.15, "medium": 0.50, "high": 0.35},
        party_type_probs={"solo": 0.30, "couple": 0.30, "friends": 0.40},
    ),
    Archetype(
        name="luxury_gastronomy",
        category_weights={"restaurant": 1.0, "wellness_spa": 0.4},
        tag_weights={"luxury": 1.0, "romantic": 0.5, "foodie": 0.8},
        touristiness_mean=0.4,
        touristiness_std=0.2,
        budget_probs={"low": 0.02, "medium": 0.18, "high": 0.80},
        party_type_probs={"couple": 0.70, "solo": 0.15, "friends": 0.15},
    ),
    Archetype(
        name="budget_backpacker",
        category_weights={"restaurant": 0.6, "nature_park": 0.5, "historic_site": 0.5},
        tag_weights={
            "budget-friendly": 1.0,
            "local": 0.7,
            "authentic": 0.6,
            "adventurous": 0.4,
        },
        touristiness_mean=-0.4,
        touristiness_std=0.3,
        budget_probs={"low": 0.75, "medium": 0.23, "high": 0.02},
        party_type_probs={"solo": 0.45, "friends": 0.45, "couple": 0.10},
    ),
]

ARCHETYPE_NAMES: list[str] = [a.name for a in ARCHETYPES]


def archetype_prototype_vector(archetype: Archetype) -> npt.NDArray[np.float64]:
    """Build the archetype's L2-normalized prototype vector in taste space."""
    vec = np.zeros(TASTE_DIM, dtype=np.float64)
    for cat, w in archetype.category_weights.items():
        vec[CATEGORY_INDEX[cat]] = w
    for tag, w in archetype.tag_weights.items():
        vec[TAG_INDEX[tag]] = w
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


PROTOTYPE_MATRIX: npt.NDArray[np.float64] = np.stack(
    [archetype_prototype_vector(a) for a in ARCHETYPES]
)
