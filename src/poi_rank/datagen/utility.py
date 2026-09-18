"""The DGP's true (latent) traveler utility function u(t,p).

spec.md section 1.2 defines:

    u(t,p) = w_taste * cos(taste_t, poi_semantic_p)
           + w_cat   * taste_t[category_p]
           + w_local * localness_p * (-touristiness_pref_t)
           + w_qual  * latent_quality_p
           + w_party * party_fit(p, party_type_t)
           + w_price * price_fit(p, budget_t)
           + w_novel * novelty_t(p)
           + eps, eps ~ N(0, sigma)

**Resolved ambiguity (see docs/DATA_CARD.md):** spec.md's literal formula has
`w_local * localness_p * touristiness_pref_t` with no negation. Since
`touristiness_pref` is defined as negative = prefers local / positive = prefers
touristy, that literal form would give a *local-preferring* traveler (negative pref)
positive utility from *touristy* (low-localness... no, high-localness) POIs — backwards.
This module implements the corrected, negated form so a local-preferring traveler
gets positive utility from high-localness POIs.

`party_fit`, `price_fit`, and `novelty_t` here are LATENT/oracle-side functions used
only to generate ground truth. They are deliberately not shared with the *observable*
`party_fit`/`budget_fit` that a later `scoring/` phase computes from visible fields —
same names, different module, different purpose, never shared code (spec.md section
1.2 note).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from poi_rank.datagen.config import UtilityWeights

FloatArray = npt.NDArray[np.float64]

BUDGET_TARGET_PRICE_LEVEL: dict[str, float] = {"low": 1.3, "medium": 2.5, "high": 3.7}

# Accessibility flags a party type is assumed to need. A party type with no listed
# need falls back to a near-constant baseline (see `party_fit_array`).
FAMILY_ACCESSIBILITY_NEEDS: dict[str, list[str]] = {
    "family_young_kids": ["kid_friendly", "stroller"],
    "family_teens": ["kid_friendly"],
}

# Small, deterministic category x party-type affinity bonus so w_party carries signal
# for non-family party types too (otherwise party_fit would be ~constant for the
# majority of travelers). Kept intentionally modest relative to the family-needs term.
CATEGORY_PARTY_BONUS: dict[str, dict[str, float]] = {
    "nightlife": {"friends": 0.15, "couple": 0.05, "solo": 0.0},
    "wellness_spa": {"couple": 0.15, "solo": 0.05, "friends": -0.05},
    "shopping": {"friends": 0.10, "solo": 0.05, "couple": 0.0},
    "entertainment": {"friends": 0.10, "couple": 0.05, "solo": 0.0},
}


def cosine_similarity_to_taste(taste_vec: FloatArray, poi_semantic: FloatArray) -> FloatArray:
    """Cosine similarity between one traveler taste vector and an (n, d) POI matrix."""
    dot = poi_semantic @ taste_vec
    poi_norms = np.linalg.norm(poi_semantic, axis=1)
    taste_norm = np.linalg.norm(taste_vec)
    denom = np.clip(poi_norms * taste_norm, 1e-9, None)
    result: FloatArray = dot / denom
    return result


def party_fit_array(
    party_type: str,
    wheelchair: npt.NDArray[np.bool_],
    stroller: npt.NDArray[np.bool_],
    kid_friendly: npt.NDArray[np.bool_],
    category: npt.NDArray[np.str_],
) -> FloatArray:
    """Latent party-compatibility score in [0, 1] for each POI.

    Family party types are scored by mean accessibility-flag match against their
    needs (partial credit). Other party types get a baseline plus a small
    category-conditioned affinity bonus (deterministic, documented above).
    """
    n = len(category)
    needs = FAMILY_ACCESSIBILITY_NEEDS.get(party_type)
    if needs:
        flag_map = {"wheelchair": wheelchair, "stroller": stroller, "kid_friendly": kid_friendly}
        match = np.mean([flag_map[need].astype(float) for need in needs], axis=0)
        result: FloatArray = 0.15 + 0.85 * match
        return result
    baseline = np.full(n, 0.85)
    bonus = np.array([CATEGORY_PARTY_BONUS.get(cat, {}).get(party_type, 0.0) for cat in category])
    clipped: FloatArray = np.clip(baseline + bonus, 0.0, 1.0)
    return clipped


def price_fit_array(budget: str, price_level_true: FloatArray) -> FloatArray:
    """Latent price-compatibility score in [0, 1]; over-budget penalized ~2x under-budget."""
    target = BUDGET_TARGET_PRICE_LEVEL[budget]
    diff = price_level_true.astype(float) - target
    penalty = np.where(diff > 0, diff * 0.35, -diff * 0.15)
    return np.clip(1.0 - penalty, 0.0, 1.0)


def novelty_array(
    poi_ids: npt.NDArray[np.str_],
    category: npt.NDArray[np.str_],
    subcategory: npt.NDArray[np.str_],
    seen_poi_ids: set[str],
    seen_category_subcategory: set[tuple[str, str]],
    same_poi_repeat_penalty: float,
    similar_category_repeat_penalty: float,
) -> FloatArray:
    """1.0 minus a repeat-visit decay. Empty history (first trip) -> all 1.0."""
    n = len(poi_ids)
    novelty = np.ones(n)
    for i in range(n):
        if poi_ids[i] in seen_poi_ids:
            novelty[i] -= same_poi_repeat_penalty
        elif (category[i], subcategory[i]) in seen_category_subcategory:
            novelty[i] -= similar_category_repeat_penalty
    return np.clip(novelty, 0.0, 1.0)


def true_utility_noise_free(
    weights: UtilityWeights,
    taste_vec: FloatArray,
    touristiness_pref: float,
    party_type: str,
    budget: str,
    poi_semantic: FloatArray,
    poi_category_taste_idx: npt.NDArray[np.intp],
    poi_latent_localness: FloatArray,
    poi_latent_quality: FloatArray,
    poi_wheelchair: npt.NDArray[np.bool_],
    poi_stroller: npt.NDArray[np.bool_],
    poi_kid_friendly: npt.NDArray[np.bool_],
    poi_category: npt.NDArray[np.str_],
    poi_price_level_true: FloatArray,
    novelty: FloatArray,
) -> FloatArray:
    """The noise-free component of u(t,p) for a traveler against an array of POIs.

    This is exactly what `eval/oracle.py` (a later phase) ranks by — the epsilon term
    is irreducible noise and is added separately by callers that need the stochastic
    version (see `interactions.py`), never baked into the oracle-exported utility.
    """
    taste_sim = cosine_similarity_to_taste(taste_vec, poi_semantic)
    cat_affinity = taste_vec[poi_category_taste_idx]
    local_term = poi_latent_localness * (-touristiness_pref)
    party = party_fit_array(
        party_type, poi_wheelchair, poi_stroller, poi_kid_friendly, poi_category
    )
    price = price_fit_array(budget, poi_price_level_true)

    result: FloatArray = (
        weights.w_taste * taste_sim
        + weights.w_cat * cat_affinity
        + weights.w_local * local_term
        + weights.w_qual * poi_latent_quality
        + weights.w_party * party
        + weights.w_price * price
        + weights.w_novel * novelty
    )
    return result
