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

**DGP remediation, Block A, RC2a (docs/DATA_CARD.md "DGP remediation, Block A"):**
each of the 7 deterministic terms is now Z-SCORED (zero-mean/unit-variance) BEFORE
its configured weight is applied, using per-destination reference statistics
(`TermStandardization`, `compute_term_standardization` below) fit once at generation
time over the FULL generation population (every traveler x every unique POI at that
destination) -- so configured weights actually mean what they say, instead of being a
lie against wildly different natural term scales (the pre-remediation bug: e.g.
`cos(taste, poi)` has near-zero natural variance, so `w_taste` barely moved `u`
regardless of its configured value). Weights are sized as `sqrt(target variance
share)` so `Var(u_total) ~= 1` by construction when terms are independent (measured
covariance causes some deviation from the target shares -- documented, not silently
forced, in docs/DATA_CARD.md).

`novelty`'s standardization reference (`TermStandardization.novelty_mean/std`) is a
documented A PRIORI constant, not measured from this catalog's realization: novelty is
a traveler-HISTORY-dependent quantity (not a fixed catalog-level distribution like the
other 6 terms), and its true empirical distribution is only knowable after running the
full history-dependent interaction simulation -- which itself depends on the
standardization this function computes (a circularity). The constant reflects a
plausible post-RC3 mix (~70% no-repeat, ~20% category-repeat, ~10% same-POI-repeat) --
see docs/DATA_CARD.md for the derivation. This is a scale reference only; the ACTUAL
novelty term's variance share is measured, never forced, via `poi_rank.cli diagnose-dgp`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.datagen.config import UtilityWeights
from poi_rank.datagen.taxonomy import CATEGORY_INDEX

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

# RC2a novelty standardization reference (module docstring): an initial a priori
# guess (mean=0.90, std=0.184, assuming ~70/20/10 no-repeat/category-repeat/
# same-POI mix) was measured, via a calibration run's own residual reconstruction
# (docs/DATA_CARD.md "DGP remediation, Block A"), to OVERSTATE the true post-RC3
# empirical spread -- most (traveler, POI) pairs never hit a repeat at all even
# with real pre-trip/earlier-trip history (most eligible POIs were simply never
# seen), so novelty's real distribution concentrates much more tightly near 1.0
# than the initial guess assumed. Replaced with the measured empirical mean/std
# (reconstructed from `weighted_novelty_residual` on a calibration run) --
# dividing by an inflated reference std was mechanically SHRINKING novelty's
# z-scored (and therefore weighted) variance contribution below its intended
# target share, which is exactly the wrong direction for a term this module
# should not artificially suppress.
NOVELTY_STANDARDIZATION_MEAN = 0.98
NOVELTY_STANDARDIZATION_STD = 0.09


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


# -----------------------------------------------------------------------------------
# DGP remediation, Block A, RC2a: term standardization.
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DestinationTermStats:
    """Mean/std of each of the 6 catalog/traveler-distributional terms (everything
    except `novelty`, which uses a fixed a priori reference -- module docstring),
    for ONE destination."""

    taste_mean: float
    taste_std: float
    cat_mean: float
    cat_std: float
    local_mean: float
    local_std: float
    qual_mean: float
    qual_std: float
    party_mean: float
    party_std: float
    price_mean: float
    price_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DestinationTermStats:
        return cls(**d)


@dataclass(frozen=True)
class TermStandardization:
    """Full standardization reference: per-destination stats for 6 terms, plus one
    global a priori reference for `novelty` (module docstring)."""

    per_destination: dict[str, DestinationTermStats]
    novelty_mean: float = NOVELTY_STANDARDIZATION_MEAN
    novelty_std: float = NOVELTY_STANDARDIZATION_STD

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_destination": {k: v.to_dict() for k, v in self.per_destination.items()},
            "novelty_mean": self.novelty_mean,
            "novelty_std": self.novelty_std,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TermStandardization:
        return cls(
            per_destination={
                k: DestinationTermStats.from_dict(v) for k, v in d["per_destination"].items()
            },
            novelty_mean=d["novelty_mean"],
            novelty_std=d["novelty_std"],
        )


def zscore(x: FloatArray, mean: float, std: float) -> FloatArray:
    """Z-score `x` against a precomputed (mean, std); std<=~0 (a degenerate,
    zero-variance reference) maps to all-zeros rather than dividing by ~0."""
    if std <= 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    result: FloatArray = (x - mean) / std
    return result


def compute_term_standardization(
    poi_true_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    taste_vectors: dict[str, FloatArray],
) -> TermStandardization:
    """Fit `TermStandardization` from the full generation population: for each
    destination, EVERY traveler (a traveler can visit any destination, not just
    their `home_destination`) crossed with EVERY unique (non-duplicate) POI at that
    destination -- a deterministic, non-circular population requiring no interaction
    simulation (module docstring's "full generation population" choice for the 6
    catalog/traveler-distributional terms; `novelty` uses the separate a priori
    constant instead, for the reasons documented there).

    `poi_true_df` is the TRUE (pre-dirtiness) catalog (`generate_all_catalogs`'s
    output) -- price_level here is never null, so every POI contributes to the
    price-term statistics (unlike a diagnostic recomputation against the exported,
    partially-nulled catalog).
    """
    taste_matrix = np.stack([taste_vectors[t] for t in travelers_df["traveler_id"]])
    touristiness_prefs = travelers_df["touristiness_pref"].to_numpy(dtype=np.float64)
    party_types = travelers_df["party_type"].tolist()
    budgets = travelers_df["budget"].tolist()

    # Precompute one raw array per distinct party_type/budget (few distinct values)
    # to tile across travelers cheaply instead of recomputing per traveler.
    unique_party_types = sorted(set(party_types))
    unique_budgets = sorted(set(budgets))

    per_destination: dict[str, DestinationTermStats] = {}
    unique_pois = poi_true_df.loc[~poi_true_df["is_duplicate"]]
    for dest, group in unique_pois.groupby("destination", sort=True):
        poi_semantic = np.stack(group["poi_semantic"].to_numpy())
        category = group["category"].to_numpy()
        category_taste_idx = np.array([CATEGORY_INDEX[c] for c in category], dtype=np.intp)
        latent_quality = group["latent_quality"].to_numpy(dtype=np.float64)
        latent_localness = group["latent_localness"].to_numpy(dtype=np.float64)
        price_level_true = group["price_level"].to_numpy(dtype=np.float64)
        wheelchair = group["wheelchair"].to_numpy(dtype=bool)
        stroller = group["stroller"].to_numpy(dtype=bool)
        kid_friendly = group["kid_friendly"].to_numpy(dtype=bool)

        n_trav = len(travelers_df)

        # taste_sim: (n_pois, n_trav) via one matmul against the full taste matrix.
        dots = poi_semantic @ taste_matrix.T
        poi_norms = np.linalg.norm(poi_semantic, axis=1, keepdims=True)
        taste_norms = np.linalg.norm(taste_matrix, axis=1, keepdims=True).T
        denom = np.clip(poi_norms * taste_norms, 1e-9, None)
        taste_sim_matrix = dots / denom

        # cat_affinity: (n_pois, n_trav) -- taste_matrix[:, category_taste_idx].T
        cat_affinity_matrix = taste_matrix[:, category_taste_idx].T

        # local_term: outer(latent_localness, -touristiness_pref).
        local_term_matrix = np.outer(latent_localness, -touristiness_prefs)

        # latent_quality: constant per POI, replicated across travelers.
        qual_matrix = np.tile(latent_quality[:, None], (1, n_trav))

        # party_fit / price_fit: compute once per distinct value, tile per traveler.
        party_by_value = {
            pt: party_fit_array(pt, wheelchair, stroller, kid_friendly, category)
            for pt in unique_party_types
        }
        party_matrix = np.stack([party_by_value[pt] for pt in party_types], axis=1)

        price_by_value = {b: price_fit_array(b, price_level_true) for b in unique_budgets}
        price_matrix = np.stack([price_by_value[b] for b in budgets], axis=1)

        per_destination[str(dest)] = DestinationTermStats(
            taste_mean=float(taste_sim_matrix.mean()),
            taste_std=float(taste_sim_matrix.std()),
            cat_mean=float(cat_affinity_matrix.mean()),
            cat_std=float(cat_affinity_matrix.std()),
            local_mean=float(local_term_matrix.mean()),
            local_std=float(local_term_matrix.std()),
            qual_mean=float(qual_matrix.mean()),
            qual_std=float(qual_matrix.std()),
            party_mean=float(party_matrix.mean()),
            party_std=float(party_matrix.std()),
            price_mean=float(price_matrix.mean()),
            price_std=float(price_matrix.std()),
        )

    return TermStandardization(per_destination=per_destination)


def standardize_and_weight(
    taste_sim: FloatArray,
    cat_affinity: FloatArray,
    local_term: FloatArray,
    latent_quality: FloatArray,
    party_fit: FloatArray,
    price_fit: FloatArray,
    novelty: FloatArray,
    destination: str,
    standardization: TermStandardization,
    weights: UtilityWeights,
) -> FloatArray:
    """Z-score each of the 7 raw terms against `standardization`'s reference stats
    for `destination`, then apply `weights` and sum -- the RC2a combination step."""
    stats = standardization.per_destination[destination]
    result: FloatArray = (
        weights.w_taste * zscore(taste_sim, stats.taste_mean, stats.taste_std)
        + weights.w_cat * zscore(cat_affinity, stats.cat_mean, stats.cat_std)
        + weights.w_local * zscore(local_term, stats.local_mean, stats.local_std)
        + weights.w_qual * zscore(latent_quality, stats.qual_mean, stats.qual_std)
        + weights.w_party * zscore(party_fit, stats.party_mean, stats.party_std)
        + weights.w_price * zscore(price_fit, stats.price_mean, stats.price_std)
        + weights.w_novel
        * zscore(novelty, standardization.novelty_mean, standardization.novelty_std)
    )
    return result


def true_utility_noise_free(
    weights: UtilityWeights,
    standardization: TermStandardization,
    destination: str,
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

    Block A RC2a: each of the 7 raw terms is now standardized (module docstring)
    before weighting -- `standardization`/`destination` select the right reference
    stats.
    """
    taste_sim = cosine_similarity_to_taste(taste_vec, poi_semantic)
    cat_affinity = taste_vec[poi_category_taste_idx]
    local_term = poi_latent_localness * (-touristiness_pref)
    party = party_fit_array(
        party_type, poi_wheelchair, poi_stroller, poi_kid_friendly, poi_category
    )
    price = price_fit_array(budget, poi_price_level_true)

    return standardize_and_weight(
        taste_sim,
        cat_affinity,
        local_term,
        poi_latent_quality,
        party,
        price,
        novelty,
        destination,
        standardization,
        weights,
    )
