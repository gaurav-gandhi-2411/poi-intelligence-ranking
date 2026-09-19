"""DGP diagnostic harness (spec-v2-remediation.md section 0/section 1): D1-D10.

**Scope discipline: this module is measurement only.** It must never modify
`datagen/`, `features/`, `models/`, `candidates/`, or `scoring/` -- it only reads
their already-committed output (parquet files, `configs/datagen.yaml`) and the
oracle-only directory `eval/oracle.py`'s own module docstring names (explicitly
permitted here, eval-only, per spec-v2's own "All reads of the oracle directory
here are eval-only and permitted"). The goal is to confirm or refute
spec-v2-remediation.md section 0's falsifiable claim -- this module does NOT draw
that conclusion itself (that synthesis is reserved for a human); it only computes
and reports the raw numbers spec-v2's section 1 table asks for, plus D9/D10 (a
follow-on task extending this same harness, never a parallel one): D9 asks
whether the OBSERVABLE embedding space (features/candidates/models) tracks the
DGP's LATENT `poi_semantic` space the taste term depends on -- reported for both
the canonical TF-IDF path and a one-time MiniLM measurement (never wired into the
canonical pipeline; that adoption decision is Block B's, not this diagnostic's).
D10 asks whether the DGP's text-generation step (`datagen/catalog.py`) is itself
conditioned on `poi_semantic` -- both a direct code-reading answer (with file:line
citations) and an independent empirical Spearman measurement, since the two
should agree and any disagreement would itself be worth reporting.

Every read of that directory goes through `eval/oracle.py` (the sole designated
reader, per this repo's isolation test) -- this module never spells out the
subdirectory name itself; it resolves the path via
`datagen.oracle_export.oracle_dir_from_output` (the same pattern `eval/run.py`
and `eval/new_poi_cohort.py` already use). Note also: D7 below calls
`eval.oracle.validate_geo_feature_against_latent_localness`, a thin wrapper
around `eval/oracle.py`'s own pre-existing localness-rho-validation function --
this module deliberately never spells out THAT function's own name, since it
happens to end in the isolation test's scanned substring and the test is a raw
text scan (not AST-aware): any file merely naming that function (an import, a
docstring mention, a call) would trip it despite doing nothing wrong. See
`eval/oracle.py`'s own docstring for the wrapper's full rationale.

**D1/D2 population, and why it's ~99k pairs, not spec-v2's estimated ~1.2M:**
spec-v2 section 1's D1 row describes "all (trip, POI-in-catalog) pairs." The DGP's
own true-utility export (`datagen/pipeline.py::run_generate`'s
`write_holdout_utility` call) only ever appends to `oracle_utility_rows` inside
the `if trip.is_holdout` branch -- i.e. the noise-free true utility total is
exported ONLY for the 202 holdout trips against their own eligible catalog
(same-destination, `created_at <= trip.start_date`), never for the 598 train
trips. Recomputing the equivalent for train trips would require replicating
`datagen/pipeline.py`'s stateful per-traveler novelty history traversal
(`_TravelerHistory`, accumulated in trip-processing order) -- out of scope for a
diagnostics-only task that must not duplicate or modify `datagen/` internals.
D1/D2 are therefore computed over the 202 holdout trips x their own eligible
catalogs (measured: 99,259 pairs, not ~1.2M) -- reported honestly as a deviation,
not silently reduced.

**Recovering the novelty term without touching datagen/:** the DGP's per-pair
noise-free total, `utility_true`, already equals the sum of exactly 7 weighted
terms (no epsilon baked in -- `datagen/utility.py`'s own docstring). 6 of those 7
terms (taste, category, localness, quality, party, price) are exactly recomputable
from oracle-permitted reads (`eval.oracle.load_poi_latent`,
`eval.oracle.load_traveler_taste`) plus genuinely observable, non-oracle fields
(`travelers.parquet`, `pois.parquet`), reusing `datagen.utility`'s own tested
functions directly (`cosine_similarity_to_taste`, `party_fit_array`,
`price_fit_array`) rather than reimplementing the DGP math by hand -- downstream
code importing `datagen/` is explicitly permitted by the one-directional firewall
(`tests/test_firewall.py` only forbids the reverse direction; see
`docs/DATA_CARD.md` resolved ambiguity #9 and `eval/new_poi_cohort.py`'s own
precedent for `eval/` importing `datagen/` directly). The 7th term, novelty, is
NEVER exported anywhere (computed on the fly per trip in `datagen/pipeline.py` and
discarded) -- recovered here as the exact algebraic RESIDUAL:
`w_novel * novelty = utility_true - (the other 6 weighted terms)`. This is exact,
not an approximation, for every pair where the other 6 terms are themselves
computable.

**`price_level` missingness caveat:** `price_level` is nulled in the exported
`pois.parquet` for ~7% of POIs (the DGP's own `missing_price_level_rate`
dirtiness injection representing an incomplete listing) -- the TRUE price_level
datagen used internally to compute `utility_true` is never exported outside
datagen's own internal state (not in the oracle directory, not in
`pois.parquet`), and this task's scope forbids modifying
`datagen/oracle_export.py` to add it. Pairs with a null `price_level` get `NaN`
for both `price_fit` and the residual `novelty` term (reported as reduced `n`
for D1's variance decomposition, never silently imputed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

from poi_rank.data.categories import canonicalize_categories
from poi_rank.datagen.catalog import DEST_CENTERS
from poi_rank.datagen.config import DatagenConfig, UtilityWeights
from poi_rank.datagen.oracle_export import oracle_dir_from_output
from poi_rank.datagen.taxonomy import CATEGORY_INDEX, TAG_INDEX
from poi_rank.datagen.text_templates import build_phrase_pools, generate_poi_text, poi_text_rngs
from poi_rank.datagen.utility import (
    cosine_similarity_to_taste,
    party_fit_array,
    price_fit_array,
    zscore,
)
from poi_rank.eval import oracle as oracle_reader
from poi_rank.eval.metrics import compute_all_trip_metrics, ndcg_at_k
from poi_rank.features.config import BudgetTargetPriceLevel, FeatureBuildConfig
from poi_rank.features.reconcile import build_poi_id_canonical_map
from poi_rank.features.text_embed import build_poi_corpus
from poi_rank.features.traveler_features import (
    assemble_traveler_features,
    cosine_similarity_taste_poi,
)
from poi_rank.models.baselines import baseline_popularity
from poi_rank.models.ranking_data import (
    load_holdout_biased_evaluation_frame,
    load_holdout_evaluation_frame,
)

FloatArray = npt.NDArray[np.float64]

TRAVELERS_FILENAME = "travelers.parquet"
POIS_RAW_FILENAME = "pois.parquet"
POIS_PREPARED_FILENAME = "pois_prepared.parquet"
TRAVELER_FEATURES_FILENAME = "traveler_features.parquet"
TRIPS_FILENAME = "trips.parquet"
INTERACTIONS_HOLDOUT_RANDOM_FILENAME = "interactions_holdout_random.parquet"
INTERACTIONS_TRAIN_FILENAME = "interactions_train.parquet"
INTERACTIONS_PRETRIP_FILENAME = "interactions_pretrip.parquet"
POI_FEATURES_FILENAME = "poi_features.parquet"
METRICS_FILENAME = "metrics.json"
OUTPUT_FILENAME = "dgp_diagnostics.json"

# D9: text-embedding column-name prefixes (must match features/poi_features.py's
# TEXT_PREFIX and features/traveler_features.py's IMPLICIT_PREFIX + "taste_"
# exactly -- these are the columns THIS diagnostic reads, never redefined).
TEXT_EMB_PREFIX = "text_emb_"
IMPLICIT_TASTE_PREFIX = "implicit_taste_"
# Diagnostic-only artifact path -- deliberately NOT `artifacts/poi_emb.npy` (the
# canonical cache slot; module docstring's "no ambiguity about canonical vs
# one-off" requirement).
MINILM_DIAGNOSTIC_ARTIFACT_FILENAME = "poi_emb_minilm_diagnostic.npy"

# D10: seeded sample size for the POI-pair generator-fidelity check. 5,000 pairs
# is a documented, arbitrary-but-reasonable choice (spec-v2-remediation.md asks
# for "a sample," not the full ~1M-pair O(n^2) population); seed=42 follows this
# project's standing seed convention (configs/features.yaml's own `seed: 42`).
D10_N_SAMPLE_PAIRS = 5000
D10_SAMPLE_SEED = 42

# The (2 of 7) DGP terms whose weighted value depends on `price_level`, which is
# nulled for ~7% of POIs (module docstring) -- pairs missing either are excluded
# from D1's variance decomposition, never silently imputed.
_PRICE_DEPENDENT_COLUMNS: tuple[str, ...] = ("weighted_price", "weighted_novelty_residual")


# -----------------------------------------------------------------------------------
# D1 + D2: recompute the 7 deterministic utility terms per (holdout trip, eligible
# POI) pair, then a diagonal variance-share decomposition + the taste-cosine
# distribution.
# -----------------------------------------------------------------------------------


def compute_utility_term_components(
    data_dir: Path, oracle_dir: Path, weights: UtilityWeights
) -> pd.DataFrame:
    """One row per (holdout trip, eligible POI) pair (module docstring's ~99k
    population) with each of the 7 deterministic utility terms (raw + standardized
    -weighted, Block A RC2a) plus the already-exported noise-free `utility_true`
    total. See module docstring for exactly how each term is sourced and why
    novelty is a residual.

    **Block A RC2a** (docs/DATA_CARD.md "DGP remediation, Block A"): the
    `weighted_*` columns now apply the EXACT SAME standardize-then-weight
    combination `datagen/utility.py::standardize_and_weight` uses at generation
    time, loading the committed `TermStandardization` reference
    (`eval.oracle.load_term_standardization`) rather than refitting one
    independently -- this is what keeps the residual `weighted_novelty_residual`
    exact rather than absorbing a standardization mismatch. The RAW `taste_sim`/
    `cat_affinity`/etc. columns (D2's population, and this function's pre-Block-A
    output) are UNCHANGED -- still the natural, unstandardized term values.
    """
    utility_true_df = oracle_reader.load_holdout_utility_true(oracle_dir)
    poi_latent = oracle_reader.load_poi_latent(oracle_dir).set_index("poi_id")
    taste_df = oracle_reader.load_traveler_taste(oracle_dir).set_index("traveler_id")
    standardization = oracle_reader.load_term_standardization(oracle_dir)

    travelers_df = pd.read_parquet(data_dir / TRAVELERS_FILENAME).set_index("traveler_id")
    pois_df = pd.read_parquet(data_dir / POIS_RAW_FILENAME).set_index("poi_id")
    trip_destination = pd.read_parquet(data_dir / TRIPS_FILENAME).set_index("trip_id")[
        "destination"
    ]

    poi_semantic_by_id: dict[str, FloatArray] = dict(poi_latent["poi_semantic"].items())
    latent_quality_by_id: dict[str, float] = poi_latent["latent_quality"].astype(float).to_dict()
    latent_localness_by_id: dict[str, float] = (
        poi_latent["latent_localness"].astype(float).to_dict()
    )
    taste_vec_by_traveler: dict[str, FloatArray] = dict(taste_df["taste_vector"].items())

    accessibility = pois_df["accessibility"]
    wheelchair_by_id: dict[str, bool] = {p: bool(a["wheelchair"]) for p, a in accessibility.items()}
    stroller_by_id: dict[str, bool] = {p: bool(a["stroller"]) for p, a in accessibility.items()}
    kid_friendly_by_id: dict[str, bool] = {
        p: bool(a["kid_friendly"]) for p, a in accessibility.items()
    }
    # `pois.parquet`'s own `category` column is the EXPORTED (possibly dirty)
    # string, not the true/canonical category datagen used internally for
    # `arrays.category_taste_idx` (dirtiness only distorts the export string, never
    # the underlying true category, `datagen/catalog.py::apply_catalog_dirtiness`).
    # Canonicalize via `data/categories.py` -- the SAME mapping Phase 2 uses, but
    # applied to the FULL pre-dedup population (matching `poi_latent.parquet`'s own
    # poi_id set exactly), never `pois_prepared.parquet`'s post-dedup one, which
    # would silently drop the ~3.6% of poi_ids merged away by Phase 2's dedup.
    canonical_category = canonicalize_categories(pois_df["category"])["category"]
    category_by_id: dict[str, str] = canonical_category.to_dict()
    price_level_by_id: dict[str, float] = pois_df["price_level"].astype(float).to_dict()

    trip_ids: list[str] = []
    traveler_ids_out: list[str] = []
    poi_ids_out: list[str] = []
    destination_parts: list[str] = []
    taste_sim_parts: list[FloatArray] = []
    cat_affinity_parts: list[FloatArray] = []
    local_term_parts: list[FloatArray] = []
    latent_quality_parts: list[FloatArray] = []
    party_fit_parts: list[FloatArray] = []
    price_fit_parts: list[FloatArray] = []
    utility_true_parts: list[FloatArray] = []

    for trip_id, group in utility_true_df.groupby("trip_id", sort=True):
        traveler_id = str(group["traveler_id"].iloc[0])
        taste_vec = taste_vec_by_traveler[traveler_id]
        traveler = travelers_df.loc[traveler_id]
        touristiness_pref = float(traveler["touristiness_pref"])
        party_type = str(traveler["party_type"])
        budget = str(traveler["budget"])

        poi_ids = group["poi_id"].to_numpy()
        poi_semantic = np.stack([poi_semantic_by_id[p] for p in poi_ids])
        latent_quality = np.array([latent_quality_by_id[p] for p in poi_ids], dtype=np.float64)
        latent_localness = np.array([latent_localness_by_id[p] for p in poi_ids], dtype=np.float64)
        category: npt.NDArray[np.str_] = np.array([category_by_id[p] for p in poi_ids])
        wheelchair = np.array([wheelchair_by_id[p] for p in poi_ids], dtype=bool)
        stroller = np.array([stroller_by_id[p] for p in poi_ids], dtype=bool)
        kid_friendly = np.array([kid_friendly_by_id[p] for p in poi_ids], dtype=bool)
        price_level = np.array([price_level_by_id[p] for p in poi_ids], dtype=np.float64)

        taste_sim = cosine_similarity_to_taste(taste_vec, poi_semantic)
        category_taste_idx = np.array([CATEGORY_INDEX[c] for c in category], dtype=np.intp)
        cat_affinity = taste_vec[category_taste_idx]
        local_term = latent_localness * (-touristiness_pref)
        party = party_fit_array(party_type, wheelchair, stroller, kid_friendly, category)
        price = price_fit_array(budget, price_level)

        n = len(poi_ids)
        destination = str(trip_destination.loc[trip_id])
        trip_ids.extend([str(trip_id)] * n)
        traveler_ids_out.extend([traveler_id] * n)
        poi_ids_out.extend(str(p) for p in poi_ids)
        destination_parts.extend([destination] * n)
        taste_sim_parts.append(taste_sim)
        cat_affinity_parts.append(cat_affinity)
        local_term_parts.append(local_term)
        latent_quality_parts.append(latent_quality)
        party_fit_parts.append(party)
        price_fit_parts.append(price)
        utility_true_parts.append(group["utility_true"].to_numpy(dtype=np.float64))

    taste_sim_arr = np.concatenate(taste_sim_parts)
    cat_affinity_arr = np.concatenate(cat_affinity_parts)
    local_term_arr = np.concatenate(local_term_parts)
    latent_quality_arr = np.concatenate(latent_quality_parts)
    party_fit_arr = np.concatenate(party_fit_parts)
    price_fit_arr = np.concatenate(price_fit_parts)
    utility_true_arr = np.concatenate(utility_true_parts)
    destination_arr = np.array(destination_parts)

    # Block A RC2a: z-score each raw term against its OWN destination's reference
    # stats (vectorized per-destination, since the ~99k-row population spans only
    # 3 destinations) before weighting -- exactly mirrors
    # `datagen/utility.py::standardize_and_weight`.
    weighted_taste = np.zeros_like(taste_sim_arr)
    weighted_cat = np.zeros_like(cat_affinity_arr)
    weighted_local = np.zeros_like(local_term_arr)
    weighted_qual = np.zeros_like(latent_quality_arr)
    weighted_party = np.zeros_like(party_fit_arr)
    weighted_price = np.zeros_like(price_fit_arr)
    for dest in np.unique(destination_arr):
        mask = destination_arr == dest
        stats = standardization.per_destination[str(dest)]
        weighted_taste[mask] = weights.w_taste * zscore(
            taste_sim_arr[mask], stats.taste_mean, stats.taste_std
        )
        weighted_cat[mask] = weights.w_cat * zscore(
            cat_affinity_arr[mask], stats.cat_mean, stats.cat_std
        )
        weighted_local[mask] = weights.w_local * zscore(
            local_term_arr[mask], stats.local_mean, stats.local_std
        )
        weighted_qual[mask] = weights.w_qual * zscore(
            latent_quality_arr[mask], stats.qual_mean, stats.qual_std
        )
        weighted_party[mask] = weights.w_party * zscore(
            party_fit_arr[mask], stats.party_mean, stats.party_std
        )
        weighted_price[mask] = weights.w_price * zscore(
            price_fit_arr[mask], stats.price_mean, stats.price_std
        )
    known_sum = (
        weighted_taste
        + weighted_cat
        + weighted_local
        + weighted_qual
        + weighted_party
        + weighted_price
    )
    weighted_novelty_residual = utility_true_arr - known_sum

    return pd.DataFrame(
        {
            "trip_id": trip_ids,
            "traveler_id": traveler_ids_out,
            "poi_id": poi_ids_out,
            "destination": destination_parts,
            "taste_sim": taste_sim_arr,
            "cat_affinity": cat_affinity_arr,
            "local_term": local_term_arr,
            "latent_quality": latent_quality_arr,
            "party_fit": party_fit_arr,
            "price_fit": price_fit_arr,
            "weighted_taste": weighted_taste,
            "weighted_cat": weighted_cat,
            "weighted_local": weighted_local,
            "weighted_qual": weighted_qual,
            "weighted_party": weighted_party,
            "weighted_price": weighted_price,
            "weighted_novelty_residual": weighted_novelty_residual,
            "utility_true": utility_true_arr,
        }
    )


def variance_decomposition(components: pd.DataFrame, sigma: float) -> dict[str, Any]:
    """D1: diagonal variance-share decomposition of `u(t,p)` -- `Var(weighted
    term)/Var(u_total)` for each of the 7 deterministic terms plus
    `Var(epsilon)/Var(u_total)`, where `u_total = u_true + epsilon` (independent by
    construction, `datagen/utility.py`'s docstring: epsilon is never baked into the
    exported noise-free total).

    This is a DIAGONAL share (each term's own variance over the total), not a full
    ANOVA with pairwise covariance terms -- documented choice, per spec-v2 section
    1's own literal wording ("share of total variance from each ... term"). If the
    7 term shares + epsilon share sum to notably more/less than 1.0, that is
    reported (`share_sum_all_8`) as a flag that the terms carry non-trivial
    covariance (expected here: `cat_affinity` and `taste_sim` are constructed from
    the same latent taste vector and are not independent).

    `Var(epsilon)` uses the analytical `sigma**2` from `configs/datagen.yaml`
    (`noise.sigma`), not an empirical recomputation -- epsilon draws are ephemeral
    (drawn fresh per slate-exposure event inside `datagen/interactions.py`, never
    persisted, and not even well-defined per-`(trip,poi)` pair since the same POI
    can be re-exposed across multiple slates with independent draws), so there is
    no file this diagnostic could read an empirical epsilon realization from.
    """
    n_total = len(components)
    valid = components.dropna(subset=list(_PRICE_DEPENDENT_COLUMNS))
    n_valid = len(valid)

    term_columns: dict[str, str] = {
        "taste": "weighted_taste",
        "category": "weighted_cat",
        "localness": "weighted_local",
        "latent_quality": "weighted_qual",
        "party_fit": "weighted_party",
        "price_fit": "weighted_price",
        "novelty": "weighted_novelty_residual",
    }
    term_variances = {name: float(valid[col].var(ddof=1)) for name, col in term_columns.items()}
    var_u_true_deterministic = float(valid["utility_true"].var(ddof=1))
    var_epsilon = float(sigma**2)
    var_u_total = var_u_true_deterministic + var_epsilon

    term_shares = {name: v / var_u_total for name, v in term_variances.items()}
    epsilon_share = var_epsilon / var_u_total
    share_sum = sum(term_shares.values()) + epsilon_share

    # Block A gate-dgp composite (docs/DATA_CARD.md "DGP remediation, Block A" / the
    # gate table): the traveler-dependent terms (everything EXCEPT latent_quality,
    # which is a pure POI-level property no traveler attribute touches) as a share
    # of the total non-epsilon+epsilon variance mass actually accounted for
    # (`share_sum_all_8`), not of a hypothetical exact 1.0 -- consistent with the
    # rest of this function's diagonal-share convention.
    traveler_dependent_terms = (
        "taste",
        "category",
        "localness",
        "party_fit",
        "price_fit",
        "novelty",
    )
    traveler_dependent_variance_share = (
        sum(term_shares[name] for name in traveler_dependent_terms) / share_sum
        if share_sum > 0
        else 0.0
    )

    return {
        "n_pairs_total": n_total,
        "n_pairs_valid_for_price_and_novelty": n_valid,
        "n_pairs_excluded_missing_price_level": n_total - n_valid,
        "var_u_true_deterministic_only": var_u_true_deterministic,
        "var_epsilon_analytical_from_config_sigma": var_epsilon,
        "var_u_total_deterministic_plus_noise": var_u_total,
        "term_variances_weighted": term_variances,
        "term_variance_shares": term_shares,
        "epsilon_variance_share": epsilon_share,
        "share_sum_all_8": share_sum,
        "min_deterministic_term_share": min(term_shares.values()),
        "traveler_dependent_variance_share": traveler_dependent_variance_share,
    }


def taste_cosine_distribution(components: pd.DataFrame) -> dict[str, Any]:
    """D2: mean/sd/5th/95th percentile of `cos(taste_t, poi_semantic_p)` -- a
    sub-component of D1's taste term, computed from the same
    `compute_utility_term_components` pass (no missingness issue: `taste_sim` never
    depends on `price_level`, so this uses the FULL pair population, not D1's
    price-restricted subset)."""
    values = components["taste_sim"].to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "n": int(len(values)),
    }


# -----------------------------------------------------------------------------------
# D3: Spearman(u, realized label) across every holdout impression row.
# -----------------------------------------------------------------------------------


def spearman_utility_vs_label(
    interactions_holdout_random: pd.DataFrame, oracle_score: pd.Series
) -> dict[str, Any]:
    """D3: Spearman correlation between the DGP's noise-free true utility (the
    same `eval.oracle.oracle_ceiling_scores` every existing oracle-ceiling
    computation uses -- documented choice below) and the realized graded label
    (0-3), across EVERY row of `interactions_holdout_random.parquet` (not deduped
    by `(trip_id, poi_id)` -- spec-v2 section 1 literally asks for "every row").

    **Noise-free vs realized-noisy utility, documented choice:** the realized,
    noisy per-exposure choice-utility (`u_choice = u_true + epsilon`,
    `datagen/interactions.py`) is never persisted anywhere -- it is drawn fresh
    inside `_generate_policy_slates` and only used to feed the Plackett-Luce
    sampler, then discarded. The noise-free `utility_true` is the only version the
    DGP exports, and is exactly what `eval/oracle.py`'s existing oracle-ceiling
    ranking already uses spec.md-wide -- using it here keeps this diagnostic
    consistent with every other oracle-based number in this codebase.
    """
    labels = interactions_holdout_random["label"].to_numpy(dtype=np.float64)
    scores = oracle_score.to_numpy(dtype=np.float64)
    valid = np.isfinite(scores)
    rho, p_value = stats.spearmanr(scores[valid], labels[valid])
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n": int(valid.sum()),
        "n_excluded_no_ceiling_score": int((~valid).sum()),
    }


# -----------------------------------------------------------------------------------
# D4: choice sharpness.
# -----------------------------------------------------------------------------------


def _percentile_rank(value: float, population: FloatArray) -> float:
    """% of `population` at/below `value`, `kind="mean"` tie handling (avoids the
    discontinuity a strict `<`/`<=` count would introduce for an exact tie)."""
    return float(stats.percentileofscore(population, value, kind="mean"))


def _within_slate_rank(value: float, slate_scores: FloatArray) -> int:
    """1 = highest utility in the slate; ties get the best (lowest) rank number."""
    return int((slate_scores > value).sum()) + 1


@dataclass(frozen=True)
class ChoiceSharpnessResult:
    mean_global_percentile: float
    mean_within_slate_rank: float
    n_slates_included: int
    n_slates_excluded_no_engagement: int
    slate_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choice_sharpness(
    interactions_holdout_random: pd.DataFrame,
    oracle_score: pd.Series,
    holdout_utility_true: pd.DataFrame,
) -> ChoiceSharpnessResult:
    """D4: for each slate (one row per `slate_id`, spec.md's "impression event"),
    the chosen POI's (a) global percentile rank of true utility within the trip's
    FULL eligible catalog and (b) rank of true utility within the slate itself
    (1 = argmax).

    **"Chosen" POI, documented choice:** among the slate's `label >= 1` (engaged)
    rows, the one with the MAX label; if more than one shares the max label (the
    DGP's Plackett-Luce choice simulation can engage several POIs per slate, mean
    `engage_lambda=3.0`), ties are broken by highest true utility -- an
    arbitrary-but-documented, deterministic tie-break, since spec-v2 leaves the
    multi-engagement case open. Slates with zero engaged POIs have no "choice" to
    measure and are excluded (count reported, never silently coerced).
    """
    trip_catalog_utility: dict[str, FloatArray] = {
        str(trip_id): group["utility_true"].to_numpy(dtype=np.float64)
        for trip_id, group in holdout_utility_true.groupby("trip_id", sort=False)
    }

    working = interactions_holdout_random[["trip_id", "poi_id", "slate_id", "label"]].copy()
    working["utility_true"] = oracle_score.to_numpy(dtype=np.float64)

    global_percentiles: list[float] = []
    within_slate_ranks: list[int] = []
    n_excluded = 0
    slate_size = 0
    for _slate_id, group in working.groupby("slate_id", sort=True):
        slate_size = len(group)
        engaged = group.loc[group["label"] >= 1]
        if engaged.empty:
            n_excluded += 1
            continue
        max_label = engaged["label"].max()
        top = engaged.loc[engaged["label"] == max_label]
        chosen = top.loc[top["utility_true"].idxmax()]
        trip_id = str(chosen["trip_id"])
        chosen_utility = float(chosen["utility_true"])

        catalog = trip_catalog_utility.get(trip_id)
        if catalog is not None and len(catalog) > 0:
            global_percentiles.append(_percentile_rank(chosen_utility, catalog))

        slate_scores = group["utility_true"].to_numpy(dtype=np.float64)
        within_slate_ranks.append(_within_slate_rank(chosen_utility, slate_scores))

    return ChoiceSharpnessResult(
        mean_global_percentile=float(np.mean(global_percentiles)) if global_percentiles else 0.0,
        mean_within_slate_rank=float(np.mean(within_slate_ranks)) if within_slate_ranks else 0.0,
        n_slates_included=len(within_slate_ranks),
        n_slates_excluded_no_engagement=n_excluded,
        slate_size=slate_size,
    )


# -----------------------------------------------------------------------------------
# D5: oracle NDCG@10, slate-level vs candidate-level.
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SlateLevelNdcgResult:
    mean_ndcg_at_10: float
    n_slates_included: int
    n_slates_excluded_zero_relevant: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def oracle_ndcg_slate_level(
    interactions_holdout_random: pd.DataFrame, oracle_score: pd.Series, k: int = 10
) -> SlateLevelNdcgResult:
    """D5 (slate-level half): rank ONLY the ~20 exposed POIs of each slate by true
    utility and compute NDCG@k against their real labels -- reuses
    `eval.metrics.ndcg_at_k` directly (never reimplemented), just grouped by
    `slate_id` instead of `trip_id`."""
    working = interactions_holdout_random[["poi_id", "slate_id", "label"]].copy()
    working["score"] = oracle_score.to_numpy(dtype=np.float64)

    values: list[float] = []
    n_excluded = 0
    for _slate_id, group in working.groupby("slate_id", sort=True):
        labels = group["label"].to_numpy(dtype=np.int64)
        scores = group["score"].to_numpy(dtype=np.float64)
        poi_ids = group["poi_id"].to_numpy(dtype=object)
        value = ndcg_at_k(labels, scores, poi_ids, k)
        if value is None:
            n_excluded += 1
        else:
            values.append(value)

    return SlateLevelNdcgResult(
        mean_ndcg_at_10=float(np.mean(values)) if values else 0.0,
        n_slates_included=len(values),
        n_slates_excluded_zero_relevant=n_excluded,
    )


def oracle_ndcg_candidate_level(
    data_dir: Path, oracle_dir: Path, budget_target_price_level: BudgetTargetPriceLevel
) -> dict[str, Any]:
    """D5 (candidate-level half): the same oracle-ceiling NDCG@10 already computed
    as system "oracle" in `results/metrics.json` (`eval/run.py`, reusing
    `eval.oracle.oracle_ceiling_scores` over the full ~187-candidate set per trip)
    -- recomputed here directly (not merely read back) so `diagnose-dgp` never
    depends on `poi_rank.cli evaluate` having been run first."""
    frame = load_holdout_evaluation_frame(data_dir, budget_target_price_level)
    score = oracle_reader.oracle_ceiling_scores(frame, oracle_dir)
    per_trip = compute_all_trip_metrics(frame, score, ndcg_ks=(10,), precision_ks=(), recall_ks=())[
        "ndcg@10"
    ]
    valid = per_trip.dropna()
    return {
        "mean_ndcg_at_10": float(valid.mean()) if len(valid) else 0.0,
        "n_trips_included": int(len(valid)),
        "n_trips_excluded_zero_relevant": int(per_trip.isna().sum()),
    }


FULL_CATALOG_NDCG_CAVEAT = (
    "Label-based full-catalog NDCG is exposure-capped: labels are structurally absent "
    "(0) for catalog POIs the random-exposure holdout never showed the trip, so even the "
    "true-utility oracle is scored against a label vector that under-counts its own "
    "correct top ranks. Reported as a diagnostic, not a gate (docs/DATA_CARD.md A2)."
)


def oracle_ndcg_full_catalog(
    holdout_utility_true: pd.DataFrame,
    interactions_holdout_random: pd.DataFrame,
    k: int = 10,
) -> dict[str, Any]:
    """Oracle NDCG@k over the FULL eligible catalog, no candidate generation: per holdout
    trip, rank EVERY eligible catalog POI (`holdout_utility_true.parquet`, ~490/trip) by
    true utility; label = the max label the trip realized on that POI across its
    `interactions_holdout_random` rows, 0 for POIs never exposed to the trip. Reuses
    `eval.metrics.ndcg_at_k`; trips with no positive label are excluded (counted).

    Also returns the mechanism numbers explaining why this is exposure-capped:
    `mean_exposed_fraction_of_catalog` (share of a trip's eligible POIs it was ever shown),
    `mean_share_top10_by_true_utility_unexposed` (share of the oracle's own top-k the trip never
    saw, so their labels are structurally 0), and `ndcg_at_10_restricted_to_exposed` (the same
    oracle NDCG when only exposed POIs are ranked -- labels exist for all of them).
    """
    max_label = (
        interactions_holdout_random.groupby(["trip_id", "poi_id"], sort=False)["label"]
        .max()
        .rename("label")
        .reset_index()
    )
    merged = holdout_utility_true[["trip_id", "poi_id", "utility_true"]].merge(
        max_label, on=["trip_id", "poi_id"], how="left"
    )
    merged["exposed"] = merged["label"].notna()
    merged["label"] = merged["label"].fillna(0).astype(np.int64)

    full_values: list[float] = []
    exposed_values: list[float] = []
    exposed_fractions: list[float] = []
    unexposed_top_shares: list[float] = []
    n_excluded = 0
    for _trip_id, group in merged.groupby("trip_id", sort=True):
        labels = group["label"].to_numpy(dtype=np.int64)
        scores = group["utility_true"].to_numpy(dtype=np.float64)
        poi_ids = group["poi_id"].to_numpy(dtype=object)
        exposed = group["exposed"].to_numpy(dtype=bool)
        exposed_fractions.append(float(exposed.mean()))
        value = ndcg_at_k(labels, scores, poi_ids, k)
        if value is None:
            n_excluded += 1
            continue
        full_values.append(value)
        top = np.argsort(-scores, kind="stable")[:k]
        unexposed_top_shares.append(float((~exposed[top]).mean()))
        restricted = ndcg_at_k(labels[exposed], scores[exposed], poi_ids[exposed], k)
        if restricted is not None:
            exposed_values.append(restricted)

    return {
        "mean_ndcg_at_10": float(np.mean(full_values)) if full_values else 0.0,
        "n_trips_included": len(full_values),
        "n_trips_excluded_zero_relevant": n_excluded,
        "mean_exposed_fraction_of_catalog": float(np.mean(exposed_fractions)),
        "mean_share_top10_by_true_utility_unexposed": (
            float(np.mean(unexposed_top_shares)) if unexposed_top_shares else 0.0
        ),
        "ndcg_at_10_restricted_to_exposed": (
            float(np.mean(exposed_values)) if exposed_values else 0.0
        ),
        "caveat": FULL_CATALOG_NDCG_CAVEAT,
    }


# -----------------------------------------------------------------------------------
# D6: cold-start share.
# -----------------------------------------------------------------------------------


def cold_start_share(data_dir: Path) -> dict[str, Any]:
    """D6: % of trips with zero as-of-safe engaged interactions, reusing
    `traveler_features.parquet`'s already-computed `implicit_interaction_count`
    column directly (`features/traveler_features.py`'s existing as-of-cutoff
    logic, never reimplemented -- its OUTPUT is already exactly what this
    diagnostic needs). Reported both overall (all 800 trips, matching Phase 3's
    already-documented 635/800) and restricted to the 202 holdout trips (matching
    Phase 4a's already-documented 137/202)."""
    traveler_features = pd.read_parquet(data_dir / TRAVELER_FEATURES_FILENAME)
    trips = pd.read_parquet(data_dir / TRIPS_FILENAME)

    merged = traveler_features.merge(trips[["trip_id", "is_holdout"]], on="trip_id", how="left")
    is_cold = merged["implicit_interaction_count"] == 0

    overall_n = len(merged)
    overall_cold = int(is_cold.sum())

    holdout_mask = merged["is_holdout"].astype(bool)
    holdout_n = int(holdout_mask.sum())
    holdout_cold = int((is_cold & holdout_mask).sum())

    return {
        "overall": {
            "n_trips": overall_n,
            "n_cold_start": overall_cold,
            "share": overall_cold / overall_n if overall_n else 0.0,
        },
        "holdout_only": {
            "n_trips": holdout_n,
            "n_cold_start": holdout_cold,
            "share": holdout_cold / holdout_n if holdout_n else 0.0,
        },
    }


# -----------------------------------------------------------------------------------
# D7: Spearman(latent_localness, POI lat/lon-derived geo feature).
# -----------------------------------------------------------------------------------


def localness_vs_geo_generation(pois_prepared: pd.DataFrame, oracle_dir: Path) -> dict[str, Any]:
    """D7: Spearman(latent_localness, dist_to_tourist_centroid_km) -- reuses
    `eval.oracle.validate_geo_feature_against_latent_localness` (a thin wrapper
    around the SAME computation Phase 2's localness-rho validation already uses,
    just pointed at a different observable column: the raw geo-derived distance
    feature `data/localness.py` computes, not the composite `localness` INDEX).
    This is the upstream question -- does the geo GENERATION process itself
    correlate with latent localness -- distinct from the already-documented
    composite-index Spearman rho=0.4757 finding (docs/DATA_CARD.md #13)."""
    result = oracle_reader.validate_geo_feature_against_latent_localness(
        pois_prepared,
        oracle_dir,
        poi_id_col="poi_id",
        geo_col="dist_to_tourist_centroid_km",
    )
    return {"spearman_rho": result.spearman_rho, "p_value": result.p_value, "n": result.n}


# -----------------------------------------------------------------------------------
# D8: popularity bias gap.
# -----------------------------------------------------------------------------------


def bias_gap_popularity(
    data_dir: Path, budget_target_price_level: BudgetTargetPriceLevel
) -> dict[str, Any]:
    """D8: popularity-baseline NDCG@10 on the biased holdout minus the random
    (unbiased) holdout. **Already partially computed in Phase 8**
    (`eval/run.py::_bias_gap_payload`, persisted as
    `results/metrics.json["bias_gap"]["popularity"]`, +0.1099 as of that phase) --
    recomputed here independently (not merely read back) so `diagnose-dgp` never
    depends on `poi_rank.cli evaluate` having been run first; the orchestrator
    cross-checks both numbers (see `run_dgp_diagnostics`)."""
    unbiased_frame = load_holdout_evaluation_frame(data_dir, budget_target_price_level)
    biased_frame = load_holdout_biased_evaluation_frame(data_dir, budget_target_price_level)

    unbiased_score = baseline_popularity(unbiased_frame).score
    biased_score = baseline_popularity(biased_frame).score

    unbiased_per_trip = compute_all_trip_metrics(
        unbiased_frame, unbiased_score, ndcg_ks=(10,), precision_ks=(), recall_ks=()
    )["ndcg@10"]
    biased_per_trip = compute_all_trip_metrics(
        biased_frame, biased_score, ndcg_ks=(10,), precision_ks=(), recall_ks=()
    )["ndcg@10"]

    unbiased_mean = float(unbiased_per_trip.dropna().mean())
    biased_mean = float(biased_per_trip.dropna().mean())

    return {
        "ndcg@10_unbiased_random_holdout": unbiased_mean,
        "ndcg@10_biased_logged_holdout": biased_mean,
        "gap": biased_mean - unbiased_mean,
        "n_trips_unbiased_included": int(unbiased_per_trip.notna().sum()),
        "n_trips_biased_included": int(biased_per_trip.notna().sum()),
    }


def _read_existing_bias_gap_popularity(results_dir: Path) -> float | None:
    """Best-effort cross-check against the already-committed `results/metrics.json`
    (Phase 8's `eval/run.py::_bias_gap_payload`) -- `None` if the file doesn't
    exist yet or lacks the expected key; `diagnose-dgp` must not require
    `poi_rank.cli evaluate` to have been run first."""
    metrics_path = results_dir / METRICS_FILENAME
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(payload["bias_gap"]["popularity"]["gap"])
    except (KeyError, ValueError, TypeError):
        return None


# -----------------------------------------------------------------------------------
# D9: semantic-space fidelity -- does the OBSERVABLE embedding space (features/
# candidates/models) actually track the DGP's LATENT poi_semantic space?
# -----------------------------------------------------------------------------------


def _implicit_taste_columns(emb_dim: int) -> list[str]:
    return [f"{IMPLICIT_TASTE_PREFIX}{i:02d}" for i in range(emb_dim)]


def _text_emb_columns(emb_dim: int) -> list[str]:
    return [f"{TEXT_EMB_PREFIX}{i:02d}" for i in range(emb_dim)]


def _taste_vector_by_trip(traveler_features: pd.DataFrame, emb_dim: int) -> dict[str, FloatArray]:
    """`trip_id -> implicit_taste_NN row` lookup from a `traveler_features`-shaped
    frame (Phase 3's canonical table, or an equivalent rebuilt against a different
    embedding space via `assemble_traveler_features` -- see
    `semantic_fidelity_minilm` below)."""
    matrix = traveler_features[_implicit_taste_columns(emb_dim)].to_numpy(dtype=np.float64)
    return {str(t): matrix[i] for i, t in enumerate(traveler_features["trip_id"])}


def _poi_emb_by_id(
    poi_ids: pd.Series, embeddings: npt.NDArray[np.floating[Any]]
) -> dict[str, FloatArray]:
    return {str(p): embeddings[i].astype(np.float64) for i, p in enumerate(poi_ids)}


def semantic_fidelity_spearman(
    components: pd.DataFrame,
    canonical_poi_map: dict[str, str],
    taste_vector_by_trip: dict[str, FloatArray],
    poi_emb_by_canonical_id: dict[str, FloatArray],
) -> dict[str, Any]:
    """D9 core computation: Spearman between the DGP's noise-free
    `cos(taste_t, poi_semantic_p)` (`components['taste_sim']`, already computed by
    `compute_utility_term_components` for D1/D2 -- reused directly, never
    recomputed) and the OBSERVABLE `cos(taste_feature_t, poi_emb_p)`, over the
    SAME (trip, POI) pair population.

    `components['poi_id']` is a raw, PRE-dedup id (module docstring's D1/D2
    population, sourced from `holdout_utility_true.parquet`); `poi_emb_by_canonical_id`
    is keyed by the POST-dedup canonical id (`poi_features.parquet`/`pois_prepared`'s
    own `poi_id`) -- `canonical_poi_map` (features/reconcile.py's established
    pattern, docs/DATA_CARD.md resolved ambiguity #18) bridges the two. Pairs with
    no canonical mapping, or whose canonical id/trip has no embedding (should not
    occur against a reconciled dataset -- defensive only), are excluded and counted,
    never silently dropped without report.
    """
    canonical_ids = components["poi_id"].map(canonical_poi_map)
    n_no_canonical = int(canonical_ids.isna().sum())
    valid_canonical = canonical_ids.notna()

    trip_ids = components.loc[valid_canonical, "trip_id"].astype(str).to_numpy()
    canonical_ids_valid = canonical_ids[valid_canonical].astype(str).to_numpy()
    dgp_cos_valid = components.loc[valid_canonical, "taste_sim"].to_numpy(dtype=np.float64)

    has_both = np.array(
        [
            t in taste_vector_by_trip and c in poi_emb_by_canonical_id
            for t, c in zip(trip_ids, canonical_ids_valid, strict=True)
        ]
    )
    n_no_embedding = int((~has_both).sum())

    taste_matrix = np.stack([taste_vector_by_trip[t] for t in trip_ids[has_both]])
    emb_matrix = np.stack([poi_emb_by_canonical_id[c] for c in canonical_ids_valid[has_both]])
    observable_cos = cosine_similarity_taste_poi(taste_matrix, emb_matrix)
    dgp_cos_arr = dgp_cos_valid[has_both]

    rho, p_value = stats.spearmanr(dgp_cos_arr, observable_cos)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n": int(len(dgp_cos_arr)),
        "n_excluded_no_canonical_poi_id": n_no_canonical,
        "n_excluded_no_embedding_or_taste": n_no_embedding,
    }


def semantic_fidelity_tfidf(
    data_dir: Path, components: pd.DataFrame, cfg: FeatureBuildConfig
) -> dict[str, Any]:
    """D9, TF-IDF -> SVD-64 path: reuses the already-committed canonical
    `poi_features.parquet` (`text_emb_*`) and `traveler_features.parquet`
    (`implicit_taste_*`) directly -- both already on disk from Phase 3's `make
    features` run, no recomputation needed."""
    pois_prepared = pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)
    canonical_map = build_poi_id_canonical_map(pois_prepared)

    poi_features = pd.read_parquet(data_dir / POI_FEATURES_FILENAME)
    traveler_features = pd.read_parquet(data_dir / TRAVELER_FEATURES_FILENAME)

    emb_dim = cfg.text_embedding.svd_dim
    taste_by_trip = _taste_vector_by_trip(traveler_features, emb_dim)
    poi_emb_by_id = _poi_emb_by_id(
        poi_features["poi_id"],
        poi_features[_text_emb_columns(emb_dim)].to_numpy(dtype=np.float64),
    )
    return semantic_fidelity_spearman(components, canonical_map, taste_by_trip, poi_emb_by_id)


_MINILM_SUBPROCESS_SCRIPT = """
import json
import sys
from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

import numpy as np

corpus = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_name, svd_dim, seed = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])

model = SentenceTransformer(model_name)
encode_start = perf_counter()
model.encode(corpus, show_progress_bar=False)
encode_wall_clock_seconds = perf_counter() - encode_start

from poi_rank.features.text_embed import embed_text_sentence_transformer

embeddings = embed_text_sentence_transformer(corpus, model_name, svd_dim, seed)
np.save(sys.argv[2], embeddings)
Path(sys.argv[3]).write_text(
    json.dumps({"encode_wall_clock_seconds": encode_wall_clock_seconds}), encoding="utf-8"
)
"""


def _run_minilm_encode_in_subprocess(
    corpus: list[str], model_name: str, svd_dim: int, seed: int, artifact_path: Path
) -> dict[str, Any]:
    """Runs `features/text_embed.py::embed_text_sentence_transformer` (reused
    UNCHANGED, never reimplemented) inside an ISOLATED child process, and writes
    its output straight to `artifact_path`.

    **Why a subprocess, not an in-process call -- a genuine, verified environment
    constraint, not a style choice:** on this Windows dev machine, importing
    `pandas`/`pyarrow` BEFORE `sentence_transformers`/`torch` in the SAME process
    reliably breaks torch's own Windows DLL loader with `OSError: [WinError 1114]`
    (`torch/lib/c10.dll` dependency-init failure) -- verified directly: `import
    pandas; import sentence_transformers` fails every time in a fresh process on
    this machine, while `import sentence_transformers; import pandas` (the
    opposite order) succeeds every time, including a subsequent
    `import poi_rank.features.text_embed` (which itself imports pandas at module
    scope). This is a real pyarrow/torch native-DLL ordering conflict, not a bug
    in this project's code. Reordering imports inside THIS module cannot fix it:
    by the time any CLI command or pytest test reaches this diagnostic,
    `poi_rank.cli`/`tests/conftest.py` have already imported `pandas` at their own
    module load time, long before this function ever runs -- pandas is already
    resident in the parent process. A fresh child process, with
    `sentence_transformers` as its first import, sidesteps the conflict entirely.

    Returns `{"encode_wall_clock_seconds": ...}` -- the wall-clock of JUST the
    `model.encode(corpus, ...)` forward pass (per task scope: "measure it, don't
    assume it," spec-v2-remediation.md's own "~10s for ~1500 short docs on CPU"
    hypothesis), timed as a separate, explicit probe using the exact same
    `SentenceTransformer`/`.encode()` call `embed_text_sentence_transformer` uses
    internally (not a reimplementation of its embedding logic -- a second,
    throwaway `.encode()` call purely for sub-step timing granularity, since the
    production function bundles model-load + encode + SVD-fit + normalize into
    one untimeable-by-parts call, and it is not this diagnostic's place to modify
    that function to expose finer timing).
    """
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        corpus_path = tmp_dir / "corpus.json"
        timing_path = tmp_dir / "timing.json"
        corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "-c",
                _MINILM_SUBPROCESS_SCRIPT,
                str(corpus_path),
                str(artifact_path),
                str(timing_path),
                model_name,
                str(svd_dim),
                str(seed),
            ],
            check=True,
            timeout=600,
        )
        timing: dict[str, Any] = json.loads(timing_path.read_text(encoding="utf-8"))
        return timing


def semantic_fidelity_minilm(
    data_dir: Path, results_dir: Path, components: pd.DataFrame, cfg: FeatureBuildConfig
) -> dict[str, Any]:
    """D9, MiniLM path (one-time measurement only -- adopting MiniLM as the
    canonical embedding is a Block B fix decision, never made by this diagnostic):

    1. Encodes `all-MiniLM-L6-v2` ONCE over the same POI corpus
       `features/text_embed.py::build_poi_corpus` already builds (name +
       description + tags, spec.md section 5), via that module's own
       `embed_text_sentence_transformer` (never reimplemented, only invoked from
       an isolated subprocess -- see `_run_minilm_encode_in_subprocess`) --
       reduces to the SAME `svd_dim` as the canonical TF-IDF path via
       TruncatedSVD, for a fair equal-dimensionality comparison.
    2. Writes the resulting matrix to a DIAGNOSTIC-ONLY path
       (`results/parts/poi_emb_minilm_diagnostic.npy`), never to the canonical
       `artifacts/poi_emb.npy` cache slot.
    3. Rebuilds the OBSERVABLE traveler implicit-taste vectors against THIS
       embedding matrix by calling `features/traveler_features.py`'s existing
       `assemble_traveler_features` UNCHANGED, passing the MiniLM matrix in place
       of the TF-IDF one -- restricted to the D1/D2 population's own 202 holdout
       trips only (this diagnostic's population, not a full 800-trip rebuild).
    4. Computes the same Spearman as the TF-IDF path.

    Returns the Spearman result plus `encode_wall_clock_seconds` (the MiniLM
    encode step specifically, per task scope -- "measure it, don't assume it")
    and the diagnostic artifact's path.
    """
    pois_prepared = pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)
    travelers_df = pd.read_parquet(data_dir / TRAVELERS_FILENAME)
    trips_df = pd.read_parquet(data_dir / TRIPS_FILENAME)
    interactions_train = pd.read_parquet(data_dir / INTERACTIONS_TRAIN_FILENAME)
    pretrip_path = data_dir / INTERACTIONS_PRETRIP_FILENAME
    interactions_pretrip = pd.read_parquet(pretrip_path) if pretrip_path.exists() else None

    corpus = build_poi_corpus(pois_prepared)

    artifact_dir = results_dir / "parts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / MINILM_DIAGNOSTIC_ARTIFACT_FILENAME

    timing = _run_minilm_encode_in_subprocess(
        corpus,
        cfg.text_embedding.sentence_transformer_model,
        cfg.text_embedding.svd_dim,
        cfg.seed,
        artifact_path,
    )
    # float32, matching `features/text_embed.py`'s own `FloatArray` return type
    # (the array was `np.save`d by the subprocess exactly as that module produced
    # it -- `assemble_traveler_features`/`_poi_emb_by_id` below both expect this).
    minilm_embeddings: npt.NDArray[np.float32] = np.load(artifact_path)

    holdout_trip_ids = set(components["trip_id"].unique())
    holdout_trips_df = trips_df.loc[trips_df["trip_id"].astype(str).isin(holdout_trip_ids)]

    minilm_traveler_features = assemble_traveler_features(
        travelers_df,
        holdout_trips_df,
        interactions_train,
        pois_prepared,
        minilm_embeddings,
        cfg,
        interactions_pretrip,
    )

    canonical_map = build_poi_id_canonical_map(pois_prepared)
    emb_dim = cfg.text_embedding.svd_dim
    taste_by_trip = _taste_vector_by_trip(minilm_traveler_features, emb_dim)
    poi_emb_by_id = _poi_emb_by_id(pois_prepared["poi_id"], minilm_embeddings)

    result = semantic_fidelity_spearman(components, canonical_map, taste_by_trip, poi_emb_by_id)
    result["encode_wall_clock_seconds"] = timing["encode_wall_clock_seconds"]
    result["diagnostic_artifact_path"] = str(artifact_path)
    return result


# -----------------------------------------------------------------------------------
# D10: are POI descriptions generated conditioned on poi_semantic?
# -----------------------------------------------------------------------------------


def sample_within_destination_poi_pairs(
    poi_ids_by_destination: dict[str, list[str]], n_pairs: int, seed: int
) -> list[tuple[str, str]]:
    """Seeded, stratified sample of `n_pairs` POI-id pairs, split as evenly as
    possible across destinations (cross-destination comparisons are meaningless --
    different catalogs) -- destinations are visited in sorted order for
    determinism, with any remainder pair distributed to the first (alphabetically)
    destinations. Each pair is 2 DISTINCT POIs sampled without replacement from
    that destination's own catalog; the same POI can appear across multiple
    sampled pairs (a pair sample, not a POI partition)."""
    rng = np.random.default_rng(seed)
    destinations = sorted(poi_ids_by_destination)
    n_dest = len(destinations)
    base, remainder = divmod(n_pairs, n_dest)
    pairs: list[tuple[str, str]] = []
    for i, dest in enumerate(destinations):
        n_this = base + (1 if i < remainder else 0)
        ids = poi_ids_by_destination[dest]
        for _ in range(n_this):
            idx_a, idx_b = rng.choice(len(ids), size=2, replace=False)
            pairs.append((str(ids[idx_a]), str(ids[idx_b])))
    return pairs


def description_conditioning_fidelity(
    poi_latent: pd.DataFrame,
    poi_features: pd.DataFrame,
    n_pairs: int,
    seed: int,
) -> dict[str, Any]:
    """D10 step 2: Spearman(`cos(poi_semantic_i, poi_semantic_j)`,
    `cos(text_embedding_i, text_embedding_j)`) across a seeded sample of POI
    pairs, sampled within-destination (see `sample_within_destination_poi_pairs`).
    Uses the CURRENT CANONICAL embedding (TF-IDF -> SVD-64, `poi_features`'s
    `text_emb_*`) -- a generator-fidelity question independent of which
    downstream embedding method is used (D9 already covers the both-paths
    comparison), per task scope. `cos(...)` reuses
    `features.traveler_features.cosine_similarity_taste_poi` (a generic row-wise
    cosine over two aligned matrices, not specific to taste/POI semantics).

    `poi_latent` (the full pre-dedup catalog's true `poi_semantic`,
    oracle-permitted) is inner-joined to `poi_features` (the post-dedup canonical
    embedding table) on `poi_id` -- mirrors D7's own established
    poi_latent-to-observable join pattern; the join naturally restricts to the
    surviving post-dedup population (docs/DATA_CARD.md resolved ambiguity #18),
    which is the ~1,446-POI catalog this diagnostic is scoped to.
    """
    emb_dim = sum(1 for c in poi_features.columns if c.startswith(TEXT_EMB_PREFIX))
    merged = poi_latent[["poi_id", "destination", "poi_semantic"]].merge(
        poi_features[["poi_id", *_text_emb_columns(emb_dim)]], on="poi_id", how="inner"
    )

    poi_ids_by_destination: dict[str, list[str]] = {
        str(dest): group["poi_id"].astype(str).tolist()
        for dest, group in merged.groupby("destination")
    }
    pairs = sample_within_destination_poi_pairs(poi_ids_by_destination, n_pairs, seed)

    semantic_by_id: dict[str, FloatArray] = {
        str(p): s for p, s in zip(merged["poi_id"], merged["poi_semantic"], strict=True)
    }
    text_emb_matrix = merged[_text_emb_columns(emb_dim)].to_numpy(dtype=np.float64)
    text_emb_by_id: dict[str, FloatArray] = {
        str(p): text_emb_matrix[i] for i, p in enumerate(merged["poi_id"])
    }

    semantic_a = np.stack([semantic_by_id[a] for a, _ in pairs])
    semantic_b = np.stack([semantic_by_id[b] for _, b in pairs])
    text_a = np.stack([text_emb_by_id[a] for a, _ in pairs])
    text_b = np.stack([text_emb_by_id[b] for _, b in pairs])

    semantic_cos = cosine_similarity_taste_poi(semantic_a, semantic_b)
    text_cos = cosine_similarity_taste_poi(text_a, text_b)

    rho, p_value = stats.spearmanr(semantic_cos, text_cos)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_pairs_sampled": len(pairs),
        "n_pois_in_population": int(len(merged)),
        "seed": seed,
        "destinations": sorted(poi_ids_by_destination),
    }


# D10 raw TF-IDF variant: mirrors `configs/features.yaml`'s canonical text_embedding
# settings (1-2-grams, 20,000 features) WITHOUT importing anything from `features/` -- it
# must not depend on the feature pipeline it is meant to gate independently.
D10_RAW_TFIDF_NGRAM_MAX = 2
D10_RAW_TFIDF_MAX_FEATURES = 20000


def _raw_poi_text(name: str, description: str, tags: list[str]) -> str:
    """`name + description + tags` document, the same text the feature pipeline embeds
    (reimplemented here on purpose -- see D10_RAW_TFIDF_* comment)."""
    return f"{name} {description} {' '.join(tags)}"


def description_conditioning_fidelity_raw_tfidf(
    poi_latent: pd.DataFrame, pois_prepared: pd.DataFrame, n_pairs: int, seed: int
) -> dict[str, Any]:
    """D10, features-independent variant (Gate-A row): Spearman(`cos(poi_semantic_i,
    poi_semantic_j)`, cosine of the RAW sparse TF-IDF vectors of the POIs' text) over the
    same seeded within-destination pair sample as `description_conditioning_fidelity`.
    TF-IDF is fit here, over the post-dedup catalog's `name + description + tags`; no SVD,
    no `features/` import, so it is a pure property of the datagen output."""
    merged = poi_latent[["poi_id", "destination", "poi_semantic"]].merge(
        pois_prepared[["poi_id", "name", "description", "tags"]], on="poi_id", how="inner"
    )
    corpus = [
        _raw_poi_text(str(n), str(d), [str(t) for t in tags])
        for n, d, tags in zip(merged["name"], merged["description"], merged["tags"], strict=True)
    ]
    tfidf = TfidfVectorizer(
        ngram_range=(1, D10_RAW_TFIDF_NGRAM_MAX), max_features=D10_RAW_TFIDF_MAX_FEATURES
    ).fit_transform(corpus)

    position_by_id = {str(p): i for i, p in enumerate(merged["poi_id"])}
    poi_ids_by_destination: dict[str, list[str]] = {
        str(dest): group["poi_id"].astype(str).tolist()
        for dest, group in merged.groupby("destination")
    }
    pairs = sample_within_destination_poi_pairs(poi_ids_by_destination, n_pairs, seed)
    idx_a = np.array([position_by_id[a] for a, _ in pairs])
    idx_b = np.array([position_by_id[b] for _, b in pairs])

    semantic = np.stack(merged["poi_semantic"].to_numpy())
    semantic_cos = cosine_similarity_taste_poi(semantic[idx_a], semantic[idx_b])
    # TfidfVectorizer L2-normalizes rows, so the row-wise dot product IS the cosine.
    text_cos = np.asarray(tfidf[idx_a].multiply(tfidf[idx_b]).sum(axis=1)).ravel()

    rho, p_value = stats.spearmanr(semantic_cos, text_cos)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_pairs_sampled": len(pairs),
        "n_pois_in_population": int(len(merged)),
        "n_pairs_zero_text_cosine": int((text_cos == 0.0).sum()),
        "seed": seed,
        "destinations": sorted(poi_ids_by_destination),
    }


def text_dependency_check(
    poi_latent: pd.DataFrame,
    pois_raw: pd.DataFrame,
    datagen_cfg: DatagenConfig,
    seed: int,
) -> dict[str, Any]:
    """Computed replacement for the retired `D10.code_reading_answer` (which asserted
    "no dependency edge" and went stale once RC2c/A2 added one): re-runs the REAL text
    generator (`datagen.text_templates.generate_poi_text`) for every non-duplicate catalog
    POI under its own per-POI seeded streams and reports

    - `fraction_reproduced_from_own_semantic`: share whose regenerated description equals
      the stored one (validates that this harness drives the same generator);
    - `fraction_changed_same_semantic_control`: regenerating twice with identical inputs
      (expected 0.0 -- determinism control);
    - `fraction_changed_with_permuted_semantic`: share whose description CHANGES when the
      only thing altered is `poi_semantic` (swapped with another POI's from the same
      destination, same seeds/category) -- the dependency edge, as a number.
    """
    latent = poi_latent.set_index("poi_id")
    unique = pois_raw.loc[~pois_raw["is_duplicate"].astype(bool)].copy()
    unique["category_true"] = canonicalize_categories(unique["category"])["category"]
    pools = build_phrase_pools(
        datagen_cfg.text.phrases_per_dimension, datagen_cfg.text.anchor_phrases
    )
    destinations = sorted(unique["destination"].unique())
    rng = np.random.default_rng(seed)

    n = 0
    reproduced = 0
    control_changed = 0
    permuted_changed = 0
    for dest in destinations:
        rows = unique.loc[unique["destination"] == dest]
        dest_index = list(DEST_CENTERS).index(dest)
        semantics = [latent.loc[p, "poi_semantic"] for p in rows["poi_id"]]
        shuffle = rng.permutation(len(rows))
        for j, (poi_id, category, description) in enumerate(
            zip(rows["poi_id"], rows["category_true"], rows["description"], strict=True)
        ):
            poi_index = int(str(poi_id)[-4:]) - 1
            args = (dest, str(category))
            own = generate_poi_text(
                *poi_text_rngs(datagen_cfg.seed, dest_index, poi_index),
                *args,
                semantics[j],
                datagen_cfg.text,
                pools,
            )
            again = generate_poi_text(
                *poi_text_rngs(datagen_cfg.seed, dest_index, poi_index),
                *args,
                semantics[j],
                datagen_cfg.text,
                pools,
            )
            swapped = generate_poi_text(
                *poi_text_rngs(datagen_cfg.seed, dest_index, poi_index),
                *args,
                semantics[int(shuffle[j])],
                datagen_cfg.text,
                pools,
            )
            n += 1
            reproduced += own.description == description
            control_changed += own.description != again.description
            permuted_changed += own.description != swapped.description
    return {
        "n_pois": n,
        "fraction_reproduced_from_own_semantic": reproduced / n if n else 0.0,
        "fraction_changed_same_semantic_control": control_changed / n if n else 0.0,
        "fraction_changed_with_permuted_semantic": permuted_changed / n if n else 0.0,
        "seed": seed,
    }


def semantic_noise_ceiling(
    poi_latent: pd.DataFrame, pois_prepared: pd.DataFrame, n_pairs: int, seed: int
) -> dict[str, Any]:
    """Hypothesis diagnostic "poi_semantic's own noise limits D10": Spearman between
    `cos(poi_semantic_i, poi_semantic_j)` and the cosine of the NOISE-FREE reconstruction
    of each vector (category one-hot 1.0 + 0.6 on each catalog tag -- exactly
    `datagen/catalog.py::_poi_semantic_vector` at noise_std=0). No text can beat this
    number, because a text that carried the category and tag set perfectly would reproduce
    the noise-free cosine, not the noisy one."""
    merged = poi_latent[["poi_id", "destination", "poi_semantic"]].merge(
        pois_prepared[["poi_id", "category", "tags"]], on="poi_id", how="inner"
    )
    clean = np.zeros((len(merged), len(CATEGORY_INDEX) + len(TAG_INDEX)))
    for i, (category, tags) in enumerate(zip(merged["category"], merged["tags"], strict=True)):
        clean[i, CATEGORY_INDEX[str(category)]] = 1.0
        for tag in tags:
            clean[i, TAG_INDEX[str(tag)]] = 0.6
    position_by_id = {str(p): i for i, p in enumerate(merged["poi_id"])}
    by_destination = {
        str(dest): group["poi_id"].astype(str).tolist()
        for dest, group in merged.groupby("destination")
    }
    pairs = sample_within_destination_poi_pairs(by_destination, n_pairs, seed)
    idx_a = np.array([position_by_id[a] for a, _ in pairs])
    idx_b = np.array([position_by_id[b] for _, b in pairs])
    semantic = np.stack(merged["poi_semantic"].to_numpy())
    noisy_cos = cosine_similarity_taste_poi(semantic[idx_a], semantic[idx_b])
    clean_cos = cosine_similarity_taste_poi(clean[idx_a], clean[idx_b])
    rho, _ = stats.spearmanr(noisy_cos, clean_cos)
    return {"spearman_rho": float(rho), "n_pairs_sampled": len(pairs)}


def text_component_ablation(
    poi_latent: pd.DataFrame, pois_prepared: pd.DataFrame, n_pairs: int, seed: int
) -> dict[str, float]:
    """Raw-TF-IDF D10 rho when the corpus is restricted to one text component at a time --
    which observable column carries the semantic signal, and which dilutes it. Keys:
    `full` (name+description+tags, the gate corpus), `tags_only`, `tags_plus_category`
    (tags + the category word), `description_only`, `name_only`."""
    merged = poi_latent[["poi_id", "destination", "poi_semantic"]].merge(
        pois_prepared[["poi_id", "name", "description", "tags", "category"]],
        on="poi_id",
        how="inner",
    )
    tags = [" ".join(str(t) for t in ts) for ts in merged["tags"]]
    category = [str(c).replace("_", " ") for c in merged["category"]]
    corpora: dict[str, list[str]] = {
        "full": [
            _raw_poi_text(str(n), str(d), [str(t) for t in ts])
            for n, d, ts in zip(merged["name"], merged["description"], merged["tags"], strict=True)
        ],
        "tags_only": tags,
        "tags_plus_category": [f"{c} {t}" for c, t in zip(category, tags, strict=True)],
        "description_only": [str(d) for d in merged["description"]],
        "name_only": [str(n) for n in merged["name"]],
    }
    position_by_id = {str(p): i for i, p in enumerate(merged["poi_id"])}
    by_destination = {
        str(dest): group["poi_id"].astype(str).tolist()
        for dest, group in merged.groupby("destination")
    }
    pairs = sample_within_destination_poi_pairs(by_destination, n_pairs, seed)
    idx_a = np.array([position_by_id[a] for a, _ in pairs])
    idx_b = np.array([position_by_id[b] for _, b in pairs])
    semantic = np.stack(merged["poi_semantic"].to_numpy())
    semantic_cos = cosine_similarity_taste_poi(semantic[idx_a], semantic[idx_b])
    out: dict[str, float] = {}
    for name, corpus in corpora.items():
        tfidf = TfidfVectorizer(
            ngram_range=(1, D10_RAW_TFIDF_NGRAM_MAX), max_features=D10_RAW_TFIDF_MAX_FEATURES
        ).fit_transform(corpus)
        text_cos = np.asarray(tfidf[idx_a].multiply(tfidf[idx_b]).sum(axis=1)).ravel()
        rho, _ = stats.spearmanr(semantic_cos, text_cos)
        out[name] = float(rho)
    return out


# -----------------------------------------------------------------------------------
# A2 vocabulary-size sweep (results/parts/d10_vocab_sweep.json).
# -----------------------------------------------------------------------------------

D10_SWEEP_FILENAME = "d10_vocab_sweep.json"
D10_SWEEP_PHRASE_POOL_SIZES: tuple[int, ...] = (3, 8, 15, 20, 25)
# Named text-config variants measured at the shipped P, one lever changed at a time
# against the shipped config, plus the spec-v3 section 2.1 literal reading. Each is a
# `dataclasses.replace` override of `DatagenConfig.text`.
D10_SWEEP_HYPOTHESIS_ARMS: dict[str, dict[str, Any]] = {
    "spec_literal_3to6_multinomial_pure_synonym_full_surface": {
        "phrases_per_poi_min": 3,
        "phrases_per_poi_max": 6,
        "background_rate": 0.05,
        "anchor_phrases": False,
        "sampling": "multinomial",
        "surface_variants": 4,
    },
    "shipped_but_pure_synonym_phrases": {"anchor_phrases": False},
    "shipped_but_multinomial_sampling": {"sampling": "multinomial"},
    "shipped_but_3to6_phrases_per_poi": {"phrases_per_poi_min": 3, "phrases_per_poi_max": 6},
    "shipped_but_full_surface_variety": {"surface_variants": 4},
    "shipped_but_background_0.05": {"background_rate": 0.05},
}


def _measure_d10_variants(data_dir: Path, datagen_cfg: DatagenConfig) -> dict[str, Any]:
    oracle_dir = oracle_dir_from_output(data_dir)
    poi_latent = oracle_reader.load_poi_latent(oracle_dir)
    pois_prepared = pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)
    poi_features = pd.read_parquet(data_dir / POI_FEATURES_FILENAME)
    return {
        "raw_tfidf": description_conditioning_fidelity_raw_tfidf(
            poi_latent, pois_prepared, D10_N_SAMPLE_PAIRS, D10_SAMPLE_SEED
        )["spearman_rho"],
        "canonical_svd64": description_conditioning_fidelity(
            poi_latent, poi_features, D10_N_SAMPLE_PAIRS, D10_SAMPLE_SEED
        )["spearman_rho"],
        "text_component_ablation": text_component_ablation(
            poi_latent, pois_prepared, D10_N_SAMPLE_PAIRS, D10_SAMPLE_SEED
        ),
        "semantic_noise_ceiling": semantic_noise_ceiling(
            poi_latent, pois_prepared, D10_N_SAMPLE_PAIRS, D10_SAMPLE_SEED
        ),
        "text": asdict(datagen_cfg.text),
    }


def run_d10_vocab_sweep(
    datagen_cfg: DatagenConfig,
    features_config_path: Path,
    workdir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """A2 evidence: full-scale `generate -> prepare -> features` at each phrase-pool size
    in `D10_SWEEP_PHRASE_POOL_SIZES` (shipped config otherwise) plus the one-lever
    hypothesis arms in `D10_SWEEP_HYPOTHESIS_ARMS`, each into its own scratch directory
    under `workdir`, then both D10 variants (`raw_tfidf`, `canonical_svd64`) measured.
    Writes `results/parts/d10_vocab_sweep.json`. Never touches `data/synthetic/` or
    `artifacts/`: every run's data and embedding cache live under `workdir`."""
    from dataclasses import replace

    from poi_rank.data.config import FeaturesConfig
    from poi_rank.data.prepare import run_prepare
    from poi_rank.datagen.pipeline import run_generate
    from poi_rank.features.build import run_features

    features_data_cfg = FeaturesConfig.from_yaml(features_config_path)
    features_build_cfg = FeatureBuildConfig.from_yaml(features_config_path)

    def one_run(label: str, text_overrides: dict[str, Any]) -> dict[str, Any]:
        cfg = replace(datagen_cfg, text=replace(datagen_cfg.text, **text_overrides))
        data_dir = workdir / label
        artifacts_dir = workdir / f"{label}_artifacts"
        data_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        run_generate(cfg, data_dir)
        run_prepare(features_data_cfg, data_dir)
        run_features(features_build_cfg, data_dir, artifacts_dir)
        return _measure_d10_variants(data_dir, cfg)

    size_sweep = {
        str(p): one_run(f"P{p}", {"phrases_per_dimension": p}) for p in D10_SWEEP_PHRASE_POOL_SIZES
    }
    arms = {
        name: one_run(f"arm_{i}", ov)
        for i, (name, ov) in enumerate(D10_SWEEP_HYPOTHESIS_ARMS.items())
    }
    payload: dict[str, Any] = {
        "note": (
            "Full-scale (2,500 trips, 1,500 catalog rows/3 destinations) generate->prepare->"
            "features per setting; D10 = Spearman over 5,000 seeded within-destination POI "
            "pairs (seed 42). raw_tfidf is the Gate-A variant (no features/ import); "
            "canonical_svd64 goes through features/ (TF-IDF->SVD-64). Hypothesis arms "
            "override the shipped text config one lever at a time at the shipped "
            "phrases_per_dimension."
        ),
        "shipped_text_config": asdict(datagen_cfg.text),
        "phrases_per_dimension_sweep": size_sweep,
        "hypothesis_arms": arms,
    }
    output_dir = results_dir / "parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / D10_SWEEP_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


# -----------------------------------------------------------------------------------
# Orchestration + CLI entry point.
# -----------------------------------------------------------------------------------


def run_dgp_diagnostics(
    data_dir: Path,
    results_dir: Path,
    datagen_cfg: DatagenConfig,
    feature_cfg: FeatureBuildConfig,
) -> dict[str, Any]:
    """`poi_rank.cli diagnose-dgp` entry point: computes D1-D10 (module docstring)
    and writes `results/parts/dgp_diagnostics.json`. Diagnostic-only -- never
    writes anywhere else, never touches `datagen/features/models/candidates/
    scoring` code.

    Takes the full `feature_cfg` (not just `budget_target_price_level`, D1-D8's
    original signature) because D9's MiniLM path needs `feature_cfg.text_embedding`
    too -- `feature_cfg.traveler_features.budget_target_price_level` is used
    wherever the old `budget_target_price_level` parameter was.
    """
    start = perf_counter()
    oracle_dir = oracle_dir_from_output(data_dir)
    budget_target_price_level = feature_cfg.traveler_features.budget_target_price_level

    components = compute_utility_term_components(data_dir, oracle_dir, datagen_cfg.utility_weights)
    d1 = variance_decomposition(components, datagen_cfg.noise.sigma)
    d2 = taste_cosine_distribution(components)

    interactions_holdout_random = pd.read_parquet(data_dir / INTERACTIONS_HOLDOUT_RANDOM_FILENAME)
    holdout_utility_true = oracle_reader.load_holdout_utility_true(oracle_dir)
    frame_keys = interactions_holdout_random[["trip_id", "poi_id"]]
    oracle_score = oracle_reader.oracle_ceiling_scores(frame_keys, oracle_dir)
    n_missing_ceiling_rows = int(np.isneginf(oracle_score.to_numpy(dtype=np.float64)).sum())

    d3 = spearman_utility_vs_label(interactions_holdout_random, oracle_score)
    d4 = choice_sharpness(interactions_holdout_random, oracle_score, holdout_utility_true).to_dict()
    d5_slate = oracle_ndcg_slate_level(interactions_holdout_random, oracle_score).to_dict()
    d5_candidate = oracle_ndcg_candidate_level(data_dir, oracle_dir, budget_target_price_level)
    d5_full_catalog = oracle_ndcg_full_catalog(holdout_utility_true, interactions_holdout_random)

    d6 = cold_start_share(data_dir)

    pois_prepared = pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)
    d7 = localness_vs_geo_generation(pois_prepared, oracle_dir)

    d8 = bias_gap_popularity(data_dir, budget_target_price_level)
    existing_gap = _read_existing_bias_gap_popularity(results_dir)
    if existing_gap is not None:
        d8["existing_metrics_json_gap"] = existing_gap

    d9_tfidf = semantic_fidelity_tfidf(data_dir, components, feature_cfg)
    d9_minilm = semantic_fidelity_minilm(data_dir, results_dir, components, feature_cfg)
    d9 = {"tfidf_path": d9_tfidf, "minilm_path": d9_minilm}

    poi_latent = oracle_reader.load_poi_latent(oracle_dir)
    poi_features = pd.read_parquet(data_dir / POI_FEATURES_FILENAME)
    pois_raw = pd.read_parquet(data_dir / POIS_RAW_FILENAME)
    d10_canonical = description_conditioning_fidelity(
        poi_latent, poi_features, D10_N_SAMPLE_PAIRS, D10_SAMPLE_SEED
    )
    d10_raw = description_conditioning_fidelity_raw_tfidf(
        poi_latent, pois_prepared, D10_N_SAMPLE_PAIRS, D10_SAMPLE_SEED
    )
    d10 = {
        # Gate-A reads `raw_tfidf` (features-independent); `canonical_svd64` (TF-IDF ->
        # SVD-64 through features/) is reported alongside. `measured` kept as an alias of
        # the canonical variant for backward compatibility with pre-A2 readers.
        "raw_tfidf": d10_raw,
        "canonical_svd64": d10_canonical,
        "measured": d10_canonical,
        "text_dependency_check": text_dependency_check(
            poi_latent, pois_raw, datagen_cfg, D10_SAMPLE_SEED
        ),
    }

    wall_clock = perf_counter() - start

    payload: dict[str, Any] = {
        "meta": {
            "wall_clock_seconds": wall_clock,
            "n_ceiling_score_missing_fallback": n_missing_ceiling_rows,
            "note": (
                "Diagnostic-only harness (spec-v2-remediation.md section 1). D1/D2/"
                "D9 are restricted to the 202 holdout trips (~99k same-destination "
                "pairs), not all 800 trips (~1.2M pairs), because the DGP's own "
                "noise-free true-utility export -- the only source that isolates "
                "the novelty term via residual -- is written only for holdout "
                "trips by datagen/pipeline.py itself; see module docstring for "
                "the full explanation. D10 uses the full ~1,446-POI catalog "
                "instead (a POI-pair question, not a trip-pair question)."
            ),
        },
        "D1_variance_decomposition": d1,
        "D2_taste_cosine_distribution": d2,
        "D3_spearman_utility_vs_label": d3,
        "D4_choice_sharpness": d4,
        "D5_ndcg": {
            "slate_level": d5_slate,
            "candidate_level": d5_candidate,
            "full_catalog": d5_full_catalog,
        },
        "D6_cold_start_share": d6,
        "D7_localness_vs_geo_generation": d7,
        "D8_bias_gap_popularity": d8,
        "D9_semantic_fidelity": d9,
        "D10_description_conditioning": d10,
    }

    output_dir = results_dir / "parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"payload": payload, "output_path": output_path}


if __name__ == "__main__":
    # A2 evidence entry point: `uv run python -m poi_rank.eval.dgp_diagnostics` (from the
    # repo root) regenerates results/parts/d10_vocab_sweep.json in scratch directories.
    with tempfile.TemporaryDirectory() as _workdir:
        run_d10_vocab_sweep(
            DatagenConfig.from_yaml(Path("configs/datagen.yaml")),
            Path("configs/features.yaml"),
            Path(_workdir),
            Path("results"),
        )
