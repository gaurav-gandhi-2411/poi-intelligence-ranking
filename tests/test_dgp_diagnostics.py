"""`eval/dgp_diagnostics.py` tests (spec-v2-remediation.md section 1, D1-D8):
hand-built, exactly-hand-verifiable examples for each diagnostic's core arithmetic,
plus a real-fixture-chain integration test with a strong correctness invariant
(the residual-recovered novelty term must equal EXACTLY 1.0, to floating-point
precision, for every pair belonging to a traveler's genuinely-first-ever trip --
`novelty_array`'s own documented behavior for an empty history, `datagen/utility.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.datagen.config import UtilityWeights
from poi_rank.datagen.oracle_export import write_term_standardization
from poi_rank.datagen.taxonomy import CATEGORY_INDEX, TASTE_DIM
from poi_rank.datagen.utility import DestinationTermStats, TermStandardization
from poi_rank.eval.dgp_diagnostics import (
    ChoiceSharpnessResult,
    SlateLevelNdcgResult,
    _read_existing_bias_gap_popularity,
    choice_sharpness,
    cold_start_share,
    compute_utility_term_components,
    description_conditioning_fidelity,
    description_conditioning_fidelity_raw_tfidf,
    localness_vs_geo_generation,
    oracle_ndcg_full_catalog,
    oracle_ndcg_slate_level,
    run_dgp_diagnostics,
    sample_within_destination_poi_pairs,
    semantic_fidelity_spearman,
    semantic_fidelity_tfidf,
    semantic_noise_ceiling,
    spearman_utility_vs_label,
    taste_cosine_distribution,
    text_component_ablation,
    text_dependency_check,
    variance_decomposition,
)
from poi_rank.features.config import (
    BudgetTargetPriceLevel,
    FeatureBuildConfig,
    PoiFeaturesConfig,
    TasteWeights,
    TextEmbeddingConfig,
    TravelerFeaturesConfig,
)

WEIGHTS = UtilityWeights(
    w_taste=2.0, w_cat=1.0, w_local=1.0, w_qual=1.5, w_party=1.0, w_price=0.8, w_novel=0.5
)

# Block A RC2a (docs/DATA_CARD.md "DGP remediation, Block A"): `compute_utility_
# term_components` now z-scores each raw term against a loaded `TermStandardization`
# before weighting (matching `datagen/utility.py::standardize_and_weight`). An
# IDENTITY standardization (mean=0, std=1 for every term) makes z-score a no-op, so
# every pre-existing hand-computed "raw_term * weight" expectation in this file's
# fixtures stays valid unchanged.
_IDENTITY_DEST_STATS = DestinationTermStats(
    taste_mean=0.0,
    taste_std=1.0,
    cat_mean=0.0,
    cat_std=1.0,
    local_mean=0.0,
    local_std=1.0,
    qual_mean=0.0,
    qual_std=1.0,
    party_mean=0.0,
    party_std=1.0,
    price_mean=0.0,
    price_std=1.0,
)


def _write_identity_standardization(oracle_dir: Path, destination: str = "seoul") -> None:
    standardization = TermStandardization(
        per_destination={destination: _IDENTITY_DEST_STATS}, novelty_mean=0.0, novelty_std=1.0
    )
    write_term_standardization(oracle_dir, standardization)


# -----------------------------------------------------------------------------------
# D1/D2 core computation: hand-built 1-trip, 2-POI example with every term
# hand-verified by direct arithmetic.
# -----------------------------------------------------------------------------------


@pytest.fixture
def hand_built_data_dir(tmp_path: Path) -> Path:
    """A tiny, fully hand-computed (data_dir, oracle_dir) pair: 1 trip (T1), 1
    traveler (U1), 2 POIs (P1 restaurant, P2 cafe). Every deterministic term is
    picked so it is exactly hand-verifiable; `utility_true` is set to the exact
    sum of the 6 known terms plus a DELIBERATE, hand-chosen novelty contribution
    (0.5*1.0 for P1, 0.5*0.7 for P2) so the residual-recovery arithmetic has a
    known-correct answer to check against.
    """
    data_dir = tmp_path
    oracle_dir = data_dir / "_oracle_test_dir"  # never named "_oracle" -- test-only, no isolation
    oracle_dir.mkdir()

    taste_vec = np.zeros(TASTE_DIM, dtype=np.float64)
    taste_vec[CATEGORY_INDEX["restaurant"]] = 1.0
    poi_semantic_p1 = taste_vec.copy()  # cos(taste, P1) == 1.0 exactly
    poi_semantic_p2 = np.zeros(TASTE_DIM, dtype=np.float64)
    poi_semantic_p2[CATEGORY_INDEX["museum"]] = 1.0  # orthogonal to taste_vec -> cos == 0.0

    pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "destination": ["seoul", "seoul"],
            "latent_quality": [0.8, 0.3],
            "latent_localness": [0.6, 0.2],
            "poi_semantic": [poi_semantic_p1, poi_semantic_p2],
        }
    ).to_parquet(oracle_dir / "poi_latent.parquet", index=False)

    pd.DataFrame({"traveler_id": ["U1"], "taste_vector": [taste_vec]}).to_parquet(
        oracle_dir / "traveler_taste.parquet", index=False
    )

    # taste_sim=[1.0, 0.0], cat_affinity=[1.0, 0.0] (taste_vec[restaurant_idx]=1.0,
    # taste_vec[cafe_idx]=0.0), local_term = localness*(-touristiness_pref=0.5) =
    # [0.3, 0.1], party_fit (solo, neither category in CATEGORY_PARTY_BONUS) =
    # [0.85, 0.85], price_fit (budget=medium target 2.5; P1 price=2 -> diff=-0.5 ->
    # penalty=0.075 -> 0.925; P2 price=4 -> diff=1.5 -> penalty=0.525 -> 0.475).
    # weighted sum (w_taste=2, w_cat=1, w_local=1, w_qual=1.5, w_party=1, w_price=0.8):
    #   P1: 2*1.0 + 1*1.0 + 1*0.3 + 1.5*0.8 + 1*0.85 + 0.8*0.925 = 6.09
    #   P2: 2*0.0 + 1*0.0 + 1*0.1 + 1.5*0.3 + 1*0.85 + 0.8*0.475 = 1.78
    # plus a hand-chosen novelty contribution (w_novel=0.5): P1 novelty=1.0 -> +0.5;
    # P2 novelty=0.7 -> +0.35.
    pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "traveler_id": ["U1", "U1"],
            "poi_id": ["P1", "P2"],
            "utility_true": [6.09 + 0.5, 1.78 + 0.35],
        }
    ).to_parquet(oracle_dir / "holdout_utility_true.parquet", index=False)

    pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "touristiness_pref": [-0.5],
            "party_type": ["solo"],
            "budget": ["medium"],
        }
    ).to_parquet(data_dir / "travelers.parquet", index=False)

    pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "category": ["restaurant", "cafe"],
            "accessibility": [
                {"wheelchair": False, "stroller": False, "kid_friendly": False},
                {"wheelchair": False, "stroller": False, "kid_friendly": False},
            ],
            "price_level": pd.array([2, 4], dtype="Int64"),
        }
    ).to_parquet(data_dir / "pois.parquet", index=False)

    pd.DataFrame({"trip_id": ["T1"], "destination": ["seoul"]}).to_parquet(
        data_dir / "trips.parquet", index=False
    )
    _write_identity_standardization(oracle_dir)

    return data_dir


def _oracle_dir_for(data_dir: Path) -> Path:
    return data_dir / "_oracle_test_dir"


def test_compute_utility_term_components_matches_hand_computation(
    hand_built_data_dir: Path,
) -> None:
    components = compute_utility_term_components(
        hand_built_data_dir, _oracle_dir_for(hand_built_data_dir), WEIGHTS
    )
    components = components.set_index("poi_id")

    assert components.loc["P1", "taste_sim"] == pytest.approx(1.0)
    assert components.loc["P2", "taste_sim"] == pytest.approx(0.0)
    assert components.loc["P1", "cat_affinity"] == pytest.approx(1.0)
    assert components.loc["P2", "cat_affinity"] == pytest.approx(0.0)
    assert components.loc["P1", "local_term"] == pytest.approx(0.3)
    assert components.loc["P2", "local_term"] == pytest.approx(0.1)
    assert components.loc["P1", "party_fit"] == pytest.approx(0.85)
    assert components.loc["P2", "party_fit"] == pytest.approx(0.85)
    assert components.loc["P1", "price_fit"] == pytest.approx(0.925)
    assert components.loc["P2", "price_fit"] == pytest.approx(0.475)

    # The residual novelty recovery: utility_true - (6 known weighted terms) should
    # exactly equal the hand-chosen w_novel * novelty contribution baked into
    # utility_true above (0.5*1.0 for P1, 0.5*0.7 for P2).
    assert components.loc["P1", "weighted_novelty_residual"] == pytest.approx(0.5, abs=1e-9)
    assert components.loc["P2", "weighted_novelty_residual"] == pytest.approx(0.35, abs=1e-9)


def test_compute_utility_term_components_propagates_nan_for_missing_price(
    tmp_path: Path,
) -> None:
    """A null `price_level` (the DGP's own dirtiness injection, module docstring)
    must yield `NaN` for `price_fit` AND the residual `weighted_novelty_residual`
    -- never a silently-imputed value -- while every other term stays computable."""
    data_dir = tmp_path
    oracle_dir = data_dir / "_oracle_test_dir"
    oracle_dir.mkdir()

    taste_vec = np.zeros(TASTE_DIM, dtype=np.float64)
    taste_vec[CATEGORY_INDEX["restaurant"]] = 1.0
    pd.DataFrame(
        {
            "poi_id": ["P1"],
            "destination": ["seoul"],
            "latent_quality": [0.5],
            "latent_localness": [0.5],
            "poi_semantic": [taste_vec.copy()],
        }
    ).to_parquet(oracle_dir / "poi_latent.parquet", index=False)
    pd.DataFrame({"traveler_id": ["U1"], "taste_vector": [taste_vec]}).to_parquet(
        oracle_dir / "traveler_taste.parquet", index=False
    )
    pd.DataFrame(
        {
            "trip_id": ["T1"],
            "traveler_id": ["U1"],
            "poi_id": ["P1"],
            "utility_true": [3.14],
        }
    ).to_parquet(oracle_dir / "holdout_utility_true.parquet", index=False)
    pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "touristiness_pref": [0.0],
            "party_type": ["solo"],
            "budget": ["medium"],
        }
    ).to_parquet(data_dir / "travelers.parquet", index=False)
    pd.DataFrame(
        {
            "poi_id": ["P1"],
            "category": ["restaurant"],
            "accessibility": [{"wheelchair": False, "stroller": False, "kid_friendly": False}],
            "price_level": pd.array([pd.NA], dtype="Int64"),
        }
    ).to_parquet(data_dir / "pois.parquet", index=False)
    pd.DataFrame({"trip_id": ["T1"], "destination": ["seoul"]}).to_parquet(
        data_dir / "trips.parquet", index=False
    )
    _write_identity_standardization(oracle_dir)

    components = compute_utility_term_components(data_dir, oracle_dir, WEIGHTS)
    assert np.isnan(components["price_fit"].iloc[0])
    assert np.isnan(components["weighted_novelty_residual"].iloc[0])
    # Every other term stays computable (not swept into NaN by the missing price).
    assert not np.isnan(components["taste_sim"].iloc[0])
    assert not np.isnan(components["local_term"].iloc[0])
    assert not np.isnan(components["latent_quality"].iloc[0])
    assert not np.isnan(components["party_fit"].iloc[0])


def test_compute_utility_term_components_canonicalizes_dirty_category(
    hand_built_data_dir: Path,
) -> None:
    """A dirty exported category string (`datagen/catalog.py`'s
    `inconsistent_category_rate` injection) must still resolve to the correct
    `CATEGORY_INDEX` slot -- proves the canonicalization fix (not the raw dirty
    string) is what feeds `cat_affinity`."""
    data_dir = hand_built_data_dir
    pois_path = data_dir / "pois.parquet"
    pois = pd.read_parquet(pois_path)
    pois.loc[pois["poi_id"] == "P1", "category"] = "Food & Dining"  # dirty variant of restaurant
    pois.to_parquet(pois_path, index=False)

    components = compute_utility_term_components(
        data_dir, _oracle_dir_for(data_dir), WEIGHTS
    ).set_index("poi_id")
    assert components.loc["P1", "cat_affinity"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------------
# D1: variance decomposition -- hand-computable shares on a tiny synthetic case.
# -----------------------------------------------------------------------------------


def test_variance_decomposition_hand_computed_shares() -> None:
    # 4 pairs, 2 deterministic terms with KNOWN variances (var(ddof=1)):
    #   term_a = [0, 2, 4, 6] -> mean 3, var = ((-3)^2+(-1)^2+1^2+3^2)/3 = 20/3
    #   term_b (novelty residual) = [1, 1, 1, 1] -> var = 0 (perfectly inert)
    # utility_true set so it's an exact sum (novelty residual computed by the
    # function as utility_true - the other 6 known terms, all zero except taste
    # here, to keep the arithmetic small).
    n = 4
    zeros = np.zeros(n)
    components = pd.DataFrame(
        {
            "weighted_taste": [0.0, 2.0, 4.0, 6.0],
            "weighted_cat": zeros,
            "weighted_local": zeros,
            "weighted_qual": zeros,
            "weighted_party": zeros,
            "weighted_price": zeros,
            "weighted_novelty_residual": [1.0, 1.0, 1.0, 1.0],
            "utility_true": [1.0, 3.0, 5.0, 7.0],
        }
    )
    sigma = 1.0  # var_epsilon = 1.0
    result = variance_decomposition(components, sigma)

    var_taste_expected = 20.0 / 3.0
    var_u_true_expected = 20.0 / 3.0  # utility_true = weighted_taste + 1, same variance
    var_u_total_expected = var_u_true_expected + 1.0

    assert result["n_pairs_total"] == 4
    assert result["n_pairs_valid_for_price_and_novelty"] == 4
    assert result["n_pairs_excluded_missing_price_level"] == 0
    assert result["var_u_true_deterministic_only"] == pytest.approx(var_u_true_expected)
    assert result["var_epsilon_analytical_from_config_sigma"] == pytest.approx(1.0)
    assert result["var_u_total_deterministic_plus_noise"] == pytest.approx(var_u_total_expected)
    assert result["term_variances_weighted"]["taste"] == pytest.approx(var_taste_expected)
    assert result["term_variances_weighted"]["novelty"] == pytest.approx(0.0)
    assert result["term_variance_shares"]["taste"] == pytest.approx(
        var_taste_expected / var_u_total_expected
    )
    assert result["term_variance_shares"]["novelty"] == pytest.approx(0.0)
    assert result["epsilon_variance_share"] == pytest.approx(1.0 / var_u_total_expected)
    assert result["min_deterministic_term_share"] == pytest.approx(0.0)
    # 6 zero-variance terms + taste + epsilon should sum to exactly 1.0 here (no
    # covariance possible: every other deterministic term is a constant).
    assert result["share_sum_all_8"] == pytest.approx(1.0)


def test_variance_decomposition_excludes_missing_price_pairs() -> None:
    components = pd.DataFrame(
        {
            "weighted_taste": [1.0, 2.0, 3.0],
            "weighted_cat": [0.0, 0.0, 0.0],
            "weighted_local": [0.0, 0.0, 0.0],
            "weighted_qual": [0.0, 0.0, 0.0],
            "weighted_party": [0.0, 0.0, 0.0],
            "weighted_price": [0.5, np.nan, 0.5],
            "weighted_novelty_residual": [0.1, np.nan, 0.1],
            "utility_true": [1.6, np.nan, 3.6],
        }
    )
    result = variance_decomposition(components, sigma=0.4)
    assert result["n_pairs_total"] == 3
    assert result["n_pairs_valid_for_price_and_novelty"] == 2
    assert result["n_pairs_excluded_missing_price_level"] == 1


# -----------------------------------------------------------------------------------
# D2: taste-cosine distribution.
# -----------------------------------------------------------------------------------


def test_taste_cosine_distribution_matches_numpy_directly() -> None:
    values = np.array([0.0, 0.5, 1.0, -0.5, 0.25])
    components = pd.DataFrame({"taste_sim": values})
    result = taste_cosine_distribution(components)
    assert result["mean"] == pytest.approx(float(np.mean(values)))
    assert result["sd"] == pytest.approx(float(np.std(values, ddof=1)))
    assert result["p5"] == pytest.approx(float(np.percentile(values, 5)))
    assert result["p95"] == pytest.approx(float(np.percentile(values, 95)))
    assert result["n"] == 5


# -----------------------------------------------------------------------------------
# D3: Spearman(u, label).
# -----------------------------------------------------------------------------------


def test_spearman_utility_vs_label_perfect_monotone_relationship() -> None:
    interactions = pd.DataFrame({"label": [0, 1, 2, 3]})
    oracle_score = pd.Series([0.1, 0.5, 1.0, 2.0])
    result = spearman_utility_vs_label(interactions, oracle_score)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n"] == 4
    assert result["n_excluded_no_ceiling_score"] == 0


def test_spearman_utility_vs_label_excludes_missing_oracle_rows() -> None:
    interactions = pd.DataFrame({"label": [0, 1, 2]})
    oracle_score = pd.Series([0.1, -np.inf, 1.0])
    result = spearman_utility_vs_label(interactions, oracle_score)
    assert result["n"] == 2
    assert result["n_excluded_no_ceiling_score"] == 1


# -----------------------------------------------------------------------------------
# D4: choice sharpness.
# -----------------------------------------------------------------------------------


def test_choice_sharpness_hand_computed_percentile_and_rank() -> None:
    # Trip T1's full eligible catalog (5 POIs): utility_true = [1, 2, 3, 4, 5].
    holdout_utility_true = pd.DataFrame(
        {
            "trip_id": ["T1"] * 5,
            "poi_id": ["C1", "C2", "C3", "C4", "C5"],
            "utility_true": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    # Slate S1 (4 of those 5 POIs exposed): C2(util=2, label=0), C3(util=3,
    # label=2, ENGAGED and chosen -- the only engaged POI), C4(util=4, label=0),
    # C5(util=5, label=0). Chosen utility=3 -> within-slate rank: 1 POI (C5=5,
    # C4=4) exceeds 3 -> wait: C4=4>3 and C5=5>3, so 2 POIs exceed -> rank=3.
    # Global percentile of 3.0 among [1,2,3,4,5] (kind="mean"): 2 strictly below,
    # 1 equal, 2 above -> percentileofscore mean = (2/5 + 3/5)/2*100 = 50.0.
    interactions = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1", "T1"],
            "poi_id": ["C2", "C3", "C4", "C5"],
            "slate_id": ["S1", "S1", "S1", "S1"],
            "label": [0, 2, 0, 0],
        }
    )
    oracle_score = pd.Series([2.0, 3.0, 4.0, 5.0], index=interactions.index)

    result = choice_sharpness(interactions, oracle_score, holdout_utility_true)
    assert isinstance(result, ChoiceSharpnessResult)
    assert result.n_slates_included == 1
    assert result.n_slates_excluded_no_engagement == 0
    assert result.mean_within_slate_rank == pytest.approx(3.0)
    assert result.mean_global_percentile == pytest.approx(50.0)
    assert result.slate_size == 4


def test_choice_sharpness_excludes_slates_with_no_engagement() -> None:
    holdout_utility_true = pd.DataFrame(
        {"trip_id": ["T1"], "poi_id": ["C1"], "utility_true": [1.0]}
    )
    interactions = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "poi_id": ["C1", "C2"],
            "slate_id": ["S1", "S1"],
            "label": [0, 0],  # nothing engaged
        }
    )
    oracle_score = pd.Series([1.0, 2.0], index=interactions.index)
    result = choice_sharpness(interactions, oracle_score, holdout_utility_true)
    assert result.n_slates_included == 0
    assert result.n_slates_excluded_no_engagement == 1


def test_choice_sharpness_ties_break_by_highest_utility() -> None:
    """Two POIs share the max label (`engage_lambda` can engage >1 POI per slate,
    module docstring) -- the documented tie-break picks the higher-utility one."""
    holdout_utility_true = pd.DataFrame(
        {"trip_id": ["T1", "T1"], "poi_id": ["A", "B"], "utility_true": [10.0, 1.0]}
    )
    interactions = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "poi_id": ["A", "B"],
            "slate_id": ["S1", "S1"],
            "label": [2, 2],  # tied max label
        }
    )
    oracle_score = pd.Series([10.0, 1.0], index=interactions.index)
    result = choice_sharpness(interactions, oracle_score, holdout_utility_true)
    # Chosen should be A (utility=10, the higher one) -> within-slate rank 1.
    assert result.mean_within_slate_rank == pytest.approx(1.0)


# -----------------------------------------------------------------------------------
# D5 (slate-level half): reuses eval.metrics.ndcg_at_k, verified via a small
# hand-checkable example.
# -----------------------------------------------------------------------------------


def test_oracle_ndcg_slate_level_perfect_ranking_gives_ndcg_one() -> None:
    interactions = pd.DataFrame(
        {
            "poi_id": ["A", "B", "C"],
            "slate_id": ["S1", "S1", "S1"],
            "label": [3, 1, 0],
        }
    )
    oracle_score = pd.Series([3.0, 2.0, 1.0], index=interactions.index)  # matches label order
    result = oracle_ndcg_slate_level(interactions, oracle_score, k=10)
    assert isinstance(result, SlateLevelNdcgResult)
    assert result.mean_ndcg_at_10 == pytest.approx(1.0)
    assert result.n_slates_included == 1
    assert result.n_slates_excluded_zero_relevant == 0


def test_oracle_ndcg_slate_level_excludes_zero_relevant_slates() -> None:
    interactions = pd.DataFrame({"poi_id": ["A", "B"], "slate_id": ["S1", "S1"], "label": [0, 0]})
    oracle_score = pd.Series([1.0, 2.0], index=interactions.index)
    result = oracle_ndcg_slate_level(interactions, oracle_score, k=10)
    assert result.n_slates_included == 0
    assert result.n_slates_excluded_zero_relevant == 1


# -----------------------------------------------------------------------------------
# D6: cold-start share.
# -----------------------------------------------------------------------------------


def test_cold_start_share_hand_counted(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "traveler_id": ["U1", "U1", "U2", "U3"],
            "trip_id": ["T1", "T2", "T3", "T4"],
            "implicit_interaction_count": [0.0, 5.0, 0.0, 0.0],
        }
    ).to_parquet(tmp_path / "traveler_features.parquet", index=False)
    pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3", "T4"],
            "is_holdout": [False, False, True, True],
        }
    ).to_parquet(tmp_path / "trips.parquet", index=False)

    result = cold_start_share(tmp_path)
    assert result["overall"]["n_trips"] == 4
    assert result["overall"]["n_cold_start"] == 3
    assert result["overall"]["share"] == pytest.approx(0.75)
    assert result["holdout_only"]["n_trips"] == 2
    assert result["holdout_only"]["n_cold_start"] == 2
    assert result["holdout_only"]["share"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------------
# D7: thin wrapper over eval.oracle.validate_localness_against_oracle.
# -----------------------------------------------------------------------------------


def test_localness_vs_geo_generation_computes_spearman(tmp_path: Path) -> None:
    oracle_dir = tmp_path / "_oracle_test_dir"
    oracle_dir.mkdir()
    pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3", "P4"],
            "destination": ["seoul"] * 4,
            "latent_quality": [0.1, 0.2, 0.3, 0.4],
            "latent_localness": [1.0, 2.0, 3.0, 4.0],
            "poi_semantic": [np.zeros(TASTE_DIM) for _ in range(4)],
        }
    ).to_parquet(oracle_dir / "poi_latent.parquet", index=False)
    pois_prepared = pd.DataFrame(
        {"poi_id": ["P1", "P2", "P3", "P4"], "dist_to_tourist_centroid_km": [4.0, 3.0, 2.0, 1.0]}
    )
    result = localness_vs_geo_generation(pois_prepared, oracle_dir)
    # Perfectly monotone-decreasing relationship -> rho = -1.0.
    assert result["spearman_rho"] == pytest.approx(-1.0)
    assert result["n"] == 4


# -----------------------------------------------------------------------------------
# D8 cross-check helper.
# -----------------------------------------------------------------------------------


def test_read_existing_bias_gap_popularity_missing_file(tmp_path: Path) -> None:
    assert _read_existing_bias_gap_popularity(tmp_path) is None


def test_read_existing_bias_gap_popularity_reads_existing_value(tmp_path: Path) -> None:
    import json

    (tmp_path / "metrics.json").write_text(
        json.dumps({"bias_gap": {"popularity": {"gap": 0.1099}}}), encoding="utf-8"
    )
    assert _read_existing_bias_gap_popularity(tmp_path) == pytest.approx(0.1099)


def test_read_existing_bias_gap_popularity_malformed_json_returns_none(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    assert _read_existing_bias_gap_popularity(tmp_path) is None


# -----------------------------------------------------------------------------------
# Real-fixture-chain integration: full `run_dgp_diagnostics` + the strong
# correctness invariant that validates the whole D1/D2 reconstruction pipeline.
# -----------------------------------------------------------------------------------


def test_run_dgp_diagnostics_end_to_end_on_real_fixture_chain(
    evaluate_ready_data_dir: Path,
    datagen_cfg: Any,
    feature_build_cfg: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    results_dir = tmp_path_factory.mktemp("dgp_diagnostics_results")
    result = run_dgp_diagnostics(
        evaluate_ready_data_dir, results_dir, datagen_cfg, feature_build_cfg
    )
    payload = result["payload"]
    output_path = result["output_path"]
    assert output_path.exists()
    assert output_path == results_dir / "parts" / "dgp_diagnostics.json"

    for key in (
        "D1_variance_decomposition",
        "D2_taste_cosine_distribution",
        "D3_spearman_utility_vs_label",
        "D4_choice_sharpness",
        "D5_ndcg",
        "D6_cold_start_share",
        "D7_localness_vs_geo_generation",
        "D8_bias_gap_popularity",
        "D9_semantic_fidelity",
        "D10_description_conditioning",
    ):
        assert key in payload

    d1 = payload["D1_variance_decomposition"]
    assert d1["n_pairs_total"] > 0
    # 7 term shares + epsilon should each be finite and (roughly) between -1 and 2
    # -- a loose sanity band, not a target; genuine negative covariance can push a
    # diagonal share slightly negative, but nothing pathological (e.g. NaN, inf).
    for share in {**d1["term_variance_shares"], "epsilon": d1["epsilon_variance_share"]}.values():
        assert np.isfinite(share)

    d6 = payload["D6_cold_start_share"]
    assert 0.0 <= d6["overall"]["share"] <= 1.0
    assert 0.0 <= d6["holdout_only"]["share"] <= 1.0

    d8 = payload["D8_bias_gap_popularity"]
    assert np.isfinite(d8["gap"])

    # D9: both paths must produce a finite Spearman rho over a genuinely nonzero
    # population -- this is the one place in the whole suite that exercises the
    # real MiniLM encode (network/model-load cost paid once, session-scoped
    # `evaluate_ready_data_dir` fixture), never repeated by a separate unit test.
    d9 = payload["D9_semantic_fidelity"]
    for path_name in ("tfidf_path", "minilm_path"):
        p = d9[path_name]
        assert np.isfinite(p["spearman_rho"])
        assert p["n"] > 0
    assert d9["minilm_path"]["encode_wall_clock_seconds"] > 0.0
    assert Path(d9["minilm_path"]["diagnostic_artifact_path"]).exists()
    # Never the canonical cache slot -- module docstring's "no ambiguity" requirement.
    assert "poi_emb_minilm_diagnostic" in d9["minilm_path"]["diagnostic_artifact_path"]
    assert "artifacts" not in d9["minilm_path"]["diagnostic_artifact_path"]

    # D10: both variants finite over a real, nonzero sample; the stale code-reading
    # constant is gone, replaced by a COMPUTED dependency check.
    d10 = payload["D10_description_conditioning"]
    assert "code_reading_answer" not in d10
    for variant in ("raw_tfidf", "canonical_svd64"):
        assert np.isfinite(d10[variant]["spearman_rho"])
        assert d10[variant]["n_pairs_sampled"] > 0
    assert d10["measured"] == d10["canonical_svd64"]
    dependency = d10["text_dependency_check"]
    assert dependency["fraction_reproduced_from_own_semantic"] == pytest.approx(1.0)
    assert dependency["fraction_changed_same_semantic_control"] == 0.0
    assert dependency["fraction_changed_with_permuted_semantic"] > 0.9

    # Full-catalog oracle NDCG: emitted as a diagnostic with its mechanism numbers + caveat.
    full_catalog = payload["D5_ndcg"]["full_catalog"]
    assert 0.0 < full_catalog["mean_ndcg_at_10"] <= 1.0
    assert 0.0 < full_catalog["mean_exposed_fraction_of_catalog"] < 1.0
    assert "exposure-capped" in full_catalog["caveat"]


def test_novelty_residual_is_exactly_one_for_true_first_trips(
    evaluate_ready_data_dir: Path, datagen_cfg: Any
) -> None:
    """The strongest available correctness check on the whole D1/D2 reconstruction
    pipeline: for a traveler's genuinely first-ever trip (`trip_sequence == 1`),
    `datagen/pipeline.py`'s own novelty history is empty by construction
    (`_TravelerHistory()` default), so `novelty_array` returns exactly 1.0 for
    EVERY POI (module docstring). If the other 6 recomputed terms were even
    slightly wrong, this residual would NOT land on the exact same z-scored
    constant for every such row.

    **Updated for Block A RC2a** (docs/DATA_CARD.md "DGP remediation, Block A"):
    the residual is now `w_novel * z(novelty)`, not `w_novel * novelty` directly
    (standardize-then-weight, `datagen/utility.py::standardize_and_weight`) -- for
    novelty=1.0 exactly, `z(1.0)` is a FIXED constant
    `(1.0 - standardization.novelty_mean) / standardization.novelty_std`, computed
    from the same loaded `TermStandardization` `compute_utility_term_components`
    itself uses, never re-derived independently. The invariant this test checks is
    unchanged: every true-first-trip row's residual must land on that exact same
    constant."""
    from poi_rank.datagen.oracle_export import oracle_dir_from_output
    from poi_rank.eval import oracle as oracle_reader

    data_dir = evaluate_ready_data_dir
    oracle_dir = oracle_dir_from_output(data_dir)
    components = compute_utility_term_components(data_dir, oracle_dir, datagen_cfg.utility_weights)
    standardization = oracle_reader.load_term_standardization(oracle_dir)

    trips = pd.read_parquet(data_dir / "trips.parquet")
    true_first_trip_holdout = set(
        trips.loc[(trips["trip_sequence"] == 1) & trips["is_holdout"], "trip_id"]
    )
    assert len(true_first_trip_holdout) > 0

    rows = components[components["trip_id"].isin(true_first_trip_holdout)]
    w_novel = datagen_cfg.utility_weights.w_novel
    implied_novelty_z = (rows["weighted_novelty_residual"] / w_novel).dropna()
    assert len(implied_novelty_z) > 0
    expected_z = (1.0 - standardization.novelty_mean) / standardization.novelty_std
    assert (implied_novelty_z - expected_z).abs().max() < 1e-6


# -----------------------------------------------------------------------------------
# D9: semantic-space fidelity -- hand-built core-computation tests. The MiniLM path
# (`semantic_fidelity_minilm`) is deliberately NOT given its own isolated hand-built
# unit test here: it has no meaningful hand-built form (its entire job is invoking
# the real `sentence_transformers` model), so it is exercised exactly once, for real,
# by `test_run_dgp_diagnostics_end_to_end_on_real_fixture_chain` above -- paying the
# model-load/network cost a single time rather than once per test.
# -----------------------------------------------------------------------------------


def _tiny_feature_cfg(svd_dim: int) -> FeatureBuildConfig:
    """A fully hand-built `FeatureBuildConfig` with a small `svd_dim`, independent
    of the real `configs/features.yaml` (64d) -- keeps hand-built embedding
    fixtures small (2-3 columns instead of 64)."""
    return FeatureBuildConfig(
        seed=42,
        text_embedding=TextEmbeddingConfig(
            method="tfidf",
            svd_dim=svd_dim,
            tfidf_max_features=50,
            tfidf_ngram_max=1,
            sentence_transformer_model="all-MiniLM-L6-v2",
            cache_path="unused.npy",
        ),
        poi_features=PoiFeaturesConfig(
            ctr_smoothing_alpha=1.0, density_radius_km=0.5, traveler_segment_clusters=2
        ),
        traveler_features=TravelerFeaturesConfig(
            taste_halflife_days=180.0,
            taste_weights=TasteWeights(
                booking=1.0,
                visit=1.0,
                navigate=1.0,
                save=1.0,
                share=1.0,
                click=1.0,
                view=0.05,
                dismiss=-0.8,
            ),
            confidence_shrinkage_k=5.0,
            budget_target_price_level=BudgetTargetPriceLevel(low=1.5, medium=2.5, high=3.5),
        ),
    )


def test_semantic_fidelity_spearman_hand_computed_perfect_anti_monotone() -> None:
    """4 (trip, poi) pairs: the DGP cosine (`taste_sim`) is strictly increasing
    while the observable cosine (hand-picked taste/embedding vectors) is strictly
    decreasing -- a perfect anti-monotone relationship, so Spearman rho must be
    EXACTLY -1.0."""
    components = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3", "T4"],
            "poi_id": ["P1", "P2", "P3", "P4"],
            "taste_sim": [0.1, 0.2, 0.3, 0.4],
        }
    )
    canonical_map = {"P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4"}
    taste_vec = np.array([1.0, 0.0])
    taste_by_trip = {"T1": taste_vec, "T2": taste_vec, "T3": taste_vec, "T4": taste_vec}
    poi_emb_by_id = {
        "P1": np.array([1.0, 0.0]),  # cos = 1.0
        "P2": np.array([1.0, 1.0]) / np.sqrt(2.0),  # cos ~= 0.7071
        "P3": np.array([0.0, 1.0]),  # cos = 0.0
        "P4": np.array([-1.0, 0.0]),  # cos = -1.0
    }
    result = semantic_fidelity_spearman(components, canonical_map, taste_by_trip, poi_emb_by_id)
    assert result["spearman_rho"] == pytest.approx(-1.0)
    assert result["n"] == 4
    assert result["n_excluded_no_canonical_poi_id"] == 0
    assert result["n_excluded_no_embedding_or_taste"] == 0


def test_semantic_fidelity_spearman_excludes_unmapped_and_missing_embeddings() -> None:
    components = pd.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3", "T4"],
            "poi_id": ["P1", "P2", "P_UNMAPPED", "P3"],
            "taste_sim": [0.1, 0.2, 0.3, 0.4],
        }
    )
    canonical_map = {"P1": "P1", "P2": "P2", "P3": "P3"}  # P_UNMAPPED deliberately absent
    taste_vec = np.array([1.0, 0.0])
    taste_by_trip = {"T1": taste_vec, "T2": taste_vec, "T4": taste_vec}  # T3 deliberately absent
    # P3 deliberately absent (no embedding for the "T4" row's canonical poi_id).
    poi_emb_by_id = {"P1": np.array([1.0, 0.0]), "P2": np.array([0.0, 1.0])}

    result = semantic_fidelity_spearman(components, canonical_map, taste_by_trip, poi_emb_by_id)
    # T3 (poi_id="P_UNMAPPED") has no canonical mapping at all.
    assert result["n_excluded_no_canonical_poi_id"] == 1
    # T4 (poi_id="P3" -> canonical "P3") has a valid canonical id and taste vector,
    # but "P3" has no embedding -- excluded via the second counter, not the first.
    assert result["n_excluded_no_embedding_or_taste"] == 1
    assert result["n"] == 2
    assert result["spearman_rho"] == pytest.approx(-1.0)


def test_semantic_fidelity_tfidf_reconciles_pre_dedup_poi_ids_and_computes_spearman(
    tmp_path: Path,
) -> None:
    """Orchestration-level test: `semantic_fidelity_tfidf` must remap a raw
    (pre-dedup) `poi_id` in `components` through `pois_prepared.merged_poi_ids`
    before looking up `poi_features.parquet`'s embedding (docs/DATA_CARD.md
    resolved ambiguity #18) -- `components`'s poi_id "P1_dup" here only exists as
    a merged-away id, not as its own `poi_features.parquet` row."""
    cfg = _tiny_feature_cfg(svd_dim=2)
    data_dir = tmp_path

    pd.DataFrame({"poi_id": ["P1", "P2"], "merged_poi_ids": [["P1", "P1_dup"], ["P2"]]}).to_parquet(
        data_dir / "pois_prepared.parquet", index=False
    )
    pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "text_emb_00": [1.0, 1.0 / np.sqrt(2.0)],
            "text_emb_01": [0.0, 1.0 / np.sqrt(2.0)],
        }
    ).to_parquet(data_dir / "poi_features.parquet", index=False)
    pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "implicit_taste_00": [1.0, 0.0],
            "implicit_taste_01": [0.0, 1.0],
        }
    ).to_parquet(data_dir / "traveler_features.parquet", index=False)

    components = pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "poi_id": ["P1_dup", "P2"],  # "P1_dup" is a raw pre-dedup id
            "taste_sim": [0.2, 0.8],  # DGP cosine, increasing
        }
    )
    result = semantic_fidelity_tfidf(data_dir, components, cfg)
    assert result["n"] == 2
    assert result["n_excluded_no_canonical_poi_id"] == 0
    assert result["n_excluded_no_embedding_or_taste"] == 0
    # T1's taste=[1,0] vs P1's emb=[1,0] -> cos=1.0; T2's taste=[0,1] vs P2's
    # emb=[0.7071,0.7071] -> cos=0.7071 -- observable cosine strictly decreasing
    # while the DGP cosine is strictly increasing -> perfect anti-monotone for
    # this 2-point case, rho = -1.0.
    assert result["spearman_rho"] == pytest.approx(-1.0)


# -----------------------------------------------------------------------------------
# D10: is POI text generation conditioned on poi_semantic?
# -----------------------------------------------------------------------------------


def test_sample_within_destination_poi_pairs_never_crosses_destinations() -> None:
    poi_ids_by_destination = {
        "seoul": [f"S{i}" for i in range(10)],
        "kyoto": [f"K{i}" for i in range(10)],
    }
    pairs = sample_within_destination_poi_pairs(poi_ids_by_destination, n_pairs=10, seed=42)
    assert len(pairs) == 10
    seoul_ids = set(poi_ids_by_destination["seoul"])
    kyoto_ids = set(poi_ids_by_destination["kyoto"])
    for a, b in pairs:
        assert a != b
        assert (a in seoul_ids and b in seoul_ids) or (a in kyoto_ids and b in kyoto_ids)
    # Even split across 2 destinations for a round n_pairs (10 // 2 == 5 each).
    assert sum(1 for a, _ in pairs if a in seoul_ids) == 5


def test_sample_within_destination_poi_pairs_is_deterministic_given_seed() -> None:
    poi_ids_by_destination = {"seoul": [f"S{i}" for i in range(20)]}
    pairs_a = sample_within_destination_poi_pairs(poi_ids_by_destination, n_pairs=50, seed=7)
    pairs_b = sample_within_destination_poi_pairs(poi_ids_by_destination, n_pairs=50, seed=7)
    assert pairs_a == pairs_b


def test_sample_within_destination_poi_pairs_distributes_remainder_to_first_destinations() -> None:
    poi_ids_by_destination = {
        "a_dest": [f"A{i}" for i in range(5)],
        "b_dest": [f"B{i}" for i in range(5)],
        "c_dest": [f"C{i}" for i in range(5)],
    }
    pairs = sample_within_destination_poi_pairs(poi_ids_by_destination, n_pairs=10, seed=1)
    a_ids = set(poi_ids_by_destination["a_dest"])
    b_ids = set(poi_ids_by_destination["b_dest"])
    c_ids = set(poi_ids_by_destination["c_dest"])
    n_a = sum(1 for x, _ in pairs if x in a_ids)
    n_b = sum(1 for x, _ in pairs if x in b_ids)
    n_c = sum(1 for x, _ in pairs if x in c_ids)
    # 10 // 3 = 3 remainder 1 -> the first destination in sorted order (a_dest)
    # gets the extra pair.
    assert (n_a, n_b, n_c) == (4, 3, 3)


def test_description_conditioning_fidelity_hand_computed_perfect_positive_relationship() -> None:
    """4 POIs, one destination, `poi_semantic` and `text_emb` set to IDENTICAL
    vectors per POI -- every sampled pair's semantic cosine exactly equals its
    text-embedding cosine by construction, so Spearman rho must be EXACTLY 1.0."""
    poi_latent = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3", "P4"],
            "destination": ["seoul"] * 4,
            "poi_semantic": [
                np.array([1.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([-1.0, 0.0]),
            ],
        }
    )
    poi_features = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3", "P4"],
            "text_emb_00": [1.0, 1.0, 0.0, -1.0],
            "text_emb_01": [0.0, 0.0, 1.0, 0.0],
        }
    )
    result = description_conditioning_fidelity(poi_latent, poi_features, n_pairs=20, seed=0)
    assert result["n_pois_in_population"] == 4
    assert result["destinations"] == ["seoul"]
    assert result["seed"] == 0
    assert result["n_pairs_sampled"] == 20
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_description_conditioning_fidelity_restricts_to_surviving_dedup_population() -> None:
    """`poi_latent` (pre-dedup, superset) inner-joined to `poi_features` (post-dedup)
    must silently restrict to the surviving population -- a poi_latent-only id
    ("P_MERGED_AWAY") must never appear in the sampled population."""
    poi_latent = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P_MERGED_AWAY"],
            "destination": ["seoul", "seoul", "seoul"],
            "poi_semantic": [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])],
        }
    )
    poi_features = pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "text_emb_00": [1.0, 0.0],
            "text_emb_01": [0.0, 1.0],
        }
    )
    result = description_conditioning_fidelity(poi_latent, poi_features, n_pairs=5, seed=3)
    assert result["n_pois_in_population"] == 2


# -----------------------------------------------------------------------------------
# A2: raw TF-IDF D10 variant, full-catalog oracle NDCG, text-dependency check.
# -----------------------------------------------------------------------------------


def _two_topic_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 POIs / 1 destination: A1, A2 share a topic word and semantic direction; B1, B2
    share another. Text and semantics agree perfectly on which pairs are similar."""
    latent = pd.DataFrame(
        {
            "poi_id": ["A1", "A2", "B1", "B2"],
            "destination": ["seoul"] * 4,
            "poi_semantic": [
                np.array([1.0, 0.0]),
                np.array([0.9, 0.1]),
                np.array([0.0, 1.0]),
                np.array([0.1, 0.9]),
            ],
        }
    )
    prepared = pd.DataFrame(
        {
            "poi_id": ["A1", "A2", "B1", "B2"],
            "name": ["Alpha", "Alpha", "Beta", "Beta"],
            "description": [
                "quiet lantern garden",
                "quiet lantern garden",
                "loud neon arcade",
                "loud neon arcade",
            ],
            "tags": [["lantern"], ["lantern"], ["neon"], ["neon"]],
            "category": ["nature_park", "nature_park", "nightlife", "nightlife"],
        }
    )
    return latent, prepared


def test_raw_tfidf_d10_hand_built_two_topic_catalog_is_positive_and_feature_free() -> None:
    latent, prepared = _two_topic_catalog()
    result = description_conditioning_fidelity_raw_tfidf(latent, prepared, n_pairs=30, seed=1)
    assert result["n_pois_in_population"] == 4
    assert result["n_pairs_sampled"] == 30
    # Same-topic pairs: text cos 1.0 and semantic cos ~1.0; cross-topic pairs: text cos
    # exactly 0 and semantic cos ~0.1 -> monotone agreement, rho must be strongly positive.
    assert result["spearman_rho"] > 0.8
    assert result["n_pairs_zero_text_cosine"] > 0


def test_raw_tfidf_d10_anti_aligned_text_gives_negative_rho() -> None:
    latent, prepared = _two_topic_catalog()
    # Swap the text of the two topics: text now says A1~B1, A2~B2 (wrong pairs).
    prepared = prepared.assign(
        description=["quiet lantern garden", "loud neon arcade"] * 2,
        tags=[["lantern"], ["neon"], ["lantern"], ["neon"]],
        name=["Alpha", "Beta", "Alpha", "Beta"],
    )
    result = description_conditioning_fidelity_raw_tfidf(latent, prepared, n_pairs=30, seed=1)
    assert result["spearman_rho"] < 0.0


def test_semantic_noise_ceiling_is_one_when_semantic_is_noise_free() -> None:
    from poi_rank.datagen.taxonomy import CATEGORIES, TAG_INDEX, TAGS

    vectors = []
    cats = ["museum", "museum", "cafe", "cafe"]
    tag_lists = [["cultural"], ["cultural", "artsy"], ["foodie"], ["foodie", "trendy"]]
    for cat, tags in zip(cats, tag_lists, strict=True):
        v = np.zeros(len(CATEGORIES) + len(TAGS))
        v[CATEGORY_INDEX[cat]] = 1.0
        for t in tags:
            v[TAG_INDEX[t]] = 0.6
        vectors.append(v)
    latent = pd.DataFrame(
        {"poi_id": ["P1", "P2", "P3", "P4"], "destination": ["seoul"] * 4, "poi_semantic": vectors}
    )
    prepared = pd.DataFrame({"poi_id": latent["poi_id"], "category": cats, "tags": tag_lists})
    result = semantic_noise_ceiling(latent, prepared, n_pairs=40, seed=3)
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_text_component_ablation_reports_all_components() -> None:
    latent, prepared = _two_topic_catalog()
    out = text_component_ablation(latent, prepared, n_pairs=30, seed=1)
    assert set(out) == {
        "full",
        "tags_only",
        "tags_plus_category",
        "description_only",
        "name_only",
    }
    assert out["tags_only"] > 0.8


def test_oracle_ndcg_full_catalog_hand_computed() -> None:
    """One trip, 4 eligible POIs with true utility u1>u2>u3>u4. The trip was exposed to
    P2 (label 2) and P4 (label 1) only. Full-catalog ranking by u: P1 (unexposed, 0),
    P2 (2), P3 (unexposed, 0), P4 (1). DCG@10 = 0/1 + 3/log2(3) + 0 + 1/log2(5) (gain
    2^l - 1); ideal order (labels 2,1,0,0) gives 3 + 1/log2(3)."""
    utility = pd.DataFrame(
        {
            "trip_id": ["T1"] * 4,
            "traveler_id": ["U1"] * 4,
            "poi_id": ["P1", "P2", "P3", "P4"],
            "utility_true": [4.0, 3.0, 2.0, 1.0],
        }
    )
    interactions = pd.DataFrame(
        {"trip_id": ["T1", "T1", "T1"], "poi_id": ["P2", "P4", "P4"], "label": [2, 1, 0]}
    )
    result = oracle_ndcg_full_catalog(utility, interactions, k=10)
    expected = (3.0 / np.log2(3) + 1.0 / np.log2(5)) / (3.0 + 1.0 / np.log2(3))
    assert result["mean_ndcg_at_10"] == pytest.approx(expected)
    assert result["mean_exposed_fraction_of_catalog"] == pytest.approx(0.5)
    # Oracle top-10 is all 4 POIs; 2 of them (P1, P3) were never exposed.
    assert result["mean_share_top10_by_true_utility_unexposed"] == pytest.approx(0.5)
    # Restricted to exposed POIs {P2, P4}: oracle order P2, P4 = ideal order -> NDCG 1.0.
    assert result["ndcg_at_10_restricted_to_exposed"] == pytest.approx(1.0)
    assert result["n_trips_included"] == 1
    assert "exposure-capped" in result["caveat"]


def test_oracle_ndcg_full_catalog_excludes_trips_without_positives() -> None:
    utility = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2", "T2"],
            "traveler_id": ["U1", "U1", "U2", "U2"],
            "poi_id": ["P1", "P2", "P1", "P2"],
            "utility_true": [2.0, 1.0, 2.0, 1.0],
        }
    )
    interactions = pd.DataFrame({"trip_id": ["T1"], "poi_id": ["P1"], "label": [3]})
    result = oracle_ndcg_full_catalog(utility, interactions, k=10)
    assert result["n_trips_included"] == 1
    assert result["n_trips_excluded_zero_relevant"] == 1
    assert result["mean_ndcg_at_10"] == pytest.approx(1.0)


def test_text_dependency_check_on_real_catalog(
    evaluate_ready_data_dir: Path, datagen_cfg: Any
) -> None:
    from poi_rank.datagen.oracle_export import oracle_dir_from_output
    from poi_rank.eval import oracle as oracle_reader

    poi_latent = oracle_reader.load_poi_latent(oracle_dir_from_output(evaluate_ready_data_dir))
    pois_raw = pd.read_parquet(evaluate_ready_data_dir / "pois.parquet")
    result = text_dependency_check(poi_latent, pois_raw, datagen_cfg, seed=42)
    assert result["n_pois"] > 1000
    assert result["fraction_reproduced_from_own_semantic"] == pytest.approx(1.0)
    assert result["fraction_changed_same_semantic_control"] == 0.0
    assert result["fraction_changed_with_permuted_semantic"] > 0.9
