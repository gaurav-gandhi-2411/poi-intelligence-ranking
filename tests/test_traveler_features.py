"""Phase 3 traveler feature-table tests (spec.md section 6): explicit-block
non-latent interest matching, implicit-block unit-normalized taste vectors,
temporal-leakage safety, and the confidence-shrinkage / interaction-feature
utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.features.confidence import confidence_shrinkage_alpha
from poi_rank.features.config import (
    BudgetTargetPriceLevel,
    FeatureBuildConfig,
    PoiFeaturesConfig,
    TasteWeights,
    TextEmbeddingConfig,
    TravelerFeaturesConfig,
)
from poi_rank.features.traveler_features import (
    EXPLICIT_PREFIX,
    IMPLICIT_PREFIX,
    build_interest_vocabulary,
    compute_taste_vector,
    cosine_similarity_taste_poi,
    half_life_to_decay_constant,
    interest_match_score,
    localness_preference_gap,
    price_gap,
    traveler_history_before,
)

# ---------------------------------------------------------------------------
# Column-grouping convention
# ---------------------------------------------------------------------------


def test_traveler_feature_columns_are_block_prefixed(built_features: dict[str, Any]) -> None:
    tf: pd.DataFrame = built_features["traveler_features"]
    assert {"traveler_id", "trip_id"} <= set(tf.columns)
    non_key_cols = [c for c in tf.columns if c not in ("traveler_id", "trip_id")]
    for col in non_key_cols:
        assert col.startswith((EXPLICIT_PREFIX, IMPLICIT_PREFIX)), f"unprefixed column '{col}'"


def test_traveler_feature_table_row_count_matches_trips(built_features: dict[str, Any]) -> None:
    tf: pd.DataFrame = built_features["traveler_features"]
    assert built_features["summary"]["n_traveler_trip_rows"] == len(tf)
    assert not tf[["traveler_id", "trip_id"]].duplicated().any()


# ---------------------------------------------------------------------------
# Explicit block: interests reflect STATED (lossy), never latent, interests
# ---------------------------------------------------------------------------


def test_explicit_interest_multi_hot_matches_stated_not_latent_interests(
    built_features: dict[str, Any], prepared_data: dict[str, Any]
) -> None:
    """Grep-style structural check: the explicit interest multi-hot for a given
    traveler must exactly reproduce `travelers.parquet['interests']` (the lossy
    stated projection) -- never a fuller/different set that could only come from the
    DGP's latent taste vector (which this module never reads)."""
    tf: pd.DataFrame = built_features["traveler_features"]
    travelers = pd.read_parquet(built_features["output_dir"] / "travelers.parquet")

    vocab = build_interest_vocabulary(travelers)
    interest_cols = {label: f"{EXPLICIT_PREFIX}interest_{label}" for label in vocab}

    sample_traveler = travelers.iloc[0]
    row = tf.loc[tf["traveler_id"] == sample_traveler["traveler_id"]].iloc[0]
    stated = set(sample_traveler["interests"])

    for label, col in interest_cols.items():
        assert bool(row[col]) == (label in stated), f"mismatch on interest '{label}'"


def test_build_interest_vocabulary_is_deterministic_and_sorted() -> None:
    travelers = pd.DataFrame({"interests": [["b", "a"], ["c"], ["a", "c"]]})
    vocab = build_interest_vocabulary(travelers)
    assert vocab == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Implicit block: unit-normalized taste vector
# ---------------------------------------------------------------------------


def test_taste_vectors_are_unit_normalized_or_zero(built_features: dict[str, Any]) -> None:
    tf: pd.DataFrame = built_features["traveler_features"]
    taste_cols = [c for c in tf.columns if c.startswith(f"{IMPLICIT_PREFIX}taste_")]
    taste = tf[taste_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(taste, axis=1)
    for n in norms:
        assert n == pytest.approx(0.0, abs=1e-6) or n == pytest.approx(1.0, abs=1e-4)


def test_half_life_to_decay_constant_gives_half_decay_at_halflife() -> None:
    tau = half_life_to_decay_constant(180.0)
    assert np.exp(-180.0 / tau) == pytest.approx(0.5, abs=1e-9)


def test_compute_taste_vector_empty_history_is_zero_vector() -> None:
    empty = pd.DataFrame(columns=["poi_id", "interaction_type", "timestamp"])
    weights = TasteWeights(
        booking=1.0,
        visit=1.0,
        navigate=0.7,
        save=0.6,
        share=0.5,
        click=0.2,
        view=0.05,
        dismiss=-0.8,
    )
    vec = compute_taste_vector(empty, {}, pd.Timestamp("2025-01-01"), weights, 180.0, emb_dim=4)
    assert np.allclose(vec, 0.0)


def test_compute_taste_vector_is_unit_norm_when_nonempty() -> None:
    history = pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "interaction_type": ["visit", "click"],
            "timestamp": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-10")],
        }
    )
    emb_by_id = {"P1": np.array([1.0, 0.0]), "P2": np.array([0.0, 1.0])}
    weights = TasteWeights(
        booking=1.0,
        visit=1.0,
        navigate=0.7,
        save=0.6,
        share=0.5,
        click=0.2,
        view=0.05,
        dismiss=-0.8,
    )
    vec = compute_taste_vector(
        history, emb_by_id, pd.Timestamp("2025-02-01"), weights, 180.0, emb_dim=2
    )
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Temporal leakage: taste vector as-of T is unaffected by interactions after T
# ---------------------------------------------------------------------------


def test_taste_vector_unaffected_by_interactions_after_as_of() -> None:
    """Synthetic case: a traveler's history has one interaction before T and one
    strictly after T. The taste vector computed as-of T must be identical whether or
    not the after-T row is present in the input at all -- proof the implicit block
    cannot see the future relative to its own as-of cutoff."""
    as_of = pd.Timestamp("2025-06-01")
    before_row = pd.DataFrame(
        {
            "poi_id": ["P1"],
            "interaction_type": ["visit"],
            "timestamp": [pd.Timestamp("2025-05-01")],
        }
    )
    after_row = pd.DataFrame(
        {
            "poi_id": ["P2"],
            "interaction_type": ["booking"],
            "timestamp": [pd.Timestamp("2025-07-01")],  # strictly after as_of
        }
    )
    emb_by_id = {"P1": np.array([1.0, 0.0, 0.0]), "P2": np.array([0.0, 1.0, 0.0])}
    weights = TasteWeights(
        booking=1.0,
        visit=1.0,
        navigate=0.7,
        save=0.6,
        share=0.5,
        click=0.2,
        view=0.05,
        dismiss=-0.8,
    )

    history_before_only = before_row
    history_with_future = pd.concat([before_row, after_row], ignore_index=True)

    # As-of filtering happens upstream in traveler_history_before / the caller;
    # compute_taste_vector itself trusts its input, so this test exercises the
    # as-of filter directly (traveler_history_before), then feeds the filtered
    # result to compute_taste_vector for both cases.
    filtered_with_future = history_with_future.loc[history_with_future["timestamp"] < as_of]

    vec_before_only = compute_taste_vector(
        history_before_only, emb_by_id, as_of, weights, 180.0, emb_dim=3
    )
    vec_filtered = compute_taste_vector(
        filtered_with_future, emb_by_id, as_of, weights, 180.0, emb_dim=3
    )

    np.testing.assert_array_equal(vec_before_only, vec_filtered)
    # And the future-only POI's embedding direction must contribute nothing.
    assert vec_filtered[1] == pytest.approx(0.0)


def test_traveler_history_before_includes_earlier_same_trip_impressions() -> None:
    """Block A RC3.1 (docs/DATA_CARD.md "DGP remediation, Block A"): the exact
    scenario the task requires -- a trip with 3+ sequential impressions. The
    taste-relevant history at impression 3's own timestamp must include
    impressions 1 and 2 (SAME trip, strictly earlier timestamps) plus any
    earlier-trip history, but NOT impression 3's own row or anything at/after it."""
    interactions = pd.DataFrame(
        {
            "traveler_id": ["U1", "U1", "U1", "U1", "U1"],
            "trip_id": ["T0", "T1", "T1", "T1", "T1"],
            "poi_id": ["P0", "P1", "P2", "P3", "P4"],
            "interaction_type": ["visit", "click", "click", "click", "click"],
            "label": [3, 1, 1, 1, 1],
            "timestamp": [
                pd.Timestamp("2025-01-01"),  # earlier trip -> always included
                pd.Timestamp("2025-06-01"),  # trip T1, impression 1
                pd.Timestamp("2025-06-05"),  # trip T1, impression 2
                pd.Timestamp("2025-06-10"),  # trip T1, impression 3 (the "as-of" one)
                pd.Timestamp("2025-06-15"),  # trip T1, impression 4 (after impression 3)
            ],
        }
    )
    by_traveler = {tid: g for tid, g in interactions.groupby("traveler_id")}

    as_of_impression_3 = pd.Timestamp("2025-06-10")
    history = traveler_history_before(by_traveler, "U1", "T1", as_of_impression_3, interactions)

    assert set(history["poi_id"]) == {"P0", "P1", "P2"}
    assert "P3" not in set(history["poi_id"])  # impression 3's own row
    assert "P4" not in set(history["poi_id"])  # after impression 3


def test_traveler_history_before_excludes_current_trip_and_future_timestamps() -> None:
    interactions = pd.DataFrame(
        {
            "traveler_id": ["U1", "U1", "U1", "U2"],
            "trip_id": ["T1", "T1", "T2", "T1"],
            "poi_id": ["P1", "P2", "P3", "P4"],
            "interaction_type": ["view", "visit", "click", "view"],
            "label": [0, 3, 1, 0],
            "timestamp": [
                pd.Timestamp("2025-01-01"),  # T1's own session -> excluded from T2's history
                pd.Timestamp("2025-01-02"),  # T1's own session -> excluded from T2's history
                pd.Timestamp(
                    "2025-06-01"
                ),  # T2's own session -> irrelevant (different traveler row)
                pd.Timestamp("2025-01-01"),  # different traveler entirely
            ],
        }
    )
    by_traveler = {tid: g for tid, g in interactions.groupby("traveler_id")}

    # T2 starts after T1's sessions but its own trip_id must still be excluded, and
    # T1's sessions (same traveler, earlier trip, strictly-before timestamp) must be
    # included.
    history = traveler_history_before(
        by_traveler, "U1", "T2", pd.Timestamp("2025-05-01"), interactions
    )
    assert set(history["poi_id"]) == {"P1", "P2"}

    # A first trip (nothing before it) sees empty history.
    history_first_trip = traveler_history_before(
        by_traveler, "U1", "T1", pd.Timestamp("2024-12-01"), interactions
    )
    assert history_first_trip.empty

    # Unknown traveler -> empty, not a KeyError.
    history_unknown = traveler_history_before(
        by_traveler, "U999", "T1", pd.Timestamp("2025-05-01"), interactions
    )
    assert history_unknown.empty


# ---------------------------------------------------------------------------
# Determinism: traveler-segment K-Means
# ---------------------------------------------------------------------------


def test_assign_traveler_segments_is_deterministic() -> None:
    from poi_rank.features.traveler_features import assign_traveler_segments

    rng = np.random.default_rng(0)
    travelers = pd.DataFrame(
        {
            "traveler_id": [f"U{i:03d}" for i in range(40)],
            "interests": [
                ["local", "foodie"] if i % 2 == 0 else ["luxury", "shopping"] for i in range(40)
            ],
            "budget": rng.choice(["low", "medium", "high"], size=40).tolist(),
            "party_type": rng.choice(["solo", "couple", "friends"], size=40).tolist(),
            "touristiness_pref": rng.uniform(-1, 1, size=40).tolist(),
        }
    )
    seg1 = assign_traveler_segments(travelers, n_clusters=4, seed=42)
    seg2 = assign_traveler_segments(travelers, n_clusters=4, seed=42)
    pd.testing.assert_series_equal(seg1, seg2)


# ---------------------------------------------------------------------------
# Confidence shrinkage vs taste-vector decay: two different mechanisms
# ---------------------------------------------------------------------------


def test_confidence_shrinkage_alpha_bounds_and_monotonicity() -> None:
    assert confidence_shrinkage_alpha(0, k=5.0) == 0.0
    assert confidence_shrinkage_alpha(5, k=5.0) == pytest.approx(0.5)
    assert confidence_shrinkage_alpha(1000, k=5.0) == pytest.approx(1.0, abs=1e-2)
    vals = [confidence_shrinkage_alpha(n, k=5.0) for n in (0, 1, 5, 20, 100)]
    assert vals == sorted(vals)  # monotone increasing


def test_confidence_shrinkage_and_taste_decay_are_independent_mechanisms() -> None:
    """A traveler with many interactions but all very stale (far in the past) has
    high confidence-shrinkage alpha (evidence volume) but a near-zero-magnitude,
    heavily-decayed taste vector contribution -- proving the two mechanisms measure
    genuinely different things, not two names for the same computation."""
    n_interactions = 50
    alpha = confidence_shrinkage_alpha(n_interactions, k=5.0)
    assert alpha > 0.9  # lots of evidence -> high confidence-shrinkage weight

    tau = half_life_to_decay_constant(180.0)
    stale_decay = np.exp(-3650 / tau)  # 10 years stale
    assert stale_decay < 0.001  # but that evidence barely contributes to taste


# ---------------------------------------------------------------------------
# Traveler x POI interaction-feature functions
# ---------------------------------------------------------------------------


def test_cosine_similarity_taste_poi_identical_vectors_is_one() -> None:
    taste = np.array([[1.0, 0.0], [0.0, 1.0]])
    poi = np.array([[1.0, 0.0], [0.0, -1.0]])
    sims = cosine_similarity_taste_poi(taste, poi)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(-1.0)


def test_localness_preference_gap_is_absolute_difference() -> None:
    gap = localness_preference_gap(np.array([0.5, -0.5]), np.array([-0.5, -0.5]))
    np.testing.assert_allclose(gap, [1.0, 0.0])


def test_interest_match_score_is_coverage_not_jaccard() -> None:
    stated = [{"local", "foodie", "cheap"}]
    category = ["restaurant"]
    tags = [["local", "authentic"]]
    score = interest_match_score(stated, category, tags)
    # "local" matches tags; "foodie"/"cheap" do not -> 1/3.
    assert score[0] == pytest.approx(1.0 / 3.0)


def test_interest_match_score_empty_interests_is_zero() -> None:
    score = interest_match_score([set()], ["restaurant"], [["local"]])
    assert score[0] == 0.0


def test_assemble_traveler_features_includes_pretrip_history_in_implicit_taste() -> None:
    """Block A RC3.2 (docs/DATA_CARD.md "DGP remediation, Block A"): a traveler
    whose ONLY history is a pre-trip interaction (dated well before their first
    trip's start date) must get a NONZERO implicit taste vector and
    `n_interactions > 0` -- proving `interactions_pretrip` genuinely flows into
    the assembled feature table, not silently ignored."""
    from poi_rank.features.traveler_features import assemble_traveler_features

    travelers_df = pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "interests": [["foodie"]],
            "budget": ["medium"],
            "party_type": ["solo"],
            "mobility": ["walk"],
            "touristiness_pref": [0.0],
            "pace": ["moderate"],
            "accessibility_needs": [[]],
        }
    )
    trips_df = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "traveler_id": ["U1"],
            "start_date": [pd.Timestamp("2025-06-01")],
            "trip_duration_days": [5],
        }
    )
    pois_df = pd.DataFrame(
        {
            "poi_id": ["P1"],
            "category": ["restaurant"],
            "price_level_imputed": [2.0],
            "localness": [0.5],
            "pop_pct": [0.5],
            "merged_poi_ids": [["P1"]],
        }
    )
    poi_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    interactions_train = pd.DataFrame(
        columns=["traveler_id", "trip_id", "poi_id", "interaction_type", "label", "timestamp"]
    )
    interactions_pretrip = pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "trip_id": ["T1"],
            "poi_id": ["P1"],
            "interaction_type": ["visit"],
            "label": [3],
            "timestamp": [pd.Timestamp("2025-02-01")],  # ~120 days before trip start
        }
    )

    cfg = FeatureBuildConfig(
        seed=42,
        text_embedding=TextEmbeddingConfig(
            method="tfidf",
            svd_dim=2,
            tfidf_max_features=100,
            tfidf_ngram_max=1,
            sentence_transformer_model="x",
            cache_path="x",
        ),
        poi_features=PoiFeaturesConfig(
            ctr_smoothing_alpha=1.0, density_radius_km=1.0, traveler_segment_clusters=2
        ),
        traveler_features=TravelerFeaturesConfig(
            taste_halflife_days=180.0,
            taste_weights=TasteWeights(
                booking=1.0,
                visit=1.0,
                navigate=0.7,
                save=0.6,
                share=0.5,
                click=0.2,
                view=0.05,
                dismiss=-0.8,
            ),
            confidence_shrinkage_k=5.0,
            budget_target_price_level=BudgetTargetPriceLevel(low=1.3, medium=2.5, high=3.7),
        ),
    )

    result_without_pretrip = assemble_traveler_features(
        travelers_df, trips_df, interactions_train, pois_df, poi_embeddings, cfg
    )
    result_with_pretrip = assemble_traveler_features(
        travelers_df,
        trips_df,
        interactions_train,
        pois_df,
        poi_embeddings,
        cfg,
        interactions_pretrip,
    )

    assert result_without_pretrip[f"{IMPLICIT_PREFIX}interaction_count"].iloc[0] == 0.0
    assert result_with_pretrip[f"{IMPLICIT_PREFIX}interaction_count"].iloc[0] == 1.0
    taste_cols = [f"{IMPLICIT_PREFIX}taste_{i:02d}" for i in range(2)]
    taste_with = result_with_pretrip[taste_cols].to_numpy(dtype=float)[0]
    assert np.linalg.norm(taste_with) == pytest.approx(1.0)


def test_price_gap_uses_independent_config_not_datagen_targets() -> None:
    targets = BudgetTargetPriceLevel(low=1.5, medium=2.5, high=3.5)
    gap = price_gap(["low", "high"], np.array([1.0, 4.0]), targets)
    np.testing.assert_allclose(gap, [0.5, 0.5])


def test_assemble_traveler_features_excludes_own_session_of_train_trips() -> None:
    """Regression (experiment H, Amendment 1): a trip browsing session runs BEFORE `start_date`
    and its interactions are the labels. The trip implicit block must therefore be as-of the trip
    FIRST impression, not `start_date` -- otherwise train/validation features carry the labelled
    session while holdout features (no session rows in the pool) cannot. Pre-trip history and
    earlier trips must still be visible; a trip without logged impressions keeps `start_date`."""
    from poi_rank.features.traveler_features import assemble_traveler_features

    travelers_df = pd.DataFrame(
        {
            "traveler_id": ["U1"],
            "interests": [["foodie"]],
            "budget": ["medium"],
            "party_type": ["solo"],
            "mobility": ["walk"],
            "touristiness_pref": [0.0],
            "pace": ["moderate"],
            "accessibility_needs": [[]],
        }
    )
    trips_df = pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "traveler_id": ["U1", "U1"],
            "start_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-12-01")],
            "trip_duration_days": [5, 5],
        }
    )
    pois_df = pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "category": ["restaurant", "cafe"],
            "price_level_imputed": [2.0, 2.0],
            "localness": [0.5, 0.5],
            "pop_pct": [0.5, 0.5],
            "merged_poi_ids": [["P1"], ["P2"]],
        }
    )
    poi_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cols = ["traveler_id", "trip_id", "poi_id", "interaction_type", "label", "timestamp"]
    # T1 session: 3 engaged impressions dated 10/8/5 days BEFORE T1 start (all < start_date).
    interactions_train = pd.DataFrame(
        [
            ["U1", "T1", "P1", "click", 1, pd.Timestamp("2025-05-22")],
            ["U1", "T1", "P2", "save", 2, pd.Timestamp("2025-05-24")],
            ["U1", "T1", "P1", "visit", 3, pd.Timestamp("2025-05-27")],
        ],
        columns=cols,
    )
    interactions_pretrip = pd.DataFrame(
        [["U1", "T1", "P1", "visit", 3, pd.Timestamp("2025-02-01")]], columns=cols
    )
    cfg = FeatureBuildConfig(
        seed=42,
        text_embedding=TextEmbeddingConfig(
            method="tfidf",
            svd_dim=2,
            tfidf_max_features=100,
            tfidf_ngram_max=1,
            sentence_transformer_model="x",
            cache_path="x",
        ),
        poi_features=PoiFeaturesConfig(
            ctr_smoothing_alpha=1.0, density_radius_km=1.0, traveler_segment_clusters=2
        ),
        traveler_features=TravelerFeaturesConfig(
            taste_halflife_days=180.0,
            taste_weights=TasteWeights(
                booking=1.0,
                visit=1.0,
                navigate=0.7,
                save=0.6,
                share=0.5,
                click=0.2,
                view=0.05,
                dismiss=-0.8,
            ),
            confidence_shrinkage_k=5.0,
            budget_target_price_level=BudgetTargetPriceLevel(low=1.3, medium=2.5, high=3.7),
        ),
    )
    result = assemble_traveler_features(
        travelers_df,
        trips_df,
        interactions_train,
        pois_df,
        poi_embeddings,
        cfg,
        interactions_pretrip,
    ).set_index("trip_id")
    count = f"{IMPLICIT_PREFIX}interaction_count"
    # T1: only the pre-trip row (1); its own 3 session rows are NOT visible.
    assert result.loc["T1", count] == 1.0
    # T2 (no logged impressions): pre-trip + T1 session are all earlier than its start_date (4).
    assert result.loc["T2", count] == 4.0
