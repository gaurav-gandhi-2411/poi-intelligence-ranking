"""Schema, scale-target, and DGP-mechanism tests for the generated synthetic data
(spec.md sections 2, 1.1). Runs the full-scale generator once per session via the
`generated_data` fixture (tests/conftest.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.datagen.timeline import build_timeline
from poi_rank.datagen.utility import (
    DestinationTermStats,
    TermStandardization,
    true_utility_noise_free,
)

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
_IDENTITY_STANDARDIZATION = TermStandardization(
    per_destination={"seoul": _IDENTITY_DEST_STATS}, novelty_mean=0.0, novelty_std=1.0
)

POI_REQUIRED_COLUMNS = {
    "poi_id",
    "destination",
    "name",
    "category",
    "subcategory",
    "lat",
    "lon",
    "description",
    "tags",
    "price_level",
    "rating",
    "review_count",
    "foreign_review_ratio",
    "popularity_raw",
    "expected_duration_min",
    "opening_hours",
    "reservation_required",
    "reservation_lead_days",
    "accessibility",
    "indoor_outdoor",
    "seasonality",
    "avg_crowd_by_hour",
    "created_at",
}
TRAVELER_REQUIRED_COLUMNS = {
    "traveler_id",
    "home_market",
    "party_type",
    "budget",
    "mobility",
    "interests",
    "touristiness_pref",
    "pace",
    "accessibility_needs",
    "explicit_preferences",
    "dietary",
}
TRIP_REQUIRED_COLUMNS = {
    "trip_id",
    "traveler_id",
    "destination",
    "start_date",
    "end_date",
    "trip_duration_days",
    "party_size",
    "stay_lat",
    "stay_lon",
}
INTERACTION_REQUIRED_COLUMNS = {
    "traveler_id",
    "trip_id",
    "poi_id",
    "slate_id",
    "interaction_type",
    "timestamp",
    "position_in_slate",
    "p_expose",
}


def _load(output_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(output_dir / name)


def _within_tolerance(actual: float, target: float, rel_tol: float = 0.15) -> bool:
    return abs(actual - target) <= rel_tol * target


class TestSchema:
    def test_poi_schema(self, generated_data: dict[str, Any]) -> None:
        df = _load(generated_data["output_dir"], "pois.parquet")
        missing = POI_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"pois.parquet missing columns: {missing}"

    def test_traveler_schema(self, generated_data: dict[str, Any]) -> None:
        df = _load(generated_data["output_dir"], "travelers.parquet")
        missing = TRAVELER_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"travelers.parquet missing columns: {missing}"

    def test_trip_schema(self, generated_data: dict[str, Any]) -> None:
        df = _load(generated_data["output_dir"], "trips.parquet")
        missing = TRIP_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"trips.parquet missing columns: {missing}"

    @pytest.mark.parametrize(
        "filename",
        [
            "interactions_train.parquet",
            "interactions_holdout_random.parquet",
            "interactions_holdout_logged.parquet",
        ],
    )
    def test_interaction_schema(self, generated_data: dict[str, Any], filename: str) -> None:
        df = _load(generated_data["output_dir"], filename)
        missing = INTERACTION_REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"{filename} missing columns: {missing}"


class TestScaleTargets:
    def test_poi_count(self, generated_data: dict[str, Any]) -> None:
        counts = generated_data["summary"]["counts"]
        cfg = generated_data["cfg"]
        target = cfg.scale.n_destinations * cfg.scale.pois_per_destination
        assert _within_tolerance(counts["n_pois"], target)

    def test_traveler_count(self, generated_data: dict[str, Any]) -> None:
        counts = generated_data["summary"]["counts"]
        cfg = generated_data["cfg"]
        assert _within_tolerance(counts["n_travelers"], cfg.scale.n_travelers)

    def test_trip_count(self, generated_data: dict[str, Any]) -> None:
        counts = generated_data["summary"]["counts"]
        target = round(generated_data["cfg"].scale.n_travelers * (1 + 1 / 3))
        assert _within_tolerance(counts["n_trips"], target)

    def test_train_impressions(self, generated_data: dict[str, Any]) -> None:
        counts = generated_data["summary"]["counts"]
        cfg = generated_data["cfg"]
        assert _within_tolerance(counts["n_train_impressions"], cfg.scale.target_train_impressions)

    def test_holdout_random_impressions(self, generated_data: dict[str, Any]) -> None:
        counts = generated_data["summary"]["counts"]
        cfg = generated_data["cfg"]
        assert _within_tolerance(
            counts["n_holdout_random_impressions"], cfg.scale.target_holdout_random_impressions
        )

    def test_total_positive_interactions(self, generated_data: dict[str, Any]) -> None:
        counts = generated_data["summary"]["counts"]
        cfg = generated_data["cfg"]
        assert _within_tolerance(
            counts["n_total_positives"], cfg.scale.target_positive_interactions
        )


class TestInterestsProjection:
    def test_interests_are_lossy_not_identical_to_latent_top_k(
        self, generated_data: dict[str, Any]
    ) -> None:
        from poi_rank.datagen.taxonomy import INTEREST_LABELS

        output_dir = generated_data["output_dir"]
        travelers = _load(output_dir, "travelers.parquet")
        taste = _load(output_dir / "_oracle", "traveler_taste.parquet")
        merged = travelers.merge(taste, on="traveler_id")

        top_k = generated_data["cfg"].interests.top_k_latent
        n_differ = 0
        for row in merged.itertuples(index=False):
            taste_vec = np.asarray(row.taste_vector)
            order = np.argsort(-taste_vec)
            top_labels = {INTEREST_LABELS[i] for i in order[:top_k]}
            stated = set(row.interests)
            if stated != top_labels:
                n_differ += 1
        # With omission_rate=0.20 and random_interest_rate=0.10 applied independently
        # per stated slot, an exact match to the raw latent top-k should be rare.
        assert n_differ / len(merged) > 0.5, "stated interests look too close to raw latent top-k"


class TestUtilitySignConvention:
    def test_local_preferring_traveler_gets_positive_utility_from_local_poi(
        self, generated_data: dict[str, Any]
    ) -> None:
        """Resolved-ambiguity spot check (docs/DATA_CARD.md): the localness term must
        use `-touristiness_pref_t`, so a local-preferring traveler (negative pref) gets
        POSITIVE utility contribution from a high-localness POI, not negative."""
        cfg = generated_data["cfg"]
        touristiness_pref = -0.8  # strongly prefers local
        latent_localness = np.array([0.95])  # very local POI
        local_term = latent_localness * (-touristiness_pref)
        assert local_term[0] > 0

        # Full utility call with a neutral setup isolates the localness contribution.
        taste_vec = np.zeros(32)
        poi_semantic = np.zeros((1, 32))
        u = true_utility_noise_free(
            weights=cfg.utility_weights,
            standardization=_IDENTITY_STANDARDIZATION,
            destination="seoul",
            taste_vec=taste_vec,
            touristiness_pref=touristiness_pref,
            party_type="solo",
            budget="medium",
            poi_semantic=poi_semantic,
            poi_category_taste_idx=np.array([0]),
            poi_latent_localness=latent_localness,
            poi_latent_quality=np.array([0.0]),
            poi_wheelchair=np.array([False]),
            poi_stroller=np.array([False]),
            poi_kid_friendly=np.array([False]),
            poi_category=np.array(["restaurant"]),
            poi_price_level_true=np.array([2.5]),
            novelty=np.array([0.0]),
        )
        # w_local * localness * (-touristiness_pref) should dominate and be positive.
        assert u[0] > 0


class TestExposureAndPropensity:
    @pytest.mark.parametrize(
        "filename",
        [
            "interactions_train.parquet",
            "interactions_holdout_random.parquet",
            "interactions_holdout_logged.parquet",
        ],
    )
    def test_p_expose_in_valid_range(self, generated_data: dict[str, Any], filename: str) -> None:
        df = _load(generated_data["output_dir"], filename)
        assert (df["p_expose"] > 0).all()
        assert (df["p_expose"] <= 1.0).all()

    def test_one_row_per_exposed_poi_per_slate(self, generated_data: dict[str, Any]) -> None:
        df = _load(generated_data["output_dir"], "interactions_train.parquet")
        dup = df.duplicated(subset=["slate_id", "poi_id"]).sum()
        assert dup == 0

    def test_unengaged_exposed_pois_get_a_view_row(self, generated_data: dict[str, Any]) -> None:
        df = _load(generated_data["output_dir"], "interactions_train.parquet")
        assert (df["interaction_type"] == "view").sum() > 0
        assert set(df["interaction_type"].unique()) <= {
            "booking",
            "visit",
            "navigate",
            "save",
            "share",
            "click",
            "view",
            "dismiss",
        }


class TestDirtiness:
    def test_near_duplicate_rate_roughly_matches_config(
        self, generated_data: dict[str, Any]
    ) -> None:
        df = _load(generated_data["output_dir"], "pois.parquet")
        rate = df["is_duplicate"].mean()
        target = generated_data["cfg"].dirtiness.near_duplicate_rate
        assert _within_tolerance(rate, target, rel_tol=0.30)

    def test_missing_price_level_rate_roughly_matches_config(
        self, generated_data: dict[str, Any]
    ) -> None:
        df = _load(generated_data["output_dir"], "pois.parquet")
        rate = df["price_level"].isna().mean()
        target = generated_data["cfg"].dirtiness.missing_price_level_rate
        assert _within_tolerance(rate, target, rel_tol=0.30)

    def test_missing_opening_hours_rate_roughly_matches_config(
        self, generated_data: dict[str, Any]
    ) -> None:
        df = _load(generated_data["output_dir"], "pois.parquet")
        rate = df["opening_hours"].isna().mean()
        target = generated_data["cfg"].dirtiness.missing_opening_hours_rate
        assert _within_tolerance(rate, target, rel_tol=0.30)

    def test_sparse_poi_rate_roughly_matches_config(self, generated_data: dict[str, Any]) -> None:
        df = _load(generated_data["output_dir"], "pois.parquet")
        rate = (df["review_count"] < 10).mean()
        target = generated_data["cfg"].dirtiness.sparse_review_count_rate
        assert _within_tolerance(rate, target, rel_tol=0.30)

    def test_new_pois_have_zero_train_interactions(self, generated_data: dict[str, Any]) -> None:
        output_dir = generated_data["output_dir"]
        cfg = generated_data["cfg"]
        timeline = build_timeline(cfg)
        pois = _load(output_dir, "pois.parquet")
        new_poi_ids = set(pois.loc[pois["created_at"] >= timeline.split, "poi_id"])
        assert len(new_poi_ids) > 0, "expected at least one new POI in this run"

        train = _load(output_dir, "interactions_train.parquet")
        assert train["poi_id"].isin(new_poi_ids).sum() == 0


# ---------------------------------------------------------------------------------
# Block A RC1: primary random holdout slate widened to a distinct, larger size.
# ---------------------------------------------------------------------------------


class TestRC1SlateSize:
    def test_random_holdout_slate_size_matches_config_not_biased_slate_size(
        self, generated_data: dict[str, Any]
    ) -> None:
        cfg = generated_data["cfg"]
        output_dir = generated_data["output_dir"]
        assert cfg.slate.random_holdout_slate_size > cfg.slate.slate_size

        random_df = _load(output_dir, "interactions_holdout_random.parquet")
        train_df = _load(output_dir, "interactions_train.parquet")
        random_slate_sizes = random_df.groupby("slate_id").size()
        train_slate_sizes = train_df.groupby("slate_id").size()
        # Most slates hit the full configured size (only clipped when a trip's
        # eligible catalog is smaller than the slate target).
        assert random_slate_sizes.max() <= cfg.slate.random_holdout_slate_size
        assert train_slate_sizes.max() <= cfg.slate.slate_size
        assert random_slate_sizes.median() > train_slate_sizes.median()


# ---------------------------------------------------------------------------------
# Block A RC2b: POI geo generation conditioned on latent_localness.
# ---------------------------------------------------------------------------------


class TestRC2bGeoConditioning:
    def test_high_localness_pois_sit_farther_from_destination_center_than_low(self) -> None:
        """Direct, hand-verifiable mechanism check (no full pipeline needed): among
        POIs generated for one destination, the mean distance-from-center of the
        top-localness decile must exceed that of the bottom-localness decile --
        the literal geo-conditioning direction RC2b implements (local POIs cluster
        AWAY from the tourist center, touristy POIs cluster TOWARD it)."""
        from poi_rank.datagen.catalog import DEST_CENTERS, generate_destination_pois
        from poi_rank.datagen.config import DatagenConfig
        from poi_rank.datagen.timeline import build_timeline

        repo_root = Path(__file__).resolve().parents[1]
        cfg = DatagenConfig.from_yaml(repo_root / "configs" / "datagen.yaml")
        timeline = build_timeline(cfg)
        rng = np.random.default_rng(123)
        df = generate_destination_pois(rng, "seoul", 600, cfg, timeline)
        df = df.loc[~df["is_duplicate"]].reset_index(drop=True)

        center_lat, center_lon = DEST_CENTERS["seoul"]
        dist = np.sqrt((df["lat"] - center_lat) ** 2 + (df["lon"] - center_lon) ** 2)

        low_decile = df["latent_localness"].quantile(0.10)
        high_decile = df["latent_localness"].quantile(0.90)
        mean_dist_low = dist[df["latent_localness"] <= low_decile].mean()
        mean_dist_high = dist[df["latent_localness"] >= high_decile].mean()
        assert mean_dist_high > mean_dist_low

    def test_dominant_flavors_from_semantic_reflects_present_tags_usually(self) -> None:
        """RC2c: for a POI whose `poi_semantic` was built from a specific tag set
        with no noise, the dominant flavor must be one of those tags exactly
        (deterministic core-arithmetic check, no full pipeline needed)."""
        from poi_rank.datagen.catalog import (
            _poi_semantic_vector,
            dominant_flavors_from_semantic,
        )

        rng = np.random.default_rng(7)
        tags = ["foodie", "local"]
        vec = _poi_semantic_vector(rng, "restaurant", tags, noise_std=0.0)
        flavors = dominant_flavors_from_semantic(vec, k=2)
        assert set(flavors) == set(tags)


# ---------------------------------------------------------------------------------
# Block A RC2a: term standardization (z-score-before-weight).
# ---------------------------------------------------------------------------------


class TestRC2aStandardization:
    def test_zscore_hand_computed(self) -> None:
        from poi_rank.datagen.utility import zscore

        x = np.array([1.0, 2.0, 3.0])
        result = zscore(x, mean=2.0, std=1.0)
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_zscore_degenerate_std_returns_zeros(self) -> None:
        from poi_rank.datagen.utility import zscore

        x = np.array([1.0, 2.0, 3.0])
        result = zscore(x, mean=2.0, std=0.0)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_compute_term_standardization_hand_computed_taste_and_quality(self) -> None:
        """A tiny, fully hand-computed (1 destination, 2 POIs, 2 travelers)
        population: taste_sim and latent_quality means/stds checked by direct
        arithmetic against `np.mean`/`np.std`."""
        import pandas as pd

        from poi_rank.datagen.taxonomy import CATEGORY_INDEX, TASTE_DIM
        from poi_rank.datagen.utility import compute_term_standardization

        taste_a = np.zeros(TASTE_DIM)
        taste_a[CATEGORY_INDEX["restaurant"]] = 1.0
        taste_b = np.zeros(TASTE_DIM)
        taste_b[CATEGORY_INDEX["cafe"]] = 1.0
        taste_vectors = {"U1": taste_a, "U2": taste_b}
        travelers_df = pd.DataFrame(
            {
                "traveler_id": ["U1", "U2"],
                "touristiness_pref": [0.2, -0.3],
                "party_type": ["solo", "solo"],
                "budget": ["medium", "medium"],
            }
        )

        poi_a_semantic = np.zeros(TASTE_DIM)
        poi_a_semantic[CATEGORY_INDEX["restaurant"]] = 1.0
        poi_b_semantic = np.zeros(TASTE_DIM)
        poi_b_semantic[CATEGORY_INDEX["cafe"]] = 1.0
        poi_true_df = pd.DataFrame(
            {
                "poi_id": ["P1", "P2"],
                "destination": ["seoul", "seoul"],
                "category": ["restaurant", "cafe"],
                "is_duplicate": [False, False],
                "poi_semantic": [poi_a_semantic, poi_b_semantic],
                "latent_quality": [0.4, 0.8],
                "latent_localness": [0.5, 0.5],
                "price_level": [2.0, 2.0],
                "wheelchair": [False, False],
                "stroller": [False, False],
                "kid_friendly": [False, False],
            }
        )

        standardization = compute_term_standardization(poi_true_df, travelers_df, taste_vectors)
        stats = standardization.per_destination["seoul"]

        # latent_quality is POI-level, replicated across both travelers: the
        # cross-product mean/std over [0.4, 0.4, 0.8, 0.8] (2 POIs x 2 travelers).
        expected_qual = np.array([0.4, 0.4, 0.8, 0.8])
        assert stats.qual_mean == pytest.approx(expected_qual.mean())
        assert stats.qual_std == pytest.approx(expected_qual.std())

        # taste_sim: P1(restaurant onehot) vs U1(restaurant onehot)=1.0,
        # P1 vs U2(cafe onehot)=0.0, P2(cafe onehot) vs U1=0.0, P2 vs U2=1.0.
        expected_taste = np.array([1.0, 0.0, 0.0, 1.0])
        assert stats.taste_mean == pytest.approx(expected_taste.mean())
        assert stats.taste_std == pytest.approx(expected_taste.std())

    def test_standardize_and_weight_zscores_each_term_before_weighting(self) -> None:
        """A single-POI check: with a standardization whose mean equals the raw
        term value, that term's z-scored contribution must be exactly 0 regardless
        of its weight -- proves z-scoring happens BEFORE weighting, not after."""
        weights_all_ones = _identity_weights()
        stats = DestinationTermStats(
            taste_mean=5.0,
            taste_std=2.0,
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
        standardization = TermStandardization(
            per_destination={"seoul": stats}, novelty_mean=0.0, novelty_std=1.0
        )
        from poi_rank.datagen.utility import standardize_and_weight

        result = standardize_and_weight(
            taste_sim=np.array([5.0]),  # equals taste_mean -> z-score exactly 0
            cat_affinity=np.array([0.0]),
            local_term=np.array([0.0]),
            latent_quality=np.array([0.0]),
            party_fit=np.array([0.0]),
            price_fit=np.array([0.0]),
            novelty=np.array([0.0]),
            destination="seoul",
            standardization=standardization,
            weights=weights_all_ones,
        )
        assert result[0] == pytest.approx(0.0)


def _identity_weights() -> Any:
    from poi_rank.datagen.config import UtilityWeights

    return UtilityWeights(
        w_taste=3.0, w_cat=3.0, w_local=3.0, w_qual=3.0, w_party=3.0, w_price=3.0, w_novel=3.0
    )


# ---------------------------------------------------------------------------------
# Block A RC3.2: pre-trip history seeding + the deliberate zero-history cohort.
# ---------------------------------------------------------------------------------


class TestRC3PretripHistory:
    def test_pretrip_file_exists_and_has_expected_columns(
        self, generated_data: dict[str, Any]
    ) -> None:
        output_dir = generated_data["output_dir"]
        df = _load(output_dir, "interactions_pretrip.parquet")
        assert len(df) > 0
        assert set(df.columns) >= INTERACTION_REQUIRED_COLUMNS

    def test_pretrip_interaction_count_within_configured_range_per_traveler(
        self, generated_data: dict[str, Any]
    ) -> None:
        """Per traveler with any pre-trip history at all, the number of ENGAGED
        (non-view) pre-trip interactions should roughly track the configured
        [min_interactions, max_interactions] range (a Poisson-driven session-count
        approximation, not an exact per-traveler guarantee -- checked in aggregate)."""
        output_dir = generated_data["output_dir"]
        cfg = generated_data["cfg"]
        df = _load(output_dir, "interactions_pretrip.parquet")
        engaged = df.loc[df["label"] > 0]
        per_traveler = engaged.groupby("traveler_id").size()
        assert len(per_traveler) > 0
        # Loose aggregate band: the Poisson-based slate-count approximation means
        # individual travelers can land outside [min, max], but the median should
        # sit inside a generously widened band around it.
        median_count = per_traveler.median()
        assert cfg.pretrip_history.min_interactions * 0.3 <= median_count
        assert median_count <= cfg.pretrip_history.max_interactions * 2.0

    def test_deliberate_cold_start_cohort_is_roughly_configured_fraction(
        self, generated_data: dict[str, Any]
    ) -> None:
        output_dir = generated_data["output_dir"]
        cfg = generated_data["cfg"]
        travelers = _load(output_dir, "travelers.parquet")
        pretrip = _load(output_dir, "interactions_pretrip.parquet")

        travelers_with_pretrip = set(pretrip["traveler_id"].unique())
        all_travelers = set(travelers["traveler_id"])
        zero_history_travelers = all_travelers - travelers_with_pretrip
        zero_history_share = len(zero_history_travelers) / len(all_travelers)
        assert _within_tolerance(
            zero_history_share, cfg.pretrip_history.cold_start_fraction, rel_tol=0.35
        )

    def test_pretrip_timestamps_strictly_precede_first_trip_in_trip_session_window(
        self, generated_data: dict[str, Any]
    ) -> None:
        """Every pre-trip row must be dated strictly before ANY of that traveler's
        own in-trip session timestamps -- the temporal separation
        `features/traveler_features.py`'s per-impression as-of cutoff relies on."""
        output_dir = generated_data["output_dir"]
        pretrip = _load(output_dir, "interactions_pretrip.parquet")
        train = _load(output_dir, "interactions_train.parquet")
        holdout_random = _load(output_dir, "interactions_holdout_random.parquet")
        holdout_logged = _load(output_dir, "interactions_holdout_logged.parquet")
        in_trip = pd.concat(
            [
                train[["traveler_id", "timestamp"]],
                holdout_random[["traveler_id", "timestamp"]],
                holdout_logged[["traveler_id", "timestamp"]],
            ],
            ignore_index=True,
        )
        earliest_in_trip = in_trip.groupby("traveler_id")["timestamp"].min()
        latest_pretrip = pretrip.groupby("traveler_id")["timestamp"].max()

        common = earliest_in_trip.index.intersection(latest_pretrip.index)
        assert len(common) > 0
        assert (latest_pretrip.loc[common] < earliest_in_trip.loc[common]).all()
