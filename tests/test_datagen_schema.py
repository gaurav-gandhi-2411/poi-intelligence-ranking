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
from poi_rank.datagen.utility import true_utility_noise_free

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
