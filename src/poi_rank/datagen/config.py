"""Typed loader for `configs/datagen.yaml`.

All DGP hyperparameters are authored in YAML (auditable/tunable without touching
code) and loaded here into frozen dataclasses so the rest of `datagen/` gets typed,
attribute-style access instead of stringly-typed dict lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScaleConfig:
    n_destinations: int
    pois_per_destination: int
    n_travelers: int
    travelers_per_destination: int
    two_trip_traveler_fraction: float
    target_train_impressions: int
    target_holdout_random_impressions: int
    target_holdout_logged_impressions: int
    target_positive_interactions: int


@dataclass(frozen=True)
class UtilityWeights:
    w_taste: float
    w_cat: float
    w_local: float
    w_qual: float
    w_party: float
    w_price: float
    w_novel: float


@dataclass(frozen=True)
class NoiseConfig:
    sigma: float


@dataclass(frozen=True)
class ChoiceConfig:
    tau: float


@dataclass(frozen=True)
class SlateConfig:
    slate_size: int


@dataclass(frozen=True)
class ExposureConfig:
    popularity_exponent: float
    geo_decay_km: float
    geo_weight_exponent: float


@dataclass(frozen=True)
class InterestsConfig:
    random_interest_rate: float
    omission_rate: float
    top_k_latent: int


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float


@dataclass(frozen=True)
class TimelineConfig:
    start_date: str
    end_date: str


@dataclass(frozen=True)
class DirtinessConfig:
    near_duplicate_rate: float
    missing_price_level_rate: float
    missing_expected_duration_rate: float
    missing_opening_hours_rate: float
    sparse_review_count_rate: float
    new_poi_rate: float
    inconsistent_category_rate: float


@dataclass(frozen=True)
class InteractionGenerationConfig:
    engage_lambda: float
    dismiss_lambda: float
    rank_decay: float


@dataclass(frozen=True)
class NoveltyConfig:
    same_poi_repeat_penalty: float
    similar_category_repeat_penalty: float


@dataclass(frozen=True)
class DatagenConfig:
    """Full, typed view of `configs/datagen.yaml`."""

    seed: int
    scale: ScaleConfig
    utility_weights: UtilityWeights
    noise: NoiseConfig
    choice: ChoiceConfig
    slate: SlateConfig
    exposure: ExposureConfig
    interests: InterestsConfig
    split: SplitConfig
    timeline: TimelineConfig
    dirtiness: DirtinessConfig
    interaction_generation: InteractionGenerationConfig
    novelty: NoveltyConfig

    @classmethod
    def from_yaml(cls, path: Path) -> DatagenConfig:
        """Parse and validate `configs/datagen.yaml` into a `DatagenConfig`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            seed=raw["seed"],
            scale=ScaleConfig(**raw["scale"]),
            utility_weights=UtilityWeights(**raw["utility_weights"]),
            noise=NoiseConfig(**raw["noise"]),
            choice=ChoiceConfig(**raw["choice"]),
            slate=SlateConfig(**raw["slate"]),
            exposure=ExposureConfig(**raw["exposure"]),
            interests=InterestsConfig(**raw["interests"]),
            split=SplitConfig(**raw["split"]),
            timeline=TimelineConfig(**raw["timeline"]),
            dirtiness=DirtinessConfig(**raw["dirtiness"]),
            interaction_generation=InteractionGenerationConfig(**raw["interaction_generation"]),
            novelty=NoveltyConfig(**raw["novelty"]),
        )
