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
    # Block A (docs/DATA_CARD.md "DGP remediation, Block A"): the primary
    # random-holdout policy needs its own (sharper) Plackett-Luce temperature --
    # `random_holdout_engage_lambda` raises the NUMBER of engaged draws per slate
    # (to keep D3's per-row Spearman signal from being diluted by RC1's wider
    # slate), but a shared `tau` means those EXTRA draws increasingly pull in
    # lower-true-utility items, which hurts D5's candidate-level oracle NDCG (more
    # "positive" labels the oracle itself can't rank all of in the top 10). A
    # sharper `random_holdout_tau` keeps even the additional draws concentrated
    # near the true top-utility items.
    random_holdout_tau: float


@dataclass(frozen=True)
class SlateConfig:
    slate_size: int
    # Block A RC1: primary unbiased holdout slate size, widened 20 -> 150
    # (docs/DATA_CARD.md "DGP remediation, Block A"). Train + secondary biased
    # holdout-logged slates keep using `slate_size` above, unchanged.
    random_holdout_slate_size: int


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
    # Block A (docs/DATA_CARD.md "DGP remediation, Block A"): RC1 widened the
    # primary random-holdout slate 20 -> 150 (`slate.random_holdout_slate_size`),
    # which mechanically dilutes D3 (Spearman(u, label) over EVERY exposed row) if
    # the expected engaged-count per slate stays fixed at `engage_lambda` -- far
    # more zero-label rows per positive in a 150-item slate than a 20-item one.
    # These give the random-holdout policy its OWN (larger) expected engaged/
    # dismissed count, independent of the biased train/logged-holdout policies.
    random_holdout_engage_lambda: float
    random_holdout_dismiss_lambda: float


@dataclass(frozen=True)
class NoveltyConfig:
    same_poi_repeat_penalty: float
    similar_category_repeat_penalty: float


@dataclass(frozen=True)
class PretripHistoryConfig:
    """Block A RC3.2: pre-trip synthetic interaction seeding knobs
    (docs/DATA_CARD.md "DGP remediation, Block A")."""

    cold_start_fraction: float
    min_interactions: int
    max_interactions: int
    lead_days_min: int
    lead_days_max: int


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
    pretrip_history: PretripHistoryConfig

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
            pretrip_history=PretripHistoryConfig(**raw["pretrip_history"]),
        )
