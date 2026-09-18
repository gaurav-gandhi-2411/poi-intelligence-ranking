"""Typed loader for the Phase 3 feature-engineering sections of `configs/features.yaml`
(`text_embedding`, `poi_features`, `traveler_features`).

Kept as its own dataclass (`FeatureBuildConfig`), independent of
`poi_rank.data.config.FeaturesConfig` (Phase 2's dedup/localness/geo config) even
though both read the same physical YAML file -- mirrors the codebase's one-typed-
config-per-phase convention (`datagen/config.py::DatagenConfig`,
`data/config.py::FeaturesConfig`) rather than coupling one phase's config class to
another's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TextEmbeddingConfig:
    """POI text representation knobs (spec.md section 5)."""

    method: str  # "tfidf" (canonical default) or "sentence_transformers" (opt-in extra)
    svd_dim: int
    tfidf_max_features: int
    tfidf_ngram_max: int
    sentence_transformer_model: str
    cache_path: str


@dataclass(frozen=True)
class PoiFeaturesConfig:
    """POI feature-table knobs (spec.md section 5)."""

    ctr_smoothing_alpha: float
    density_radius_km: float
    traveler_segment_clusters: int


@dataclass(frozen=True)
class TasteWeights:
    """Per-interaction-type weight for the implicit taste vector (spec.md section 6)."""

    booking: float
    visit: float
    navigate: float
    save: float
    share: float
    click: float
    view: float
    dismiss: float

    def get(self, interaction_type: str) -> float:
        value: float = getattr(self, interaction_type)
        return value


@dataclass(frozen=True)
class BudgetTargetPriceLevel:
    """Independently-authored budget -> target price_level mapping used by the
    traveler x POI `price_gap` interaction feature. Deliberately NOT imported from
    `datagen/utility.py::BUDGET_TARGET_PRICE_LEVEL` (see docs/DATA_CARD.md)."""

    low: float
    medium: float
    high: float

    def get(self, budget: str) -> float:
        value: float = getattr(self, budget)
        return value


@dataclass(frozen=True)
class TravelerFeaturesConfig:
    """Traveler feature-table knobs (spec.md section 6)."""

    taste_halflife_days: float
    taste_weights: TasteWeights
    confidence_shrinkage_k: float
    budget_target_price_level: BudgetTargetPriceLevel


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Full, typed view of `configs/features.yaml`'s Phase 3 sections."""

    seed: int
    text_embedding: TextEmbeddingConfig
    poi_features: PoiFeaturesConfig
    traveler_features: TravelerFeaturesConfig

    @classmethod
    def from_yaml(cls, path: Path) -> FeatureBuildConfig:
        """Parse and validate `configs/features.yaml`'s Phase 3 sections."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        tf_raw = raw["traveler_features"]
        return cls(
            seed=raw["seed"],
            text_embedding=TextEmbeddingConfig(**raw["text_embedding"]),
            poi_features=PoiFeaturesConfig(**raw["poi_features"]),
            traveler_features=TravelerFeaturesConfig(
                taste_halflife_days=tf_raw["taste_halflife_days"],
                taste_weights=TasteWeights(**tf_raw["taste_weights"]),
                confidence_shrinkage_k=tf_raw["confidence_shrinkage_k"],
                budget_target_price_level=BudgetTargetPriceLevel(
                    **tf_raw["budget_target_price_level"]
                ),
            ),
        )
