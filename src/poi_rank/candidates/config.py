"""Typed loader for the `candidates` section of `configs/features.yaml` (candidate
generation channel quotas and thresholds, spec.md section 7).

No dedicated `configs/candidates.yaml` file exists -- spec.md section 14's repository
layout lists exactly `datagen.yaml, features.yaml, model.yaml, scoring.yaml,
eval.yaml`. Mirrors the codebase's established one-typed-config-per-phase convention
(`data.config.FeaturesConfig` and `features.config.FeatureBuildConfig` already read
two independent sections of this SAME physical `configs/features.yaml` file) rather
than inventing a new yaml file spec.md never asked for -- see docs/DATA_CARD.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MOBILITY_TYPES: tuple[str, ...] = ("walk", "public_transport", "car", "mixed")


@dataclass(frozen=True)
class GeoChannelConfig:
    """H3 k-ring + exact-radius geo channel (spec.md section 7)."""

    quota: int
    h3_resolution: int
    radius_km_walk: float
    radius_km_public_transport: float
    radius_km_car: float
    radius_km_mixed: float

    def radius_km_for_mobility(self, mobility: str) -> float:
        mapping = {
            "walk": self.radius_km_walk,
            "public_transport": self.radius_km_public_transport,
            "car": self.radius_km_car,
            "mixed": self.radius_km_mixed,
        }
        if mobility not in mapping:
            raise ValueError(
                f"unknown mobility mode '{mobility}', expected one of {MOBILITY_TYPES}"
            )
        return mapping[mobility]


@dataclass(frozen=True)
class InterestChannelConfig:
    quota: int


@dataclass(frozen=True)
class SemanticChannelConfig:
    quota: int


@dataclass(frozen=True)
class CollaborativeChannelConfig:
    """Item-item kNN collaborative-filtering channel."""

    quota: int
    top_k_neighbors: int


@dataclass(frozen=True)
class LongtailChannelConfig:
    """Long-tail exploration channel -- HARD FLOOR quota, never backfilled."""

    quota: int
    pop_pct_cutoff: float
    semantic_sim_threshold: float
    epsilon: float


@dataclass(frozen=True)
class ArchetypeChannelConfig:
    quota: int


@dataclass(frozen=True)
class LearnedChannelConfig:
    """Learned first-stage retriever (`candidates/retriever.py`). `quota` is the per-trip
    top-K taken from its full-catalog scores; the rest fit the IPS-weighted LightGBM scorer."""

    quota: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    n_folds: int
    ips_clip_min: float
    num_threads: int


@dataclass(frozen=True)
class CandidatesConfig:
    """Full, typed view of `configs/features.yaml`'s `candidates` section, plus the
    two upstream knobs candidate generation must stay consistent with:
    `seed` (top-level) and `poi_features.traveler_segment_clusters` (must match the
    K used to build `poi_features.parquet`'s `behav_archetype_affinity_NN` columns,
    which the archetype-prior channel directly consumes)."""

    seed: int
    traveler_segment_clusters: int
    geo: GeoChannelConfig
    interest: InterestChannelConfig
    semantic: SemanticChannelConfig
    collaborative: CollaborativeChannelConfig
    longtail: LongtailChannelConfig
    archetype: ArchetypeChannelConfig
    # None = legacy 6-channel behaviour (kept so hand-built configs in unit tests and the
    # pre-A3 union stay valid).
    learned: LearnedChannelConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> CandidatesConfig:
        """Parse and validate `configs/features.yaml`'s `candidates` section."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        c = raw["candidates"]
        return cls(
            seed=raw["seed"],
            traveler_segment_clusters=raw["poi_features"]["traveler_segment_clusters"],
            geo=GeoChannelConfig(**c["geo"]),
            interest=InterestChannelConfig(**c["interest"]),
            semantic=SemanticChannelConfig(**c["semantic"]),
            collaborative=CollaborativeChannelConfig(**c["collaborative"]),
            longtail=LongtailChannelConfig(**c["longtail"]),
            archetype=ArchetypeChannelConfig(**c["archetype"]),
            learned=LearnedChannelConfig(**c["learned"]) if "learned" in c else None,
        )
