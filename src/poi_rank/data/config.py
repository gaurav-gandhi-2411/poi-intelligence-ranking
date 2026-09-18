"""Typed loader for `configs/features.yaml` (data-prep hyperparameters).

Mirrors `datagen/config.py`'s convention: YAML is the single source of truth for
tunable knobs, loaded here into frozen dataclasses for typed, attribute-style access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DedupPrepConfig:
    h3_resolution: int
    name_similarity_threshold: float
    max_distance_m: float


@dataclass(frozen=True)
class LocalnessPrepConfig:
    weight_popularity: float
    weight_foreign: float
    weight_geo: float
    weight_tag: float
    tourist_centroid_top_decile: float


@dataclass(frozen=True)
class GeoPrepConfig:
    transit_nodes_per_destination: int
    transit_node_spread_deg: float


@dataclass(frozen=True)
class FeaturesConfig:
    """Full, typed view of `configs/features.yaml`."""

    seed: int
    dedup: DedupPrepConfig
    localness: LocalnessPrepConfig
    geo: GeoPrepConfig

    @classmethod
    def from_yaml(cls, path: Path) -> FeaturesConfig:
        """Parse and validate `configs/features.yaml` into a `FeaturesConfig`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            seed=raw["seed"],
            dedup=DedupPrepConfig(**raw["dedup"]),
            localness=LocalnessPrepConfig(**raw["localness"]),
            geo=GeoPrepConfig(**raw["geo"]),
        )
