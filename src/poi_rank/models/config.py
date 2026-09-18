"""Typed loader for `configs/model.yaml` (spec.md section 14): baseline
hyperparameters for Phase 4b (this phase's 6 baselines) and later phases
(LambdaMART/IPS/calibration). Mirrors `candidates/config.py`'s / `features/config.py`'s
established one-typed-config-per-phase convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ContentCosineConfig:
    """Baseline 4 (content cosine, explicit interests only) blend weights."""

    interest_weight: float
    price_fit_weight: float


@dataclass(frozen=True)
class ItemKnnCfConfig:
    """Baseline 5 (item-item kNN collaborative filtering) cold-start fallback."""

    cf_score_missing_fallback: str


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Baseline 6 (pointwise logistic regression, full feature set) sklearn knobs."""

    max_iter: int
    C: float


@dataclass(frozen=True)
class BaselinesConfig:
    content_cosine: ContentCosineConfig
    item_knn_cf: ItemKnnCfConfig
    logistic_regression: LogisticRegressionConfig


@dataclass(frozen=True)
class ModelConfig:
    """Full, typed view of `configs/model.yaml`."""

    seed: int
    baselines: BaselinesConfig

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Parse and validate `configs/model.yaml`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        b = raw["baselines"]
        return cls(
            seed=raw["seed"],
            baselines=BaselinesConfig(
                content_cosine=ContentCosineConfig(**b["content_cosine"]),
                item_knn_cf=ItemKnnCfConfig(**b["item_knn_cf"]),
                logistic_regression=LogisticRegressionConfig(**b["logistic_regression"]),
            ),
        )
