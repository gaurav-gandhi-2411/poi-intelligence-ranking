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
    C: float  # used as-is when `c_grid` is empty
    # S2: non-empty => choose the L2 strength by trip-grouped CV on a trip subsample (below).
    c_grid: tuple[float, ...] = ()
    cv_folds: int = 3
    cv_trip_fraction: float = 0.3


@dataclass(frozen=True)
class BaselinesConfig:
    content_cosine: ContentCosineConfig
    item_knn_cf: ItemKnnCfConfig
    logistic_regression: LogisticRegressionConfig


@dataclass(frozen=True)
class LambdaMartConfig:
    """Systems 7/8 (`models/lambdamart.py`, spec.md section 8): LightGBM
    `lambdarank` hyperparameters, IPS clip bounds, the train-only validation split
    used for early stopping, and the new-POI-robustness behavioral-dropout knobs.
    `num_threads: 1` + `deterministic: true` (hardcoded in `lambdamart.py`, not a
    yaml knob) are the CPU-determinism settings LightGBM's own docs require for
    byte-identical reruns -- see `docs/DATA_CARD.md`.
    """

    num_leaves: int
    learning_rate: float
    n_estimators: int
    early_stopping_rounds: int
    lambdarank_truncation_level: int
    eval_ndcg_at: tuple[int, ...]
    label_gain: tuple[int, ...]
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    num_threads: int
    val_fraction: float
    val_split_seed: int
    ips_clip_low: float
    ips_clip_high: float
    behavioral_dropout_rate: float
    behavioral_dropout_seed: int
    # LightGBM histogram resolution (255 = library default; kept as the default so hand-built
    # configs, e.g. the fast test config, are unchanged).
    max_bin: int = 255


@dataclass(frozen=True)
class ModelConfig:
    """Full, typed view of `configs/model.yaml`."""

    seed: int
    baselines: BaselinesConfig
    lambdamart: LambdaMartConfig

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Parse and validate `configs/model.yaml`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        b = raw["baselines"]
        lm = raw["lambdamart"]
        return cls(
            seed=raw["seed"],
            baselines=BaselinesConfig(
                content_cosine=ContentCosineConfig(**b["content_cosine"]),
                item_knn_cf=ItemKnnCfConfig(**b["item_knn_cf"]),
                logistic_regression=LogisticRegressionConfig(
                    **{
                        **b["logistic_regression"],
                        "c_grid": tuple(b["logistic_regression"].get("c_grid", ())),
                    }
                ),
            ),
            lambdamart=LambdaMartConfig(
                num_leaves=lm["num_leaves"],
                learning_rate=lm["learning_rate"],
                n_estimators=lm["n_estimators"],
                early_stopping_rounds=lm["early_stopping_rounds"],
                lambdarank_truncation_level=lm["lambdarank_truncation_level"],
                eval_ndcg_at=tuple(lm["eval_ndcg_at"]),
                label_gain=tuple(lm["label_gain"]),
                min_data_in_leaf=lm["min_data_in_leaf"],
                feature_fraction=lm["feature_fraction"],
                bagging_fraction=lm["bagging_fraction"],
                bagging_freq=lm["bagging_freq"],
                num_threads=lm["num_threads"],
                val_fraction=lm["val_fraction"],
                val_split_seed=lm["val_split_seed"],
                ips_clip_low=lm["ips_clip_low"],
                ips_clip_high=lm["ips_clip_high"],
                behavioral_dropout_rate=lm["behavioral_dropout_rate"],
                behavioral_dropout_seed=lm["behavioral_dropout_seed"],
                max_bin=lm.get("max_bin", 255),
            ),
        )
