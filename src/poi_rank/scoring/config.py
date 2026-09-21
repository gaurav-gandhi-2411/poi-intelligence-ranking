"""Typed loader for `configs/scoring.yaml` (spec.md section 14): the scoring-layer
hyperparameters (spec.md section 9) -- multiplicative utility, the 6 compatibility
sub-score formulas + hard gates, isotonic calibration, confidence, and MMR diversity
re-ranking. Mirrors the codebase's established one-typed-config-per-phase convention
(`models/config.py`, `eval/config.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UtilityConfig:
    """`utility = hard_gate * relevance^alpha * compatibility^beta` (spec.md section
    9.1)."""

    alpha: float
    beta: float
    beta_sweep: tuple[float, ...]
    # Experiment L1 (`docs/experiments/L-final.md`): exponent of the per-trip stated-touristiness
    # factor `pref_align`, and its two label-free train-frame constants. gamma = 0 disables it.
    gamma: float = 0.0
    pref_align_center: float = 0.0
    pref_align_scale: float = 1.0


@dataclass(frozen=True)
class BudgetFitConfig:
    over_budget_multiplier: float
    normalization_range: float


@dataclass(frozen=True)
class MobilityFitConfig:
    speed_kmh: dict[str, float]
    half_life_min: dict[str, float]

    def speed_for(self, mobility: str) -> float:
        return self.speed_kmh[mobility]

    def half_life_for(self, mobility: str) -> float:
        return self.half_life_min[mobility]


@dataclass(frozen=True)
class HoursFitConfig:
    plausible_window_start_hour: int
    plausible_window_end_hour: int


@dataclass(frozen=True)
class ReservationFitConfig:
    assumed_planning_lead_days: float
    decay_days: float


@dataclass(frozen=True)
class PartyFitConfig:
    kid_friendly_penalty: float
    kid_component_weight: float


@dataclass(frozen=True)
class DurationFitConfig:
    pace_budget_min: dict[str, float]

    def budget_for(self, pace: str) -> float:
        return self.pace_budget_min[pace]


@dataclass(frozen=True)
class CompatibilityConfig:
    """The 6 sub-score formulas' constants (spec.md section 9.1)."""

    budget_fit: BudgetFitConfig
    mobility_fit: MobilityFitConfig
    hours_fit: HoursFitConfig
    reservation_fit: ReservationFitConfig
    party_fit: PartyFitConfig
    duration_fit: DurationFitConfig


@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int
    calibration_fraction: float
    calibration_split_seed: int


@dataclass(frozen=True)
class ConfidenceConfig:
    """`g(n_interactions_traveler, poi_impression_count, review_count, ensemble_std,
    calibration_bin_width)` (spec.md section 9.3)."""

    traveler_evidence_k: float
    poi_impression_k: float
    review_count_k: float
    ensemble_seeds: tuple[int, ...]
    ensemble_std_tau: float
    calibration_bin_width_tau: float
    weight_traveler: float
    weight_poi: float
    weight_reviews: float
    weight_ensemble: float
    weight_calibration: float


@dataclass(frozen=True)
class DiversityConfig:
    """MMR re-rank knobs (spec.md section 9.4)."""

    lambda_default: float
    lambda_sweep: tuple[float, ...]
    top_pool_size: int
    similarity_cosine_weight: float
    similarity_category_weight: float


@dataclass(frozen=True)
class OutputConfig:
    top_k: int
    model_version: str


@dataclass(frozen=True)
class ScoringConfig:
    """Full, typed view of `configs/scoring.yaml`."""

    seed: int
    utility: UtilityConfig
    compatibility: CompatibilityConfig
    calibration: CalibrationConfig
    confidence: ConfidenceConfig
    diversity: DiversityConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: Path) -> ScoringConfig:
        """Parse and validate `configs/scoring.yaml`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        u = raw["utility"]
        c = raw["compatibility"]
        cal = raw["calibration"]
        conf = raw["confidence"]
        div = raw["diversity"]
        out = raw["output"]
        return cls(
            seed=raw["seed"],
            utility=UtilityConfig(
                alpha=u["alpha"],
                beta=u["beta"],
                beta_sweep=tuple(u["beta_sweep"]),
                gamma=u.get("gamma", 0.0),
                pref_align_center=u.get("pref_align_center", 0.0),
                pref_align_scale=u.get("pref_align_scale", 1.0),
            ),
            compatibility=CompatibilityConfig(
                budget_fit=BudgetFitConfig(**c["budget_fit"]),
                mobility_fit=MobilityFitConfig(
                    speed_kmh=dict(c["mobility_fit"]["speed_kmh"]),
                    half_life_min=dict(c["mobility_fit"]["half_life_min"]),
                ),
                hours_fit=HoursFitConfig(**c["hours_fit"]),
                reservation_fit=ReservationFitConfig(**c["reservation_fit"]),
                party_fit=PartyFitConfig(**c["party_fit"]),
                duration_fit=DurationFitConfig(
                    pace_budget_min=dict(c["duration_fit"]["pace_budget_min"])
                ),
            ),
            calibration=CalibrationConfig(**cal),
            confidence=ConfidenceConfig(
                traveler_evidence_k=conf["traveler_evidence_k"],
                poi_impression_k=conf["poi_impression_k"],
                review_count_k=conf["review_count_k"],
                ensemble_seeds=tuple(conf["ensemble_seeds"]),
                ensemble_std_tau=conf["ensemble_std_tau"],
                calibration_bin_width_tau=conf["calibration_bin_width_tau"],
                weight_traveler=conf["weight_traveler"],
                weight_poi=conf["weight_poi"],
                weight_reviews=conf["weight_reviews"],
                weight_ensemble=conf["weight_ensemble"],
                weight_calibration=conf["weight_calibration"],
            ),
            diversity=DiversityConfig(
                lambda_default=div["lambda_default"],
                lambda_sweep=tuple(div["lambda_sweep"]),
                top_pool_size=div["top_pool_size"],
                similarity_cosine_weight=div["similarity_cosine_weight"],
                similarity_category_weight=div["similarity_category_weight"],
            ),
            output=OutputConfig(**out),
        )
