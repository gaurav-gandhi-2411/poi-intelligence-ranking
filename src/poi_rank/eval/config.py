"""Typed loader for `configs/eval.yaml` (spec.md section 14): metrics/bootstrap knobs
for the evaluation harness (spec.md section 11). Mirrors `candidates/config.py`'s /
`models/config.py`'s established one-typed-config-per-phase convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap-CI knobs (spec.md section 8: "2,000-sample bootstrap 95% CI")."""

    n_resamples: int
    ci_low_pct: float
    ci_high_pct: float
    resample_unit: str  # "trip" -- see docs/DATA_CARD.md for the resolved ambiguity


@dataclass(frozen=True)
class MetricsConfig:
    """Which @k cutoffs to compute for each metric family (spec.md section 11.1)."""

    ndcg_ks: tuple[int, ...]
    precision_ks: tuple[int, ...]
    recall_ks: tuple[int, ...]


@dataclass(frozen=True)
class EvalConfig:
    """Full, typed view of `configs/eval.yaml`."""

    seed: int
    metrics: MetricsConfig
    bootstrap: BootstrapConfig
    long_tail_pop_pct_cutoff: float

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        """Parse and validate `configs/eval.yaml`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        m = raw["metrics"]
        bo = raw["bootstrap"]
        return cls(
            seed=raw["seed"],
            metrics=MetricsConfig(
                ndcg_ks=tuple(m["ndcg_ks"]),
                precision_ks=tuple(m["precision_ks"]),
                recall_ks=tuple(m["recall_ks"]),
            ),
            bootstrap=BootstrapConfig(**bo),
            long_tail_pop_pct_cutoff=raw["long_tail_pop_pct_cutoff"],
        )
