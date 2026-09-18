"""Two-factor multiplicative utility (spec.md section 9.1):

```
utility = hard_gate * relevance^alpha * compatibility^beta   # alpha=1.0, beta=0.7
```

**Multiplicative, not additive -- the brief's section 11 example is additive, we
deviate and justify it** (spec.md section 9.1): additive scoring lets
`preference=0.95 + availability=0.00` survive ranking, exactly the failure mode the
brief itself warns about. Multiplicative + a hard gate makes a zero in ANY factor
(closed, inaccessible, unreachable) propagate to zero utility, not get averaged
away by a high preference score.

`relevance` is the isotonic-calibrated `preference_score` (`scoring/calibration.py`),
never the raw LambdaMART score directly -- raw scores are not comparable across
travelers (`scoring/calibration.py`'s own module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.scoring._ranking_metrics import ndcg_at_k
from poi_rank.scoring.config import UtilityConfig

FloatArray = npt.NDArray[np.float64]


def compute_utility(
    hard_gate: FloatArray,
    relevance: FloatArray,
    compatibility: FloatArray,
    alpha: float,
    beta: float,
) -> FloatArray:
    """`hard_gate * relevance^alpha * compatibility^beta` (spec.md section 9.1,
    verbatim). `relevance`/`compatibility` are expected in `[0, 1]`; `hard_gate` in
    `{0.0, 1.0}`."""
    result: FloatArray = hard_gate * np.power(relevance, alpha) * np.power(compatibility, beta)
    return result


@dataclass(frozen=True)
class BetaSensitivityRow:
    beta: float
    ndcg_at_10_mean: float
    n_trips_included: int
    n_trips_excluded: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "beta": self.beta,
            "ndcg@10_mean": self.ndcg_at_10_mean,
            "n_trips_included": self.n_trips_included,
            "n_trips_excluded": self.n_trips_excluded,
        }


def beta_sensitivity_table(
    frame: pd.DataFrame,
    hard_gate: FloatArray,
    relevance: FloatArray,
    compatibility: FloatArray,
    cfg: UtilityConfig,
    k: int = 10,
) -> list[BetaSensitivityRow]:
    """Sweep `cfg.beta_sweep` (spec.md section 9.1: "swept, sensitivity reported"):
    for each beta, recompute `utility` and its per-trip NDCG@k (via this package's
    own internal `_ranking_metrics.ndcg_at_k` duplicate -- see that module's
    docstring for why), aggregated to one mean-NDCG@k row per beta. `frame` must
    carry `trip_id`, `poi_id`, `label`, all row-aligned with `hard_gate`/
    `relevance`/`compatibility`.
    """
    rows: list[BetaSensitivityRow] = []
    working = frame[["trip_id", "poi_id", "label"]].copy()
    for beta in cfg.beta_sweep:
        utility = compute_utility(hard_gate, relevance, compatibility, cfg.alpha, beta)
        working["utility"] = utility
        per_trip: list[float] = []
        n_excluded = 0
        for _trip_id, group in working.groupby("trip_id", sort=True):
            labels = group["label"].to_numpy(dtype=np.int64)
            scores = group["utility"].to_numpy(dtype=np.float64)
            poi_ids = group["poi_id"].to_numpy(dtype=object)
            value = ndcg_at_k(labels, scores, poi_ids, k)
            if value is None:
                n_excluded += 1
            else:
                per_trip.append(value)
        mean_ndcg = float(np.mean(per_trip)) if per_trip else 0.0
        rows.append(
            BetaSensitivityRow(
                beta=float(beta),
                ndcg_at_10_mean=mean_ndcg,
                n_trips_included=len(per_trip),
                n_trips_excluded=n_excluded,
            )
        )
    return rows


def beta_sensitivity_payload(rows: list[BetaSensitivityRow]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]
