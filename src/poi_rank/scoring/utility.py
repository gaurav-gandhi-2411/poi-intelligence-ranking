"""Two-factor multiplicative utility (spec.md section 9.1):

```
utility = hard_gate * relevance^alpha * compatibility^beta * pref_align^gamma
```

`pref_align` (experiment L1, `docs/experiments/L-final.md`) is the per-trip stated-touristiness
instruction as a scoring-layer factor: `sigmoid(localness_centered * (-touristiness_pref) / s)`.
A repeat visitor who says "this time, local only" must get local results even when their history is
landmark-heavy; the ranker alone does not do that (`docs/TECHNICAL.md` section 3.1), and the brief
section 11 puts trip context in the scoring layer. At `gamma = 0` the utility is the original
two-factor formula exactly.

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


def pref_align(
    localness: FloatArray, touristiness_pref: FloatArray, center: float, scale: float
) -> FloatArray:
    """`sigmoid((localness - center) * (-touristiness_pref) / scale)` in `(0, 1)`.

    Sign: negative `touristiness_pref` = prefers local (the resolved DGP convention,
    `docs/DATA_CARD.md` ambiguity 1), so a local-preferring traveler scores a local POI above 0.5.
    `localness` is the OBSERVABLE localness index (`num_localness`); `center` and `scale` are the
    train-frame mean of that index and the train-frame sd of the numerator (`configs/scoring.yaml`).
    A traveler with `pref = 0` gets the constant 0.5, which cannot reorder their candidates."""
    z = (localness - center) * (-touristiness_pref) / scale
    out: FloatArray = 1.0 / (1.0 + np.exp(-z))
    return out


def compute_utility(
    hard_gate: FloatArray,
    relevance: FloatArray,
    compatibility: FloatArray,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    pref_align_factor: FloatArray | None = None,
) -> FloatArray:
    """`hard_gate * relevance^alpha * compatibility^beta * pref_align^gamma` (spec.md section 9.1
    plus experiment L1). `relevance`/`compatibility` are expected in `[0, 1]`; `hard_gate` in
    `{0.0, 1.0}`. With `gamma == 0` (the default) `pref_align_factor` is not read and the result is
    the original two-factor utility exactly."""
    result: FloatArray = hard_gate * np.power(relevance, alpha) * np.power(compatibility, beta)
    if gamma != 0.0:
        if pref_align_factor is None:
            raise ValueError("gamma != 0 requires pref_align_factor")
        result = result * np.power(pref_align_factor, gamma)
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
