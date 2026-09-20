"""Compatibility-style traveler x POI scores used as RANKER features (experiment H).

The assignment's section 11 compatibility dimensions (budget fit, travel-time/mobility fit,
opening hours, party fit, visit duration) are computed after ranking by `scoring/compatibility.py`.
The ranker is also given them as features, so they are implemented here -- `features/` may not
import `scoring/` (firewall) -- with the SAME formulas and constants: `CompatParams` reads the
constants from `configs/scoring.yaml` (single source of truth, no duplicated numbers), and
`tests/test_compat_scores.py` asserts value-for-value equality with `scoring/compatibility.py` on a
real sample, so the two implementations cannot silently drift.

Only stated traveler attributes and derived POI columns are used; nothing here touches the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from poi_rank.data.geo_prep import haversine_km
from poi_rank.features.config import BudgetTargetPriceLevel

FloatArray = npt.NDArray[np.float64]

FAMILY_PARTY_TYPES: tuple[str, ...] = ("family_young_kids", "family_teens")
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7


@dataclass(frozen=True)
class CompatParams:
    """The constants of the compatibility formulas, as in `configs/scoring.yaml`."""

    over_budget_multiplier: float
    normalization_range: float
    speed_kmh: dict[str, float]
    half_life_min: dict[str, float]
    window_start_hour: int
    window_end_hour: int
    kid_friendly_penalty: float
    kid_component_weight: float
    pace_budget_min: dict[str, float]

    @classmethod
    def from_scoring_yaml(cls, path: Path) -> CompatParams:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["compatibility"]
        return cls(
            over_budget_multiplier=float(raw["budget_fit"]["over_budget_multiplier"]),
            normalization_range=float(raw["budget_fit"]["normalization_range"]),
            speed_kmh={k: float(v) for k, v in raw["mobility_fit"]["speed_kmh"].items()},
            half_life_min={k: float(v) for k, v in raw["mobility_fit"]["half_life_min"].items()},
            window_start_hour=int(raw["hours_fit"]["plausible_window_start_hour"]),
            window_end_hour=int(raw["hours_fit"]["plausible_window_end_hour"]),
            kid_friendly_penalty=float(raw["party_fit"]["kid_friendly_penalty"]),
            kid_component_weight=float(raw["party_fit"]["kid_component_weight"]),
            pace_budget_min={
                k: float(v) for k, v in raw["duration_fit"]["pace_budget_min"].items()
            },
        )


def build_context(
    pairs: pd.DataFrame,
    trips_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    pois_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per `(trip_id, poi_id)` of `pairs`, aligned to `pairs` row order, with the raw
    trip/traveler/POI fields the scores need."""
    keys = pairs[["trip_id", "poi_id"]]
    trip_ctx = trips_df[
        ["trip_id", "traveler_id", "stay_lat", "stay_lon", "start_date", "trip_duration_days"]
    ]
    ctx = keys.merge(trip_ctx, on="trip_id", how="left")
    ctx = ctx.merge(
        travelers_df[
            ["traveler_id", "mobility", "budget", "party_type", "pace", "accessibility_needs"]
        ],
        on="traveler_id",
        how="left",
    )
    poi_ctx = pois_df[
        [
            "poi_id",
            "lat",
            "lon",
            "hours_mask",
            "price_level_imputed",
            "accessibility",
            "expected_duration_min_imputed",
        ]
    ].rename(columns={"lat": "poi_lat", "lon": "poi_lon"})
    return ctx.merge(poi_ctx, on="poi_id", how="left")


def travel_minutes(ctx: pd.DataFrame, p: CompatParams) -> tuple[FloatArray, FloatArray]:
    """Haversine travel time under the traveler's mobility mode, and the mode half-life."""
    dist_km = haversine_km(
        ctx["poi_lat"].to_numpy(dtype=np.float64),
        ctx["poi_lon"].to_numpy(dtype=np.float64),
        ctx["stay_lat"].to_numpy(dtype=np.float64),
        ctx["stay_lon"].to_numpy(dtype=np.float64),
    )
    speed = ctx["mobility"].map(p.speed_kmh).to_numpy(dtype=np.float64)
    half_life = ctx["mobility"].map(p.half_life_min).to_numpy(dtype=np.float64)
    return dist_km / speed * 60.0, half_life


def budget_fit(ctx: pd.DataFrame, target: BudgetTargetPriceLevel, p: CompatParams) -> FloatArray:
    tgt = ctx["budget"].map(target.get).to_numpy(dtype=np.float64)
    gap = ctx["price_level_imputed"].to_numpy(dtype=np.float64) - tgt
    over = np.clip(gap, 0.0, None)
    under = np.clip(-gap, 0.0, None)
    penalty = (over * p.over_budget_multiplier + under) / p.normalization_range
    out: FloatArray = np.clip(1.0 - penalty, 0.0, 1.0)
    return out


def mobility_fit(ctx: pd.DataFrame, p: CompatParams) -> FloatArray:
    travel_min, half_life = travel_minutes(ctx, p)
    out: FloatArray = np.power(0.5, travel_min / half_life)
    return out


def hours_fit(ctx: pd.DataFrame, p: CompatParams) -> FloatArray:
    """Fraction of trip days x plausible-visit-window hours the POI is open (mean over the
    calendar days the trip spans), vectorised per unique (start weekday, duration, POI mask)."""
    s, e = p.window_start_hour, p.window_end_hour
    out = np.zeros(len(ctx), dtype=np.float64)
    start_wd = pd.to_datetime(ctx["start_date"]).dt.weekday.to_numpy()
    n_days = np.maximum(ctx["trip_duration_days"].to_numpy(dtype=np.float64).astype(np.int64), 1)
    poi_ids = ctx["poi_id"].to_numpy()
    # per-POI, per-weekday open fraction within the window (computed once per POI)
    uniq = pd.unique(poi_ids)
    frac = {
        pid: np.asarray(mask, dtype=bool).reshape(DAYS_PER_WEEK, HOURS_PER_DAY)[:, s:e].mean(axis=1)
        for pid, mask in zip(
            uniq,
            ctx.drop_duplicates("poi_id").set_index("poi_id").loc[uniq, "hours_mask"],
            strict=True,
        )
    }
    key = pd.DataFrame({"pid": poi_ids, "wd": start_wd, "n": n_days})
    cache: dict[tuple[str, int, int], float] = {}
    for i, (pid, wd, n) in enumerate(key.itertuples(index=False, name=None)):
        k = (pid, int(wd), int(n))
        if k not in cache:
            days = (int(wd) + np.arange(int(n))) % DAYS_PER_WEEK
            cache[k] = float(np.mean(frac[pid][days]))
        out[i] = cache[k]
    return out


def party_fit(ctx: pd.DataFrame, p: CompatParams) -> FloatArray:
    """Weighted blend of a kid-friendliness component (steep penalty for a family party at a POI
    without the `kid_friendly` flag) and an accessibility component (1 if every stated need is
    met)."""
    kid = np.ones(len(ctx), dtype=np.float64)
    family = ctx["party_type"].isin(FAMILY_PARTY_TYPES).to_numpy()
    flags = np.array([bool(a.get("kid_friendly", False)) for a in ctx["accessibility"]])
    kid[family & ~flags] = p.kid_friendly_penalty
    access = np.array(
        [
            1.0 if len(needs) == 0 or all(acc.get(n, False) for n in needs) else 0.0
            for needs, acc in zip(ctx["accessibility_needs"], ctx["accessibility"], strict=True)
        ]
    )
    out: FloatArray = p.kid_component_weight * kid + (1.0 - p.kid_component_weight) * access
    return out


def duration_fit(ctx: pd.DataFrame, p: CompatParams) -> FloatArray:
    budget_min = ctx["pace"].map(p.pace_budget_min).to_numpy(dtype=np.float64)
    duration = ctx["expected_duration_min_imputed"].to_numpy(dtype=np.float64)
    excess = np.clip(duration - budget_min, 0.0, None)
    out: FloatArray = np.exp(-excess / budget_min)
    return out
