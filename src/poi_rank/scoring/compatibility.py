"""Compatibility sub-scores + hard gates (spec.md section 9.1).

```
hard_gate     = 0 if (closed_entire_trip) or (accessibility_need unmet)
                   or (unreachable by mobility) else 1
compatibility = (budget_fit . mobility_fit . hours_fit . reservation_fit
                  . party_fit . duration_fit) ^ (1/6)   # geometric mean
```

**Firewall**: like every other `scoring/` module, must never import from
`poi_rank.explain` and must never reference the oracle-only export directory
(`tests/test_firewall_scoring.py`). Legitimately imports from `poi_rank.data`
(`haversine_km`), `poi_rank.features` (`BudgetTargetPriceLevel`, the SAME mapping
`features/traveler_features.py::price_gap` already uses -- see docs/DATA_CARD.md
resolved ambiguity #2), and `poi_rank.candidates` (`GeoChannelConfig`, the SAME
mobility-conditioned radius `candidates/channels.py`'s geo channel and
`models/baselines.py`'s popularity+geo baseline already use).

**`party_fit` here is a fresh, purely observable computation** from
`party_type`/`accessibility_needs` vs the POI's own `accessibility` flags --
deliberately independent from the DGP's LATENT `party_fit`
(`datagen/utility.py`), same naming, never shared code (docs/DATA_CARD.md resolved
ambiguity #2's established precedent, extended here to a 5th sub-score).

**`days_until_trip` (reservation_fit)**: this synthetic dataset carries no
"recommendation generation / booking" timestamp distinct from `trip.start_date` --
mirrors the exact gap `features/traveler_features.py`'s `explicit_days_remaining`
docstring already documents for a different field. Resolved the same way: a fixed,
configured assumed planning lead time (`configs/scoring.yaml`
`compatibility.reservation_fit.assumed_planning_lead_days`), applied uniformly to
every trip, rather than an invented per-trip value with no grounding in the data.
See docs/DATA_CARD.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.candidates.config import GeoChannelConfig
from poi_rank.data.geo_prep import haversine_km
from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.scoring.config import (
    BudgetFitConfig,
    CompatibilityConfig,
    DurationFitConfig,
    HoursFitConfig,
    MobilityFitConfig,
    PartyFitConfig,
    ReservationFitConfig,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Day-major 168-bit weekly hours mask layout: index = day_idx * 24 + hour, in
# `data/hours.py::DAYS` order ("mon", "tue", ..., "sun"). Python's `date.weekday()`
# returns 0=Monday..6=Sunday -- the SAME order, so no re-mapping is needed anywhere
# in this module.
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

FAMILY_PARTY_TYPES: tuple[str, ...] = ("family_young_kids", "family_teens")

CONTEXT_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "poi_id",
    "traveler_id",
    "stay_lat",
    "stay_lon",
    "start_date",
    "trip_duration_days",
    "mobility",
    "budget",
    "party_type",
    "pace",
    "accessibility_needs",
    "poi_lat",
    "poi_lon",
    "hours_mask",
    "price_level_imputed",
    "reservation_lead_days",
    "accessibility",
    "expected_duration_min_imputed",
)


# -----------------------------------------------------------------------------------
# Context frame assembly
# -----------------------------------------------------------------------------------


def build_compatibility_context(
    candidates_df: pd.DataFrame,
    trip_ids: set[str],
    trips_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    pois_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per `(trip_id, poi_id)` candidate pair (restricted to `trip_ids`),
    carrying every raw field the hard gate + 6 sub-scores need. Independent of
    `models.ranking_data.build_ranking_frame` (which does not carry `party_type`,
    `pace`, `accessibility_needs`, POI `accessibility`, or the raw `hours_mask`) --
    a dedicated, minimal join rather than growing that module's frame with columns
    only this phase needs.
    """
    cand = candidates_df.loc[candidates_df["trip_id"].isin(trip_ids), ["trip_id", "poi_id"]]

    trip_ctx = trips_df[
        ["trip_id", "traveler_id", "stay_lat", "stay_lon", "start_date", "trip_duration_days"]
    ]
    frame = cand.merge(trip_ctx, on="trip_id", how="left")

    traveler_ctx = travelers_df[
        ["traveler_id", "mobility", "budget", "party_type", "pace", "accessibility_needs"]
    ]
    frame = frame.merge(traveler_ctx, on="traveler_id", how="left")

    poi_ctx = pois_df[
        [
            "poi_id",
            "lat",
            "lon",
            "hours_mask",
            "price_level_imputed",
            "reservation_lead_days",
            "accessibility",
            "expected_duration_min_imputed",
        ]
    ].rename(columns={"lat": "poi_lat", "lon": "poi_lon"})
    frame = frame.merge(poi_ctx, on="poi_id", how="left")

    return frame.sort_values(["trip_id", "poi_id"]).reset_index(drop=True)


# -----------------------------------------------------------------------------------
# Hard gate
# -----------------------------------------------------------------------------------


def _covered_weekdays(start_date: pd.Timestamp, trip_duration_days: float) -> list[int]:
    """Weekday indices (0=Monday..6=Sunday, matching `hours_mask`'s day-major
    layout) for every calendar day the trip spans, in order (may repeat once
    `trip_duration_days > 7`)."""
    start_wd = int(pd.Timestamp(start_date).weekday())
    n_days = max(int(trip_duration_days), 1)
    return [(start_wd + d) % DAYS_PER_WEEK for d in range(n_days)]


def closed_entire_trip_mask(frame: pd.DataFrame) -> BoolArray:
    """`True` iff the POI has ZERO open hours (any hour of any day, the full 24h,
    not just the plausible-visit window) across every calendar day the trip spans --
    the hard-gate condition (spec.md section 9.1: "closed_entire_trip"), a much
    lower bar than the soft `hours_fit` sub-score below."""
    out = np.zeros(len(frame), dtype=bool)
    for i, (start, dur, mask) in enumerate(
        zip(frame["start_date"], frame["trip_duration_days"], frame["hours_mask"], strict=True)
    ):
        arr = np.asarray(mask, dtype=bool).reshape(DAYS_PER_WEEK, HOURS_PER_DAY)
        covered = set(_covered_weekdays(start, dur))
        out[i] = not any(arr[d].any() for d in covered)
    return out


def accessibility_unmet_mask(
    accessibility_needs: pd.Series, poi_accessibility: pd.Series
) -> BoolArray:
    """`True` iff the traveler has a stated accessibility need
    (`accessibility_needs`, e.g. `"wheelchair"`/`"stroller"`) the candidate POI's
    own `accessibility` dict does not satisfy -- the hard-gate condition (spec.md
    section 9.1: "accessibility_need unmet")."""
    out = np.zeros(len(accessibility_needs), dtype=bool)
    for i, (needs, acc) in enumerate(zip(accessibility_needs, poi_accessibility, strict=True)):
        for need in needs:
            if not acc.get(need, False):
                out[i] = True
                break
    return out


def unreachable_mask(frame: pd.DataFrame, geo_cfg: GeoChannelConfig) -> BoolArray:
    """`True` iff the candidate POI sits outside the traveler's mobility-conditioned
    radius from the trip's stay point -- the hard-gate condition (spec.md section
    9.1: "unreachable by mobility"), reusing `GeoChannelConfig.radius_km_for_mobility`
    directly (the SAME radius `candidates/channels.py::channel_geo` and
    `models/baselines.py::baseline_popularity_geo_filter` already use)."""
    dist = haversine_km(
        frame["poi_lat"].to_numpy(dtype=np.float64),
        frame["poi_lon"].to_numpy(dtype=np.float64),
        frame["stay_lat"].to_numpy(dtype=np.float64),
        frame["stay_lon"].to_numpy(dtype=np.float64),
    )
    radius = frame["mobility"].map(geo_cfg.radius_km_for_mobility).to_numpy(dtype=np.float64)
    result: BoolArray = dist > radius
    return result


def compute_hard_gate(frame: pd.DataFrame, geo_cfg: GeoChannelConfig) -> FloatArray:
    """`hard_gate` (spec.md section 9.1), 1.0/0.0 (not bool) so it composes directly
    into the multiplicative `utility` formula without an explicit cast at every call
    site."""
    closed = closed_entire_trip_mask(frame)
    unmet = accessibility_unmet_mask(frame["accessibility_needs"], frame["accessibility"])
    unreachable = unreachable_mask(frame, geo_cfg)
    survives = ~(closed | unmet | unreachable)
    return survives.astype(np.float64)


# -----------------------------------------------------------------------------------
# Compatibility sub-scores
# -----------------------------------------------------------------------------------


def budget_fit_score(
    frame: pd.DataFrame, budget_target_price_level: BudgetTargetPriceLevel, cfg: BudgetFitConfig
) -> FloatArray:
    """`1 - normalized|price_level - budget_target|`, asymmetric: over-budget gaps
    (POI pricier than the traveler's target) penalized `cfg.over_budget_multiplier`x
    an equal-magnitude under-budget gap (spec.md section 9.1)."""
    target = frame["budget"].map(budget_target_price_level.get).to_numpy(dtype=np.float64)
    price = frame["price_level_imputed"].to_numpy(dtype=np.float64)
    gap = price - target
    over = np.clip(gap, 0.0, None)
    under = np.clip(-gap, 0.0, None)
    penalty = (over * cfg.over_budget_multiplier + under) / cfg.normalization_range
    result: FloatArray = np.clip(1.0 - penalty, 0.0, 1.0)
    return result


def mobility_fit_score(frame: pd.DataFrame, cfg: MobilityFitConfig) -> FloatArray:
    """Exponential-half-life decaying function of travel time under the traveler's
    mobility mode (spec.md section 9.1): `0.5 ** (travel_time_min / half_life_min)`."""
    dist_km = haversine_km(
        frame["poi_lat"].to_numpy(dtype=np.float64),
        frame["poi_lon"].to_numpy(dtype=np.float64),
        frame["stay_lat"].to_numpy(dtype=np.float64),
        frame["stay_lon"].to_numpy(dtype=np.float64),
    )
    speed = frame["mobility"].map(cfg.speed_for).to_numpy(dtype=np.float64)
    half_life = frame["mobility"].map(cfg.half_life_for).to_numpy(dtype=np.float64)
    travel_min = dist_km / speed * 60.0
    result: FloatArray = np.power(0.5, travel_min / half_life)
    return result


def hours_fit_score(frame: pd.DataFrame, cfg: HoursFitConfig) -> FloatArray:
    """Fraction of trip days x plausible visit windows the POI is open (spec.md
    section 9.1, literal): for every calendar day the trip spans, the fraction of
    the plausible-visit-window hours the POI is open, averaged across those days
    (a day repeated because `trip_duration_days > 7` is counted once per
    occurrence, matching "fraction of trip DAYS" literally, not "fraction of
    distinct weekdays")."""
    out = np.zeros(len(frame), dtype=np.float64)
    start_h, end_h = cfg.plausible_window_start_hour, cfg.plausible_window_end_hour
    for i, (start, dur, mask) in enumerate(
        zip(frame["start_date"], frame["trip_duration_days"], frame["hours_mask"], strict=True)
    ):
        arr = np.asarray(mask, dtype=bool).reshape(DAYS_PER_WEEK, HOURS_PER_DAY)
        days = _covered_weekdays(start, dur)
        fracs = [float(arr[d, start_h:end_h].mean()) for d in days]
        out[i] = float(np.mean(fracs)) if fracs else 0.0
    return out


def reservation_fit_score(frame: pd.DataFrame, cfg: ReservationFitConfig) -> FloatArray:
    """`1 if reservation_lead_days <= days_until_trip else steep decay` (spec.md
    section 9.1, literal): `exp(-shortfall_days / decay_days)`, `shortfall_days =
    max(0, reservation_lead_days - days_until_trip)` -- reduces to exactly 1.0 when
    there is no shortfall."""
    shortfall = frame["reservation_lead_days"].to_numpy(dtype=np.float64) - float(
        cfg.assumed_planning_lead_days
    )
    shortfall = np.clip(shortfall, 0.0, None)
    result: FloatArray = np.exp(-shortfall / cfg.decay_days)
    return result


def party_fit_score(frame: pd.DataFrame, cfg: PartyFitConfig) -> FloatArray:
    """Kid/stroller/accessibility compatibility (spec.md section 9.1) -- a weighted
    blend of a kid-friendliness component (steep, non-zero penalty for a
    family-with-kids party at a non-`kid_friendly`-flagged POI) and an accessibility
    component (near-redundant with the HARD gate for any surviving candidate, kept
    for transparency in `compatibility_breakdown`, under-weighted accordingly --
    see `configs/scoring.yaml`)."""
    out = np.zeros(len(frame), dtype=np.float64)
    for i, (party_type, needs, acc) in enumerate(
        zip(frame["party_type"], frame["accessibility_needs"], frame["accessibility"], strict=True)
    ):
        if party_type in FAMILY_PARTY_TYPES:
            kid_component = 1.0 if acc.get("kid_friendly", False) else cfg.kid_friendly_penalty
        else:
            kid_component = 1.0
        if len(needs) == 0:
            access_component = 1.0
        else:
            access_component = 1.0 if all(acc.get(n, False) for n in needs) else 0.0
        out[i] = (
            cfg.kid_component_weight * kid_component
            + (1.0 - cfg.kid_component_weight) * access_component
        )
    return out


def duration_fit_score(frame: pd.DataFrame, cfg: DurationFitConfig) -> FloatArray:
    """`expected_duration` vs the remaining daily time budget implied by `pace`
    (spec.md section 9.1): `1.0` while the POI's expected visit duration is within
    the pace's per-POI time budget, exponential decay beyond it."""
    budget_min = frame["pace"].map(cfg.budget_for).to_numpy(dtype=np.float64)
    duration = frame["expected_duration_min_imputed"].to_numpy(dtype=np.float64)
    excess = np.clip(duration - budget_min, 0.0, None)
    result: FloatArray = np.exp(-excess / budget_min)
    return result


def compatibility_geometric_mean(sub_scores: list[FloatArray]) -> FloatArray:
    """Geometric mean of exactly the 6 sub-scores (spec.md section 9.1, literal
    `^(1/6)`). A true 0.0 in any one sub-score collapses `compatibility` to exactly
    0.0 -- the geometric mean's own well-known compensatory-but-zero-sensitive
    behavior, not a bug: matches spec.md's own formula verbatim."""
    stacked = np.stack(sub_scores, axis=1)
    product = np.prod(stacked, axis=1)
    result: FloatArray = np.power(product, 1.0 / len(sub_scores))
    return result


# -----------------------------------------------------------------------------------
# Top-level assembly
# -----------------------------------------------------------------------------------

COMPATIBILITY_SUB_SCORE_NAMES: tuple[str, ...] = (
    "budget_fit",
    "mobility_fit",
    "hours_fit",
    "reservation_fit",
    "party_fit",
    "duration_fit",
)


def compute_compatibility_frame(
    candidates_df: pd.DataFrame,
    trip_ids: set[str],
    trips_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    budget_target_price_level: BudgetTargetPriceLevel,
    geo_cfg: GeoChannelConfig,
    cfg: CompatibilityConfig,
) -> pd.DataFrame:
    """Full compatibility assembly (spec.md section 9.1): one row per
    `(trip_id, poi_id)` candidate pair in `trip_ids`, with `hard_gate`, all 6
    sub-scores, and their geometric-mean `compatibility`."""
    ctx = build_compatibility_context(candidates_df, trip_ids, trips_df, travelers_df, pois_df)

    hard_gate = compute_hard_gate(ctx, geo_cfg)
    sub_scores = {
        "budget_fit": budget_fit_score(ctx, budget_target_price_level, cfg.budget_fit),
        "mobility_fit": mobility_fit_score(ctx, cfg.mobility_fit),
        "hours_fit": hours_fit_score(ctx, cfg.hours_fit),
        "reservation_fit": reservation_fit_score(ctx, cfg.reservation_fit),
        "party_fit": party_fit_score(ctx, cfg.party_fit),
        "duration_fit": duration_fit_score(ctx, cfg.duration_fit),
    }
    compatibility = compatibility_geometric_mean(
        [sub_scores[name] for name in COMPATIBILITY_SUB_SCORE_NAMES]
    )

    return pd.DataFrame(
        {
            "trip_id": ctx["trip_id"],
            "poi_id": ctx["poi_id"],
            "hard_gate": hard_gate,
            **sub_scores,
            "compatibility": compatibility,
        }
    )
