"""`scoring/compatibility.py` unit tests (spec.md section 9.1): hard gate + each of
the 6 sub-scores, on small, hand-verifiable synthetic examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poi_rank.candidates.config import GeoChannelConfig
from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.scoring.compatibility import (
    accessibility_unmet_mask,
    budget_fit_score,
    closed_entire_trip_mask,
    compatibility_geometric_mean,
    compute_hard_gate,
    duration_fit_score,
    hours_fit_score,
    mobility_fit_score,
    party_fit_score,
    reservation_fit_score,
    unreachable_mask,
)
from poi_rank.scoring.config import (
    BudgetFitConfig,
    DurationFitConfig,
    HoursFitConfig,
    MobilityFitConfig,
    PartyFitConfig,
    ReservationFitConfig,
)

ALWAYS_OPEN = np.ones(168, dtype=bool)
ALWAYS_CLOSED = np.zeros(168, dtype=bool)


def _open_9_to_17_every_day() -> np.ndarray:
    mask = np.zeros((7, 24), dtype=bool)
    mask[:, 9:17] = True
    return mask.reshape(168)


def _open_only_sunday() -> np.ndarray:
    mask = np.zeros((7, 24), dtype=bool)
    mask[6, :] = True  # sunday = weekday index 6
    return mask.reshape(168)


# -----------------------------------------------------------------------------------
# Hard gate: closed_entire_trip
# -----------------------------------------------------------------------------------


def test_closed_entire_trip_true_when_always_closed() -> None:
    # Monday 2024-01-01 is a real Monday.
    frame = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01")],
            "trip_duration_days": [3],
            "hours_mask": [ALWAYS_CLOSED],
        }
    )
    result = closed_entire_trip_mask(frame)
    assert result[0] is np.bool_(True) or bool(result[0]) is True


def test_closed_entire_trip_false_when_open_any_day_in_range() -> None:
    """POI open ONLY on Sunday; a Monday-start 3-day trip (Mon/Tue/Wed) never
    touches Sunday -> closed_entire_trip True. A Friday-start 3-day trip
    (Fri/Sat/Sun) DOES touch Sunday -> False."""
    only_sunday = _open_only_sunday()
    frame_no_sunday = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01")],  # Monday
            "trip_duration_days": [3],
            "hours_mask": [only_sunday],
        }
    )
    frame_with_sunday = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-05")],  # Friday
            "trip_duration_days": [3],
            "hours_mask": [only_sunday],
        }
    )
    assert bool(closed_entire_trip_mask(frame_no_sunday)[0]) is True
    assert bool(closed_entire_trip_mask(frame_with_sunday)[0]) is False


def test_closed_entire_trip_false_when_always_open() -> None:
    frame = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01")],
            "trip_duration_days": [2],
            "hours_mask": [ALWAYS_OPEN],
        }
    )
    assert bool(closed_entire_trip_mask(frame)[0]) is False


# -----------------------------------------------------------------------------------
# Hard gate: accessibility_unmet
# -----------------------------------------------------------------------------------


def test_accessibility_unmet_true_when_poi_lacks_stated_need() -> None:
    needs = pd.Series([["wheelchair"]])
    acc = pd.Series([{"wheelchair": False, "stroller": True, "kid_friendly": True}])
    result = accessibility_unmet_mask(needs, acc)
    assert bool(result[0]) is True


def test_accessibility_unmet_false_when_poi_meets_all_stated_needs() -> None:
    needs = pd.Series([["wheelchair", "stroller"]])
    acc = pd.Series([{"wheelchair": True, "stroller": True, "kid_friendly": False}])
    assert bool(accessibility_unmet_mask(needs, acc)[0]) is False


def test_accessibility_unmet_false_when_no_needs_stated() -> None:
    needs = pd.Series([[]])
    acc = pd.Series([{"wheelchair": False, "stroller": False, "kid_friendly": False}])
    assert bool(accessibility_unmet_mask(needs, acc)[0]) is False


# -----------------------------------------------------------------------------------
# Hard gate: unreachable_by_mobility
# -----------------------------------------------------------------------------------


def _geo_cfg() -> GeoChannelConfig:
    return GeoChannelConfig(
        quota=60,
        h3_resolution=8,
        radius_km_walk=2.0,
        radius_km_public_transport=8.0,
        radius_km_car=25.0,
        radius_km_mixed=25.0,
    )


def test_unreachable_true_beyond_walk_radius() -> None:
    # ~0.9 degrees latitude ~= 100 km -- far beyond the 2 km walk radius.
    frame = pd.DataFrame(
        {
            "poi_lat": [1.0],
            "poi_lon": [0.0],
            "stay_lat": [0.0],
            "stay_lon": [0.0],
            "mobility": ["walk"],
        }
    )
    assert bool(unreachable_mask(frame, _geo_cfg())[0]) is True


def test_unreachable_false_within_car_radius() -> None:
    frame = pd.DataFrame(
        {
            "poi_lat": [0.01],  # ~1.1 km
            "poi_lon": [0.0],
            "stay_lat": [0.0],
            "stay_lon": [0.0],
            "mobility": ["car"],
        }
    )
    assert bool(unreachable_mask(frame, _geo_cfg())[0]) is False


# -----------------------------------------------------------------------------------
# compute_hard_gate: composition
# -----------------------------------------------------------------------------------


def test_compute_hard_gate_zero_on_any_single_violated_condition() -> None:
    frame = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "trip_duration_days": [2, 2],
            "hours_mask": [ALWAYS_OPEN, ALWAYS_CLOSED],
            "accessibility_needs": [[], []],
            "accessibility": [
                {"wheelchair": True, "stroller": True, "kid_friendly": True},
                {"wheelchair": True, "stroller": True, "kid_friendly": True},
            ],
            "poi_lat": [0.01, 0.01],
            "poi_lon": [0.0, 0.0],
            "stay_lat": [0.0, 0.0],
            "stay_lon": [0.0, 0.0],
            "mobility": ["car", "car"],
        }
    )
    gate = compute_hard_gate(frame, _geo_cfg())
    np.testing.assert_array_equal(gate, np.array([1.0, 0.0]))


# -----------------------------------------------------------------------------------
# budget_fit: asymmetric penalty (over-budget ~2x under-budget)
# -----------------------------------------------------------------------------------


def test_budget_fit_over_budget_penalized_twice_under_budget() -> None:
    """Traveler target 2.0 (medium budget maps to 2.5 by default config, but we
    pass an explicit mapping here); POI A is 1.0 UNDER target (gap -1), POI B is
    1.0 OVER target (gap +1). Hand-computed: with multiplier=2.0, range=3.0 ->
    under penalty = 1/3, budget_fit_under = 1 - 1/3 = 0.6667; over penalty =
    2/3, budget_fit_over = 1 - 2/3 = 0.3333 -- exactly half."""
    target_map = BudgetTargetPriceLevel(low=2.0, medium=2.0, high=2.0)
    cfg = BudgetFitConfig(over_budget_multiplier=2.0, normalization_range=3.0)
    frame = pd.DataFrame({"budget": ["low", "low"], "price_level_imputed": [1.0, 3.0]})
    result = budget_fit_score(frame, target_map, cfg)
    np.testing.assert_allclose(result, [2.0 / 3.0, 1.0 / 3.0], rtol=1e-10)
    # Explicit "~2x" check: the PENALTY (1 - fit) for over-budget is exactly 2x
    # the under-budget penalty for an equal-magnitude gap.
    under_penalty = 1.0 - result[0]
    over_penalty = 1.0 - result[1]
    assert over_penalty == pytest.approx(2.0 * under_penalty)


def test_budget_fit_perfect_match_is_one() -> None:
    target_map = BudgetTargetPriceLevel(low=1.5, medium=2.5, high=3.5)
    cfg = BudgetFitConfig(over_budget_multiplier=2.0, normalization_range=3.0)
    frame = pd.DataFrame({"budget": ["medium"], "price_level_imputed": [2.5]})
    result = budget_fit_score(frame, target_map, cfg)
    assert result[0] == pytest.approx(1.0)


def test_budget_fit_clips_at_zero_for_extreme_gap() -> None:
    target_map = BudgetTargetPriceLevel(low=1.0, medium=1.0, high=1.0)
    cfg = BudgetFitConfig(over_budget_multiplier=2.0, normalization_range=3.0)
    frame = pd.DataFrame({"budget": ["low"], "price_level_imputed": [4.0]})
    result = budget_fit_score(frame, target_map, cfg)
    assert result[0] == pytest.approx(0.0)


# -----------------------------------------------------------------------------------
# mobility_fit
# -----------------------------------------------------------------------------------


def test_mobility_fit_one_at_zero_distance() -> None:
    cfg = MobilityFitConfig(speed_kmh={"walk": 4.5}, half_life_min={"walk": 20.0})
    frame = pd.DataFrame(
        {
            "poi_lat": [0.0],
            "poi_lon": [0.0],
            "stay_lat": [0.0],
            "stay_lon": [0.0],
            "mobility": ["walk"],
        }
    )
    result = mobility_fit_score(frame, cfg)
    assert result[0] == pytest.approx(1.0)


def test_mobility_fit_half_at_half_life_travel_time() -> None:
    """Speed 4.5 km/h, half-life 20 min -> a distance travelled in EXACTLY 20 min
    at 4.5 km/h (1.5 km) should give fit == 0.5."""
    cfg = MobilityFitConfig(speed_kmh={"walk": 6.0}, half_life_min={"walk": 10.0})
    # distance covered in exactly 10 minutes at 6 km/h = 1.0 km.
    from poi_rank.data.geo_prep import haversine_km

    # Solve for a lat offset giving ~1.0 km using haversine at the equator.
    lat_offset = 1.0 / 111.19  # ~1 km in degrees latitude
    dist = haversine_km(lat_offset, 0.0, 0.0, 0.0)
    frame = pd.DataFrame(
        {
            "poi_lat": [lat_offset],
            "poi_lon": [0.0],
            "stay_lat": [0.0],
            "stay_lon": [0.0],
            "mobility": ["walk"],
        }
    )
    result = mobility_fit_score(frame, cfg)
    travel_min = dist / 6.0 * 60.0
    expected = 0.5 ** (travel_min / 10.0)
    assert result[0] == pytest.approx(expected, rel=1e-6)
    assert result[0] == pytest.approx(0.5, abs=0.02)


# -----------------------------------------------------------------------------------
# hours_fit
# -----------------------------------------------------------------------------------


def test_hours_fit_one_when_open_throughout_plausible_window() -> None:
    cfg = HoursFitConfig(plausible_window_start_hour=9, plausible_window_end_hour=17)
    frame = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01")],
            "trip_duration_days": [2],
            "hours_mask": [_open_9_to_17_every_day()],
        }
    )
    result = hours_fit_score(frame, cfg)
    assert result[0] == pytest.approx(1.0)


def test_hours_fit_partial_when_open_half_the_window() -> None:
    cfg = HoursFitConfig(plausible_window_start_hour=9, plausible_window_end_hour=21)
    # Open 9-17 (8 of the 12-hour 9-21 window) every day -> fraction 8/12.
    frame = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01")],
            "trip_duration_days": [1],
            "hours_mask": [_open_9_to_17_every_day()],
        }
    )
    result = hours_fit_score(frame, cfg)
    assert result[0] == pytest.approx(8.0 / 12.0)


def test_hours_fit_zero_when_closed_during_plausible_window() -> None:
    cfg = HoursFitConfig(plausible_window_start_hour=9, plausible_window_end_hour=17)
    frame = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2024-01-01")],
            "trip_duration_days": [1],
            "hours_mask": [ALWAYS_CLOSED],
        }
    )
    result = hours_fit_score(frame, cfg)
    assert result[0] == pytest.approx(0.0)


# -----------------------------------------------------------------------------------
# reservation_fit
# -----------------------------------------------------------------------------------


def test_reservation_fit_one_when_lead_time_sufficient() -> None:
    cfg = ReservationFitConfig(assumed_planning_lead_days=21, decay_days=2.0)
    frame = pd.DataFrame({"reservation_lead_days": [5.0, 21.0]})
    result = reservation_fit_score(frame, cfg)
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_reservation_fit_decays_steeply_beyond_lead_time() -> None:
    cfg = ReservationFitConfig(assumed_planning_lead_days=21, decay_days=2.0)
    frame = pd.DataFrame({"reservation_lead_days": [23.0]})  # 2-day shortfall
    result = reservation_fit_score(frame, cfg)
    assert result[0] == pytest.approx(np.exp(-1.0))  # shortfall/decay_days == 1


# -----------------------------------------------------------------------------------
# party_fit
# -----------------------------------------------------------------------------------


def test_party_fit_penalizes_non_kid_friendly_for_family_with_kids() -> None:
    cfg = PartyFitConfig(kid_friendly_penalty=0.3, kid_component_weight=0.6)
    frame = pd.DataFrame(
        {
            "party_type": ["family_young_kids"],
            "accessibility_needs": [[]],
            "accessibility": [{"wheelchair": False, "stroller": False, "kid_friendly": False}],
        }
    )
    result = party_fit_score(frame, cfg)
    # kid_component=0.3, access_component=1.0 (no stated needs) -> 0.6*0.3+0.4*1.0
    assert result[0] == pytest.approx(0.6 * 0.3 + 0.4 * 1.0)


def test_party_fit_full_for_non_family_party_at_non_kid_friendly_poi() -> None:
    cfg = PartyFitConfig(kid_friendly_penalty=0.3, kid_component_weight=0.6)
    frame = pd.DataFrame(
        {
            "party_type": ["solo"],
            "accessibility_needs": [[]],
            "accessibility": [{"wheelchair": False, "stroller": False, "kid_friendly": False}],
        }
    )
    result = party_fit_score(frame, cfg)
    assert result[0] == pytest.approx(1.0)


# -----------------------------------------------------------------------------------
# duration_fit
# -----------------------------------------------------------------------------------


def test_duration_fit_one_within_pace_budget() -> None:
    cfg = DurationFitConfig(pace_budget_min={"packed": 75.0})
    frame = pd.DataFrame({"pace": ["packed"], "expected_duration_min_imputed": [60.0]})
    result = duration_fit_score(frame, cfg)
    assert result[0] == pytest.approx(1.0)


def test_duration_fit_decays_beyond_pace_budget() -> None:
    cfg = DurationFitConfig(pace_budget_min={"packed": 75.0})
    frame = pd.DataFrame({"pace": ["packed"], "expected_duration_min_imputed": [150.0]})
    result = duration_fit_score(frame, cfg)
    assert result[0] == pytest.approx(np.exp(-1.0))  # excess == budget -> ratio 1


# -----------------------------------------------------------------------------------
# compatibility_geometric_mean
# -----------------------------------------------------------------------------------


def test_compatibility_geometric_mean_all_ones_is_one() -> None:
    subs = [np.ones(3) for _ in range(6)]
    result = compatibility_geometric_mean(subs)
    np.testing.assert_allclose(result, np.ones(3))


def test_compatibility_geometric_mean_hand_computed() -> None:
    subs = [np.array([0.5]) for _ in range(6)]
    result = compatibility_geometric_mean(subs)
    assert result[0] == pytest.approx(0.5)


def test_compatibility_geometric_mean_zero_collapses_to_zero() -> None:
    subs = [np.array([1.0])] * 5 + [np.array([0.0])]
    result = compatibility_geometric_mean(subs)
    assert result[0] == pytest.approx(0.0)
