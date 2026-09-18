"""`eval/longtail.py` tests: hand-computed long-tail share/precision examples
(spec.md section 11.4)."""

from __future__ import annotations

import pytest

from poi_rank.eval.longtail import longtail_share_and_precision


def test_longtail_share_and_precision_hand_computed() -> None:
    lists_by_trip = {"T1": ["A", "B", "C"], "T2": ["A", "D"]}
    pop_pct_by_poi = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.05}
    label_by_trip_poi = {("T1", "A"): 1, ("T1", "C"): 0, ("T2", "A"): 3, ("T2", "D"): 0}

    result = longtail_share_and_precision(lists_by_trip, pop_pct_by_poi, 0.5, label_by_trip_poi)

    assert result.n_total_recommended == 5
    # longtail rows: (T1,A), (T1,C), (T2,A), (T2,D) = 4
    assert result.n_longtail_recommended == 4
    assert result.share == pytest.approx(4 / 5)
    # relevant among longtail: (T1,A)=1 relevant, (T1,C)=0 not, (T2,A)=3 relevant,
    # (T2,D)=0 not (missing from dict treated as 0 anyway) -> 2 relevant
    assert result.n_longtail_relevant == 2
    assert result.precision == pytest.approx(2 / 4)


def test_longtail_share_zero_recommended_is_zero_not_nan() -> None:
    result = longtail_share_and_precision({}, {}, 0.5, {})
    assert result.n_total_recommended == 0
    assert result.share == 0.0
    assert result.precision is None


def test_longtail_precision_undefined_when_no_longtail_recommended() -> None:
    lists_by_trip = {"T1": ["B"]}
    pop_pct_by_poi = {"B": 0.9}  # not long-tail
    result = longtail_share_and_precision(lists_by_trip, pop_pct_by_poi, 0.5, {})
    assert result.n_longtail_recommended == 0
    assert result.precision is None
    assert result.to_dict()["precision_target_met"] is False


def test_longtail_missing_label_treated_as_zero() -> None:
    lists_by_trip = {"T1": ["A"]}
    pop_pct_by_poi = {"A": 0.1}
    result = longtail_share_and_precision(lists_by_trip, pop_pct_by_poi, 0.5, {})
    assert result.n_longtail_relevant == 0
    assert result.precision == pytest.approx(0.0)
