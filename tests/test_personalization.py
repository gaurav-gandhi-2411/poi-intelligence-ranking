"""`eval/personalization.py` tests: hand-computed Jaccard/RBO examples (spec.md
section 11.2), following this project's established hand-verification discipline."""

from __future__ import annotations

import pytest

from poi_rank.eval.personalization import (
    jaccard,
    mean_pairwise_jaccard,
    rank_biased_overlap,
    top10_lists_from_payload,
    within_cross_archetype_jaccard,
)


def test_jaccard_full_overlap_is_one() -> None:
    assert jaccard({"A", "B", "C"}, {"A", "B", "C"}) == pytest.approx(1.0)


def test_jaccard_hand_computed_partial_overlap() -> None:
    # {A,B,C} vs {B,C,D}: intersection={B,C} (2), union={A,B,C,D} (4) -> 0.5
    assert jaccard({"A", "B", "C"}, {"B", "C", "D"}) == pytest.approx(0.5)


def test_jaccard_disjoint_is_zero() -> None:
    assert jaccard({"A", "B"}, {"C", "D"}) == pytest.approx(0.0)


def test_jaccard_both_empty_is_one_by_convention() -> None:
    assert jaccard(set(), set()) == pytest.approx(1.0)


def test_mean_pairwise_jaccard_hand_computed() -> None:
    lists_by_trip = {
        "T1": ["A", "B"],
        "T2": ["A", "B"],  # jaccard(T1,T2) = 1.0
        "T3": ["C", "D"],  # jaccard(T1,T3) = 0.0, jaccard(T2,T3) = 0.0
    }
    mean, n_pairs = mean_pairwise_jaccard(lists_by_trip)
    assert n_pairs == 3
    assert mean == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)


def test_rank_biased_overlap_identical_lists_is_one() -> None:
    assert rank_biased_overlap(["A", "B", "C"], ["A", "B", "C"], p=0.9) == pytest.approx(1.0)


def test_rank_biased_overlap_disjoint_lists_is_zero() -> None:
    assert rank_biased_overlap(["A", "B", "C"], ["D", "E", "F"], p=0.9) == pytest.approx(0.0)


def test_rank_biased_overlap_hand_computed_reordering() -> None:
    # S=[A,B,C], T=[B,A,C], p=0.9 -- hand-derived exactly (see module docstring's
    # formula): X_1=0 (A vs B), X_2=2 ({A,B} both), X_3=3 (all match) ->
    # RBO = (3/3)*0.9^3 + (0.1/0.9)*[(0/1)*0.9 + (2/2)*0.81 + (3/3)*0.729]
    #     = 0.729 + (1/9)*1.539 = 0.729 + 0.171 = 0.9 exactly.
    result = rank_biased_overlap(["A", "B", "C"], ["B", "A", "C"], p=0.9)
    assert result == pytest.approx(0.9, abs=1e-9)


def test_rank_biased_overlap_truncates_to_shorter_list() -> None:
    # Shorter list wins depth (k=2): S[:2]={A,B}, T[:2]={A,B} -> full overlap at k.
    result = rank_biased_overlap(["A", "B"], ["A", "B", "Z", "Y"], p=0.9)
    assert result == pytest.approx(1.0)


def test_top10_lists_from_payload_preserves_rank_order() -> None:
    payload = {
        "T1": {
            "recommendations": [
                {"poi_id": "P3", "rank": 1},
                {"poi_id": "P1", "rank": 2},
            ]
        }
    }
    assert top10_lists_from_payload(payload) == {"T1": ["P3", "P1"]}


def test_within_cross_archetype_jaccard_hand_computed() -> None:
    lists_by_trip = {
        "T1": ["A", "B"],
        "T2": ["A", "B"],  # same segment as T1, jaccard=1.0
        "T3": ["C", "D"],  # different segment, jaccard(T1,T3)=jaccard(T2,T3)=0.0
    }
    segments = {"T1": 0, "T2": 0, "T3": 1}
    result = within_cross_archetype_jaccard(lists_by_trip, segments)
    assert result.within_n_pairs == 1
    assert result.within_mean == pytest.approx(1.0)
    assert result.cross_n_pairs == 2
    assert result.cross_mean == pytest.approx(0.0)
    assert result.ratio == float("inf")


def test_within_cross_archetype_jaccard_to_dict_serializes_infinite_ratio_as_none() -> None:
    lists_by_trip = {"T1": ["A"], "T2": ["A"], "T3": ["B"]}
    segments = {"T1": 0, "T2": 0, "T3": 1}
    result = within_cross_archetype_jaccard(lists_by_trip, segments)
    d = result.to_dict()
    assert d["within_cross_ratio"] is None
    assert d["ratio_target_met"] is True
