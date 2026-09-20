"""Derived quantities that the docs quote (`eval/compose.py::derived_numbers`)."""

from __future__ import annotations

import pytest

from poi_rank.eval.compose import COMPOSED_PARTS, derived_numbers


def test_within_cross_fraction_uses_no_signal_floor_of_one() -> None:
    # ratio / perfect_ratio (1.20 / 1.89 = 63%) would overstate the recovered share: 1.0 is "no
    # archetype signal", so the achievable gap is (perfect - 1) and the share is (ratio - 1) / it.
    metrics = {
        "scenario4_diagnosis": {
            "post_fix": {
                "holdout_personalization_reference": {
                    "within_cross_ratio_true_labels": 1.2,
                    "perfect_ranker_within_cross_ratio": 1.9,
                }
            }
        }
    }
    got = derived_numbers(metrics)["within_cross_fraction_of_achievable"]
    assert got == pytest.approx(0.2 / 0.9)
    assert got != pytest.approx(1.2 / 1.9)


def test_within_cross_fraction_absent_without_the_diagnosis_part() -> None:
    assert "within_cross_fraction_of_achievable" not in derived_numbers({})


def test_mmr_lift_endpoints_are_scalars_of_the_lambda_curve() -> None:
    stage = {"long_tail_precision": 0.1}
    metrics = {
        "longtail_stages": {
            "stages": {
                "0_candidate_pool (base)": {"long_tail_precision": 0.1},
                "1_raw_ranker_top10": stage,
                "2_after_hard_gate_raw_order": stage,
                "3_after_utility": stage,
                "4_final_after_mmr": stage,
            },
            "mmr_lambda_diagnostic_post_hoc_on_holdout": {
                "0.5": {"long_tail_precision": 0.2},
                "1": {"long_tail_precision": 0.3},
            },
        }
    }
    out = derived_numbers(metrics)
    assert out["mmr_lift_lambda_0_5"] == pytest.approx(2.0)
    assert out["mmr_lift_lambda_1_0"] == pytest.approx(3.0)


def test_touristiness_axis_part_is_composed_under_its_own_key() -> None:
    assert COMPOSED_PARTS["touristiness_axis"] == "touristiness_axis"
