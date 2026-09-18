"""`scoring/confidence.py` unit tests (spec.md section 9.3): `g(...)`'s individual
components on hand-checkable examples. The full monotonicity validation (confidence
decile vs NDCG@10, Spearman rho >= 0.7) is exercised end-to-end in
`tests/test_hard_constraints.py`'s sibling real-pipeline test module (`scoring/output
.py`'s `confidence_decile_validation`, invoked via `run_recommend` in
`tests/test_scoring_output.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from poi_rank.scoring.confidence import (
    calibration_stability,
    compute_confidence,
    ensemble_agreement,
    evidence_shrinkage,
)
from poi_rank.scoring.config import ConfidenceConfig


def test_evidence_shrinkage_half_at_k() -> None:
    x = np.array([0.0, 5.0, 20.0])
    result = evidence_shrinkage(x, k=5.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 20.0 / 25.0])


def test_ensemble_agreement_one_at_zero_std() -> None:
    result = ensemble_agreement(np.array([0.0, 0.5]), tau=0.5)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(np.exp(-1.0))


def test_calibration_stability_decreases_with_wider_bins() -> None:
    result = calibration_stability(np.array([0.0, 0.15, 0.3]), tau=0.15)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(np.exp(-1.0))
    assert result[0] > result[1] > result[2]


def _cfg(**overrides: float) -> ConfidenceConfig:
    base = {
        "traveler_evidence_k": 5.0,
        "poi_impression_k": 20.0,
        "review_count_k": 2.0,
        "ensemble_seeds": (1, 2, 3, 4, 5),
        "ensemble_std_tau": 0.5,
        "calibration_bin_width_tau": 0.15,
        "weight_traveler": 0.2,
        "weight_poi": 0.2,
        "weight_reviews": 0.2,
        "weight_ensemble": 0.2,
        "weight_calibration": 0.2,
    }
    base.update(overrides)
    return ConfidenceConfig(**base)  # type: ignore[arg-type]


def test_compute_confidence_all_max_evidence_near_one() -> None:
    """Very high evidence on every dimension, zero ensemble std, zero bin width ->
    confidence should be close to 1.0 (all 5 terms near 1.0)."""
    n = 3
    huge = np.full(n, 1_000_000.0)
    zero = np.zeros(n)
    result = compute_confidence(huge, huge, huge, zero, zero, _cfg())
    assert np.all(result > 0.95)


def test_compute_confidence_zero_evidence_and_high_disagreement_is_low() -> None:
    n = 2
    zero = np.zeros(n)
    high_std = np.full(n, 100.0)
    high_bin_width = np.full(n, 100.0)
    result = compute_confidence(zero, zero, zero, high_std, high_bin_width, _cfg())
    assert np.all(result < 0.05)


def test_compute_confidence_clipped_to_unit_interval() -> None:
    result = compute_confidence(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), _cfg()
    )
    assert 0.0 <= result[0] <= 1.0


def test_compute_confidence_hand_computed_single_row() -> None:
    """One row, hand-computed: traveler=5 (k=5 -> 0.5), poi=20 (k=20 -> 0.5),
    review log1p(e-1)=1 (k=2 -> 1/3), ensemble_std=0 (-> 1.0), bin_width=0 (-> 1.0).
    Equal weights 0.2 each."""
    review_count = np.array([np.e - 1.0])  # log1p == 1.0
    result = compute_confidence(
        np.array([5.0]),
        np.array([20.0]),
        review_count,
        np.array([0.0]),
        np.array([0.0]),
        _cfg(),
    )
    expected = 0.2 * (0.5 + 0.5 + (1.0 / 3.0) + 1.0 + 1.0)
    assert result[0] == pytest.approx(expected, rel=1e-6)
