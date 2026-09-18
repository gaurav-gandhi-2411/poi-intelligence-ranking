"""`scoring/calibration.py` unit tests (spec.md section 9.2): ECE/Brier on small
hand-checkable examples, isotonic calibration improving ECE vs a naive baseline on a
synthetic monotone-but-miscalibrated score, and the calibration-split carving
discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from poi_rank.scoring.calibration import (
    brier_score,
    calibration_bin_width,
    carve_calibration_split,
    expected_calibration_error,
    fit_isotonic_calibrator,
    naive_probability_from_raw_score,
)

# -----------------------------------------------------------------------------------
# ECE / Brier: hand-checkable examples
# -----------------------------------------------------------------------------------


def test_ece_zero_for_perfectly_calibrated_bins() -> None:
    """2 bins, each perfectly calibrated: bin A predicts 0.2 and has 20% positive
    rate; bin B predicts 0.8 and has 80% positive rate -> ECE == 0."""
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
    labels = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=np.float64)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0)


def test_ece_hand_computed_single_bin_example() -> None:
    """`n_bins=1` puts every point in one bin (avoids edge-placement ambiguity from
    `np.digitize`'s bin-edge convention): mean predicted conf = 0.55, mean observed
    accuracy = 0.5 (2 of 4 positive) -> ECE == |0.5 - 0.55| == 0.05."""
    probs = np.array([0.5, 0.55, 0.55, 0.6])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    ece = expected_calibration_error(probs, labels, n_bins=1)
    assert ece == pytest.approx(0.05, abs=1e-9)


def test_brier_score_hand_computed() -> None:
    probs = np.array([1.0, 0.0, 0.5])
    labels = np.array([1.0, 1.0, 0.0])
    # (1-1)^2=0, (0-1)^2=1, (0.5-0)^2=0.25 -> mean = 1.25/3
    expected = (0.0 + 1.0 + 0.25) / 3.0
    assert brier_score(probs, labels) == pytest.approx(expected)


def test_brier_score_zero_for_perfect_predictions() -> None:
    probs = np.array([1.0, 0.0, 1.0])
    labels = np.array([1.0, 0.0, 1.0])
    assert brier_score(probs, labels) == pytest.approx(0.0)


# -----------------------------------------------------------------------------------
# Isotonic calibration improves ECE vs naive on a synthetic miscalibrated score
# -----------------------------------------------------------------------------------


def test_isotonic_calibration_improves_ece_on_synthetic_miscalibrated_scores() -> None:
    """Synthetic raw scores that are monotone in the true probability but
    systematically miscalibrated (a sigmoid-squashed, offset transform) -- isotonic
    regression fit on a calibration split should reduce ECE relative to the naive
    min-max-normalized-as-probability baseline, measured on a DISJOINT holdout
    slice (never the calibration-fit rows)."""
    rng = np.random.default_rng(7)
    n = 4000
    true_p = rng.uniform(0.02, 0.98, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.float64)
    # Raw score: monotone in true_p but badly miscalibrated -- exponential, not
    # linear, so a naive min-max-normalize-as-probability transform is far off the
    # true probability everywhere except the two endpoints, while isotonic
    # regression (which learns an arbitrary monotone mapping from the calibration
    # split) can recover it.
    raw = np.exp(4.0 * true_p) + rng.normal(0.0, 0.05, size=n)

    split = n // 2
    calib_raw, holdout_raw = raw[:split], raw[split:]
    calib_labels, holdout_labels = labels[:split], labels[split:]

    ir = fit_isotonic_calibrator(pd.Series(calib_raw), pd.Series(calib_labels.astype(np.int64)))
    calibrated_holdout = ir.predict(holdout_raw)
    naive_holdout = naive_probability_from_raw_score(pd.Series(holdout_raw)).to_numpy()

    ece_before = expected_calibration_error(naive_holdout, holdout_labels, n_bins=15)
    ece_after = expected_calibration_error(calibrated_holdout, holdout_labels, n_bins=15)

    assert ece_after < ece_before


# -----------------------------------------------------------------------------------
# naive_probability_from_raw_score
# -----------------------------------------------------------------------------------


def test_naive_probability_min_max_normalizes() -> None:
    raw = pd.Series([0.0, 5.0, 10.0])
    naive = naive_probability_from_raw_score(raw)
    np.testing.assert_allclose(naive.to_numpy(), [0.0, 0.5, 1.0])


def test_naive_probability_constant_input_returns_half() -> None:
    raw = pd.Series([3.0, 3.0, 3.0])
    naive = naive_probability_from_raw_score(raw)
    np.testing.assert_allclose(naive.to_numpy(), [0.5, 0.5, 0.5])


# -----------------------------------------------------------------------------------
# calibration_bin_width
# -----------------------------------------------------------------------------------


def test_calibration_bin_width_hand_checked() -> None:
    x = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 1.0])
    y = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(x, y)
    # ir.X_thresholds_ == [0.1, 0.3, 0.4, 0.5, 0.6, 1.0] (knots at the fit points)
    widths = calibration_bin_width(ir, np.array([0.15, 0.45, 0.99]))
    # 0.15 falls in [0.1, 0.3) -> width 0.2; 0.45 in [0.4, 0.5) -> width 0.1;
    # 0.99 in [0.6, 1.0) -> width 0.4.
    np.testing.assert_allclose(widths, [0.2, 0.1, 0.4], rtol=1e-6)


# -----------------------------------------------------------------------------------
# carve_calibration_split: disjoint from remaining fit trips
# -----------------------------------------------------------------------------------


def test_carve_calibration_split_is_disjoint_by_trip() -> None:
    trip_ids = [f"T{i:04d}" for i in range(40)]
    rows = []
    for tid in trip_ids:
        rows.extend([{"trip_id": tid, "poi_id": f"P{tid}{j}"} for j in range(3)])
    frame = pd.DataFrame(rows)

    remaining, calib = carve_calibration_split(frame, calibration_fraction=0.25, seed=11)

    remaining_trips = set(remaining["trip_id"])
    calib_trips = set(calib["trip_id"])
    assert remaining_trips.isdisjoint(calib_trips)
    assert remaining_trips | calib_trips == set(trip_ids)
    assert len(calib_trips) == pytest.approx(10, abs=1)  # ~25% of 40 trips
