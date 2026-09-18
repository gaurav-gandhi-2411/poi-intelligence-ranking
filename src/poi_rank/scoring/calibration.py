"""Isotonic calibration (spec.md section 9.2): raw LambdaMART scores are not
probabilities and are not comparable across travelers -- which breaks a downstream
planner that consumes them as *weights* (`planner_weight`, spec.md section 9.5). Fit
**isotonic regression** on a calibration split mapping raw score -> P(positive
engagement, i.e. `label >= 1`). Report ECE (15 bins), Brier score, and a reliability
plot.

**Calibration split, carved to never touch the holdout being reported on and never
overlap LightGBM's own early-stopping validation split**: `carve_calibration_split`
reuses `models.lambdamart.train_val_split_by_trip` (the SAME function, same by-trip-
group discipline) TWICE -- once (already done by `models.lambdamart` itself when
training) to get `fit_frame`/`val_frame` from the TRAIN ranking frame, then a SECOND
time on `fit_frame` to carve `calib_frame` out of it. `calib_frame`'s trips are
therefore disjoint from BOTH the LightGBM validation trips (both carved from the same
`fit_frame` after removing val trips) AND every holdout trip (`fit_frame` itself is
already TRAIN-only, by construction of `load_train_ranking_frame`). ECE/Brier are then
reported on the actual HOLDOUT evaluation frame -- never on `calib_frame` itself
(which would trivially flatter ECE by evaluating the isotonic fit on the exact rows it
was fit to) -- so calibration-fitting data and calibration-QUALITY-reporting data are
always disjoint, per spec.md section 9.2 / this phase's own instructions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/deterministic -- no display backend in CI or `uv run`
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from poi_rank.models.lambdamart import train_val_split_by_trip

FloatArray = npt.NDArray[np.float64]


def carve_calibration_split(
    fit_frame: pd.DataFrame, calibration_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(remaining_fit_frame, calib_frame)` -- `calib_frame` is a
    `calibration_fraction` slice of `fit_frame`'s OWN trips (module docstring),
    carved by `models.lambdamart.train_val_split_by_trip` reused directly (never
    reimplemented)."""
    remaining, calib = train_val_split_by_trip(fit_frame, calibration_fraction, seed)
    return remaining, calib


def fit_isotonic_calibrator(raw_scores: pd.Series, labels: pd.Series) -> IsotonicRegression:
    """Fit `IsotonicRegression` mapping raw LambdaMART score -> P(label >= 1)
    (spec.md section 9.2). `out_of_bounds="clip"` so a holdout raw score outside the
    calibration split's observed range gets the nearest endpoint's calibrated
    probability rather than extrapolating or raising."""
    y = (labels.to_numpy(dtype=np.int64) >= 1).astype(np.float64)
    x = raw_scores.to_numpy(dtype=np.float64)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(x, y)
    return ir


def apply_calibrator(ir: IsotonicRegression, raw_scores: pd.Series) -> pd.Series:
    """Apply a fitted calibrator to any raw-score series, returning a `[0, 1]`
    calibrated-probability `pd.Series` aligned to `raw_scores.index`."""
    calibrated = ir.predict(raw_scores.to_numpy(dtype=np.float64))
    return pd.Series(calibrated, index=raw_scores.index, name="preference_score")


def naive_probability_from_raw_score(raw_scores: pd.Series) -> pd.Series:
    """The "naive" (uncalibrated) baseline spec.md section 9.2 implicitly compares
    against: raw LambdaMART scores are unbounded reals (not `[0, 1]`), so treating
    them "naively as probabilities" requires SOME minimal transform just to make
    ECE/Brier computable at all -- min-max normalized over the SAME evaluation set
    (never fit against labels, unlike the isotonic calibrator), the simplest
    label-free rescaling. Documented, not asserted -- see docs/DATA_CARD.md."""
    values = raw_scores.to_numpy(dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return pd.Series(np.full(len(values), 0.5), index=raw_scores.index)
    return pd.Series((values - lo) / (hi - lo), index=raw_scores.index)


def calibration_bin_width(ir: IsotonicRegression, raw_scores: FloatArray) -> FloatArray:
    """Width (in raw-score units) of the isotonic step function's constant segment
    containing each raw score -- used by `scoring/confidence.py` as a
    calibration-stability signal (a narrow segment means the local calibrated
    estimate rests on a tightly-bounded range of training evidence; a wide one means
    it's an extrapolation-prone, sparsely-supported region). `ir.X_thresholds_` are
    the fitted isotonic function's knot x-values (sklearn >= 1.3), sorted
    ascending; fewer than 2 knots (degenerate fit) returns all-zero widths."""
    thresholds = ir.X_thresholds_
    if len(thresholds) < 2:
        return np.zeros(len(raw_scores), dtype=np.float64)
    idx = np.searchsorted(thresholds, raw_scores, side="right") - 1
    idx = np.clip(idx, 0, len(thresholds) - 2)
    result: FloatArray = thresholds[idx + 1] - thresholds[idx]
    return result


# -----------------------------------------------------------------------------------
# ECE / Brier / reliability diagram
# -----------------------------------------------------------------------------------


def expected_calibration_error(probs: FloatArray, labels: FloatArray, n_bins: int) -> float:
    """Standard equal-width-bin ECE: `sum_b (n_b / n) * |acc_b - conf_b|` (spec.md
    section 9.2: "ECE (15 bins)"). Empty bins contribute 0 (excluded from the sum,
    not treated as a 0-vs-0 perfect-calibration bin)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins[1:-1], right=True), 0, n_bins - 1)
    n = len(probs)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (float(mask.sum()) / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: FloatArray, labels: FloatArray) -> float:
    """Mean squared error between predicted probability and the binary outcome
    (spec.md section 9.2)."""
    return float(np.mean((probs - labels) ** 2))


def reliability_diagram(
    probs_before: FloatArray,
    probs_after: FloatArray,
    labels: FloatArray,
    n_bins: int,
    output_path: Path,
) -> None:
    """Save a before-vs-after-isotonic reliability diagram (spec.md section 9.2:
    "reliability plot") to `output_path` -- deterministic (no random jitter,
    `Agg` backend, fixed bin edges)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    def _bin_accuracies(probs: FloatArray) -> FloatArray:
        bin_idx = np.clip(np.digitize(probs, bins[1:-1], right=True), 0, n_bins - 1)
        acc = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                acc[b] = labels[mask].mean()
        return acc

    acc_before = _bin_accuracies(probs_before)
    acc_after = _bin_accuracies(probs_after)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(centers, acc_before, marker="o", label="raw (naive)", color="tab:red")
    ax.plot(centers, acc_after, marker="o", label="isotonic-calibrated", color="tab:blue")
    ax.set_xlabel("predicted P(positive engagement)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram: before vs after isotonic calibration")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
