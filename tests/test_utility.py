"""`scoring/utility.py` unit tests (spec.md section 9.1): the multiplicative utility
formula on hand-computed examples, plus the beta-sensitivity table's shape/behavior
on a small synthetic ranking frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poi_rank.scoring.config import UtilityConfig
from poi_rank.scoring.utility import beta_sensitivity_table, compute_utility


def test_compute_utility_hand_computed() -> None:
    hard_gate = np.array([1.0, 1.0, 0.0])
    relevance = np.array([0.5, 0.8, 0.9])
    compatibility = np.array([0.4, 1.0, 1.0])
    result = compute_utility(hard_gate, relevance, compatibility, alpha=1.0, beta=0.7)
    expected = np.array(
        [
            1.0 * (0.5**1.0) * (0.4**0.7),
            1.0 * (0.8**1.0) * (1.0**0.7),
            0.0,  # hard_gate == 0 -> utility forced to exactly 0 regardless of the rest
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_compute_utility_zero_relevance_or_compatibility_zeroes_utility() -> None:
    hard_gate = np.array([1.0, 1.0])
    relevance = np.array([0.0, 0.9])
    compatibility = np.array([0.9, 0.0])
    result = compute_utility(hard_gate, relevance, compatibility, alpha=1.0, beta=0.7)
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_compute_utility_alpha_beta_exponents_applied_independently() -> None:
    hard_gate = np.array([1.0])
    relevance = np.array([0.25])
    compatibility = np.array([0.25])
    result = compute_utility(hard_gate, relevance, compatibility, alpha=2.0, beta=0.5)
    assert result[0] == pytest.approx(0.25**2.0 * 0.25**0.5)


def _small_frame() -> pd.DataFrame:
    # 2 trips x 3 candidates. Trip A: one strong relevant/compatible POI + 2 weak.
    # Trip B: all label 0 (no relevant POI -> undefined NDCG, excluded).
    return pd.DataFrame(
        {
            "trip_id": ["A", "A", "A", "B", "B", "B"],
            "poi_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "label": [3, 0, 0, 0, 0, 0],
        }
    )


def test_beta_sensitivity_table_shape_and_exclusion_count() -> None:
    frame = _small_frame()
    hard_gate = np.ones(6)
    relevance = np.array([0.9, 0.2, 0.1, 0.5, 0.5, 0.5])
    compatibility = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    cfg = UtilityConfig(alpha=1.0, beta=0.7, beta_sweep=(0.3, 0.7, 1.1))

    rows = beta_sensitivity_table(frame, hard_gate, relevance, compatibility, cfg, k=10)

    assert [r.beta for r in rows] == [0.3, 0.7, 1.1]
    for row in rows:
        # Trip A has a relevant POI (defined NDCG); trip B has none (excluded).
        assert row.n_trips_included == 1
        assert row.n_trips_excluded == 1


def test_beta_sensitivity_table_perfect_ranking_gives_ndcg_one() -> None:
    """Trip A's highest-relevance/compatibility POI (P1) IS the relevant one
    (label=3) -- for ANY beta, utility ranks P1 first, giving NDCG@10 == 1.0 for
    that trip."""
    frame = _small_frame()
    hard_gate = np.ones(6)
    relevance = np.array([0.99, 0.01, 0.01, 0.5, 0.5, 0.5])
    compatibility = np.array([0.99, 0.01, 0.01, 0.5, 0.5, 0.5])
    cfg = UtilityConfig(alpha=1.0, beta=0.7, beta_sweep=(0.3, 0.7, 1.1))

    rows = beta_sensitivity_table(frame, hard_gate, relevance, compatibility, cfg, k=10)
    for row in rows:
        assert row.ndcg_at_10_mean == pytest.approx(1.0)
