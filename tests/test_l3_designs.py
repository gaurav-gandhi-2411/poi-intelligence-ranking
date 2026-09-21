"""Experiment L3b: the design-selection rule and the candidate-set-independent NDCG denominator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poi_rank.eval.l3_designs import (
    ADOPTION_BAR,
    candidate_recall,
    fixed_denominator_ndcg10,
    ideal_dcg_by_trip,
    select_design,
)


def _d(ndcg: float, lt: float = 0.9, k: float = 250.0) -> dict[str, float]:
    return {"v_ndcg10": ndcg, "long_tail_recall": lt, "effective_k": k}


def test_incumbent_stays_unless_the_winner_clears_the_adoption_bar() -> None:
    below = select_design({"A": _d(0.20), "B": _d(0.20 + ADOPTION_BAR - 1e-4), "C": _d(0.199)})
    assert below["adopted"] == "A"
    above = select_design({"A": _d(0.20), "B": _d(0.20 + ADOPTION_BAR + 1e-4), "C": _d(0.199)})
    assert above["adopted"] == "B"


def test_constraints_exclude_designs_before_the_bar_is_applied() -> None:
    # B has the best NDCG but fails long-tail recall; C fails the serving budget.
    picked = select_design({"A": _d(0.20), "B": _d(0.30, lt=0.74), "C": _d(0.29, k=300.5)})
    assert picked["adopted"] == "A"


def test_an_infeasible_incumbent_is_replaced_without_the_bar() -> None:
    picked = select_design({"A": _d(0.30, lt=0.5), "B": _d(0.10), "C": _d(0.11, k=400.0)})
    assert picked["adopted"] == "B"


def test_ndcg_denominator_is_independent_of_the_candidate_set() -> None:
    exposed = pd.DataFrame(
        {
            "trip_id": ["t"] * 3,
            "poi_id": ["p1", "p2", "p3"],
            "label": [3, 2, 1],
            "p_expose": [0.5, 0.5, 0.5],
        }
    )
    ideal = ideal_dcg_by_trip(exposed)
    full = exposed.copy()
    missing_best = exposed.iloc[1:].copy()  # a design that never retrieves the best POI
    p = pd.Series([0.5] * 3, index=full.index)
    score = np.array([3.0, 2.0, 1.0])
    ndcg_full = fixed_denominator_ndcg10(full, score, p, ideal)
    ndcg_missing = fixed_denominator_ndcg10(
        missing_best, score[1:], p.iloc[1:].reset_index(drop=True), ideal
    )
    assert ndcg_full == pytest.approx(1.0)
    assert ndcg_missing < ndcg_full  # the missed positive costs DCG instead of vanishing


def test_candidate_recall_is_ips_weighted_and_reports_chance() -> None:
    exposed = pd.DataFrame(
        {
            "trip_id": ["t", "t"],
            "poi_id": ["hit", "miss"],
            "label": [2, 2],
            "p_expose": [0.5, 0.1],  # the rarely exposed miss carries five times the weight
        }
    )
    out = candidate_recall(
        {"t": {"hit", "x"}},
        exposed,
        {"t"},
        {"hit": 0.1, "miss": 0.2, "x": 0.3},
        0.5,
        {"d": {"hit", "miss", "x", "y"}},
        {"t": "d"},
    )
    assert out["overall_recall"] == pytest.approx(2.0 / (2.0 + 10.0))
    assert out["overall_chance"] == pytest.approx(0.5)
    assert out["overall_lift"] == pytest.approx(out["overall_recall"] - 0.5)
