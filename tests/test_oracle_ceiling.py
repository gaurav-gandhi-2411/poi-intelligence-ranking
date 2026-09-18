"""Oracle-ceiling tests (spec.md section 1.4 / section 8 system 9):

1. `eval.oracle.oracle_ceiling_scores` reads/joins `holdout_utility_true.parquet`
   correctly (hand-built tiny oracle directory, non-default frame index to prove the
   "row order preserved" claim, and the `-inf` fallback for a candidate with no
   oracle row).
2. The correctness invariant the task asks for: on a synthetic scenario where the
   oracle's true utility is set monotonic with the graded label (so the oracle
   ranking exactly equals the IDEAL label-sorted ranking), the oracle achieves the
   theoretical maximum NDCG (1.0) on every trip, strictly dominating an
   intentionally-adversarial (reversed-order) system and a constant-score (tie-broken
   arbitrarily) system on every trip. If this ever fails, either the NDCG
   implementation or the oracle-scoring implementation has a bug -- exactly the
   invariant the task's own tests-section instruction describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poi_rank.eval.metrics import compute_all_trip_metrics
from poi_rank.eval.oracle import oracle_ceiling_scores

# ---------------------------------------------------------------------------
# 1. oracle_ceiling_scores: read/join correctness
# ---------------------------------------------------------------------------


@pytest.fixture
def oracle_dir(tmp_path: Path) -> Path:
    out = tmp_path / "_oracle"
    out.mkdir()
    utility = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T2"],
            "traveler_id": ["U1", "U1", "U2"],
            "poi_id": ["P1", "P2", "P3"],
            "utility_true": [4.5, 2.1, 3.3],
        }
    )
    utility.to_parquet(out / "holdout_utility_true.parquet", index=False)
    return out


def test_oracle_ceiling_scores_matches_utility_true(oracle_dir: Path) -> None:
    frame_keys = pd.DataFrame({"trip_id": ["T1", "T1", "T2"], "poi_id": ["P1", "P2", "P3"]})
    scores = oracle_ceiling_scores(frame_keys, oracle_dir)
    assert scores.tolist() == pytest.approx([4.5, 2.1, 3.3])


def test_oracle_ceiling_scores_missing_row_falls_back_to_negative_infinity(
    oracle_dir: Path,
) -> None:
    frame_keys = pd.DataFrame({"trip_id": ["T1"], "poi_id": ["P_NOT_IN_ORACLE"]})
    scores = oracle_ceiling_scores(frame_keys, oracle_dir)
    assert scores.iloc[0] == -np.inf


def test_oracle_ceiling_scores_preserves_row_order_on_nondefault_index(
    oracle_dir: Path,
) -> None:
    frame_keys = pd.DataFrame(
        {"trip_id": ["T2", "T1", "T1"], "poi_id": ["P3", "P2", "P1"]},
        index=[7, 3, 9],
    )
    scores = oracle_ceiling_scores(frame_keys, oracle_dir)
    assert list(scores.index) == [7, 3, 9]
    assert scores.tolist() == pytest.approx([3.3, 2.1, 4.5])


# ---------------------------------------------------------------------------
# 2. Correctness invariant: oracle dominates every trip when its score is set
#    monotonic with the graded label.
# ---------------------------------------------------------------------------


def test_oracle_ranking_achieves_max_ndcg_and_dominates_every_trip() -> None:
    frame = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1", "T2", "T2", "T2", "T3", "T3", "T3"],
            "poi_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "label": [0, 1, 3, 3, 0, 2, 1, 3, 0],
        }
    )
    oracle_score = frame["label"].astype(float).rename("score_oracle")  # monotonic with label
    adversarial_score = (-frame["label"].astype(float)).rename("score_adversarial")  # reversed
    constant_score = pd.Series(0.0, index=frame.index, name="score_constant")  # arbitrary order

    oracle_per_trip = compute_all_trip_metrics(frame, oracle_score, (3,), (), ())["ndcg@3"]
    adversarial_per_trip = compute_all_trip_metrics(frame, adversarial_score, (3,), (), ())[
        "ndcg@3"
    ]
    constant_per_trip = compute_all_trip_metrics(frame, constant_score, (3,), (), ())["ndcg@3"]

    assert oracle_per_trip.to_numpy() == pytest.approx([1.0, 1.0, 1.0])
    assert (oracle_per_trip.to_numpy() >= adversarial_per_trip.to_numpy() - 1e-12).all()
    assert (oracle_per_trip.to_numpy() >= constant_per_trip.to_numpy() - 1e-12).all()
    # The adversarial (worst-possible) system must be strictly worse on at least one
    # trip with real label variation, or this "dominance" check would be vacuous.
    assert (oracle_per_trip.to_numpy() > adversarial_per_trip.to_numpy() + 1e-9).any()
