"""Candidate-recall metric tests (spec.md sections 7 / 11.10): hand-constructed
examples with a known correct answer -- verified before trusting the real-data
numbers `poi_rank.cli candidates` reports.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poi_rank.candidates.channels import CHANNEL_NAMES
from poi_rank.candidates.recall_metrics import (
    RecallResult,
    candidate_recall_at_k,
    marginal_recall_per_channel,
    overall_and_longtail_recall,
)

# ---------------------------------------------------------------------------
# candidate_recall_at_k: pure sanity checks on hand-built dicts
# ---------------------------------------------------------------------------


def test_candidate_recall_at_k_full_hit_is_one() -> None:
    relevant = {"T1": {"P1", "P2"}}
    candidates = {"T1": {"P1", "P2", "P3"}}
    result = candidate_recall_at_k(relevant, candidates)
    assert result.recall_mean == pytest.approx(1.0)
    assert result.n_trips_evaluated == 1
    assert result.n_trips_excluded_no_relevant == 0


def test_candidate_recall_at_k_partial_hit() -> None:
    relevant = {"T1": {"P1", "P2"}}
    candidates = {"T1": {"P1"}}
    result = candidate_recall_at_k(relevant, candidates)
    assert result.recall_mean == pytest.approx(0.5)


def test_candidate_recall_at_k_zero_hit_is_zero() -> None:
    relevant = {"T1": {"P1"}}
    candidates = {"T1": {"P9"}}
    result = candidate_recall_at_k(relevant, candidates)
    assert result.recall_mean == pytest.approx(0.0)


def test_candidate_recall_at_k_excludes_trips_with_no_relevant_pois() -> None:
    """A trip with zero holdout-positive POIs has no ground truth to check
    against -- excluded from the mean, but counted, never silently dropped."""
    relevant = {"T1": {"P1"}, "T2": set()}
    candidates = {"T1": {"P1"}, "T2": {"P9"}}
    result = candidate_recall_at_k(relevant, candidates)
    assert result.recall_mean == pytest.approx(1.0)  # only T1 contributes
    assert result.n_trips_evaluated == 1
    assert result.n_trips_excluded_no_relevant == 1


def test_candidate_recall_at_k_missing_trip_in_candidates_is_zero_recall() -> None:
    relevant = {"T1": {"P1"}}
    candidates: dict[str, set[str]] = {}
    result = candidate_recall_at_k(relevant, candidates)
    assert result.recall_mean == pytest.approx(0.0)


def test_recall_result_is_frozen_dataclass() -> None:
    r = RecallResult(recall_mean=0.5, n_trips_evaluated=2, n_trips_excluded_no_relevant=0)
    assert r.recall_mean == 0.5


# ---------------------------------------------------------------------------
# overall_and_longtail_recall / marginal_recall_per_channel: a small, fully
# hand-verified scenario -- includes a deliberately-included (via merged_poi_ids
# canonical remap) and a deliberately-excluded holdout-positive POI per trip.
# ---------------------------------------------------------------------------


@pytest.fixture
def small_scenario() -> dict[str, pd.DataFrame]:
    # P1 was merged away from a raw duplicate "P1_DUP" during Phase 2 dedup --
    # `interactions_holdout_random` below references the RAW id, exactly like the
    # real committed dataset (docs/DATA_CARD.md resolved ambiguity #18).
    pois = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3"],
            "pop_pct": [0.1, 0.6, 0.2],  # P1, P3 are long-tail (< 0.5); P2 is not
            "merged_poi_ids": [["P1", "P1_DUP"], ["P2"], ["P3"]],
        }
    )

    # Trip TA: candidates = {P1 (geo only), P2 (interest only)}.
    # Trip TB: candidates = {P2 (semantic only)}.
    def _row(trip_id: str, poi_id: str, **flags: bool) -> dict[str, object]:
        record: dict[str, object] = {"trip_id": trip_id, "poi_id": poi_id}
        for ch in CHANNEL_NAMES:
            record[ch] = flags.get(ch, False)
        return record

    candidates = pd.DataFrame(
        [
            _row("TA", "P1", channel_geo=True),
            _row("TA", "P2", channel_interest=True),
            _row("TB", "P2", channel_semantic=True),
        ]
    )

    # TA's holdout-positive POIs: "P1_DUP" (remaps to canonical P1, deliberately
    # INCLUDED in TA's candidate set) and "P3" (deliberately EXCLUDED -- never a
    # TA candidate). TB's holdout-positive POI: "P2" (not long-tail).
    holdout_random = pd.DataFrame(
        {
            "trip_id": ["TA", "TA", "TB"],
            "poi_id": ["P1_DUP", "P3", "P2"],
            "label": [3, 2, 1],
        }
    )
    return {"pois": pois, "candidates": candidates, "holdout_random": holdout_random}


def test_overall_and_longtail_recall_hand_verified(small_scenario: dict[str, pd.DataFrame]) -> None:
    result = overall_and_longtail_recall(
        small_scenario["pois"],
        small_scenario["candidates"],
        small_scenario["holdout_random"],
        long_tail_cutoff=0.5,
    )

    # TA: relevant={P1,P3} (post-remap), candidate={P1,P2} -> hit=1/2=0.5.
    # TB: relevant={P2}, candidate={P2} -> hit=1/1=1.0.
    # overall mean = (0.5 + 1.0) / 2 = 0.75.
    assert result["overall"].recall_mean == pytest.approx(0.75)
    assert result["overall"].n_trips_evaluated == 2

    # Long-tail stratum (pop_pct < 0.5): TA's relevant restricted to {P1,P3} (both
    # qualify) -> recall 0.5. TB's relevant {P2} does NOT qualify (pop_pct=0.6) ->
    # TB excluded entirely from the long-tail mean.
    assert result["long_tail"].recall_mean == pytest.approx(0.5)
    assert result["long_tail"].n_trips_evaluated == 1
    assert result["long_tail"].n_trips_excluded_no_relevant == 1


def test_marginal_recall_per_channel_hand_verified(small_scenario: dict[str, pd.DataFrame]) -> None:
    marginal = marginal_recall_per_channel(
        small_scenario["pois"], small_scenario["candidates"], small_scenario["holdout_random"]
    )

    # channel_geo is the ONLY source of P1 in TA -- removing it drops TA's recall
    # from 0.5 to 0.0 (TB unaffected) -> mean drops from 0.75 to 0.5 -> marginal=0.25.
    assert marginal["channel_geo"]["marginal_recall"] == pytest.approx(0.25)

    # channel_interest contributes P2 to TA, but P2 was never TA's relevant POI --
    # removing it changes nothing: a genuine, honestly-measured ~0-marginal-recall
    # channel in this toy scenario (spec.md section 7's "gets deleted" case).
    assert marginal["channel_interest"]["marginal_recall"] == pytest.approx(0.0)

    # channel_semantic is the ONLY source of P2 in TB -- removing it drops TB's
    # candidate set to empty (recall 0.0) -> mean drops from 0.75 to 0.25 ->
    # marginal = 0.5.
    assert marginal["channel_semantic"]["marginal_recall"] == pytest.approx(0.5)

    # Channels never used anywhere in this toy scenario contribute exactly zero.
    for ch in ("channel_cf", "channel_longtail", "channel_archetype"):
        assert marginal[ch]["marginal_recall"] == pytest.approx(0.0)

    for ch in CHANNEL_NAMES:
        assert marginal[ch]["recall_full"] == pytest.approx(0.75)
