"""Counterfactual explanation line (spec.md section 10): *"Would rank #3 instead of
#11 if your trip included a weekday"* -- generalized to any of the 6 compatibility
sub-scores when it is the clear BINDING constraint on a recommended POI's utility.

**A genuine recomputation, not a scripted-sounding fake number**: `compute_utility`
and `compatibility_geometric_mean` are the EXACT same functions
`scoring/output.py::run_scoring_pipeline` used to produce the real utility/rank in
the first place (`poi_rank.scoring.utility`/`poi_rank.scoring.compatibility`,
legitimately importable from `explain/` per this package's firewall). The
counterfactual sets exactly ONE sub-score to 1.0 (perfect), recomputes that one
POI's `compatibility`/`utility`, and re-derives its rank within the SAME trip's full
survivors pool (every other candidate's utility held fixed) -- a real re-rank, not an
assumption about what the new rank "should" be.

**Binding-constraint detection**: a sub-score is BINDING only if it is (a) clearly
the minimum of the 6 (at least `BINDING_MIN_GAP` below the second-lowest) AND (b)
itself below `BINDING_MAX_VALUE` -- both judgment-call thresholds (spec.md gives no
exact numbers), documented in docs/DATA_CARD.md, not spec-mandated. If no sub-score
clears both bars, OR the resulting counterfactual re-rank does not actually improve
(can happen: the "binding" sub-score heuristic can be wrong for a POI where the
other 5 sub-scores are already so strong that removing this one drag still doesn't
move it up), no counterfactual line is produced for that recommendation --
`explain/output_enrichment.py` treats `None` here as "omit," never fabricates a line
that doesn't hold up under the real recomputation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.scoring import compatibility as compat
from poi_rank.scoring.utility import compute_utility

FloatArray = npt.NDArray[np.float64]

BINDING_MIN_GAP = 0.15  # judgment call, docs/DATA_CARD.md
BINDING_MAX_VALUE = 0.7  # judgment call, docs/DATA_CARD.md

SUBSCORE_COUNTERFACTUAL_CLAUSE: dict[str, str] = {
    "budget_fit": "if it fit your budget exactly",
    "mobility_fit": "if it were closer to your stay",
    "hours_fit": "if your trip included a day when it's open during your preferred hours",
    "reservation_fit": "if you booked further in advance",
    "party_fit": "if it better accommodated your travel party",
    "duration_fit": "if it fit better within your day's pace",
}


def find_binding_subscore(compatibility_breakdown: dict[str, float]) -> str | None:
    """The name of the sub-score that is a CLEAR binding constraint (module
    docstring), or `None` if compatibility is already uniformly high (no sub-score
    stands out as the dominant drag)."""
    ordered = sorted(compatibility_breakdown.items(), key=lambda kv: kv[1])
    (lowest_name, lowest_value), (_, second_value) = ordered[0], ordered[1]
    if lowest_value <= BINDING_MAX_VALUE and (second_value - lowest_value) >= BINDING_MIN_GAP:
        return lowest_name
    return None


def _rank_within_trip(
    poi_ids: npt.NDArray[np.object_], utilities: FloatArray, target_poi_id: str
) -> int:
    """1-indexed rank of `target_poi_id` when the trip's survivors pool is sorted by
    utility descending, tie-broken by `poi_id` ascending (same deterministic
    convention `assemble_output_payload`'s MMR-rank ordering already relies on
    upstream)."""
    pairs = zip(poi_ids.tolist(), utilities.tolist(), strict=True)
    order = sorted(pairs, key=lambda pu: (-pu[1], pu[0]))
    for position, (poi_id, _utility) in enumerate(order, start=1):
        if poi_id == target_poi_id:
            return position
    raise ValueError(f"poi_id {target_poi_id!r} not found in its own trip's survivors pool")


def compute_counterfactual_line(
    trip_survivors: pd.DataFrame,
    poi_id: str,
    alpha: float,
    beta: float,
) -> str | None:
    """`trip_survivors`: every `hard_gate == 1` candidate for ONE trip, carrying
    `poi_id`, `hard_gate`, `relevance`, `utility`, and all 6
    `compat.COMPATIBILITY_SUB_SCORE_NAMES` columns (the exact frame
    `scoring/output.py::run_scoring_pipeline` already builds as `full`, filtered to
    one trip and to survivors). Returns the rendered counterfactual line, or `None`
    if no genuine, improving binding constraint exists (module docstring)."""
    row = trip_survivors.loc[trip_survivors["poi_id"] == poi_id].iloc[0]
    breakdown = {name: float(row[name]) for name in compat.COMPATIBILITY_SUB_SCORE_NAMES}

    binding = find_binding_subscore(breakdown)
    if binding is None:
        return None

    poi_ids = trip_survivors["poi_id"].to_numpy(dtype=object)
    utilities = trip_survivors["utility"].to_numpy(dtype=np.float64).copy()
    actual_rank = _rank_within_trip(poi_ids, utilities, poi_id)

    modified = dict(breakdown)
    modified[binding] = 1.0
    new_compatibility = float(
        compat.compatibility_geometric_mean(
            [np.array([modified[name]]) for name in compat.COMPATIBILITY_SUB_SCORE_NAMES]
        )[0]
    )
    new_utility = float(
        compute_utility(
            np.array([float(row["hard_gate"])]),
            np.array([float(row["relevance"])]),
            np.array([new_compatibility]),
            alpha,
            beta,
        )[0]
    )

    idx = int(np.where(poi_ids == poi_id)[0][0])
    utilities[idx] = new_utility
    new_rank = _rank_within_trip(poi_ids, utilities, poi_id)

    if new_rank >= actual_rank:
        # The binding-heuristic's pick didn't actually move the needle once
        # recomputed for real -- honest handling per module docstring, not forced.
        return None

    clause = SUBSCORE_COUNTERFACTUAL_CLAUSE[binding]
    return f"Would rank #{new_rank} instead of #{actual_rank} {clause}"
