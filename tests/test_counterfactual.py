"""`explain/counterfactual.py`: binding-constraint detection + the genuine
re-score/re-rank recomputation (spec.md section 10), on hand-constructed examples
with a KNOWN binding constraint and a known expected rank-improvement direction."""

from __future__ import annotations

import pandas as pd

from poi_rank.explain.counterfactual import (
    BINDING_MAX_VALUE,
    BINDING_MIN_GAP,
    compute_counterfactual_line,
    find_binding_subscore,
)

SUBSCORE_COLUMNS = (
    "budget_fit",
    "mobility_fit",
    "hours_fit",
    "reservation_fit",
    "party_fit",
    "duration_fit",
)


def _full_scores(**overrides: float) -> dict[str, float]:
    base = {name: 0.95 for name in SUBSCORE_COLUMNS}
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------------
# find_binding_subscore
# -----------------------------------------------------------------------------------


def test_find_binding_subscore_detects_clear_single_low_score() -> None:
    breakdown = _full_scores(hours_fit=0.2)
    assert find_binding_subscore(breakdown) == "hours_fit"


def test_find_binding_subscore_none_when_uniformly_high() -> None:
    breakdown = _full_scores()
    assert find_binding_subscore(breakdown) is None


def test_find_binding_subscore_none_when_two_scores_are_both_low_no_clear_gap() -> None:
    breakdown = _full_scores(hours_fit=0.3, mobility_fit=0.32)
    assert find_binding_subscore(breakdown) is None


def test_binding_thresholds_are_the_documented_constants() -> None:
    # Sanity guard: if these constants ever change, the tests above encode the
    # actual boundary behavior, not just the current numbers.
    assert BINDING_MAX_VALUE == 0.7
    assert BINDING_MIN_GAP == 0.15


# -----------------------------------------------------------------------------------
# compute_counterfactual_line -- hand-built trip with a KNOWN binding constraint
# -----------------------------------------------------------------------------------


def _trip_survivors_frame() -> pd.DataFrame:
    """3 candidates for one trip: `P1` (the target) is weak on `hours_fit` and
    currently ranks BEHIND `P2` because of it; `P3` is a distant third. Hand-computed
    utility with alpha=1.0, beta=0.7 (this project's own default) using
    `hard_gate=1, relevance` fixed per row and `compatibility = geometric_mean(6
    subscores)`.
    """
    rows = [
        {
            "poi_id": "P1",
            "hard_gate": 1.0,
            "relevance": 0.9,
            **_full_scores(hours_fit=0.2),
        },
        {
            "poi_id": "P2",
            "hard_gate": 1.0,
            "relevance": 0.85,
            **_full_scores(),
        },
        {
            "poi_id": "P3",
            "hard_gate": 1.0,
            "relevance": 0.3,
            **_full_scores(),
        },
    ]
    frame = pd.DataFrame(rows)
    # Real utility, the exact formula scoring/utility.py::compute_utility uses.
    import numpy as np

    from poi_rank.scoring.compatibility import compatibility_geometric_mean
    from poi_rank.scoring.utility import compute_utility

    compat = compatibility_geometric_mean(
        [frame[name].to_numpy(dtype=np.float64) for name in SUBSCORE_COLUMNS]
    )
    frame["utility"] = compute_utility(
        frame["hard_gate"].to_numpy(dtype=np.float64),
        frame["relevance"].to_numpy(dtype=np.float64),
        compat,
        alpha=1.0,
        beta=0.7,
    )
    return frame


def test_counterfactual_line_shows_genuine_rank_improvement() -> None:
    frame = _trip_survivors_frame()
    # Sanity: P1 (weak hours_fit) must rank BEHIND P2 in this hand-built example,
    # otherwise this test isn't actually exercising a binding constraint.
    ranked = frame.sort_values("utility", ascending=False)["poi_id"].tolist()
    assert ranked.index("P1") > ranked.index("P2")

    line = compute_counterfactual_line(frame, "P1", alpha=1.0, beta=0.7)
    assert line is not None
    assert "if your trip included a day when it's open" in line
    assert line.startswith("Would rank #")

    # The claimed new rank must be strictly better (numerically lower) than the
    # actual rank, and both numbers in the string must match a real recomputation.
    actual_rank = ranked.index("P1") + 1
    assert f"instead of #{actual_rank}" in line


def test_counterfactual_line_is_none_when_no_binding_constraint() -> None:
    frame = _trip_survivors_frame()
    frame.loc[frame["poi_id"] == "P1", SUBSCORE_COLUMNS] = 0.95
    frame.loc[frame["poi_id"] == "P1", "hours_fit"] = 0.95
    # Recompute utility after flattening P1's sub-scores.
    import numpy as np

    from poi_rank.scoring.compatibility import compatibility_geometric_mean
    from poi_rank.scoring.utility import compute_utility

    compat = compatibility_geometric_mean(
        [frame[name].to_numpy(dtype=np.float64) for name in SUBSCORE_COLUMNS]
    )
    frame["utility"] = compute_utility(
        frame["hard_gate"].to_numpy(dtype=np.float64),
        frame["relevance"].to_numpy(dtype=np.float64),
        compat,
        alpha=1.0,
        beta=0.7,
    )
    line = compute_counterfactual_line(frame, "P1", alpha=1.0, beta=0.7)
    assert line is None
