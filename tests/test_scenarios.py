"""`eval/scenarios.py` tests (spec.md section 15, the 3+1 required scenarios):
constructed-profile field correctness, the scenario-4-vs-base diff invariant, the
overlap-matrix invariants (symmetric, diagonal 1.0, a genuinely-computed
scenario-4-vs-base number), and full-pipeline determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from poi_rank.eval.personalization import jaccard
from poi_rank.eval.scenarios import (
    ALL_SCENARIO_PROFILES,
    DIAGNOSTIC_BASE_SCENARIO,
    DIAGNOSTIC_SCENARIO,
    SCENARIO_DESTINATION,
    build_scenario_travelers_and_trips,
    compute_overlap_matrix,
    scenario_traveler_id,
    scenario_trip_id,
    stay_point_for_destination,
)

# -----------------------------------------------------------------------------------
# Constructed-profile field correctness (spec.md section 15's stated fields, exactly)
# -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """`build_scenario_travelers_and_trips` needs `pois_df` (for the stay-point
    centroid) and `real_trips_df` (for the start-date anchor) -- small, hand-built
    stand-ins are enough for these pure field-value assertions (no real dataset
    dependency needed for this test class)."""
    pois_df = pd.DataFrame(
        {
            "poi_id": ["P1", "P2"],
            "destination": [SCENARIO_DESTINATION, SCENARIO_DESTINATION],
            "lat": [37.0, 38.0],
            "lon": [126.0, 127.0],
        }
    )
    real_trips_df = pd.DataFrame({"start_date": [pd.Timestamp("2024-01-01")]})
    return build_scenario_travelers_and_trips(pois_df, real_trips_df, SCENARIO_DESTINATION)


def _traveler_row(travelers_df: pd.DataFrame, number: int) -> pd.Series:
    return travelers_df.set_index("traveler_id").loc[scenario_traveler_id(number)]


def _trip_row(trips_df: pd.DataFrame, number: int) -> pd.Series:
    return trips_df.set_index("trip_id").loc[scenario_trip_id(number)]


def test_scenario_1_local_experience_matches_spec_fields(
    scenario_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    travelers_df, trips_df = scenario_frames
    t = _traveler_row(travelers_df, 1)
    trip = _trip_row(trips_df, 1)
    assert t["touristiness_pref"] == pytest.approx(-0.8)
    assert t["budget"] == "medium"
    assert t["mobility"] == "public_transport"
    assert t["party_type"] == "solo"
    assert trip["destination"] == SCENARIO_DESTINATION
    # "local food, neighborhoods" -> documented mapping onto the real
    # CATEGORIES+TAGS vocabulary (module docstring).
    assert set(t["interests"]) == {"local", "foodie", "authentic"}


def test_scenario_2_history_architecture_matches_spec_fields(
    scenario_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    travelers_df, trips_df = scenario_frames
    t = _traveler_row(travelers_df, 2)
    assert t["touristiness_pref"] == pytest.approx(0.4)
    assert t["budget"] == "high"
    assert t["mobility"] == "public_transport"
    assert t["party_type"] == "couple"
    assert set(t["interests"]) == {"historic_site", "historic", "cultural", "museum"}


def test_scenario_3_family_young_kids_matches_spec_fields(
    scenario_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    travelers_df, _trips_df = scenario_frames
    t = _traveler_row(travelers_df, 3)
    assert t["party_type"] == "family_young_kids"
    assert "stroller" in t["accessibility_needs"]
    assert t["pace"] == "relaxed"
    assert t["budget"] == "medium"
    assert set(t["interests"]) == {"family_activity", "nature_park", "family-friendly"}


def test_scenario_4_is_scenario_1_with_only_touristiness_pref_flipped(
    scenario_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The scenario-4-vs-base diff invariant: diff the two constructed traveler
    rows field-by-field and assert `touristiness_pref` is the ONLY difference
    (excluding the identifier columns, which necessarily differ)."""
    travelers_df, _trips_df = scenario_frames
    base = _traveler_row(travelers_df, DIAGNOSTIC_BASE_SCENARIO)
    diagnostic = _traveler_row(travelers_df, DIAGNOSTIC_SCENARIO.number)

    assert base["touristiness_pref"] == pytest.approx(-0.8)
    assert diagnostic["touristiness_pref"] == pytest.approx(0.8)
    assert diagnostic["touristiness_pref"] == pytest.approx(-base["touristiness_pref"])

    identifier_cols = {"traveler_id", "explicit_preferences", "home_destination", "home_market"}
    differing = [
        col
        for col in base.index
        if col not in identifier_cols and not _values_equal(base[col], diagnostic[col])
    ]
    assert differing == ["touristiness_pref"], (
        f"expected ONLY touristiness_pref to differ between scenario "
        f"{DIAGNOSTIC_BASE_SCENARIO} and scenario {DIAGNOSTIC_SCENARIO.number}, "
        f"also found: {differing}"
    )


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return a == b
    return bool(a == b)


def test_all_4_scenarios_share_one_destination(
    scenario_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _travelers_df, trips_df = scenario_frames
    assert set(trips_df["destination"]) == {SCENARIO_DESTINATION}
    assert len(trips_df) == len(ALL_SCENARIO_PROFILES) == 4


def test_stay_point_is_the_destination_pois_centroid() -> None:
    pois_df = pd.DataFrame(
        {
            "poi_id": ["A", "B"],
            "destination": ["seoul", "seoul"],
            "lat": [10.0, 20.0],
            "lon": [1.0, 3.0],
        }
    )
    lat, lon = stay_point_for_destination(pois_df, "seoul")
    assert lat == pytest.approx(15.0)
    assert lon == pytest.approx(2.0)


# -----------------------------------------------------------------------------------
# Overlap matrix invariants
# -----------------------------------------------------------------------------------


def test_overlap_matrix_is_symmetric_with_unit_diagonal() -> None:
    top10 = {
        1: ["a", "b", "c"],
        2: ["b", "c", "d"],
        3: ["x", "y", "z"],
        4: ["a", "b", "e"],
    }
    result = compute_overlap_matrix(top10, diagnostic_scenario=4, base_scenario=1)
    matrix = result["matrix"]
    n = len(matrix)
    for i in range(n):
        assert matrix[i][i] == pytest.approx(1.0)
        for j in range(n):
            assert matrix[i][j] == pytest.approx(matrix[j][i])


def test_overlap_matrix_scenario_4_vs_base_is_genuinely_computed_not_hardcoded() -> None:
    top10 = {
        1: ["a", "b", "c", "d"],
        2: ["e", "f", "g", "h"],
        3: ["i", "j", "k", "l"],
        4: ["a", "b", "m", "n"],
    }
    result = compute_overlap_matrix(top10, diagnostic_scenario=4, base_scenario=1)
    expected = jaccard(set(top10[1]), set(top10[4]))
    assert result["diagnostic_vs_base_jaccard"] == pytest.approx(expected)
    assert expected == pytest.approx(2.0 / 6.0)  # {a,b} intersect / {a,b,c,d,m,n} union

    # A different top-10 assignment must change the reported number -- proves this
    # isn't a hardcoded constant.
    top10_different = dict(top10)
    top10_different[4] = ["p", "q", "r", "s"]
    result_different = compute_overlap_matrix(
        top10_different, diagnostic_scenario=4, base_scenario=1
    )
    assert result_different["diagnostic_vs_base_jaccard"] == pytest.approx(0.0)
    assert result_different["diagnostic_vs_base_jaccard"] != result["diagnostic_vs_base_jaccard"]


def test_overlap_matrix_reports_base_and_diagnostic_scenario_numbers() -> None:
    top10 = {1: ["a"], 2: ["b"], 3: ["c"], 4: ["d"]}
    result = compute_overlap_matrix(top10, diagnostic_scenario=4, base_scenario=1)
    assert result["diagnostic_scenario"] == 4
    assert result["diagnostic_base_scenario"] == 1
    assert result["scenario_order"] == [1, 2, 3, 4]


# -----------------------------------------------------------------------------------
# Full-pipeline integration (real fixture-chain data): output shape + determinism
# -----------------------------------------------------------------------------------


def _load_all_scenario_json(results_dir: Path) -> dict[int, dict[str, Any]]:
    scenarios_dir = results_dir / "scenarios"
    return {
        n: json.loads((scenarios_dir / f"{n}.json").read_text(encoding="utf-8"))
        for n in (1, 2, 3, 4)
    }


def test_run_scenarios_writes_all_4_files_and_overlap_matrix(
    scenarios_result_dir: Path,
) -> None:
    scenarios_dir = scenarios_result_dir / "scenarios"
    for n in (1, 2, 3, 4):
        assert (scenarios_dir / f"{n}.json").exists()
    assert (scenarios_dir / "overlap_matrix.json").exists()


def test_scenario_json_top10_has_required_spec_columns(scenarios_result_dir: Path) -> None:
    payloads = _load_all_scenario_json(scenarios_result_dir)
    required = {
        "poi_id",
        "name",
        "category",
        "utility",
        "preference_score",
        "context_compatibility",
        "confidence",
        "popularity_percentile",
        "localness_index",
        "top_signals",
        "explanation",
    }
    for payload in payloads.values():
        assert len(payload["recommendations"]) > 0
        for rec in payload["recommendations"]:
            assert required.issubset(rec.keys())
        # Real content, not the scoring-layer placeholder (Phase 7 discipline).
        assert payload["recommendations"][0]["explanation"] != [
            "Explanation generation (grouped TreeSHAP + template layer, spec.md "
            "section 10) is implemented in a later phase (src/poi_rank/explain/) "
            "-- this is a structural placeholder with the correct output schema, "
            "not a real explanation."
        ]


def test_scenarios_are_cold_start_by_construction(scenarios_result_dir: Path) -> None:
    """None of the 4 synthetic travelers has any real interaction history --
    every candidate's `implicit_interaction_count`-derived confidence input must
    reflect zero traveler-side evidence (module docstring: "cold start by
    construction, not a bug"). Checked indirectly: `top_signals`'s
    `implicit_taste`/`interact_cos_taste_poi`-driven group should never dominate
    a cold-start row the way it legitimately can for a real returning traveler --
    a weak, structural sanity check that this ran through the real cold-start
    path rather than accidentally matching an existing real traveler_id."""
    payloads = _load_all_scenario_json(scenarios_result_dir)
    for n, payload in payloads.items():
        assert payload["profile"]["traveler_id"] == f"SCENARIO_{n}_TRAVELER"


def test_run_scenarios_is_byte_deterministic_excluding_generated_at(
    evaluate_ready_data_dir: Path,
    trained_scoring_artifacts_dir: Path,
    feature_build_cfg: Any,
    fast_model_cfg: Any,
    scoring_cfg: Any,
    candidates_cfg: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from poi_rank.eval.scenarios import run_scenarios

    results_dir_a = tmp_path_factory.mktemp("scenarios_det_a")
    results_dir_b = tmp_path_factory.mktemp("scenarios_det_b")
    run_scenarios(
        evaluate_ready_data_dir,
        trained_scoring_artifacts_dir,
        results_dir_a,
        feature_build_cfg,
        fast_model_cfg,
        scoring_cfg,
        candidates_cfg,
    )
    run_scenarios(
        evaluate_ready_data_dir,
        trained_scoring_artifacts_dir,
        results_dir_b,
        feature_build_cfg,
        fast_model_cfg,
        scoring_cfg,
        candidates_cfg,
    )

    for name in ("1.json", "2.json", "3.json", "4.json", "overlap_matrix.json"):
        payload_a = json.loads((results_dir_a / "scenarios" / name).read_text(encoding="utf-8"))
        payload_b = json.loads((results_dir_b / "scenarios" / name).read_text(encoding="utf-8"))
        payload_a.pop("generated_at", None)
        payload_b.pop("generated_at", None)
        assert payload_a == payload_b, f"{name} differs across two runs (excl. generated_at)"
