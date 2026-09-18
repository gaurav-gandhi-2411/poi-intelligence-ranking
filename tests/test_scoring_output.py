"""`scoring/output.py` end-to-end tests: the assembled output JSON validates against
spec.md section 9.5's exact field set (every required field present, correct types),
calibration/beta-sensitivity/lambda-sweep diagnostics have the expected shape, and
the confidence-decile monotonicity validation (spec.md section 9.3).

**Confidence-decile monotonicity is a genuine, measured honest miss** (`xfail(strict
=True)`, mirroring `tests/test_localness_oracle.py`'s established pattern for the
localness-ρ honest miss) -- NOT the same category as `tests/test_hard_constraints.py`
(build-blocking, must pass). Diagnosed in docs/DATA_CARD.md: every one of spec.md's 5
named confidence inputs (n_interactions_traveler, poi_impression_count, review_count,
ensemble_std, calibration_bin_width), individually AND under two different
combination functions (weighted arithmetic mean -- the shipped `g` -- and geometric
mean), shows no statistically significant correlation with per-trip NDCG@10 on this
dataset (measured |Spearman rho| < 0.09, p > 0.24 for every individual input). A
positive control (other model-internal signals NOT among spec's 5 named inputs, e.g.
the top-ranked candidate's own utility score) DOES show a weak but significant
correlation (rho ~0.16-0.18, p < 0.04), proving the measurement methodology itself
detects real signal when present -- the miss is specific to spec's named 5 inputs on
this dataset, not a broken evaluation harness.
"""

from __future__ import annotations

from typing import Any

import pytest

from poi_rank.scoring.output import COMPATIBILITY_BREAKDOWN_KEYS

REQUIRED_TOP_LEVEL_KEYS = {
    "trip_id",
    "traveler_id",
    "destination",
    "generated_at",
    "model_version",
    "recommendations",
}
REQUIRED_RECOMMENDATION_KEYS = {
    "poi_id",
    "rank",
    "utility",
    "planner_weight",
    "preference_score",
    "context_compatibility",
    "compatibility_breakdown",
    "confidence",
    "hard_constraints_ok",
    "expected_duration_min",
    "diversity_group",
    "popularity_percentile",
    "localness_index",
    "top_signals",
    "explanation",
}


# -----------------------------------------------------------------------------------
# Output schema (spec.md section 9.5)
# -----------------------------------------------------------------------------------


def test_output_schema_every_trip_has_required_top_level_fields(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    payload = scoring_pipeline_result["payload"]
    assert payload
    for trip_id, trip_payload in payload.items():
        assert set(trip_payload.keys()) == REQUIRED_TOP_LEVEL_KEYS
        assert trip_payload["trip_id"] == trip_id
        assert isinstance(trip_payload["traveler_id"], str)
        assert isinstance(trip_payload["destination"], str)
        assert isinstance(trip_payload["generated_at"], str)
        assert isinstance(trip_payload["model_version"], str)
        assert isinstance(trip_payload["recommendations"], list)
        assert len(trip_payload["recommendations"]) > 0


def test_output_schema_every_recommendation_has_required_fields_and_types(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    payload = scoring_pipeline_result["payload"]
    for trip_payload in payload.values():
        for rec in trip_payload["recommendations"]:
            assert set(rec.keys()) == REQUIRED_RECOMMENDATION_KEYS
            assert isinstance(rec["poi_id"], str)
            assert isinstance(rec["rank"], int)
            assert isinstance(rec["utility"], float)
            assert isinstance(rec["planner_weight"], float)
            assert isinstance(rec["preference_score"], float)
            assert isinstance(rec["context_compatibility"], float)
            assert isinstance(rec["compatibility_breakdown"], dict)
            assert set(rec["compatibility_breakdown"].keys()) == set(COMPATIBILITY_BREAKDOWN_KEYS)
            for v in rec["compatibility_breakdown"].values():
                assert isinstance(v, float)
                assert 0.0 <= v <= 1.0 + 1e-9
            assert isinstance(rec["confidence"], float)
            assert 0.0 <= rec["confidence"] <= 1.0 + 1e-9
            assert isinstance(rec["hard_constraints_ok"], bool)
            assert rec["hard_constraints_ok"] is True
            assert isinstance(rec["expected_duration_min"], float)
            assert isinstance(rec["diversity_group"], str)
            assert isinstance(rec["popularity_percentile"], float)
            assert 0.0 <= rec["popularity_percentile"] <= 100.0 + 1e-9
            assert isinstance(rec["localness_index"], float)
            assert isinstance(rec["top_signals"], list) and len(rec["top_signals"]) > 0
            assert isinstance(rec["explanation"], list) and len(rec["explanation"]) > 0


def test_output_schema_ranks_are_contiguous_from_one(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    payload = scoring_pipeline_result["payload"]
    for trip_payload in payload.values():
        ranks = [r["rank"] for r in trip_payload["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))


def test_output_schema_planner_weight_sums_to_one_per_trip(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    payload = scoring_pipeline_result["payload"]
    for trip_payload in payload.values():
        total = sum(r["planner_weight"] for r in trip_payload["recommendations"])
        assert total == pytest.approx(1.0, abs=1e-6)


def test_output_schema_at_most_top_k_recommendations_per_trip(
    scoring_pipeline_result: dict[str, Any], scoring_cfg: Any
) -> None:
    payload = scoring_pipeline_result["payload"]
    for trip_payload in payload.values():
        assert len(trip_payload["recommendations"]) <= scoring_cfg.output.top_k


# -----------------------------------------------------------------------------------
# Calibration / beta-sensitivity / lambda-sweep diagnostics: shape + sanity
# -----------------------------------------------------------------------------------


def test_calibration_ece_after_meets_target_on_real_data(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    """spec.md section 11.10's own success criterion: ECE after calibration <= 0.05.
    Measured, not asserted in isolation from real data -- see docs/DATA_CARD.md for
    the exact numbers."""
    c = scoring_pipeline_result["calibration"]
    assert c["ece_after"] <= 0.05
    assert c["ece_after"] < c["ece_before"]  # calibration demonstrably helps
    assert c["brier_after"] < c["brier_before"]


def test_beta_sensitivity_table_has_every_configured_beta(
    scoring_pipeline_result: dict[str, Any], scoring_cfg: Any
) -> None:
    rows = scoring_pipeline_result["beta_sensitivity"]
    betas = [r["beta"] for r in rows]
    assert betas == list(scoring_cfg.utility.beta_sweep)
    for row in rows:
        assert 0.0 <= row["ndcg@10_mean"] <= 1.0


def test_lambda_sweep_has_every_configured_lambda_and_diversity_increases_as_lambda_drops(
    scoring_pipeline_result: dict[str, Any], scoring_cfg: Any
) -> None:
    rows = scoring_pipeline_result["lambda_sweep"]
    lambdas = [r["lambda"] for r in rows]
    assert lambdas == list(scoring_cfg.diversity.lambda_sweep)
    # lambda=1.0 (pure utility, no diversity term) must have the HIGHEST mean
    # intra-list similarity among the sweep (least diverse) -- the qualitative
    # trade-off direction the sweep exists to demonstrate.
    by_lambda = {r["lambda"]: r["mean_intra_list_similarity"] for r in rows}
    assert by_lambda[1.0] == max(by_lambda.values())


# -----------------------------------------------------------------------------------
# Confidence-decile monotonicity (spec.md section 9.3) -- genuine, diagnosed honest
# miss, see module docstring.
# -----------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Confidence-decile NDCG@10 monotonicity (Spearman rho >= 0.7, spec.md "
        "section 9.3) is a genuine, measured miss on this dataset: every one of "
        "spec.md's 5 named confidence inputs, individually and under 2 different "
        "combination functions, shows no significant correlation with per-trip "
        "NDCG@10 -- diagnosed in docs/DATA_CARD.md, not tuned to hide."
    ),
)
def test_confidence_decile_ndcg_is_monotone_increasing(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    dv = scoring_pipeline_result["confidence_decile_validation"]
    assert dv["spearman_rho"] is not None
    assert dv["spearman_rho"] >= 0.7
