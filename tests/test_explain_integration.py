"""End-to-end `explain/` integration: real content (not `scoring/output.py`'s
placeholders) on the real session fixture-chain data, spec.md section 9.5 schema
shape, cold-start honesty across every real recommendation (not just a
hand-built example), and payload determinism.

**Determinism scope, an honest deviation from a literal reading of "byte-identical
`recommendations.json`"**: `assemble_output_payload` (Phase 6, unchanged by this
phase) stamps every payload with a real wall-clock `generated_at` --
`datetime.now(UTC).isoformat()` -- which by construction differs between two
independent process runs unless they land in the same microsecond. This is a
pre-existing Phase 6 schema property (spec.md section 9.5 requires the field), not
something Phase 7 introduces or can fix within scope. `test_payload_is_deterministic
_excluding_generated_at` below verifies determinism of everything ELSE -- every
`top_signals`/`explanation` value, every score -- across two fully independent
`run_scoring_pipeline` + `enrich_recommend_result` calls, which is the genuinely
testable determinism claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.explain.output_enrichment import enrich_recommend_result
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.config import ModelConfig
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import run_scoring_pipeline

_PLACEHOLDER_GROUP = "placeholder_pending_explainability_phase"

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
# Schema + real (non-placeholder) content
# -----------------------------------------------------------------------------------


def test_enriched_payload_matches_spec_schema(enriched_payload: dict[str, Any]) -> None:
    assert enriched_payload
    for trip_id, trip_payload in enriched_payload.items():
        assert set(trip_payload.keys()) == REQUIRED_TOP_LEVEL_KEYS
        assert trip_payload["trip_id"] == trip_id
        for rec in trip_payload["recommendations"]:
            assert set(rec.keys()) == REQUIRED_RECOMMENDATION_KEYS


def test_top_signals_are_real_not_placeholder(enriched_payload: dict[str, Any]) -> None:
    for trip_payload in enriched_payload.values():
        for rec in trip_payload["recommendations"]:
            assert len(rec["top_signals"]) > 0
            for signal in rec["top_signals"]:
                assert signal["feature_group"] != _PLACEHOLDER_GROUP
                assert isinstance(signal["contribution"], float)


def test_explanation_lines_are_real_not_placeholder(enriched_payload: dict[str, Any]) -> None:
    for trip_payload in enriched_payload.values():
        for rec in trip_payload["recommendations"]:
            assert len(rec["explanation"]) > 0
            for line in rec["explanation"]:
                assert "structural placeholder" not in line
                assert "placeholder_pending_explainability_phase" not in line


def test_top_signals_feature_groups_are_real_group_names(enriched_payload: dict[str, Any]) -> None:
    from poi_rank.explain.shap_groups import FEATURE_GROUPS

    for trip_payload in enriched_payload.values():
        for rec in trip_payload["recommendations"]:
            for signal in rec["top_signals"]:
                assert signal["feature_group"] in FEATURE_GROUPS


# -----------------------------------------------------------------------------------
# Cold-start honesty across every real recommendation (not just the hand-built
# templates.py unit test)
# -----------------------------------------------------------------------------------


def test_no_cold_start_recommendation_falsely_claims_past_trips(
    enriched_payload: dict[str, Any], scoring_pipeline_result: dict[str, Any]
) -> None:
    full = scoring_pipeline_result["full_frame"]
    interaction_count_by_key = {
        (str(tid), str(pid)): float(count)
        for tid, pid, count in zip(
            full["trip_id"], full["poi_id"], full["implicit_interaction_count"], strict=True
        )
    }

    n_cold_start_checked = 0
    for trip_id, trip_payload in enriched_payload.items():
        for rec in trip_payload["recommendations"]:
            count = interaction_count_by_key[(trip_id, rec["poi_id"])]
            if count == 0.0:
                n_cold_start_checked += 1
                assert not any("past trips" in line for line in rec["explanation"])

    # This is a real-data assertion, not a tautology: the fixture-chain dataset must
    # actually contain at least one cold-start recommendation for this test to be
    # exercising anything (mirrors `features/traveler_features.py`'s own documented
    # cold-start-trip population).
    assert n_cold_start_checked > 0


# -----------------------------------------------------------------------------------
# Determinism (module docstring: excluding the spec-mandated wall-clock stamp)
# -----------------------------------------------------------------------------------


def test_payload_is_deterministic_excluding_generated_at(
    evaluate_ready_data_dir: Path,
    trained_scoring_artifacts_dir: Path,
    feature_build_cfg: FeatureBuildConfig,
    fast_model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
    scoring_pipeline_result: dict[str, Any],
    tmp_path_factory: Any,
) -> None:
    trip_id_filter = {next(iter(scoring_pipeline_result["payload"]))}

    def _run(tag: str) -> dict[str, Any]:
        result = run_scoring_pipeline(
            evaluate_ready_data_dir,
            trained_scoring_artifacts_dir,
            feature_build_cfg,
            fast_model_cfg,
            scoring_cfg,
            candidates_cfg.geo,
            candidates_cfg.longtail.pop_pct_cutoff,
            trip_id_filter=trip_id_filter,
        )
        figures_dir = tmp_path_factory.mktemp(f"determinism_figs_{tag}")
        return enrich_recommend_result(
            result,
            evaluate_ready_data_dir,
            trained_scoring_artifacts_dir,
            figures_dir,
            scoring_cfg,
            feature_build_cfg,
        )

    payload_1 = _run("a")
    payload_2 = _run("b")

    def _strip_generated_at(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            trip_id: {k: v for k, v in trip_payload.items() if k != "generated_at"}
            for trip_id, trip_payload in payload.items()
        }

    assert _strip_generated_at(payload_1) == _strip_generated_at(payload_2)
