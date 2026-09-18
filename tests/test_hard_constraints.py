"""**Build-blocking** (spec.md section 11.5): "Hard-constraint violation rate in
top-10 must be 0.000 -- this is a correctness assertion, enforced by a test that
fails the build if violated." This is NOT an honest-miss candidate like the
localness/candidate-recall metrics elsewhere in this project -- it must genuinely
pass, no `xfail`, no tolerance.

Runs the full, real scoring pipeline (`scoring.output.run_scoring_pipeline`,
`tests/conftest.py`'s session-scoped `scoring_pipeline_result` fixture) on real
fixture-chain data and asserts that ZERO top-10 recommendations across ALL scored
trips have `hard_gate == 0` -- i.e. a closed/inaccessible/unreachable POI never
appears in anyone's actual output.
"""

from __future__ import annotations

from typing import Any


def test_zero_hard_constraint_violations_in_output_top_k(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    """Every recommendation actually assembled into the output JSON has
    `hard_constraints_ok == True` -- direct, end-to-end proof against the real
    pipeline, not just a unit test of `compute_hard_gate` in isolation."""
    payload = scoring_pipeline_result["payload"]
    assert payload, "expected at least one trip in the scoring output"

    violations: list[tuple[str, str]] = []
    n_recommendations = 0
    for trip_id, trip_payload in payload.items():
        for rec in trip_payload["recommendations"]:
            n_recommendations += 1
            if not rec["hard_constraints_ok"]:
                violations.append((trip_id, rec["poi_id"]))

    assert n_recommendations > 0, "expected at least one recommendation across all trips"
    assert not violations, (
        f"hard-constraint violation rate in top-10 must be 0.000 "
        f"(spec.md section 11.5) -- found {len(violations)} violation(s): {violations}"
    )


def test_zero_hard_constraint_violations_via_full_frame_hard_gate(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    """A second, independent check straight against `hard_gate` on the assembled
    `full_frame`, restricted to whatever candidates actually appear in the output
    payload -- catches a violation even if `assemble_output_payload`'s own internal
    assertion were ever silently removed."""
    payload = scoring_pipeline_result["payload"]
    full = scoring_pipeline_result["full_frame"]
    hard_gate_by_key = full.set_index(["trip_id", "poi_id"])["hard_gate"]

    violations: list[tuple[str, str]] = []
    for trip_id, trip_payload in payload.items():
        for rec in trip_payload["recommendations"]:
            gate = hard_gate_by_key.loc[(trip_id, rec["poi_id"])]
            if gate != 1.0:
                violations.append((trip_id, rec["poi_id"]))

    assert not violations, f"hard_gate != 1.0 for output rows: {violations}"
