"""Gate-B after the L3a amendment: blocking rows are requirements, lift only reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poi_rank.eval.gate_representation import run_gate_representation


def _write_parts(
    tmp_path: Path, long_tail: float, k: float, overall: float = 0.5, lift: float = 0.0
) -> Path:
    parts = tmp_path / "parts"
    parts.mkdir(parents=True)
    stratum = {"recall": 0.0, "chance_recall": 0.0, "lift_abs": lift, "n_trips": 10.0}
    evaluate: dict[str, Any] = {
        "candidate_recall": {
            "overall": {"recall_mean": overall},
            "long_tail": {"recall_mean": long_tail},
            "by_stratum": {"overall": stratum, "long_tail": stratum},
            "mean_candidates_per_trip": k,
        },
        "systems": {"oracle": {"metrics": {"ndcg@10": {"mean": 0.3}}}},
    }
    representation = {
        "d9_within_trip": {
            "shipped_chain_d9": {"mean_within_trip_spearman": 0.1},
            "taste_estimator_alone": {"mean_within_trip_spearman": 0.1},
            "text_alone": {"mean_within_trip_spearman": 0.1},
        },
        "d11_text_only": {"ridge_r2_oof": 0.5, "cca_first_corr_oof": 0.5},
        "oracle_relevance_recall": {"overall": 0.9, "long_tail": 0.9},
    }
    (parts / "evaluate.json").write_text(json.dumps(evaluate), encoding="utf-8")
    (parts / "representation.json").write_text(json.dumps(representation), encoding="utf-8")
    return tmp_path


def test_low_overall_recall_and_low_lift_no_longer_block(tmp_path: Path) -> None:
    results = _write_parts(tmp_path, long_tail=0.80, k=290.0, overall=0.60, lift=0.05)
    payload = run_gate_representation(results)["payload"]
    assert payload["overall_pass"] is True
    assert set(payload["blocking_checks"]) == {
        "candidate_recall_long_tail",
        "effective_candidates_per_trip",
    }
    reporting = payload["reporting_checks"]
    assert reporting["candidate_recall_overall"]["meets_reference"] is False
    assert reporting["candidate_recall_lift_overall"]["meets_reference"] is False


def test_long_tail_recall_below_the_brief_requirement_blocks(tmp_path: Path) -> None:
    results = _write_parts(tmp_path, long_tail=0.74, k=200.0)
    assert run_gate_representation(results)["payload"]["overall_pass"] is False


def test_serving_budget_blocks_above_300_and_passes_at_300(tmp_path: Path) -> None:
    over = run_gate_representation(_write_parts(tmp_path / "a", 0.9, 300.5))["payload"]
    at = run_gate_representation(_write_parts(tmp_path / "b", 0.9, 300.0))["payload"]
    assert over["blocking_checks"]["effective_candidates_per_trip"]["passed"] is False
    assert at["blocking_checks"]["effective_candidates_per_trip"]["passed"] is True
