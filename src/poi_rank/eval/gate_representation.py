"""Gate-B: representation / candidate-generation acceptance gate (spec-v3 section 4, GG rulings).

Two tiers, kept structurally apart so the oracle can never sit in a decision:

  BLOCKING, oracle-free -- measured against EXPOSED holdout positives (uniform-random-exposure
  log; no oracle read). AMENDED before experiment L3b ran (`docs/experiments/L-final.md`): a
  blocking gate encodes a REQUIREMENT, not an a-priori guess.
    * candidate recall, long-tail stratum  >= 0.75  (the brief section 9 requirement: do not
      eliminate relevant long-tail POIs)
    * effective candidates per trip        <= 300   (the serving budget the chance-lift gate was
      standing in for: larger sets raise the chance baseline, so a lift gate penalises them
      for the wrong reason)
  REPORTING, still computed for every design and shown with the old reference thresholds, but no
  longer blocking (both were disclosed as miscalibrated a-priori targets, `docs/TECHNICAL.md`
  section 10.2): overall candidate recall (reference 0.85) and the recall lift over the chance
  baseline, overall and long-tail (reference +0.35 absolute).
  Every representation / retrieval hyperparameter was selected on a train-carved validation
  split (`results/parts/a3_*.json`), never on these numbers.

  REPORTING ONLY, computed once after the selection was frozen: D9 (within-trip), D11 ridge R^2
  / CCA, oracle-relevance recall, oracle candidate-level NDCG ceiling. They can never fail the
  gate -- TECHNICAL.md states the oracle never touched a decision, it only ever scored the result.

Reads results/parts/{evaluate,representation}.json; writes results/parts/gate_representation.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poi_rank.eval.compose import PARTS_DIRNAME, write_part

BLOCKING_THRESHOLDS: dict[str, dict[str, Any]] = {
    "candidate_recall_long_tail": {"op": ">=", "threshold": 0.75},
    "effective_candidates_per_trip": {"op": "<=", "threshold": 300.0},
}
# Former blocking rows, demoted by the L3a amendment: computed and shown, never failing the gate.
REPORTING_REFERENCE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "candidate_recall_overall": {"op": ">=", "threshold": 0.85},
    "candidate_recall_lift_overall": {"op": ">=", "threshold": 0.35},
    "candidate_recall_lift_long_tail": {"op": ">=", "threshold": 0.35},
}


def _read_part(results_dir: Path, stem: str) -> dict[str, Any]:
    path = results_dir / PARTS_DIRNAME / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"results/{PARTS_DIRNAME}/{stem}.json missing -- run `{stem}` first"
        )
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def run_gate_representation(results_dir: Path) -> dict[str, Any]:
    evaluate = _read_part(results_dir, "evaluate")
    representation = _read_part(results_dir, "representation")
    recall = evaluate["candidate_recall"]
    strata = recall["by_stratum"]

    measured = {
        "candidate_recall_overall": recall["overall"]["recall_mean"],
        "candidate_recall_long_tail": recall["long_tail"]["recall_mean"],
        "candidate_recall_lift_overall": strata["overall"]["lift_abs"],
        "candidate_recall_lift_long_tail": strata["long_tail"]["lift_abs"],
        "effective_candidates_per_trip": recall["mean_candidates_per_trip"],
    }

    def judge(spec: dict[str, Any], value: float) -> bool:
        threshold = float(spec["threshold"])
        return value >= threshold if spec["op"] == ">=" else value <= threshold

    checks: dict[str, dict[str, Any]] = {}
    for name, spec in BLOCKING_THRESHOLDS.items():
        value = float(measured[name])
        checks[name] = {**spec, "measured": value, "passed": bool(judge(spec, value))}
    reporting_checks: dict[str, dict[str, Any]] = {}
    for name, spec in REPORTING_REFERENCE_THRESHOLDS.items():
        value = float(measured[name])
        reporting_checks[name] = {
            **spec,
            "measured": value,
            "meets_reference": bool(judge(spec, value)),
        }

    reporting = {
        "d9_within_trip_shipped_chain": representation["d9_within_trip"]["shipped_chain_d9"][
            "mean_within_trip_spearman"
        ],
        "d9_within_trip_taste_estimator_alone": representation["d9_within_trip"][
            "taste_estimator_alone"
        ]["mean_within_trip_spearman"],
        "d9_within_trip_text_alone": representation["d9_within_trip"]["text_alone"][
            "mean_within_trip_spearman"
        ],
        "d11_ridge_r2_text_only": representation["d11_text_only"]["ridge_r2_oof"],
        "d11_cca_first_corr_text_only": representation["d11_text_only"]["cca_first_corr_oof"],
        "oracle_relevance_recall_overall": representation["oracle_relevance_recall"]["overall"],
        "oracle_relevance_recall_long_tail": representation["oracle_relevance_recall"]["long_tail"],
        "oracle_ceiling_ndcg10_candidate_level": evaluate["systems"]["oracle"]["metrics"][
            "ndcg@10"
        ]["mean"],
        "recall_by_stratum": strata,
    }
    payload: dict[str, Any] = {
        "overall_pass": all(c["passed"] for c in checks.values()),
        "blocking_checks": checks,
        "reporting_checks": reporting_checks,
        "reporting_only_after_freeze": reporting,
        "n_passed": sum(1 for c in checks.values() if c["passed"]),
        "n_total": len(checks),
    }
    path = write_part(results_dir, "gate_representation", payload)
    return {"payload": payload, "output_path": path}
