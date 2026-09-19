"""DGP acceptance gate (spec-v2-remediation.md section 3, thresholds superseded by
the orchestrator's task prompt -- docs/DATA_CARD.md "DGP remediation, Block A").

`poi_rank.cli gate-dgp`: reuses `eval/dgp_diagnostics.py::run_dgp_diagnostics`'s
already-computed D1-D10 numbers directly (never reimplements a measurement) and
applies the exact threshold table below, writing `results/parts/dgp_gate.json` with a
pass/fail verdict per threshold plus an overall pass/fail. Designed to run
immediately after `generate` and before `prepare` in the eventual `make reproduce`
chain -- see this module's docstring on `run_gate_dgp` for the documented precondition
tension (several checks need `pois_prepared.parquet`/`poi_features.parquet`, which
`prepare`/`features` produce, i.e. this command's OWN checks require steps the gate is
meant to run BEFORE -- flagged explicitly, not silently resolved, since the eventual
Makefile wiring is a later task's decision).

**Freeze discipline**: this module reads `features/`/`candidates/` OUTPUT PARQUET
FILES (already on disk, produced by a prior `prepare`/`features` run against the
CURRENT dataset) but never imports or calls into `features/`/`candidates/`/`models/`/
`scoring/` code itself, and never re-tunes anything on its own -- it only measures and
reports. Any DGP-side re-tuning happens by editing `configs/datagen.yaml` and
re-running `generate` -> `prepare` -> `features` -> `diagnose-dgp`/`gate-dgp`, not by
this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poi_rank.datagen.config import DatagenConfig
from poi_rank.eval.dgp_diagnostics import run_dgp_diagnostics
from poi_rank.features.config import FeatureBuildConfig

OUTPUT_FILENAME = "dgp_gate.json"

# Exact thresholds (supersedes spec-v2-remediation.md section 3's original table
# entirely -- per the task prompt's explicit instruction to use ONLY this list).
# Each entry: (gate name, comparison, threshold, path into the diagnostics payload
# as a tuple of keys, plus an optional transform to derive the compared value).
GATE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "traveler_dependent_variance_share": {"op": ">=", "threshold": 0.75},
    "spearman_u_vs_label": {"op": ">=", "threshold": 0.40},
    "var_epsilon_over_var_u": {"op": "<=", "threshold": 0.15},
    "min_non_epsilon_term_share": {"op": ">=", "threshold": 0.02},
    "spearman_localness_vs_geo": {"op": ">=", "threshold": 0.55},
    "d9_semantic_fidelity_best": {"op": ">=", "threshold": 0.50},
    "d10_description_conditioning": {"op": ">=", "threshold": 0.50},
    "oracle_ndcg10_candidate_level": {"op": ">=", "threshold": 0.45},
    "cold_start_trip_share": {"op": "<=", "threshold": 0.25},
}


def _compare(op: str, value: float, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"unknown comparison op: {op}")  # pragma: no cover - defensive


def _extract_measured_values(payload: dict[str, Any]) -> dict[str, float]:
    """Pull the exact 9 gate quantities out of an already-computed
    `run_dgp_diagnostics` payload -- no diagnostic is recomputed here."""
    d1 = payload["D1_variance_decomposition"]
    d3 = payload["D3_spearman_utility_vs_label"]
    d6 = payload["D6_cold_start_share"]
    d7 = payload["D7_localness_vs_geo_generation"]
    d9 = payload["D9_semantic_fidelity"]
    d10 = payload["D10_description_conditioning"]
    d5 = payload["D5_ndcg"]

    var_epsilon_over_var_u = d1["epsilon_variance_share"]
    d9_best = max(d9["tfidf_path"]["spearman_rho"], d9["minilm_path"]["spearman_rho"])

    return {
        "traveler_dependent_variance_share": d1["traveler_dependent_variance_share"],
        "spearman_u_vs_label": d3["spearman_rho"],
        "var_epsilon_over_var_u": var_epsilon_over_var_u,
        "min_non_epsilon_term_share": d1["min_deterministic_term_share"],
        "spearman_localness_vs_geo": d7["spearman_rho"],
        "d9_semantic_fidelity_best": d9_best,
        "d10_description_conditioning": d10["measured"]["spearman_rho"],
        "oracle_ndcg10_candidate_level": d5["candidate_level"]["mean_ndcg_at_10"],
        "cold_start_trip_share": d6["holdout_only"]["share"],
    }


def run_gate_dgp(
    data_dir: Path,
    results_dir: Path,
    datagen_cfg: DatagenConfig,
    feature_cfg: FeatureBuildConfig,
) -> dict[str, Any]:
    """Run `diagnose-dgp`'s full D1-D10 computation, apply the gate thresholds, and
    write `results/parts/dgp_gate.json`.

    **Documented precondition** (flagged per task scope, not silently resolved):
    this command requires `poi_rank.cli generate` -> `poi_rank.cli prepare` ->
    `poi_rank.cli features` to have ALL already run against the current dataset --
    D7/D9/D10 read `pois_prepared.parquet`/`poi_features.parquet`/
    `traveler_features.parquet`, none of which exist until `prepare`/`features`
    run. This is in tension with the "gate runs before prepare" framing in the
    eventual `make reproduce` chain; resolving that tension (e.g. by having the
    gate internally trigger a minimal prepare/features run, or by accepting the
    gate only ever runs AFTER those steps and re-documenting the chain order) is a
    Block B/D decision, not made here.

    Returns `{"payload": <gate result dict>, "output_path": <Path>,
    "diagnostics_payload": <the full underlying D1-D10 payload>}`.
    """
    diagnostics_result = run_dgp_diagnostics(data_dir, results_dir, datagen_cfg, feature_cfg)
    diagnostics_payload = diagnostics_result["payload"]
    measured = _extract_measured_values(diagnostics_payload)

    checks: dict[str, dict[str, Any]] = {}
    for name, spec in GATE_THRESHOLDS.items():
        value = measured[name]
        passed = _compare(spec["op"], value, spec["threshold"])
        checks[name] = {
            "measured": value,
            "op": spec["op"],
            "threshold": spec["threshold"],
            "passed": passed,
        }

    overall_pass = all(check["passed"] for check in checks.values())

    payload: dict[str, Any] = {
        "overall_pass": overall_pass,
        "checks": checks,
        "n_passed": sum(1 for c in checks.values() if c["passed"]),
        "n_total": len(checks),
    }

    output_dir = results_dir / "parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "payload": payload,
        "output_path": output_path,
        "diagnostics_payload": diagnostics_payload,
    }
