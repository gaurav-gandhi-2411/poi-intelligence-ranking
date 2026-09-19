"""`eval/gate_dgp.py` unit tests: hand-constructed diagnostics payloads straddling
each of the 9 exact gate thresholds (docs/DATA_CARD.md "DGP remediation, Block A"),
verifying `run_gate_dgp`'s threshold application without needing a real pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from poi_rank.eval.gate_dgp import GATE_THRESHOLDS, _compare, _extract_measured_values


def _base_payload(**overrides: Any) -> dict[str, Any]:
    """A payload with every gate quantity set exactly AT its passing boundary
    (or comfortably inside it), so a single overridden field can be tested in
    isolation against a known-passing baseline."""
    payload: dict[str, Any] = {
        "D1_variance_decomposition": {
            "traveler_dependent_variance_share": 0.80,
            "epsilon_variance_share": 0.10,
            "min_deterministic_term_share": 0.05,
        },
        "D3_spearman_utility_vs_label": {"spearman_rho": 0.45},
        "D5_ndcg": {"candidate_level": {"mean_ndcg_at_10": 0.50}},
        "D6_cold_start_share": {"holdout_only": {"share": 0.15}},
        "D7_localness_vs_geo_generation": {"spearman_rho": 0.60},
        "D9_semantic_fidelity": {
            "tfidf_path": {"spearman_rho": 0.55},
            "minilm_path": {"spearman_rho": 0.30},
        },
        "D10_description_conditioning": {"measured": {"spearman_rho": 0.55}},
    }
    for key, value in overrides.items():
        section, field = key.split(".", 1)
        payload[section] = {**payload[section]}
        if "." in field:
            subsection, subfield = field.split(".", 1)
            payload[section][subsection] = {**payload[section][subsection], subfield: value}
        else:
            payload[section][field] = value
    return payload


def test_compare_greater_equal() -> None:
    assert _compare(">=", 0.5, 0.5) is True
    assert _compare(">=", 0.49, 0.5) is False


def test_compare_less_equal() -> None:
    assert _compare("<=", 0.15, 0.15) is True
    assert _compare("<=", 0.16, 0.15) is False


def test_compare_unknown_op_raises() -> None:
    with pytest.raises(ValueError):
        _compare("!=", 1.0, 1.0)


def test_extract_measured_values_all_gates_present() -> None:
    payload = _base_payload()
    measured = _extract_measured_values(payload)
    assert set(measured.keys()) == set(GATE_THRESHOLDS.keys())


def test_extract_measured_values_d9_takes_the_higher_of_the_two_paths() -> None:
    payload = _base_payload()
    payload["D9_semantic_fidelity"]["tfidf_path"]["spearman_rho"] = 0.20
    payload["D9_semantic_fidelity"]["minilm_path"]["spearman_rho"] = 0.60
    measured = _extract_measured_values(payload)
    assert measured["d9_semantic_fidelity_best"] == pytest.approx(0.60)


@pytest.mark.parametrize(
    ("gate_name", "passing_value", "failing_value"),
    [
        ("traveler_dependent_variance_share", 0.76, 0.70),
        ("spearman_u_vs_label", 0.41, 0.30),
        ("var_epsilon_over_var_u", 0.10, 0.20),
        ("min_non_epsilon_term_share", 0.03, 0.01),
        ("spearman_localness_vs_geo", 0.60, 0.40),
        ("d9_semantic_fidelity_best", 0.55, 0.30),
        ("d10_description_conditioning", 0.55, 0.30),
        ("oracle_ndcg10_candidate_level", 0.50, 0.30),
        ("cold_start_trip_share", 0.15, 0.40),
    ],
)
def test_each_gate_passes_and_fails_at_expected_boundary(
    gate_name: str, passing_value: float, failing_value: float
) -> None:
    spec = GATE_THRESHOLDS[gate_name]
    assert _compare(spec["op"], passing_value, spec["threshold"]) is True
    assert _compare(spec["op"], failing_value, spec["threshold"]) is False


def test_run_gate_dgp_writes_report_and_computes_overall_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_gate_dgp` itself, with `run_dgp_diagnostics` monkeypatched to return a
    hand-built payload -- verifies the write path, the per-check structure, and
    the overall pass/fail aggregation without running a real pipeline."""
    import poi_rank.eval.gate_dgp as gate_dgp_module

    all_passing_payload = _base_payload()

    def _fake_run_dgp_diagnostics(
        data_dir: Path, results_dir: Path, datagen_cfg: Any, feature_cfg: Any
    ) -> dict[str, Any]:
        return {"payload": all_passing_payload, "output_path": results_dir / "parts" / "x.json"}

    monkeypatch.setattr(gate_dgp_module, "run_dgp_diagnostics", _fake_run_dgp_diagnostics)

    result = gate_dgp_module.run_gate_dgp(tmp_path, tmp_path, object(), object())
    payload = result["payload"]

    assert result["output_path"] == tmp_path / "parts" / "dgp_gate.json"
    assert result["output_path"].exists()
    on_disk = json.loads(result["output_path"].read_text(encoding="utf-8"))
    assert on_disk == payload

    assert payload["overall_pass"] is True
    assert payload["n_total"] == len(GATE_THRESHOLDS)
    assert payload["n_passed"] == len(GATE_THRESHOLDS)
    for check in payload["checks"].values():
        assert check["passed"] is True


def test_run_gate_dgp_overall_fails_if_any_single_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import poi_rank.eval.gate_dgp as gate_dgp_module

    failing_payload = _base_payload(**{"D3_spearman_utility_vs_label.spearman_rho": 0.10})

    def _fake_run_dgp_diagnostics(
        data_dir: Path, results_dir: Path, datagen_cfg: Any, feature_cfg: Any
    ) -> dict[str, Any]:
        return {"payload": failing_payload, "output_path": results_dir / "parts" / "x.json"}

    monkeypatch.setattr(gate_dgp_module, "run_dgp_diagnostics", _fake_run_dgp_diagnostics)

    result = gate_dgp_module.run_gate_dgp(tmp_path, tmp_path, object(), object())
    payload = result["payload"]

    assert payload["overall_pass"] is False
    assert payload["checks"]["spearman_u_vs_label"]["passed"] is False
    assert payload["n_passed"] == len(GATE_THRESHOLDS) - 1
