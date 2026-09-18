"""`eval/report.py` tests: `docs/RESULTS.md` is GENERATED, never hand-written
(spec.md section 0/11: "No number appears in `docs/` that is not produced by
`results/metrics.json`"). This is the actual enforcement mechanism for that rule --
a real, automated grep-and-cross-check, not a manual promise, per this phase's own
task brief ("the verifier agent must grep for it")."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from poi_rank.eval.report import (
    load_metrics,
    render_results_md,
    render_success_criteria,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATAGEN_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"

# Numbers that are structural (section references, target thresholds explicitly
# stated in spec.md section 11.10 itself, bin/decile/rank indices, table
# scaffolding) rather than measured pipeline output -- excluded from the
# cross-check. Every one of these is either (a) a literal target FROM spec.md's
# own success-criteria table (not something the pipeline measures) or (b) a
# small structural index (decile 1-10, "9" systems, markdown table width, spec
# section numbers like "11.10").
_STRUCTURAL_ALLOWLIST: frozenset[float] = frozenset(
    {
        0.05,
        0.25,
        0.40,
        0.5,
        0.7,
        0.8,
        0.9,
        1.0,
        2.0,
        2.5,
        5.0,
        10.0,
        15.0,
        20.0,
        50.0,
        90.0,
        97.5,
        100.0,
        250.0,
        0.01,
        0.6,
    }
    | {float(i) for i in range(0, 12)}  # decile/rank/system indices, section numbers
)

_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
_SECTION_REF_RE = re.compile(r"section \d+(\.\d+)?", re.IGNORECASE)


def _flatten_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, out)


def _extract_doc_numbers(markdown: str) -> list[float]:
    # Strip "section N.N" references (spec cross-references, not measurements)
    # before scanning for numeric tokens.
    cleaned = _SECTION_REF_RE.sub("", markdown)
    values: list[float] = []
    for match in _NUMBER_RE.finditer(cleaned):
        try:
            values.append(float(match.group()))
        except ValueError:
            continue
    return values


def _traces_to_source(value: float, source_values: list[float], ratio_pairs: list[float]) -> bool:
    """`value` traces to `results/metrics.json` if it matches a raw leaf (tol
    matched to `_fmt`'s 4-decimal rendering), a percentage of a raw leaf (tol
    matched to `_fmt_pct`'s 1-decimal rendering -- up to 0.05 rounding error in
    percentage units, so a tighter tolerance would false-positive on this module's
    OWN correct rounding), a signed delta, or a relative-lift percentage
    `(a/b - 1) * 100` between two raw leaves (`render_success_criteria`'s own
    derived quantity, precomputed into `ratio_pairs` by the caller)."""
    raw_tol = 0.0005
    pct_tol = 0.06
    if any(abs(value - s) < raw_tol for s in _STRUCTURAL_ALLOWLIST):
        return True
    for s in source_values:
        if abs(value - s) < raw_tol:
            return True
        if abs(value - s * 100.0) < pct_tol:  # percentage form
            return True
        if abs(value + s) < raw_tol:  # signed delta rendered with an explicit +/- sign
            return True
    return any(abs(value - r) < pct_tol for r in ratio_pairs)


def _relative_lift_percentages(source_values: list[float]) -> list[float]:
    """`(a/b - 1) * 100` for every ordered pair of positive source leaves --
    `render_success_criteria`'s own "NDCG@10 vs popularity" relative-lift
    percentage is exactly this derived form applied to two raw NDCG@10 means
    already present in `results/metrics.json`. Restricted to values in a plausible
    metric range (positive fractions <= 5) to keep the O(n^2) sweep meaningful
    rather than combinatorially matching unrelated large counts (e.g. n_pairs)."""
    small_positive = [s for s in source_values if 0.0 < s <= 5.0]
    out: list[float] = []
    for a in small_positive:
        for b in small_positive:
            if b != 0.0:
                out.append((a / b - 1.0) * 100.0)
    return out


@pytest.fixture(scope="module")
def fast_metrics_payload(
    evaluate_ready_data_dir: Any,
    feature_build_cfg: Any,
    fast_model_cfg: Any,
    scoring_cfg: Any,
    candidates_cfg: Any,
    trained_scoring_artifacts_dir: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """One fast full `run_evaluate` pass (fixture-chain data, small LightGBM
    settings) -- shared across every test in this module, mirrors
    `tests/test_evaluate.py`'s own fixture-reuse convention."""
    from poi_rank.datagen.config import DatagenConfig
    from poi_rank.eval.config import BootstrapConfig, EvalConfig, MetricsConfig
    from poi_rank.eval.run import run_evaluate

    eval_cfg = EvalConfig(
        seed=42,
        metrics=MetricsConfig(ndcg_ks=(5, 10), precision_ks=(5,), recall_ks=(10,)),
        bootstrap=BootstrapConfig(
            n_resamples=50, ci_low_pct=2.5, ci_high_pct=97.5, resample_unit="trip"
        ),
        long_tail_pop_pct_cutoff=0.5,
    )
    datagen_cfg = DatagenConfig.from_yaml(DATAGEN_CONFIG_PATH)
    results_dir = tmp_path_factory.mktemp("report_results")

    summary = run_evaluate(
        evaluate_ready_data_dir,
        results_dir,
        trained_scoring_artifacts_dir,
        fast_model_cfg,
        eval_cfg,
        feature_build_cfg,
        candidates_cfg,
        datagen_cfg,
        scoring_cfg,
    )
    result: dict[str, Any] = summary["payload"]
    return result


def test_render_results_md_produces_nonempty_markdown(fast_metrics_payload: dict[str, Any]) -> None:
    md = render_results_md(fast_metrics_payload)
    assert "# RESULTS.md" in md
    assert "## Success criteria" in md
    assert "## Ablations" in md
    assert len(md) > 500


def test_every_number_in_results_md_traces_to_metrics_json(
    fast_metrics_payload: dict[str, Any],
) -> None:
    """The actual enforcement mechanism for "no hand-typed numbers in docs/":
    every numeric token in the generated markdown must trace back to
    `results/metrics.json` (directly, as a percentage, or as a signed delta),
    or be one of a small documented set of structural/target constants."""
    md = render_results_md(fast_metrics_payload)
    source_values: list[float] = []
    _flatten_numbers(fast_metrics_payload, source_values)
    ratio_pairs = _relative_lift_percentages(source_values)

    doc_numbers = _extract_doc_numbers(md)
    untraceable = [n for n in doc_numbers if not _traces_to_source(n, source_values, ratio_pairs)]

    assert not untraceable, (
        f"{len(untraceable)} number(s) in docs/RESULTS.md do not trace to "
        f"results/metrics.json (first 20): {untraceable[:20]}"
    )


def test_render_success_criteria_has_nine_rows(fast_metrics_payload: dict[str, Any]) -> None:
    section = render_success_criteria(fast_metrics_payload)
    data_rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and "Target" not in line and "---" not in line
    ]
    assert len(data_rows) == 9


def test_load_metrics_missing_file_raises_actionable_error(tmp_path: Any) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="evaluate"):
        load_metrics(missing)
