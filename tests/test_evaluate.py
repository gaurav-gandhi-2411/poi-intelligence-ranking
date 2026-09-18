"""`eval/run.py::run_evaluate` orchestration tests: full-pipeline determinism (two
runs -> byte-identical `metrics.json`, spec.md section 14's reproducibility
contract) and payload-structure sanity, against the real full-scale fixture chain
(`evaluate_ready_data_dir`). Runs with a reduced bootstrap resample count (spec.md
explicitly permits fewer resamples than the production 2,000 "if runtime matters" --
this is exactly that case: a test, not the reported headline number) to keep the
suite fast while still exercising the SAME code path (including the LogisticRegression
fit and the bootstrap RNG) the real `poi_rank.cli evaluate` command runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.eval.config import BootstrapConfig, EvalConfig, MetricsConfig
from poi_rank.eval.run import ALL_SYSTEM_NAMES, run_evaluate
from poi_rank.models.config import (
    BaselinesConfig,
    ContentCosineConfig,
    ItemKnnCfConfig,
    LogisticRegressionConfig,
    ModelConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"


@pytest.fixture(scope="module")
def fast_eval_cfg() -> EvalConfig:
    """Same shape as `configs/eval.yaml`, fewer resamples for test speed."""
    return EvalConfig(
        seed=42,
        metrics=MetricsConfig(ndcg_ks=(5, 10), precision_ks=(5,), recall_ks=(10,)),
        bootstrap=BootstrapConfig(
            n_resamples=100, ci_low_pct=2.5, ci_high_pct=97.5, resample_unit="trip"
        ),
        long_tail_pop_pct_cutoff=0.5,
    )


@pytest.fixture(scope="module")
def fast_model_cfg() -> ModelConfig:
    return ModelConfig(
        seed=42,
        baselines=BaselinesConfig(
            content_cosine=ContentCosineConfig(interest_weight=0.7, price_fit_weight=0.3),
            item_knn_cf=ItemKnnCfConfig(cf_score_missing_fallback="popularity"),
            logistic_regression=LogisticRegressionConfig(max_iter=300, C=1.0),
        ),
    )


def test_run_evaluate_payload_structure(
    evaluate_ready_data_dir: Path,
    feature_build_cfg: Any,
    fast_model_cfg: ModelConfig,
    fast_eval_cfg: EvalConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    geo_cfg = CandidatesConfig.from_yaml(FEATURES_CONFIG_PATH).geo
    results_dir = tmp_path_factory.mktemp("results")

    summary = run_evaluate(
        evaluate_ready_data_dir,
        results_dir,
        fast_model_cfg,
        fast_eval_cfg,
        feature_build_cfg,
        geo_cfg,
    )
    payload = summary["payload"]

    assert set(payload["systems"]) == set(ALL_SYSTEM_NAMES)
    for name in ALL_SYSTEM_NAMES:
        sys_payload = payload["systems"][name]
        assert "ndcg@10" in sys_payload["metrics"]
        assert "precision@5" in sys_payload["metrics"]
        assert "recall@10" in sys_payload["metrics"]
        assert "map" in sys_payload["metrics"]
        assert "mrr" in sys_payload["metrics"]
        for metric_payload in sys_payload["metrics"].values():
            assert metric_payload["ci_low"] <= metric_payload["mean"] <= metric_payload["ci_high"]
        assert "diagnostics" in sys_payload
        assert "pct_of_ceiling_ndcg10" in sys_payload

    assert payload["systems"]["oracle"]["pct_of_ceiling_ndcg10"] == pytest.approx(1.0)

    expected_wilcoxon_keys = {
        "content_cosine_vs_popularity",
        "item_knn_cf_vs_popularity",
        "logistic_regression_vs_popularity",
        "oracle_vs_popularity",
    }
    assert set(payload["wilcoxon"]) == expected_wilcoxon_keys
    for w in payload["wilcoxon"].values():
        assert 0.0 <= w["p_value"] <= 1.0

    assert payload["meta"]["n_holdout_trips"] > 0
    assert "candidate_recall@250" in payload["meta"]["note"] or "0.44" in payload["meta"]["note"]

    assert summary["output_path"].exists()
    written = json.loads(summary["output_path"].read_text(encoding="utf-8"))
    assert written == payload


def test_run_evaluate_is_deterministic(
    evaluate_ready_data_dir: Path,
    feature_build_cfg: Any,
    fast_model_cfg: ModelConfig,
    fast_eval_cfg: EvalConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    geo_cfg = CandidatesConfig.from_yaml(FEATURES_CONFIG_PATH).geo
    results_dir1 = tmp_path_factory.mktemp("results1")
    results_dir2 = tmp_path_factory.mktemp("results2")

    summary1 = run_evaluate(
        evaluate_ready_data_dir,
        results_dir1,
        fast_model_cfg,
        fast_eval_cfg,
        feature_build_cfg,
        geo_cfg,
    )
    summary2 = run_evaluate(
        evaluate_ready_data_dir,
        results_dir2,
        fast_model_cfg,
        fast_eval_cfg,
        feature_build_cfg,
        geo_cfg,
    )

    bytes1 = summary1["output_path"].read_bytes()
    bytes2 = summary2["output_path"].read_bytes()
    assert bytes1 == bytes2
