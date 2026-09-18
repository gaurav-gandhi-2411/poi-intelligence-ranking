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
from poi_rank.datagen.config import DatagenConfig
from poi_rank.eval.config import BootstrapConfig, EvalConfig, MetricsConfig
from poi_rank.eval.run import ALL_SYSTEM_NAMES, run_evaluate
from poi_rank.models.config import (
    BaselinesConfig,
    ContentCosineConfig,
    ItemKnnCfConfig,
    LambdaMartConfig,
    LogisticRegressionConfig,
    ModelConfig,
)
from poi_rank.models.lambdamart import run_train_lambdamart
from poi_rank.scoring.config import ScoringConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
DATAGEN_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
SCORING_CONFIG_PATH = REPO_ROOT / "configs" / "scoring.yaml"


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
        # Small/fast LightGBM settings -- exercises the SAME code path
        # (`models.lambdamart`) real `poi_rank.cli train` runs, just cheaper.
        lambdamart=LambdaMartConfig(
            num_leaves=15,
            learning_rate=0.1,
            n_estimators=30,
            early_stopping_rounds=10,
            lambdarank_truncation_level=20,
            eval_ndcg_at=(10,),
            label_gain=(0, 1, 3, 7),
            min_data_in_leaf=5,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            num_threads=1,
            val_fraction=0.15,
            val_split_seed=43,
            ips_clip_low=1.0,
            ips_clip_high=20.0,
            behavioral_dropout_rate=0.15,
            behavioral_dropout_seed=44,
        ),
    )


@pytest.fixture(scope="module")
def datagen_cfg_module() -> DatagenConfig:
    return DatagenConfig.from_yaml(DATAGEN_CONFIG_PATH)


@pytest.fixture(scope="module")
def trained_artifacts_dir(
    evaluate_ready_data_dir: Path,
    feature_build_cfg: Any,
    fast_model_cfg: ModelConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Fits systems 7/8 + the dropout-ablation booster once per module (mirrors
    `poi_rank.cli train`) into a shared tmp `artifacts/` dir every test in this
    module reads from -- `run_evaluate` only ever LOADS boosters, never retrains."""
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    run_train_lambdamart(evaluate_ready_data_dir, artifacts_dir, fast_model_cfg, feature_build_cfg)
    return artifacts_dir


def test_run_evaluate_payload_structure(
    evaluate_ready_data_dir: Path,
    feature_build_cfg: Any,
    fast_model_cfg: ModelConfig,
    fast_eval_cfg: EvalConfig,
    trained_artifacts_dir: Path,
    datagen_cfg_module: DatagenConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    candidates_cfg = CandidatesConfig.from_yaml(FEATURES_CONFIG_PATH)
    scoring_cfg = ScoringConfig.from_yaml(SCORING_CONFIG_PATH)
    results_dir = tmp_path_factory.mktemp("results")

    summary = run_evaluate(
        evaluate_ready_data_dir,
        results_dir,
        trained_artifacts_dir,
        fast_model_cfg,
        fast_eval_cfg,
        feature_build_cfg,
        candidates_cfg,
        datagen_cfg_module,
        scoring_cfg,
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
        "lambdamart_vs_popularity",
        "lambdamart_ips_vs_popularity",
        "oracle_vs_popularity",
        "lambdamart_ips_vs_content_cosine",
        "lambdamart_vs_lambdamart_ips",
    }
    assert set(payload["wilcoxon"]) == expected_wilcoxon_keys
    for w in payload["wilcoxon"].values():
        assert 0.0 <= w["p_value"] <= 1.0

    assert payload["meta"]["n_holdout_trips"] > 0
    assert "candidate_recall@250" in payload["meta"]["note"] or "0.44" in payload["meta"]["note"]

    cohort = payload["new_poi_cohort"]
    assert cohort["n_new_pois_in_catalog"] > 0
    assert "ndcg@10_lambdamart_ips_with_dropout" in cohort
    assert "ndcg@10_lambdamart_ips_no_dropout" in cohort
    assert "wilcoxon_with_vs_without_dropout" in cohort
    assert 0.0 <= cohort["wilcoxon_with_vs_without_dropout"]["p_value"] <= 1.0

    # Phase 8 additions.
    assert set(payload["bias_gap"]) == set(ALL_SYSTEM_NAMES)
    for row in payload["bias_gap"].values():
        assert "ndcg@10_unbiased" in row
        assert "ndcg@10_biased" in row

    pers = payload["personalization"]
    assert 0.0 <= pers["mean_pairwise_jaccard_at_10"] <= 1.0
    assert 0.0 <= pers["mean_pairwise_rbo"] <= 1.0
    assert "archetype" in pers

    cov = payload["coverage"]
    assert "primary_system" in cov and "popularity_baseline" in cov
    assert 0.0 <= cov["primary_system"]["catalog_coverage_at_10"] <= 1.0

    lg = payload["longtail"]
    assert 0.0 <= lg["share"] <= 1.0

    cc = payload["constraint_compatibility"]
    assert cc["n_hard_constraint_violations"] == 0

    dv = payload["diversity"]
    assert dv["category_entropy_at_10_bits"] >= 0.0
    assert len(dv["lambda_sweep"]) > 0

    assert "ece_after" in payload["calibration"]
    assert "spearman_rho" in payload["confidence_decile_validation"]
    assert isinstance(payload["beta_sensitivity"], list) and len(payload["beta_sensitivity"]) > 0

    cs_payload = payload["cold_start"]
    assert set(cs_payload["ndcg@10_by_interaction_count_bucket"]) == {"0", "1-3", "4-10", ">10"}

    ablations = payload["ablations"]
    assert len(ablations) == 9
    assert all(row["status"] == "measured" for row in ablations)

    assert "candidate_recall" in payload
    assert "overall" in payload["candidate_recall"] and "long_tail" in payload["candidate_recall"]

    assert summary["output_path"].exists()
    written = json.loads(summary["output_path"].read_text(encoding="utf-8"))
    assert written == payload


def test_run_evaluate_is_deterministic(
    evaluate_ready_data_dir: Path,
    feature_build_cfg: Any,
    fast_model_cfg: ModelConfig,
    fast_eval_cfg: EvalConfig,
    trained_artifacts_dir: Path,
    datagen_cfg_module: DatagenConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    candidates_cfg = CandidatesConfig.from_yaml(FEATURES_CONFIG_PATH)
    scoring_cfg = ScoringConfig.from_yaml(SCORING_CONFIG_PATH)
    results_dir1 = tmp_path_factory.mktemp("results1")
    results_dir2 = tmp_path_factory.mktemp("results2")

    summary1 = run_evaluate(
        evaluate_ready_data_dir,
        results_dir1,
        trained_artifacts_dir,
        fast_model_cfg,
        fast_eval_cfg,
        feature_build_cfg,
        candidates_cfg,
        datagen_cfg_module,
        scoring_cfg,
    )
    summary2 = run_evaluate(
        evaluate_ready_data_dir,
        results_dir2,
        trained_artifacts_dir,
        fast_model_cfg,
        fast_eval_cfg,
        feature_build_cfg,
        candidates_cfg,
        datagen_cfg_module,
        scoring_cfg,
    )

    bytes1 = summary1["output_path"].read_bytes()
    bytes2 = summary2["output_path"].read_bytes()
    assert bytes1 == bytes2
