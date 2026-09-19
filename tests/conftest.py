"""Shared pytest fixtures for the datagen test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.candidates.retriever import (
    RetrieverInputs,
    crossfit_retriever_scores,
    top_k_by_trip,
)
from poi_rank.candidates.union import generate_candidates
from poi_rank.data.config import FeaturesConfig
from poi_rank.data.prepare import prepare_pois
from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate
from poi_rank.explain.output_enrichment import enrich_recommend_result
from poi_rank.explain.shap_groups import GroupedShapResult, compute_grouped_shap
from poi_rank.features.build import run_features
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import LambdaMartConfig, ModelConfig
from poi_rank.models.lambdamart import load_boosters, save_boosters, train_lambdamart_systems
from poi_rank.models.ranking_data import load_train_ranking_frame
from poi_rank.scoring.confidence import save_ensemble, train_ensemble_boosters
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import run_scoring_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"
SCORING_CONFIG_PATH = REPO_ROOT / "configs" / "scoring.yaml"


@pytest.fixture(scope="session")
def datagen_cfg() -> DatagenConfig:
    return DatagenConfig.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig.from_yaml(FEATURES_CONFIG_PATH)


@pytest.fixture(scope="session")
def generated_data(
    tmp_path_factory: pytest.TempPathFactory, datagen_cfg: DatagenConfig
) -> dict[str, Any]:
    """Run the full-scale generator once per test session (~8s) into a tmp dir."""
    output_dir = tmp_path_factory.mktemp("synthetic")
    summary = run_generate(datagen_cfg, output_dir)
    return {"summary": summary, "output_dir": output_dir, "cfg": datagen_cfg}


@pytest.fixture(scope="session")
def prepared_data(generated_data: dict[str, Any], features_cfg: FeaturesConfig) -> dict[str, Any]:
    """Run the full-scale data-prep pipeline once per test session on top of
    `generated_data`'s output."""
    output_dir: Path = generated_data["output_dir"]
    pois_raw = pd.read_parquet(output_dir / "pois.parquet")
    prepared, report = prepare_pois(pois_raw, features_cfg)
    return {
        "pois_raw": pois_raw,
        "prepared": prepared,
        "report": report,
        "output_dir": output_dir,
        "cfg": features_cfg,
    }


@pytest.fixture(scope="session")
def feature_build_cfg() -> FeatureBuildConfig:
    return FeatureBuildConfig.from_yaml(FEATURES_CONFIG_PATH)


@pytest.fixture(scope="session")
def built_features(
    prepared_data: dict[str, Any],
    feature_build_cfg: FeatureBuildConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Run the full-scale Phase 3 feature-build pipeline once per test session, on
    top of `prepared_data`'s output (writing `pois_prepared.parquet` into that same
    tmp `data/synthetic/`-shaped directory first, matching what `make prepare` would
    have produced on disk)."""
    output_dir: Path = prepared_data["output_dir"]
    prepared_path = output_dir / "pois_prepared.parquet"
    if not prepared_path.exists():
        prepared_data["prepared"].to_parquet(prepared_path, index=False)

    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    summary = run_features(feature_build_cfg, output_dir, artifacts_dir)

    return {
        "summary": summary,
        "poi_features": pd.read_parquet(summary["poi_features_path"]),
        "traveler_features": pd.read_parquet(summary["traveler_features_path"]),
        "output_dir": output_dir,
        "artifacts_dir": artifacts_dir,
        "cfg": feature_build_cfg,
    }


@pytest.fixture(scope="session")
def candidates_cfg() -> CandidatesConfig:
    return CandidatesConfig.from_yaml(FEATURES_CONFIG_PATH)


@pytest.fixture(scope="session")
def generated_candidates(
    built_features: dict[str, Any], candidates_cfg: CandidatesConfig
) -> dict[str, Any]:
    """Run the full-scale Phase 4a candidate-generation pipeline once per test
    session, on top of `built_features`'s output."""
    output_dir: Path = built_features["output_dir"]
    pois_df = pd.read_parquet(output_dir / "pois_prepared.parquet")
    travelers_df = pd.read_parquet(output_dir / "travelers.parquet")
    trips_df = pd.read_parquet(output_dir / "trips.parquet")
    interactions_train = pd.read_parquet(output_dir / "interactions_train.parquet")

    learned_top_k = None
    retriever = None
    if candidates_cfg.learned is not None and candidates_cfg.learned.quota > 0:
        scores, retriever = crossfit_retriever_scores(
            RetrieverInputs(
                pois_df,
                travelers_df,
                trips_df,
                built_features["poi_features"],
                built_features["traveler_features"],
                interactions_train,
                built_features["cfg"].traveler_features.budget_target_price_level,
            ),
            candidates_cfg.learned,
            candidates_cfg.seed,
            candidates_cfg.learned.num_threads,
        )
        learned_top_k = top_k_by_trip(scores, candidates_cfg.learned.quota)

    candidates_df = generate_candidates(
        pois_df,
        travelers_df,
        trips_df,
        built_features["poi_features"],
        built_features["traveler_features"],
        interactions_train,
        candidates_cfg,
        learned_top_k=learned_top_k,
    )
    return {
        "candidates": candidates_df,
        "pois": pois_df,
        "travelers": travelers_df,
        "trips": trips_df,
        "interactions_train": interactions_train,
        "output_dir": output_dir,
        "cfg": candidates_cfg,
        "retriever": retriever,
    }


@pytest.fixture(scope="session")
def evaluate_ready_data_dir(generated_candidates: dict[str, Any]) -> Path:
    """`generated_candidates`'s `output_dir`, with `candidates.parquet` additionally
    written to disk (that fixture keeps the candidate frame in-memory only) -- the
    full, on-disk `data/synthetic/`-shaped directory `models.ranking_data`'s
    file-based loaders and `poi_rank.cli evaluate` (`eval.run.run_evaluate`) expect.
    Session-scoped: shared read-only by every Phase 4b test that needs real-scale
    data, mirroring `built_features`/`generated_candidates`'s own convention."""
    output_dir: Path = generated_candidates["output_dir"]
    candidates_path = output_dir / "candidates.parquet"
    if not candidates_path.exists():
        generated_candidates["candidates"].to_parquet(candidates_path, index=False)
    return output_dir


# -----------------------------------------------------------------------------------
# Phase 6 (scoring/): a real, fast-config-trained artifacts dir + one full scoring-
# pipeline run, shared (session-scoped) across every `scoring/` test that needs real
# data -- mirrors `test_lambdamart.py`'s own `fast_lambdamart_cfg` convention (few
# boosting rounds, same recipe otherwise) so the ensemble-of-5 confidence step stays
# fast; NEVER reuses the repo's own committed `artifacts/*.txt` (tests must pass
# against a freshly-generated dataset, not depend on committed-artifact content).
# -----------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fast_model_cfg() -> ModelConfig:
    real = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    fast_lambdamart = LambdaMartConfig(
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
    )
    return ModelConfig(seed=real.seed, baselines=real.baselines, lambdamart=fast_lambdamart)


@pytest.fixture(scope="session")
def scoring_cfg() -> ScoringConfig:
    return ScoringConfig.from_yaml(SCORING_CONFIG_PATH)


@pytest.fixture(scope="session")
def trained_scoring_artifacts_dir(
    generated_candidates: dict[str, Any],
    evaluate_ready_data_dir: Path,
    feature_build_cfg: FeatureBuildConfig,
    fast_model_cfg: ModelConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Train systems 7/8 (+ the dropout-ablation booster) with `fast_model_cfg` on
    the real fixture-chain data, once per test session, and persist them exactly as
    `poi_rank.cli train` would -- `scoring/output.py`'s `run_scoring_pipeline`
    LOADS boosters from disk, never trains them itself, so a real artifacts dir is
    required."""
    train_frame = load_train_ranking_frame(
        evaluate_ready_data_dir, feature_build_cfg.traveler_features.budget_target_price_level
    )
    interactions_train = pd.read_parquet(evaluate_ready_data_dir / "interactions_train.parquet")
    pois_df = pd.read_parquet(evaluate_ready_data_dir / "pois_prepared.parquet")
    artifacts = train_lambdamart_systems(
        train_frame, interactions_train, pois_df, fast_model_cfg.lambdamart, fast_model_cfg.seed
    )
    artifacts_dir = tmp_path_factory.mktemp("scoring_artifacts")
    save_boosters(artifacts, artifacts_dir)
    # `poi_rank.cli candidates` persists the full-train retriever (scenarios score unseen trips
    # with it).
    if generated_candidates["retriever"] is not None:
        generated_candidates["retriever"].save(artifacts_dir)
    # `poi_rank.cli train` also fits + persists the confidence ensemble; scoring loads it.
    seeds = ScoringConfig.from_yaml(SCORING_CONFIG_PATH).confidence.ensemble_seeds
    save_ensemble(
        train_ensemble_boosters(
            train_frame, interactions_train, pois_df, fast_model_cfg.lambdamart, seeds
        ),
        seeds,
        artifacts_dir,
    )
    return artifacts_dir


@pytest.fixture(scope="session")
def scoring_pipeline_result(
    evaluate_ready_data_dir: Path,
    trained_scoring_artifacts_dir: Path,
    feature_build_cfg: FeatureBuildConfig,
    fast_model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
) -> dict[str, Any]:
    """One full `run_scoring_pipeline` run (spec.md section 9) over the real
    fixture-chain data's holdout trips, shared read-only across every test that
    needs real scoring output -- mirrors `evaluate_ready_data_dir`'s own
    session-scope sharing convention."""
    return run_scoring_pipeline(
        evaluate_ready_data_dir,
        trained_scoring_artifacts_dir,
        feature_build_cfg,
        fast_model_cfg,
        scoring_cfg,
        candidates_cfg.geo,
        candidates_cfg.longtail.pop_pct_cutoff,
    )


# -----------------------------------------------------------------------------------
# Phase 7 (explain/): grouped TreeSHAP + enrichment, built on top of the SAME
# session-scoped `scoring_pipeline_result`/`trained_scoring_artifacts_dir` fixtures
# above -- mirrors their own "real fixture-chain data, computed once per session"
# convention.
# -----------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def explain_ips_booster(trained_scoring_artifacts_dir: Path) -> Any:
    """The system-8 (LambdaMART+IPS) booster, loaded exactly as `scoring/output.py
    ::run_scoring_pipeline` loads it -- never retrained here."""
    return load_boosters(trained_scoring_artifacts_dir)["lambdamart_ips"]


@pytest.fixture(scope="session")
def explain_grouped_shap(
    scoring_pipeline_result: dict[str, Any], explain_ips_booster: Any
) -> GroupedShapResult:
    """Grouped TreeSHAP over `scoring_pipeline_result["full_frame"]` (every holdout
    candidate this session's fixture chain scored), computed once and shared
    read-only across every `explain/` test that needs it."""
    full = scoring_pipeline_result["full_frame"]
    numeric_columns = numeric_feature_columns(full)
    categorical_columns = categorical_feature_columns(full)
    return compute_grouped_shap(explain_ips_booster, full, numeric_columns, categorical_columns)


@pytest.fixture(scope="session")
def scenarios_result_dir(
    evaluate_ready_data_dir: Path,
    trained_scoring_artifacts_dir: Path,
    feature_build_cfg: FeatureBuildConfig,
    fast_model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """One full `poi_rank.eval.scenarios.run_scenarios` run (spec.md section 15,
    the 3+1 required scenarios) over the real fixture-chain data, once per test
    session -- mirrors `trained_scoring_artifacts_dir`/`scoring_pipeline_result`'s
    own session-scope sharing convention. Returns the `results_dir` passed to
    `run_scenarios` (which contains `results_dir/scenarios/{1,2,3,4}.json` +
    `overlap_matrix.json` on return), not the summary dict, so both
    `tests/test_scenarios.py` and `tests/test_report.py` can independently
    re-read the same on-disk files `eval.report.load_scenarios` itself reads."""
    from poi_rank.eval.scenarios import run_scenarios

    results_dir = tmp_path_factory.mktemp("scenarios_results")
    run_scenarios(
        evaluate_ready_data_dir,
        trained_scoring_artifacts_dir,
        results_dir,
        feature_build_cfg,
        fast_model_cfg,
        scoring_cfg,
        candidates_cfg,
    )
    return results_dir


@pytest.fixture(scope="session")
def enriched_payload(
    scoring_pipeline_result: dict[str, Any],
    evaluate_ready_data_dir: Path,
    trained_scoring_artifacts_dir: Path,
    scoring_cfg: ScoringConfig,
    feature_build_cfg: FeatureBuildConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """The real, fully-enriched output payload (`explain/output_enrichment
    ::enrich_recommend_result`) built from the session's real scoring-pipeline
    result -- `top_signals`/`explanation` here are genuine content, not
    `scoring/output.py`'s placeholders."""
    figures_dir = tmp_path_factory.mktemp("explain_figures")
    return enrich_recommend_result(
        scoring_pipeline_result,
        evaluate_ready_data_dir,
        trained_scoring_artifacts_dir,
        figures_dir,
        scoring_cfg,
        feature_build_cfg,
    )


# --- test tiers ---------------------------------------------------------------------------------
# Any test whose fixture closure reaches the full-scale pipeline chain (prepare -> features ->
# candidates, built once per session and shared) is `slow`: it runs in the serial lane, because
# every xdist worker would otherwise rebuild that chain itself (N x the time, N x ~5 GB).
# `-m "not slow" -n 12` therefore runs only the light tests, in parallel.
_HEAVY_FIXTURES = frozenset({"prepared_data", "built_features", "generated_candidates"})
_SLOW_MODULES = frozenset({"test_determinism.py", "test_readme_quickstart.py"})


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        fixtures = set(getattr(item, "fixturenames", ()))
        if fixtures & _HEAVY_FIXTURES or item.path.name in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
