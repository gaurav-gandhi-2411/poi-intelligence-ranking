"""Shared pytest fixtures for the datagen test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.candidates.union import generate_candidates
from poi_rank.data.config import FeaturesConfig
from poi_rank.data.prepare import prepare_pois
from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate
from poi_rank.features.build import run_features
from poi_rank.features.config import FeatureBuildConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"


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

    candidates_df = generate_candidates(
        pois_df,
        travelers_df,
        trips_df,
        built_features["poi_features"],
        built_features["traveler_features"],
        interactions_train,
        candidates_cfg,
    )
    return {
        "candidates": candidates_df,
        "pois": pois_df,
        "travelers": travelers_df,
        "trips": trips_df,
        "interactions_train": interactions_train,
        "output_dir": output_dir,
        "cfg": candidates_cfg,
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
