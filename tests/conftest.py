"""Shared pytest fixtures for the datagen test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

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
