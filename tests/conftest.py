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
