"""Shared pytest fixtures for the datagen test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"


@pytest.fixture(scope="session")
def datagen_cfg() -> DatagenConfig:
    return DatagenConfig.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def generated_data(
    tmp_path_factory: pytest.TempPathFactory, datagen_cfg: DatagenConfig
) -> dict[str, Any]:
    """Run the full-scale generator once per test session (~8s) into a tmp dir."""
    output_dir = tmp_path_factory.mktemp("synthetic")
    summary = run_generate(datagen_cfg, output_dir)
    return {"summary": summary, "output_dir": output_dir, "cfg": datagen_cfg}
