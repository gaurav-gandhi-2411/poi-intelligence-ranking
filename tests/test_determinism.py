"""Two full-scale generator runs must produce byte-identical output files
(spec.md section 14). Run at real scale since it completes in ~8s.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate
from poi_rank.features.build import run_features
from poi_rank.features.config import FeatureBuildConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"


def test_two_runs_produce_byte_identical_files(tmp_path_factory: pytest.TempPathFactory) -> None:
    cfg = DatagenConfig.from_yaml(CONFIG_PATH)

    out1 = tmp_path_factory.mktemp("run1")
    out2 = tmp_path_factory.mktemp("run2")

    summary1 = run_generate(cfg, out1)
    summary2 = run_generate(cfg, out2)

    sha1 = summary1["sha256"]
    sha2 = summary2["sha256"]

    assert set(sha1.keys()) == set(sha2.keys())
    mismatches = [name for name in sha1 if sha1[name] != sha2[name]]
    assert not mismatches, f"non-deterministic output files: {mismatches}"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_two_feature_build_runs_produce_byte_identical_files(
    built_features: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Two independent `run_features` runs against the same prepared data (each with
    its own fresh, empty `artifacts/` cache dir -- proving determinism of a genuine
    recompute, not just of cache-reuse) must produce byte-identical
    `poi_features.parquet` / `traveler_features.parquet`."""
    cfg: FeatureBuildConfig = built_features["cfg"]
    data_dir: Path = built_features["output_dir"]

    artifacts1 = tmp_path_factory.mktemp("features_artifacts1")
    artifacts2 = tmp_path_factory.mktemp("features_artifacts2")

    # Two separate output dirs too: `run_features` writes both parquet files into
    # `data_dir` by convention, so reusing one `data_dir` for both runs would let the
    # second run's write silently shadow the first before either hash is taken.
    data_dir_1 = tmp_path_factory.mktemp("features_run1")
    data_dir_2 = tmp_path_factory.mktemp("features_run2")
    for name in (
        "pois_prepared.parquet",
        "travelers.parquet",
        "trips.parquet",
        "interactions_train.parquet",
    ):
        for dest in (data_dir_1, data_dir_2):
            (dest / name).write_bytes((data_dir / name).read_bytes())

    summary1 = run_features(cfg, data_dir_1, artifacts1)
    summary2 = run_features(cfg, data_dir_2, artifacts2)

    for key in ("poi_features_path", "traveler_features_path"):
        sha1 = _sha256_of_file(summary1[key])
        sha2 = _sha256_of_file(summary2[key])
        assert sha1 == sha2, f"non-deterministic feature output: {key}"
