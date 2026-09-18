"""Two full-scale generator runs must produce byte-identical output files
(spec.md section 14). Run at real scale since it completes in ~8s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate

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
