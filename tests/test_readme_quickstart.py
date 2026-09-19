"""Integration test for README.md's documented quick-start command sequence
(spec.md section 14's reproducibility contract) -- actually invokes the real
`poi_rank.cli` Typer app, in the documented order, against a temp sandbox, rather
than merely describing what should happen. This is a genuinely new check: every
other test in this suite calls the underlying pipeline functions directly
(`run_generate`, `run_evaluate`, etc., per `tests/conftest.py`'s own fixture
chain) -- none exercises the CLI wrapper layer itself (argument parsing, config
loading from a YAML path, writing to a caller-supplied directory), which is
exactly what a reviewer following README.md's quick-start actually runs.

Uses the REAL, full-scale `configs/datagen.yaml` / `configs/features.yaml` /
`configs/scoring.yaml` for `generate`/`prepare`/`features`/`candidates` (already
fast at this project's committed dataset scale -- `tests/conftest.py`'s own
`generated_data` fixture docstring: "~8s"; `docs/DATA_CARD.md`'s Phase 10 measured
table: candidates ~45s) and temporary FAST `model.yaml`/`eval.yaml` configs
(fewer LightGBM boosting rounds, fewer bootstrap resamples -- mirroring
`tests/conftest.py::fast_model_cfg` and `tests/test_report.py`'s own
reduced-bootstrap `EvalConfig`) for `train`/`evaluate`/`lodo`/`scenarios`, which
are the genuinely slow steps at production settings. Every command from
README.md's quick-start table is invoked, in the documented order, writing only
into a `tmp_path` sandbox -- never the repo's own committed `data/synthetic/`,
`artifacts/`, or `results/`."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from poi_rank.cli import app
from poi_rank.eval.report import run_report

REPO_ROOT = Path(__file__).resolve().parents[1]
DATAGEN_CONFIG = REPO_ROOT / "configs" / "datagen.yaml"
FEATURES_CONFIG = REPO_ROOT / "configs" / "features.yaml"
SCORING_CONFIG = REPO_ROOT / "configs" / "scoring.yaml"

runner = CliRunner()


def _write_fast_model_config(path: Path) -> None:
    """Mirrors `tests/conftest.py::fast_model_cfg`'s hyperparameters exactly
    (few boosting rounds, same recipe otherwise) so `poi_rank.cli train` finishes
    quickly in a test -- never the committed `configs/model.yaml`'s production
    settings (1000 max rounds), which is what makes the real `train` step ~25s
    even though this test's LightGBM configuration is much cheaper."""
    payload = {
        "seed": 42,
        "baselines": {
            "content_cosine": {"interest_weight": 0.7, "price_fit_weight": 0.3},
            "item_knn_cf": {"cf_score_missing_fallback": "popularity"},
            "logistic_regression": {"max_iter": 200, "C": 1.0},
        },
        "lambdamart": {
            "num_leaves": 15,
            "learning_rate": 0.1,
            "n_estimators": 30,
            "early_stopping_rounds": 10,
            "lambdarank_truncation_level": 20,
            "eval_ndcg_at": [10],
            "label_gain": [0, 1, 3, 7],
            "min_data_in_leaf": 5,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "num_threads": 1,
            "val_fraction": 0.15,
            "val_split_seed": 43,
            "ips_clip_low": 1.0,
            "ips_clip_high": 20.0,
            "behavioral_dropout_rate": 0.15,
            "behavioral_dropout_seed": 44,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_fast_eval_config(path: Path) -> None:
    """Mirrors `tests/test_report.py::fast_metrics_payload`'s reduced bootstrap
    settings (50 resamples, `ndcg_ks=(5, 10)`) -- the real `configs/eval.yaml`'s
    2000-resample table over 9 systems + 4 full ablation retrains is what makes
    the real `evaluate` step ~2m40s; this config keeps the same code path but at
    a cost cheap enough for a test."""
    payload = {
        "seed": 42,
        "metrics": {"ndcg_ks": [5, 10], "precision_ks": [5], "recall_ks": [10]},
        "bootstrap": {
            "n_resamples": 30,
            "ci_low_pct": 2.5,
            "ci_high_pct": 97.5,
            "resample_unit": "trip",
        },
        "long_tail_pop_pct_cutoff": 0.5,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_readme_quickstart_command_sequence_runs_end_to_end(tmp_path: Path) -> None:
    """Runs the exact 9-command sequence README.md documents as the primary,
    verified quick-start, via the real `poi_rank.cli` Typer app, entirely inside
    `tmp_path` -- asserts every step exits 0 and produces the file README.md
    claims it produces, in the documented order (each step's own inputs are the
    previous step's real on-disk outputs, not hand-built fixtures)."""
    data_dir = tmp_path / "data" / "synthetic"
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    fast_model_config = tmp_path / "model.yaml"
    fast_eval_config = tmp_path / "eval.yaml"
    _write_fast_model_config(fast_model_config)
    _write_fast_eval_config(fast_eval_config)

    # Step 1: generate
    result = runner.invoke(
        app,
        [
            "generate",
            "--config-path",
            str(DATAGEN_CONFIG),
            "--output-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (data_dir / "pois.parquet").exists()
    assert (data_dir / "trips.parquet").exists()

    # Step 2: prepare
    result = runner.invoke(
        app,
        ["prepare", "--config-path", str(FEATURES_CONFIG), "--data-dir", str(data_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (data_dir / "pois_prepared.parquet").exists()

    # Step 3: features
    result = runner.invoke(
        app,
        [
            "features",
            "--config-path",
            str(FEATURES_CONFIG),
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (data_dir / "poi_features.parquet").exists()
    assert (data_dir / "traveler_features.parquet").exists()

    # Step 4: candidates
    result = runner.invoke(
        app,
        ["candidates", "--config-path", str(FEATURES_CONFIG), "--data-dir", str(data_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (data_dir / "candidates.parquet").exists()

    # Step 5: train
    result = runner.invoke(
        app,
        [
            "train",
            "--features-config-path",
            str(FEATURES_CONFIG),
            "--model-config-path",
            str(fast_model_config),
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (artifacts_dir / "model.txt").exists()

    # Step 6: evaluate
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--features-config-path",
            str(FEATURES_CONFIG),
            "--model-config-path",
            str(fast_model_config),
            "--eval-config-path",
            str(fast_eval_config),
            "--datagen-config-path",
            str(DATAGEN_CONFIG),
            "--scoring-config-path",
            str(SCORING_CONFIG),
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--results-dir",
            str(results_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    metrics_path = results_dir / "metrics.json"
    assert metrics_path.exists()

    # Step 7: lodo (must run AFTER evaluate -- merges into the existing metrics.json)
    result = runner.invoke(
        app,
        [
            "lodo",
            "--features-config-path",
            str(FEATURES_CONFIG),
            "--model-config-path",
            str(fast_model_config),
            "--eval-config-path",
            str(fast_eval_config),
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--results-dir",
            str(results_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    # Step 8: scenarios
    result = runner.invoke(
        app,
        [
            "scenarios",
            "--features-config-path",
            str(FEATURES_CONFIG),
            "--model-config-path",
            str(fast_model_config),
            "--scoring-config-path",
            str(SCORING_CONFIG),
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--results-dir",
            str(results_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (results_dir / "scenarios" / "1.json").exists()
    assert (results_dir / "scenarios" / "overlap_matrix.json").exists()

    # Step 9: docs (poi_rank.eval.report -- README's `uv run python -m poi_rank.eval.report`)
    output_path = run_report(
        metrics_path=metrics_path,
        output_path=tmp_path / "docs" / "RESULTS.md",
        scenarios_dir=results_dir / "scenarios",
    )
    assert output_path.exists()
    rendered = output_path.read_text(encoding="utf-8")
    assert "# RESULTS.md" in rendered
    assert "## Success criteria" in rendered
    # lodo ran before this step, so the LODO section must be populated, not the
    # "not yet run" fallback (docs/DATA_CARD.md #75's documented behavior).
    assert "not yet run" not in rendered.lower()
