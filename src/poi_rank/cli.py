"""Typer CLI entry point for the poi-rank pipeline.

`make generate` runs `python -m poi_rank.cli generate`. Later phases (prepare,
features, candidates, train, evaluate, scenarios) are wired in as they're built.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.candidates.recall_metrics import (
    marginal_recall_per_channel,
    overall_and_longtail_recall,
)
from poi_rank.candidates.union import run_candidates
from poi_rank.data.config import FeaturesConfig
from poi_rank.data.prepare import run_prepare
from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate
from poi_rank.features.build import run_features
from poi_rank.features.config import FeatureBuildConfig

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
DEFAULT_FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"


@app.callback()
def main() -> None:
    """poi-rank pipeline CLI. Subcommands are added as each build phase lands."""


@app.command()
def generate(
    # typer's documented idiom requires the call in the default; B008 is a false
    # positive here (typer reads the Option() at import time, not per-call).
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to datagen.yaml"),  # noqa: B008
    output_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Output directory for data/synthetic/"
    ),
) -> None:
    """Run the full synthetic data-generation pipeline and print a summary report."""
    cfg = DatagenConfig.from_yaml(config_path)
    summary = run_generate(cfg, output_dir)

    counts = summary["counts"]
    typer.echo("=== poi-rank datagen: generation summary ===")
    for key, value in counts.items():
        typer.echo(f"  {key}: {value}")

    typer.echo("\n=== SHA256 of generated files ===")
    for name, digest in sorted(summary["sha256"].items()):
        typer.echo(f"  {name}: {digest}")


@app.command()
def prepare(
    config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
) -> None:
    """Run the data-prep pipeline (dedup, categories, shrinkage, popularity,
    localness, hours, imputation, geo) and write `pois_prepared.parquet`."""
    cfg = FeaturesConfig.from_yaml(config_path)
    summary = run_prepare(cfg, data_dir)

    typer.echo("=== poi-rank data prep: summary ===")
    typer.echo(f"  output: {summary['output_path']}")
    typer.echo(f"  n_input_pois: {summary['n_input_pois']}")
    typer.echo(f"  n_output_pois: {summary['n_output_pois']}")
    typer.echo(
        f"  n_merged_away: {summary['n_merged_away']} "
        f"(dedup_merge_rate={summary['dedup_merge_rate']:.4f})"
    )
    typer.echo(
        f"  n_other_category: {summary['n_other_category']} "
        f"(other_category_rate={summary['other_category_rate']:.4f})"
    )


@app.command()
def features(
    config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory for the committed artifacts/poi_emb.npy cache"
    ),
) -> None:
    """Run the Phase 3 feature-build pipeline (POI text embedding, POI feature
    table, traveler feature table) and write both feature parquet files."""
    cfg = FeatureBuildConfig.from_yaml(config_path)
    summary = run_features(cfg, data_dir, artifacts_dir)

    typer.echo("=== poi-rank features: summary ===")
    typer.echo(f"  text_embedding_method: {summary['text_embedding_method']}")
    typer.echo(f"  text_embedding_cache: {summary['text_embedding_cache']}")
    typer.echo(f"  poi_features: {summary['poi_features_path']}")
    typer.echo(f"    n_pois={summary['n_pois']}, n_columns={summary['n_poi_feature_columns']}")
    typer.echo(f"  traveler_features: {summary['traveler_features_path']}")
    typer.echo(
        f"    n_traveler_trip_rows={summary['n_traveler_trip_rows']}, "
        f"n_columns={summary['n_traveler_feature_columns']}"
    )


@app.command()
def candidates(
    config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
) -> None:
    """Run the Phase 4a candidate-generation pipeline (6 channels + quotas) and
    print candidate-set-size + candidate-recall@250 (overall, long-tail-stratum,
    per-channel marginal) summaries."""
    cfg = CandidatesConfig.from_yaml(config_path)
    summary = run_candidates(cfg, data_dir)

    typer.echo("=== poi-rank candidates: summary ===")
    typer.echo(f"  output: {summary['output_path']}")
    typer.echo(f"  n_trips={summary['n_trips']}, n_candidate_rows={summary['n_rows']}")
    typer.echo(
        f"  candidates_per_trip: mean={summary['mean_candidates_per_trip']:.1f} "
        f"median={summary['median_candidates_per_trip']:.1f} "
        f"min={summary['min_candidates_per_trip']} max={summary['max_candidates_per_trip']}"
    )

    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    candidates_df = pd.read_parquet(summary["output_path"])
    holdout_random = pd.read_parquet(data_dir / "interactions_holdout_random.parquet")

    recall = overall_and_longtail_recall(pois_df, candidates_df, holdout_random)
    typer.echo("\n=== candidate_recall@250 ===")
    for name, result in recall.items():
        typer.echo(
            f"  {name}: recall={result.recall_mean:.4f} "
            f"(n_trips_evaluated={result.n_trips_evaluated}, "
            f"n_trips_excluded_no_relevant={result.n_trips_excluded_no_relevant})"
        )

    typer.echo("\n=== per-channel marginal recall (leave-one-channel-out) ===")
    marginal = marginal_recall_per_channel(pois_df, candidates_df, holdout_random)
    for channel, stats in marginal.items():
        typer.echo(
            f"  {channel}: recall_without={stats['recall_without_channel']:.4f} "
            f"marginal={stats['marginal_recall']:.4f}"
        )


if __name__ == "__main__":
    app()
