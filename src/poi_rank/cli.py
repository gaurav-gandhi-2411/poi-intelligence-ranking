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
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.run import ALL_SYSTEM_NAMES, WILCOXON_METRIC, run_evaluate
from poi_rank.features.build import run_features
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.config import ModelConfig
from poi_rank.models.lambdamart import run_train_lambdamart

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
DEFAULT_FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "eval.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


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


@app.command()
def train(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory to write artifacts/*.txt LightGBM boosters"
    ),
) -> None:
    """Fit systems 7 (LambdaMART) and 8 (LambdaMART+IPS, primary) plus the
    dropout-ablation-only booster (`models.lambdamart`) on the TRAIN ranking frame,
    and persist all 3 as LightGBM native `.txt` boosters under `artifacts/`
    (spec.md section 14). `poi_rank.cli evaluate` loads these, never retrains."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)

    summary = run_train_lambdamart(data_dir, artifacts_dir, model_cfg, feature_cfg)
    diagnostics = summary["diagnostics"]

    typer.echo("=== poi-rank train: lambdamart / lambdamart_ips ===")
    for name, path in summary["paths"].items():
        typer.echo(f"  {name}: {path}")
    typer.echo(
        f"  n_fit_rows={diagnostics['n_fit_rows']} n_val_rows={diagnostics['n_val_rows']} "
        f"n_fit_trips={diagnostics['n_fit_trips']} n_val_trips={diagnostics['n_val_trips']}"
    )
    typer.echo(
        f"  behavioral_dropout: n_rows={diagnostics['n_dropout_rows']} "
        f"actual_rate={diagnostics['dropout_rate_actual']:.4f}"
    )
    typer.echo(
        f"  ips: n_exposed_rows={diagnostics['n_ips_exposed_rows']} "
        f"exposure_rate={diagnostics['ips_exposure_rate']:.4f} "
        f"mean_weight={diagnostics['mean_ips_weight']:.4f}"
    )
    typer.echo(
        f"  best_iteration: lambdamart={diagnostics['best_iteration_lambdamart']} "
        f"lambdamart_ips={diagnostics['best_iteration_lambdamart_ips']} "
        f"lambdamart_ips_no_dropout={diagnostics['best_iteration_lambdamart_ips_no_dropout']}"
    )
    typer.echo(
        f"  val ndcg@10: lambdamart={diagnostics['best_score_ndcg10_lambdamart']:.4f} "
        f"lambdamart_ips={diagnostics['best_score_ndcg10_lambdamart_ips']:.4f}"
    )


@app.command()
def evaluate(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    eval_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_EVAL_CONFIG_PATH, help="Path to eval.yaml"
    ),
    datagen_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_CONFIG_PATH, help="Path to datagen.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory containing artifacts/*.txt LightGBM boosters"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/metrics.json"
    ),
) -> None:
    """Run the full evaluation harness (baselines 1-6 + LambdaMART systems 7/8,
    loaded from `artifacts/` -- run `poi_rank.cli train` first -- + oracle ceiling,
    full metric table with bootstrap 95% CIs, paired Wilcoxon comparisons, new-POI
    cohort robustness) on the PRIMARY unbiased holdout, and write
    `results/metrics.json`."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    eval_cfg = EvalConfig.from_yaml(eval_config_path)
    candidates_cfg = CandidatesConfig.from_yaml(features_config_path)
    datagen_cfg = DatagenConfig.from_yaml(datagen_config_path)

    summary = run_evaluate(
        data_dir,
        results_dir,
        artifacts_dir,
        model_cfg,
        eval_cfg,
        feature_cfg,
        candidates_cfg.geo,
        datagen_cfg,
    )
    payload = summary["payload"]

    typer.echo("=== poi-rank evaluate: summary ===")
    typer.echo(f"  output: {summary['output_path']}")
    typer.echo(f"  n_holdout_trips: {payload['meta']['n_holdout_trips']}")
    typer.echo(f"  bootstrap_n_resamples: {payload['meta']['bootstrap_n_resamples']}")

    typer.echo(f"\n=== {WILCOXON_METRIC} (mean, 95% CI, % of oracle ceiling) ===")
    for name in ALL_SYSTEM_NAMES:
        m = payload["systems"][name]["metrics"][WILCOXON_METRIC]
        pct = payload["systems"][name]["pct_of_ceiling_ndcg10"]
        typer.echo(
            f"  {name}: {m['mean']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}] "
            f"({pct * 100:.1f}% of ceiling, n_excluded={m['n_excluded']})"
        )

    typer.echo(f"\n=== paired Wilcoxon ({WILCOXON_METRIC}) ===")
    for key, w in payload["wilcoxon"].items():
        typer.echo(f"  {key}: p={w['p_value']:.4g} (n_pairs={w['n_pairs']})")

    cohort = payload["new_poi_cohort"]
    typer.echo("\n=== new-POI cohort (spec.md section 8: new-POI robustness) ===")
    typer.echo(
        f"  n_new_pois_in_catalog={cohort['n_new_pois_in_catalog']} "
        f"n_holdout_rows_in_cohort={cohort['n_holdout_rows_in_cohort']} "
        f"n_trips_with_relevant_cohort_candidate="
        f"{cohort['n_holdout_trips_with_relevant_cohort_candidate']}"
    )
    with_dropout = cohort["ndcg@10_lambdamart_ips_with_dropout"]
    without_dropout = cohort["ndcg@10_lambdamart_ips_no_dropout"]
    typer.echo(
        f"  ndcg@10 with dropout:    {with_dropout['mean']:.4f} "
        f"[{with_dropout['ci_low']:.4f}, {with_dropout['ci_high']:.4f}] "
        f"(n_included={with_dropout['n_included']})"
    )
    typer.echo(
        f"  ndcg@10 without dropout: {without_dropout['mean']:.4f} "
        f"[{without_dropout['ci_low']:.4f}, {without_dropout['ci_high']:.4f}] "
        f"(n_included={without_dropout['n_included']})"
    )
    dropout_w = cohort["wilcoxon_with_vs_without_dropout"]
    typer.echo(f"  wilcoxon with vs without dropout: p={dropout_w['p_value']:.4g}")

    typer.echo(f"\n  {payload['meta']['note']}")


if __name__ == "__main__":
    app()
