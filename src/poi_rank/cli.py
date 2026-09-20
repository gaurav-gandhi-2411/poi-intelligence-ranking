"""Typer CLI entry point for the poi-rank pipeline.

`make generate` runs `python -m poi_rank.cli generate`. Later phases (prepare,
features, candidates, train, evaluate, scenarios) are wired in as they're built.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
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
from poi_rank.eval.audit import run_audit
from poi_rank.eval.cold_start import run_lodo
from poi_rank.eval.compose import PARTS_DIRNAME, compose_metrics, write_part
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.decision_register import (
    DR_CATALOG,
    compose_register,
    load_lab,
    run_dr_scoring,
)
from poi_rank.eval.decomposition import run_decomposition
from poi_rank.eval.demo import (
    DemoInputError,
    build_profile,
    format_recommendations,
    recommend_for_profile,
    recommend_for_traveler,
)
from poi_rank.eval.dgp_diagnostics import run_dgp_diagnostics
from poi_rank.eval.dr_experiments import (
    run_dr2,
    run_dr3,
    run_dr4,
    run_dr7,
    run_dr8,
    run_dr9,
    run_dr11,
)
from poi_rank.eval.gate_dgp import run_gate_dgp
from poi_rank.eval.gate_representation import run_gate_representation
from poi_rank.eval.longtail_stages import run_longtail_stages
from poi_rank.eval.ranker_sweep import run_sweep
from poi_rank.eval.representation import run_representation_report
from poi_rank.eval.run import ALL_SYSTEM_NAMES, WILCOXON_METRIC, run_evaluate
from poi_rank.eval.scenarios import run_scenarios
from poi_rank.explain.output_enrichment import build_payload_enricher
from poi_rank.features.build import run_features
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.config import ModelConfig
from poi_rank.models.lambdamart import run_train_lambdamart
from poi_rank.scoring.confidence import run_train_ensemble
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import run_recommend, run_scoring_pipeline

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
DEFAULT_FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "eval.yaml"
DEFAULT_SCORING_CONFIG_PATH = REPO_ROOT / "configs" / "scoring.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
# `recommend` output is for human inspection (sample of served slates + explanations), so it
# is capped; `evaluate` -- the metric source -- always runs the full holdout.
RECOMMEND_MAX_TRIPS = 300
SAMPLE_SEED = 42


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


@app.command(name="diagnose-dgp")
def diagnose_dgp(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to datagen.yaml"),  # noqa: B008
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/parts/dgp_diagnostics.json"
    ),
) -> None:
    """Run the DIAGNOSTIC-ONLY DGP harness (spec-v2-remediation.md section 1,
    D1-D10): measures the falsifiable claim in section 0 (evaluation-set mismatch,
    utility-term scale imbalance, cold-start share) plus D9 (semantic-space
    fidelity, TF-IDF vs one-time MiniLM measurement) and D10 (is POI text
    generation conditioned on `poi_semantic`) WITHOUT changing any datagen/
    features/models/candidates/scoring code. Writes
    `results/parts/dgp_diagnostics.json`."""
    datagen_cfg = DatagenConfig.from_yaml(config_path)
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)

    result = run_dgp_diagnostics(data_dir, results_dir, datagen_cfg, feature_cfg)
    payload = result["payload"]

    typer.echo("=== poi-rank diagnose-dgp: DGP diagnostic harness (measurement only) ===")
    typer.echo(f"  output: {result['output_path']}")
    typer.echo(f"  wall_clock_seconds: {payload['meta']['wall_clock_seconds']:.2f}")

    d1 = payload["D1_variance_decomposition"]
    typer.echo("\n=== D1: variance decomposition of u(t,p) ===")
    typer.echo(
        f"  n_pairs_total={d1['n_pairs_total']} "
        f"n_valid={d1['n_pairs_valid_for_price_and_novelty']}"
    )
    for name, share in d1["term_variance_shares"].items():
        typer.echo(f"  share[{name}]={share:.4f}")
    typer.echo(f"  share[epsilon]={d1['epsilon_variance_share']:.4f}")
    typer.echo(f"  share_sum_all_8={d1['share_sum_all_8']:.4f}")

    d2 = payload["D2_taste_cosine_distribution"]
    typer.echo("\n=== D2: cos(taste, poi_semantic) distribution ===")
    typer.echo(
        f"  mean={d2['mean']:.4f} sd={d2['sd']:.4f} p5={d2['p5']:.4f} "
        f"p95={d2['p95']:.4f} n={d2['n']}"
    )

    d3 = payload["D3_spearman_utility_vs_label"]
    typer.echo("\n=== D3: Spearman(u, label) ===")
    typer.echo(f"  rho={d3['spearman_rho']:.4f} p={d3['p_value']:.4g} n={d3['n']}")

    d4 = payload["D4_choice_sharpness"]
    typer.echo("\n=== D4: choice sharpness ===")
    typer.echo(
        f"  mean_global_percentile={d4['mean_global_percentile']:.2f} "
        f"mean_within_slate_rank={d4['mean_within_slate_rank']:.2f} "
        f"(n_slates_included={d4['n_slates_included']}, "
        f"n_excluded={d4['n_slates_excluded_no_engagement']})"
    )

    d5 = payload["D5_ndcg"]
    typer.echo("\n=== D5: oracle NDCG@10, slate-level vs candidate-level ===")
    typer.echo(
        f"  slate_level={d5['slate_level']['mean_ndcg_at_10']:.4f} "
        f"(n={d5['slate_level']['n_slates_included']})"
    )
    typer.echo(
        f"  candidate_level={d5['candidate_level']['mean_ndcg_at_10']:.4f} "
        f"(n={d5['candidate_level']['n_trips_included']})"
    )

    d6 = payload["D6_cold_start_share"]
    typer.echo("\n=== D6: cold-start share ===")
    typer.echo(
        f"  overall={d6['overall']['share']:.4f} "
        f"({d6['overall']['n_cold_start']}/{d6['overall']['n_trips']}) "
        f"holdout_only={d6['holdout_only']['share']:.4f} "
        f"({d6['holdout_only']['n_cold_start']}/{d6['holdout_only']['n_trips']})"
    )

    d7 = payload["D7_localness_vs_geo_generation"]
    typer.echo("\n=== D7: Spearman(latent_localness, dist_to_tourist_centroid_km) ===")
    typer.echo(f"  rho={d7['spearman_rho']:.4f} p={d7['p_value']:.4g} n={d7['n']}")

    d8 = payload["D8_bias_gap_popularity"]
    typer.echo("\n=== D8: popularity bias gap ===")
    typer.echo(
        f"  gap={d8['gap']:.4f} (biased={d8['ndcg@10_biased_logged_holdout']:.4f}, "
        f"unbiased={d8['ndcg@10_unbiased_random_holdout']:.4f})"
    )
    if "existing_metrics_json_gap" in d8:
        typer.echo(f"  existing results/metrics.json gap={d8['existing_metrics_json_gap']:.4f}")

    d9 = payload["D9_semantic_fidelity"]
    typer.echo("\n=== D9: semantic-space fidelity (observable vs DGP latent) ===")
    for path_name in ("tfidf_path", "minilm_path"):
        p = d9[path_name]
        extra = (
            f" encode_wall_clock_seconds={p['encode_wall_clock_seconds']:.2f}"
            if "encode_wall_clock_seconds" in p
            else ""
        )
        typer.echo(
            f"  {path_name}: rho={p['spearman_rho']:.4f} p={p['p_value']:.4g} n={p['n']}{extra}"
        )

    d10 = payload["D10_description_conditioning"]
    typer.echo("\n=== D10: is POI text generation conditioned on poi_semantic? ===")
    for variant in ("raw_tfidf", "canonical_svd64"):
        v = d10[variant]
        typer.echo(
            f"  {variant}: rho={v['spearman_rho']:.4f} p={v['p_value']:.4g} "
            f"n_pairs_sampled={v['n_pairs_sampled']} seed={v['seed']}"
        )
    dep = d10["text_dependency_check"]
    typer.echo(
        f"  text_dependency_check: fraction_changed_with_permuted_semantic="
        f"{dep['fraction_changed_with_permuted_semantic']:.4f} "
        f"(control {dep['fraction_changed_same_semantic_control']:.4f}, n={dep['n_pois']})"
    )


@app.command(name="gate-dgp")
def gate_dgp(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to datagen.yaml"),  # noqa: B008
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/parts/dgp_gate.json"
    ),
) -> None:
    """DGP acceptance gate (docs/DATA_CARD.md "DGP remediation, Block A"): reuses
    `diagnose-dgp`'s D1-D10 computation and applies the 8 Gate-A thresholds, writing
    `results/parts/dgp_gate.json`. Exits non-zero if any threshold fails. Requires
    `generate` -> `prepare` -> `features` to have already run (see
    `eval/gate_dgp.py::run_gate_dgp`'s docstring for the documented precondition
    tension against the "runs before prepare" framing)."""
    datagen_cfg = DatagenConfig.from_yaml(config_path)
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)

    result = run_gate_dgp(data_dir, results_dir, datagen_cfg, feature_cfg)
    payload = result["payload"]

    typer.echo("=== poi-rank gate-dgp: DGP acceptance gate ===")
    typer.echo(f"  output: {result['output_path']}")
    for name, check in payload["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        typer.echo(
            f"  [{status}] {name}: measured={check['measured']:.4f} "
            f"{check['op']} {check['threshold']}"
        )
    typer.echo(
        f"\n  overall: {'PASS' if payload['overall_pass'] else 'FAIL'} "
        f"({payload['n_passed']}/{payload['n_total']})"
    )
    if not payload["overall_pass"]:
        raise typer.Exit(code=1)


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
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory the fitted retriever is saved to"
    ),
) -> None:
    """Run candidate generation (learned retriever + long-tail floor + interest; the legacy
    geo/semantic/CF/archetype channels have quota 0) and
    print candidate-set-size + candidate-recall@250 (overall, long-tail-stratum,
    per-channel marginal) summaries."""
    cfg = CandidatesConfig.from_yaml(config_path)
    budget = FeatureBuildConfig.from_yaml(config_path).traveler_features.budget_target_price_level
    summary = run_candidates(cfg, data_dir, budget, artifacts_dir)

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
    scoring_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_SCORING_CONFIG_PATH, help="Path to scoring.yaml (confidence-ensemble seeds)"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/parts/train.json"
    ),
) -> None:
    """Fit systems 7 (LambdaMART) and 8 (LambdaMART+IPS, primary) plus the
    dropout-ablation-only booster (`models.lambdamart`) on the TRAIN ranking frame,
    and persist all 3 as LightGBM native `.txt` boosters under `artifacts/`
    (spec.md section 14). `poi_rank.cli evaluate` loads these, never retrains."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)

    summary = run_train_lambdamart(data_dir, artifacts_dir, model_cfg, feature_cfg)
    write_part(results_dir, "train", summary["diagnostics"])
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)
    ensemble_paths = run_train_ensemble(
        data_dir,
        artifacts_dir,
        model_cfg,
        feature_cfg,
        scoring_cfg,
        num_boost_round=summary["diagnostics"]["best_iteration_lambdamart_ips"],
    )
    summary["paths"].update({p.stem: p for p in ensemble_paths})
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
    scoring_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_SCORING_CONFIG_PATH, help="Path to scoring.yaml"
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
    cohort robustness, bias-gap table, personalization, coverage, long-tail,
    constraint compatibility, diversity, calibration, confidence-decile validation,
    cold-start cohorts, and the 9-row ablation table) on the PRIMARY unbiased
    holdout, and write `results/metrics.json`."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    eval_cfg = EvalConfig.from_yaml(eval_config_path)
    candidates_cfg = CandidatesConfig.from_yaml(features_config_path)
    datagen_cfg = DatagenConfig.from_yaml(datagen_config_path)
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)

    summary = run_evaluate(
        data_dir,
        results_dir,
        artifacts_dir,
        model_cfg,
        eval_cfg,
        feature_cfg,
        candidates_cfg,
        datagen_cfg,
        scoring_cfg,
    )
    payload = summary["payload"]
    metrics_path = compose_metrics(results_dir)

    typer.echo("=== poi-rank evaluate: summary ===")
    typer.echo(f"  output: {summary['output_path']} (composed into {metrics_path})")
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

    pers = payload["personalization"]
    arch = pers["archetype"]
    typer.echo("\n=== personalization (spec.md section 11.2) ===")
    typer.echo(
        f"  mean_pairwise_jaccard@10={pers['mean_pairwise_jaccard_at_10']:.4f} "
        f"mean_pairwise_rbo={pers['mean_pairwise_rbo']:.4f}"
    )
    typer.echo(
        f"  within_archetype_jaccard={arch['within_archetype_jaccard_mean']:.4f} "
        f"cross_archetype_jaccard={arch['cross_archetype_jaccard_mean']:.4f} "
        f"ratio={arch['within_cross_ratio']}"
    )

    cov = payload["coverage"]
    cov_primary = cov["primary_system"]
    cov_pop = cov["popularity_baseline"]
    typer.echo("\n=== coverage (spec.md section 11.3) ===")
    typer.echo(
        f"  primary: coverage@10={cov_primary['catalog_coverage_at_10']:.4f} "
        f"gini={cov_primary['gini']:.4f} entropy_bits={cov_primary['entropy_bits']:.4f}"
    )
    typer.echo(
        f"  popularity_baseline: coverage@10={cov_pop['catalog_coverage_at_10']:.4f} "
        f"gini={cov_pop['gini']:.4f}"
    )

    lg = payload["longtail"]
    typer.echo("\n=== long-tail (spec.md section 11.4) ===")
    typer.echo(f"  share={lg['share']:.4f} precision={lg['precision']}")

    cc = payload["constraint_compatibility"]
    typer.echo("\n=== constraint compatibility (spec.md section 11.5) ===")
    typer.echo(f"  share_above_target={cc['share_above_target']:.4f}")

    dv2 = payload["diversity"]
    typer.echo("\n=== diversity (spec.md section 11.6) ===")
    typer.echo(
        f"  category_entropy_at_10_bits={dv2['category_entropy_at_10_bits']:.4f} "
        f"intra_list_mean_distance={dv2['intra_list_mean_distance']:.4f}"
    )

    typer.echo("\n=== ablations (spec.md section 11.9) ===")
    for row in payload["ablations"]:
        typer.echo(f"  {row['ablation']}: delta_ndcg@10={row['delta_ndcg@10']}")

    typer.echo(f"\n  {payload['meta']['note']}")


@app.command()
def lodo(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    eval_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_EVAL_CONFIG_PATH, help="Path to eval.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory containing artifacts/model.txt (run `train` first)"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory containing results/metrics.json (run `evaluate` first)"
    ),
) -> None:
    """Leave-one-destination-out (spec.md section 11.8): train on 2 destinations,
    evaluate on the 3rd, for all 3 destinations -- 3 full LightGBM retrains,
    genuinely expensive, NOT part of `make reproduce`'s default chain
    (`eval/cold_start.py`'s module docstring). Requires `poi_rank.cli evaluate` to
    have already run: merges a `lodo` key into the existing `results/metrics.json`
    rather than writing a separate file."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    eval_cfg = EvalConfig.from_yaml(eval_config_path)

    if not (results_dir / PARTS_DIRNAME / "evaluate.json").exists():
        typer.echo(
            "ERROR: results/parts/evaluate.json missing -- run `poi_rank.cli evaluate` first."
        )
        raise typer.Exit(code=1)

    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    destinations = tuple(sorted(trips_df["destination"].unique().tolist()))

    result = run_lodo(
        data_dir,
        artifacts_dir,
        model_cfg,
        feature_cfg,
        eval_cfg.seed,
        eval_cfg.bootstrap.n_resamples,
        eval_cfg.bootstrap.ci_low_pct,
        eval_cfg.bootstrap.ci_high_pct,
        destinations,
    )

    write_part(results_dir, "lodo", result)
    metrics_path = compose_metrics(results_dir)

    typer.echo("=== poi-rank lodo: summary ===")
    typer.echo(f"  wall_clock_seconds={result['wall_clock_seconds']:.1f}")
    for row in result["per_destination"]:
        typer.echo(
            f"  {row['destination']}: lodo_ndcg@10={row['ndcg@10_lodo']['mean']:.4f} "
            f"full_training_ndcg@10={row['ndcg@10_full_training']['mean']:.4f} "
            f"p={row['wilcoxon_lodo_vs_full_training']['p_value']:.4g}"
        )
    typer.echo(f"  part written; composed into: {metrics_path}")


@app.command()
def recommend(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    scoring_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_SCORING_CONFIG_PATH, help="Path to scoring.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory containing artifacts/model.txt (run `train` first)"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/recommendations.json + figures/"
    ),
    trip_id: str | None = typer.Option(  # noqa: B008
        None, help="Restrict to a single trip_id (default: a seeded sample of holdout trips)"
    ),
    max_trips: int = typer.Option(  # noqa: B008
        RECOMMEND_MAX_TRIPS,
        help="Cap on holdout trips explained (0 = all). `recommend` writes an inspection "
        "artifact, not a metric source -- `evaluate` scores the FULL holdout.",
    ),
) -> None:
    """Run the full scoring pipeline (spec.md section 9: compatibility + hard gates,
    isotonic calibration, confidence, MMR diversity) over the primary unbiased
    holdout trips, and write `results/recommendations.json` + `artifacts/
    calibrator.pkl` + `results/figures/{calibration_reliability,mmr_lambda_sweep}
    .png`. Requires `poi_rank.cli train` to have already written `artifacts/
    model.txt` (system 8, LOADED here, never retrained)."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)
    candidates_cfg = CandidatesConfig.from_yaml(features_config_path)

    trip_id_filter = {trip_id} if trip_id else None
    if trip_id_filter is None and max_trips > 0:
        trips = pd.read_parquet(data_dir / "trips.parquet")
        holdout_ids = sorted(trips.loc[trips["is_holdout"], "trip_id"].astype(str))
        if len(holdout_ids) > max_trips:
            picked = np.random.default_rng(SAMPLE_SEED).choice(
                holdout_ids, size=max_trips, replace=False
            )
            trip_id_filter = {str(t) for t in picked}
    payload_enricher = build_payload_enricher(
        data_dir, artifacts_dir, results_dir / "figures", scoring_cfg, feature_cfg
    )
    summary = run_recommend(
        data_dir,
        artifacts_dir,
        results_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        candidates_cfg.geo,
        candidates_cfg.longtail.pop_pct_cutoff,
        trip_id_filter=trip_id_filter,
        payload_enricher=payload_enricher,
    )

    typer.echo("=== poi-rank recommend: summary ===")
    typer.echo(f"  recommendations: {summary['recommendations_path']}")
    typer.echo(f"  calibrator: {summary['calibrator_path']}")
    typer.echo(
        f"  n_holdout_trips={summary['n_holdout_trips']} n_trips_output={summary['n_trips_output']}"
    )

    c = summary["calibration"]
    typer.echo("\n=== calibration (ECE 15-bin / Brier, before vs after isotonic) ===")
    typer.echo(f"  ECE:   before={c['ece_before']:.4f}  after={c['ece_after']:.4f}")
    typer.echo(f"  Brier: before={c['brier_before']:.4f}  after={c['brier_after']:.4f}")
    typer.echo(
        f"  calibration split: n_rows={c['n_calibration_rows']} n_trips={c['n_calibration_trips']}"
    )

    typer.echo("\n=== utility beta sensitivity (NDCG@10) ===")
    for row in summary["beta_sensitivity"]:
        typer.echo(f"  beta={row['beta']:.2f}: ndcg@10={row['ndcg@10_mean']:.4f}")

    dv = summary["confidence_decile_validation"]
    typer.echo("\n=== confidence-decile validation ===")
    typer.echo(
        f"  spearman_rho={dv['spearman_rho']} target_met(>=0.7)={dv['target_met']} "
        f"n_trips_included={dv['n_trips_included']}"
    )

    typer.echo("\n=== MMR lambda sweep (NDCG@10 vs mean intra-list similarity) ===")
    for row in summary["lambda_sweep"]:
        typer.echo(
            f"  lambda={row['lambda']:.2f}: ndcg@10={row['ndcg@10_mean']:.4f} "
            f"mean_intra_list_similarity={row['mean_intra_list_similarity']:.4f}"
        )


@app.command()
def scenarios(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    scoring_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_SCORING_CONFIG_PATH, help="Path to scoring.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory containing artifacts/model.txt (run `train` first)"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/scenarios/*.json"
    ),
) -> None:
    """Run the 3+1 required scenarios (spec.md section 15): 4 hand-specified,
    synthetic traveler/trip profiles run through the exact same live pipeline
    `recommend` runs for real holdout trips, writing `results/scenarios/{1,2,3,4}
    .json` + `results/scenarios/overlap_matrix.json`. Requires `poi_rank.cli
    train` to have already written `artifacts/model.txt` (LOADED here, never
    retrained)."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)
    candidates_cfg = CandidatesConfig.from_yaml(features_config_path)

    summary = run_scenarios(
        data_dir, artifacts_dir, results_dir, feature_cfg, model_cfg, scoring_cfg, candidates_cfg
    )

    typer.echo("=== poi-rank scenarios: summary ===")
    for number, path in sorted(summary["scenario_paths"].items()):
        top10 = summary["top10_by_scenario"][number]
        typer.echo(f"  scenario {number}: {path} ({len(top10)} recommendations)")

    overlap = summary["overlap_matrix"]
    typer.echo("\n=== pairwise top-10 Jaccard overlap ===")
    for key, value in sorted(overlap["pairwise_jaccard_top10"].items()):
        typer.echo(f"  {key}: {value:.4f}")
    typer.echo(
        f"\n  scenario {overlap['diagnostic_scenario']} vs base "
        f"{overlap['diagnostic_base_scenario']} (touristiness_pref flip): "
        f"{overlap['diagnostic_vs_base_jaccard']:.4f}"
    )
    typer.echo(f"  overlap matrix: {summary['overlap_matrix_path']}")


@app.command()
def compose(
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory holding results/parts/*.json"
    ),
) -> None:
    """Merge every `results/parts/<stage>.json` into `results/metrics.json` -- the only writer
    of that file (`eval/compose.py`). `evaluate`/`lodo` call it after writing their part, so
    metrics.json is always current; run it directly after adding or refreshing any other part."""
    path = compose_metrics(results_dir)
    typer.echo(f"composed: {path}")


@app.command()
def representation(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/parts/representation.json"
    ),
) -> None:
    """D11 / within-trip D9 / ablations / oracle-relevance recall (REPORTING ONLY -- computed
    after representation choices are frozen; `eval/representation.py`)."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    payload = run_representation_report(data_dir, feature_cfg)
    path = write_part(results_dir, "representation", payload)
    typer.echo(f"=== poi-rank representation: {path} ===")
    typer.echo(f"  D11 ridge R2 (text-only): {payload['d11_text_only']['ridge_r2_oof']:.4f}")
    for name, d in payload["d9_within_trip"].items():
        typer.echo(f"  {name}: within-trip Spearman={d['mean_within_trip_spearman']:.4f}")
    orr = payload["oracle_relevance_recall"]
    typer.echo(
        f"  oracle-relevance recall: overall={orr['overall']:.4f} long_tail={orr['long_tail']:.4f}"
    )


@app.command(name="gate-representation")
def gate_representation(
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory holding results/parts/{evaluate,representation}.json"
    ),
) -> None:
    """Gate-B (blocking, oracle-free candidate-recall rows; D9/D11/oracle rows reported only).
    Exits non-zero if any blocking row fails."""
    result = run_gate_representation(results_dir)
    payload = result["payload"]
    typer.echo("=== poi-rank gate-representation: Gate-B ===")
    for name, check in payload["blocking_checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        typer.echo(
            f"  [{status}] {name}: {check['measured']:.4f} {check['op']} {check['threshold']}"
        )
    for name, value in payload["reporting_only_after_freeze"].items():
        if isinstance(value, float):
            typer.echo(f"  [report] {name}: {value:.4f}")
    typer.echo(f"\n  overall: {'PASS' if payload['overall_pass'] else 'FAIL'}")
    if not payload["overall_pass"]:
        raise typer.Exit(code=1)


@app.command()
def audit(
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory holding results/parts + metrics.json"
    ),
    deep: bool = typer.Option(  # noqa: B008
        False, help="Also regenerate the dataset twice and compare SHA-256 manifests (~1 min)"
    ),
) -> None:
    """Deterministic invariant audit (`eval/audit.py`); prints JSON and exits non-zero on any
    failed check. Replaces the verifier subagent: a script cannot hallucinate a PASS."""
    payload = run_audit(results_dir, deep=deep)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise typer.Exit(code=1)


@app.command()
def dr(
    which: str = typer.Option("all", help="Comma-separated DR ids (DR1,DR2,...) or 'all'"),  # noqa: B008
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    eval_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_EVAL_CONFIG_PATH, help="Path to eval.yaml"
    ),
    scoring_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_SCORING_CONFIG_PATH, help="Path to scoring.yaml"
    ),
    data_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Directory containing data/synthetic/*.parquet"
    ),
    artifacts_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_ARTIFACTS_DIR, help="Directory containing trained artifacts"
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_RESULTS_DIR, help="Directory to write results/parts/dr/*.json"
    ),
) -> None:
    """Run Decision-Register experiments (eval/decision_register.py, eval/dr_experiments.py),
    then compose results/parts/decision_register.json (missing rows become explicit NOT RUN)."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    eval_cfg = EvalConfig.from_yaml(eval_config_path)
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)
    candidates_cfg = CandidatesConfig.from_yaml(features_config_path)
    wanted = (
        set(DR_CATALOG) if which == "all" else {w.strip().upper() for w in which.split(",") if w}
    )

    lab = load_lab(data_dir, feature_cfg, model_cfg, eval_cfg)
    if wanted & {"DR1", "DR6", "DR10"}:
        run_dr_scoring(
            data_dir,
            artifacts_dir,
            results_dir,
            feature_cfg,
            model_cfg,
            scoring_cfg,
            candidates_cfg,
        )
    if "DR3" in wanted:
        run_dr3(lab, results_dir)
    if "DR7" in wanted:
        run_dr7(lab, results_dir)
    if "DR9" in wanted:
        run_dr9(data_dir, artifacts_dir, results_dir, feature_cfg, candidates_cfg, lab)
    if "DR11" in wanted:
        run_dr11(results_dir)
    if "DR8" in wanted:
        run_dr8(data_dir, results_dir, feature_cfg, lab)
    if "DR4" in wanted:
        run_dr4(data_dir, results_dir, feature_cfg, lab)
    if "DR2" in wanted:
        if "torch" in sys.modules:
            run_dr2(lab, results_dir)
        else:
            # torch must be imported BEFORE pandas/pyarrow in a process on this Windows box
            # (WinError 1114 on torch's c10.dll otherwise -- same constraint as the MiniLM
            # subprocess in eval/dgp_diagnostics.py), and pandas is already loaded here, so
            # DR2 runs in a fresh interpreter whose first import is torch.
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-c",
                    "import torch\nfrom poi_rank.cli import app\napp()",
                    "dr",
                    "--which",
                    "DR2",
                    "--data-dir",
                    str(data_dir),
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )
    path = compose_register(results_dir)
    compose_metrics(results_dir)
    typer.echo(f"decision register: {path}")


@app.command()
def decompose(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    legacy_config_path: Path = typer.Option(  # noqa: B008
        REPO_ROOT / "configs" / "features_legacy6.yaml", help="Pre-retriever 6-channel config"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    eval_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_EVAL_CONFIG_PATH, help="Path to eval.yaml"
    ),
    data_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="data/synthetic"),  # noqa: B008
    artifacts_dir: Path = typer.Option(DEFAULT_ARTIFACTS_DIR, help="artifacts"),  # noqa: B008
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, help="results"),  # noqa: B008
) -> None:
    """E1: retrieval-vs-ranking gain decomposition over both candidate sets."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    lab = load_lab(
        data_dir,
        feature_cfg,
        ModelConfig.from_yaml(model_config_path),
        EvalConfig.from_yaml(eval_config_path),
    )
    payload = run_decomposition(
        data_dir,
        artifacts_dir,
        results_dir,
        feature_cfg,
        CandidatesConfig.from_yaml(legacy_config_path),
        lab,
        lab.eval_cfg,
    )
    compose_metrics(results_dir)
    typer.echo(
        json.dumps(payload["decomposition_of_popularity_to_primary_gain_end_to_end"], indent=1)
    )


@app.command()
def sweep(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    eval_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_EVAL_CONFIG_PATH, help="Path to eval.yaml"
    ),
    data_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="data/synthetic"),  # noqa: B008
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, help="results"),  # noqa: B008
) -> None:
    """E2: validation-only joint sweep over ranker objective / IPS clip / feature blocks."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    lab = load_lab(
        data_dir,
        feature_cfg,
        ModelConfig.from_yaml(model_config_path),
        EvalConfig.from_yaml(eval_config_path),
    )
    payload = run_sweep(lab, results_dir)
    typer.echo(json.dumps(payload["winner"], indent=1))


@app.command(name="longtail-stages")
def longtail_stages(
    features_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_FEATURES_CONFIG_PATH, help="Path to features.yaml"
    ),
    model_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MODEL_CONFIG_PATH, help="Path to model.yaml"
    ),
    eval_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_EVAL_CONFIG_PATH, help="Path to eval.yaml"
    ),
    scoring_config_path: Path = typer.Option(  # noqa: B008
        DEFAULT_SCORING_CONFIG_PATH, help="Path to scoring.yaml"
    ),
    data_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="data/synthetic"),  # noqa: B008
    artifacts_dir: Path = typer.Option(DEFAULT_ARTIFACTS_DIR, help="artifacts"),  # noqa: B008
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, help="results"),  # noqa: B008
) -> None:
    """E3: long-tail share/precision after each serving stage."""
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)
    cand_cfg = CandidatesConfig.from_yaml(features_config_path)
    eval_cfg = EvalConfig.from_yaml(eval_config_path)
    result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        cand_cfg.geo,
        cand_cfg.longtail.pop_pct_cutoff,
    )
    pois = pd.read_parquet(data_dir / "pois_prepared.parquet")
    payload = run_longtail_stages(
        result,
        pois,
        eval_cfg.long_tail_pop_pct_cutoff,
        scoring_cfg.diversity,
        scoring_cfg.diversity.lambda_default,
        scoring_cfg.output.top_k,
        results_dir,
    )
    compose_metrics(results_dir)
    typer.echo(json.dumps(payload["stages"], indent=1))


@app.command()
def demo(
    traveler: str = typer.Option(
        "", "--traveler", help="A traveler_id from the dataset (e.g. U0005)"
    ),
    interests: str = typer.Option("", help="Comma list, e.g. local_food,neighborhoods"),
    budget: str = typer.Option("medium", help="low | medium | high"),
    mobility: str = typer.Option("public_transport", help="walk | public_transport | car | mixed"),
    touristiness: float = typer.Option(0.0, help="Preference in [-1, 1]; -1 = avoid touristy"),
    party: str = typer.Option(
        "solo", help="solo | couple | family_young_kids | family_teens | friends"
    ),
    dest: str = typer.Option("seoul", help="Destination in the catalog"),
    features_config_path: Path = typer.Option(DEFAULT_FEATURES_CONFIG_PATH),  # noqa: B008
    model_config_path: Path = typer.Option(DEFAULT_MODEL_CONFIG_PATH),  # noqa: B008
    scoring_config_path: Path = typer.Option(DEFAULT_SCORING_CONFIG_PATH),  # noqa: B008
    data_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR),  # noqa: B008
    artifacts_dir: Path = typer.Option(DEFAULT_ARTIFACTS_DIR),  # noqa: B008
) -> None:
    """Live top-10 from the committed model artifacts: `--traveler U0005` (real trip) or a
    stated profile (`--interests`, `--budget`, `--mobility`, `--touristiness`, `--party`,
    `--dest`)."""
    warnings.simplefilter("ignore", FutureWarning)  # pandas dtype-downcast noise, not actionable
    feature_cfg = FeatureBuildConfig.from_yaml(features_config_path)
    model_cfg = ModelConfig.from_yaml(model_config_path)
    scoring_cfg = ScoringConfig.from_yaml(scoring_config_path)
    candidates_cfg = CandidatesConfig.from_yaml(features_config_path)
    try:
        if traveler:
            recs = recommend_for_traveler(
                traveler,
                data_dir,
                artifacts_dir,
                feature_cfg,
                model_cfg,
                scoring_cfg,
                candidates_cfg,
            )
        else:
            travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
            vocabulary = {label for xs in travelers_df["interests"] for label in xs}
            profile = build_profile(interests, budget, mobility, touristiness, party, vocabulary)
            recs = recommend_for_profile(
                profile,
                dest,
                data_dir,
                artifacts_dir,
                feature_cfg,
                model_cfg,
                scoring_cfg,
                candidates_cfg,
            )
    except DemoInputError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(format_recommendations(recs))


if __name__ == "__main__":
    app()
