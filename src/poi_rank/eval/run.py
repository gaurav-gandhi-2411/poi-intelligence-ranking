"""Evaluation-harness orchestration (spec.md section 11): the `poi_rank.cli evaluate`
entry point.

Builds the PRIMARY (unbiased holdout) and train-side ranking frames
(`models.ranking_data`), runs baselines 1-6 (`models.baselines`) plus the oracle
ceiling (`eval.oracle`), computes the full metric table (`eval.metrics`) with
bootstrap 95% CIs, runs the paired-Wilcoxon comparisons spec.md section 8 asks for
(this phase: baselines 4/5/6 and the oracle ceiling vs the popularity baseline --
LambdaMART/LambdaMART+IPS comparisons are a later phase's addition, not a rework of
this harness), and writes `results/metrics.json` under a top-level `"systems"` dict
keyed by system name so later phases can add entries without breaking this phase's
keys.

Determinism: every stochastic step (baseline 1's per-trip RNG, the logistic
regression fit, the bootstrap resampling) is seeded from `configs/model.yaml`'s /
`configs/eval.yaml`'s own `seed`; two runs of `poi_rank.cli evaluate` produce a
byte-identical `results/metrics.json` (`tests/test_determinism.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from poi_rank.candidates.config import GeoChannelConfig
from poi_rank.datagen.oracle_export import oracle_dir_from_output
from poi_rank.eval import oracle as oracle_reader
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.metrics import SystemMetrics, WilcoxonResult, evaluate_system, paired_wilcoxon
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import baselines as bl
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame

METRICS_FILENAME = "metrics.json"

BASELINE_SYSTEM_NAMES: tuple[str, ...] = (
    "random",
    "popularity",
    "popularity_geo_filter",
    "content_cosine",
    "item_knn_cf",
    "logistic_regression",
)
ORACLE_SYSTEM_NAME = "oracle"
ALL_SYSTEM_NAMES: tuple[str, ...] = (*BASELINE_SYSTEM_NAMES, ORACLE_SYSTEM_NAME)

WILCOXON_METRIC = "ndcg@10"
# spec.md section 8: paired Wilcoxon of (8) vs (2) and (8) vs (4) -- (8) is
# LambdaMART+IPS, a later phase. This phase proves the harness out on every system
# that already exists: baselines 4/5/6 and the oracle ceiling, all vs popularity (2).
POPULARITY_COMPARISON_SYSTEMS: tuple[str, ...] = (
    "content_cosine",
    "item_knn_cf",
    "logistic_regression",
    "oracle",
)

CANDIDATE_RECALL_NOTE = (
    "All metrics in this table are bounded by candidate_recall@250 (overall 0.4413, "
    "long-tail-stratum 0.3936 -- see candidates/recall_metrics.py output and "
    "docs/DATA_CARD.md): a candidate set missing a trip's true-relevant POIs caps "
    "every ranking metric computed over it (NDCG/Recall/MAP/MRR), regardless of "
    "ranking quality within the candidate set that IS present. This is a known, "
    "already-diagnosed property of Phase 4a's candidate generation, not a bug in "
    "this phase's baselines or metrics."
)


def _compute_all_baseline_scores(
    holdout_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    data_dir: Path,
    model_cfg: ModelConfig,
    geo_cfg: GeoChannelConfig,
) -> dict[str, bl.BaselineResult]:
    """Run baselines 1-6 against the holdout evaluation frame (baseline 6 is fit on
    the train frame first, per spec.md section 8's training-data requirement)."""
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    interactions_train = pd.read_parquet(data_dir / "interactions_train.parquet")

    results: dict[str, bl.BaselineResult] = {
        "random": bl.baseline_random(holdout_frame, model_cfg.seed),
        "popularity": bl.baseline_popularity(holdout_frame),
        "popularity_geo_filter": bl.baseline_popularity_geo_filter(holdout_frame, geo_cfg),
        "content_cosine": bl.baseline_content_cosine(
            holdout_frame, model_cfg.baselines.content_cosine
        ),
        "item_knn_cf": bl.baseline_item_knn_cf(
            holdout_frame, pois_df, interactions_train, trips_df
        ),
    }
    lr_model = bl.fit_logistic_regression(
        train_frame, model_cfg.baselines.logistic_regression, model_cfg.seed
    )
    results["logistic_regression"] = bl.score_logistic_regression(lr_model, holdout_frame)
    return results


def _evaluate_all_systems(
    holdout_frame: pd.DataFrame,
    scores: dict[str, bl.BaselineResult],
    oracle_score: pd.Series,
    eval_cfg: EvalConfig,
) -> dict[str, SystemMetrics]:
    system_metrics: dict[str, SystemMetrics] = {}
    for name in BASELINE_SYSTEM_NAMES:
        result = scores[name]
        system_metrics[name] = evaluate_system(
            holdout_frame,
            result.score,
            eval_cfg.metrics.ndcg_ks,
            eval_cfg.metrics.precision_ks,
            eval_cfg.metrics.recall_ks,
            eval_cfg.bootstrap.n_resamples,
            eval_cfg.seed,
            eval_cfg.bootstrap.ci_low_pct,
            eval_cfg.bootstrap.ci_high_pct,
            diagnostics=result.diagnostics,
        )
    system_metrics[ORACLE_SYSTEM_NAME] = evaluate_system(
        holdout_frame,
        oracle_score,
        eval_cfg.metrics.ndcg_ks,
        eval_cfg.metrics.precision_ks,
        eval_cfg.metrics.recall_ks,
        eval_cfg.bootstrap.n_resamples,
        eval_cfg.seed,
        eval_cfg.bootstrap.ci_low_pct,
        eval_cfg.bootstrap.ci_high_pct,
        diagnostics={},
    )
    return system_metrics


def _build_payload(
    system_metrics: dict[str, SystemMetrics],
    wilcoxon: dict[str, WilcoxonResult],
    model_cfg: ModelConfig,
    eval_cfg: EvalConfig,
    n_holdout_trips: int,
) -> dict[str, Any]:
    oracle_ndcg10 = system_metrics[ORACLE_SYSTEM_NAME].metrics[WILCOXON_METRIC].mean
    systems_payload: dict[str, Any] = {}
    for name, sm in system_metrics.items():
        ndcg10_mean = sm.metrics[WILCOXON_METRIC].mean
        pct_of_ceiling = ndcg10_mean / oracle_ndcg10 if oracle_ndcg10 > 0 else 0.0
        systems_payload[name] = {
            "metrics": {metric: agg.to_dict() for metric, agg in sm.metrics.items()},
            "diagnostics": sm.diagnostics,
            "pct_of_ceiling_ndcg10": pct_of_ceiling,
        }

    wilcoxon_payload = {
        key: {
            "metric": WILCOXON_METRIC,
            "statistic": r.statistic,
            "p_value": r.p_value,
            "n_pairs": r.n_pairs,
        }
        for key, r in wilcoxon.items()
    }

    return {
        "systems": systems_payload,
        "wilcoxon": wilcoxon_payload,
        "meta": {
            "phase": "4b",
            "n_holdout_trips": n_holdout_trips,
            "model_seed": model_cfg.seed,
            "eval_seed": eval_cfg.seed,
            "bootstrap_n_resamples": eval_cfg.bootstrap.n_resamples,
            "bootstrap_resample_unit": eval_cfg.bootstrap.resample_unit,
            "wilcoxon_metric": WILCOXON_METRIC,
            "note": CANDIDATE_RECALL_NOTE,
        },
    }


def run_evaluate(
    data_dir: Path,
    results_dir: Path,
    model_cfg: ModelConfig,
    eval_cfg: EvalConfig,
    feature_cfg: FeatureBuildConfig,
    geo_cfg: GeoChannelConfig,
) -> dict[str, Any]:
    """Run the full Phase 4b evaluation and write `results/<results_dir>/metrics.json`.
    Returns a summary dict (`output_path`, `payload`) for the CLI report and tests.
    """
    budget_target_price_level = feature_cfg.traveler_features.budget_target_price_level
    holdout_frame = load_holdout_evaluation_frame(data_dir, budget_target_price_level)
    train_frame = load_train_ranking_frame(data_dir, budget_target_price_level)

    scores = _compute_all_baseline_scores(holdout_frame, train_frame, data_dir, model_cfg, geo_cfg)

    oracle_dir = oracle_dir_from_output(data_dir)
    oracle_score = oracle_reader.oracle_ceiling_scores(
        holdout_frame[["trip_id", "poi_id"]], oracle_dir
    )

    system_metrics = _evaluate_all_systems(holdout_frame, scores, oracle_score, eval_cfg)

    wilcoxon: dict[str, WilcoxonResult] = {}
    popularity_per_trip = system_metrics["popularity"].per_trip[WILCOXON_METRIC]
    for name in POPULARITY_COMPARISON_SYSTEMS:
        other_per_trip = system_metrics[name].per_trip[WILCOXON_METRIC]
        wilcoxon[f"{name}_vs_popularity"] = paired_wilcoxon(other_per_trip, popularity_per_trip)

    n_holdout_trips = int(holdout_frame["trip_id"].nunique())
    payload = _build_payload(system_metrics, wilcoxon, model_cfg, eval_cfg, n_holdout_trips)

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / METRICS_FILENAME
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"output_path": output_path, "payload": payload}
