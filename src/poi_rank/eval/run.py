"""Evaluation-harness orchestration (spec.md section 11): the `poi_rank.cli evaluate`
entry point.

Builds the PRIMARY (unbiased holdout) and train-side ranking frames
(`models.ranking_data`), runs baselines 1-6 (`models.baselines`) plus systems 7/8
(`models.lambdamart` -- LOADED from `artifacts/`, never retrained here; `poi_rank.cli
train` is the only place systems 7/8 are fit, spec.md section 14) and the oracle
ceiling (`eval.oracle`), computes the full metric table (`eval.metrics`) with
bootstrap 95% CIs, runs the paired-Wilcoxon comparisons spec.md section 8 asks for
(every system vs popularity, plus lambdamart_ips vs content_cosine -- the best
baseline -- and lambdamart vs lambdamart_ips, the IPS ablation itself), evaluates the
new-POI cohort (`eval.new_poi_cohort`: dropout-on system 8 vs the dedicated
dropout-off ablation booster), and writes `results/metrics.json` under a top-level
`"systems"` dict keyed by system name so later phases can add entries without
breaking this phase's keys.

**Phase 8 additions** (spec.md section 11.2-11.9, the full eval suite -- this is the
module every downstream `docs/RESULTS.md` number ultimately traces back to):
  - `bias_gap`: every system's NDCG@10 on the SECONDARY biased holdout
    (`models.ranking_data.load_holdout_biased_evaluation_frame`) vs the primary
    unbiased holdout, reusing each system's ALREADY-COMPUTED score (the two frames
    are row-for-row `(trip_id, poi_id)`-aligned by construction, see that
    function's own docstring) -- no rescoring.
  - One call to `scoring.output.run_scoring_pipeline` (spec.md section 9's full
    pipeline, over EVERY primary-holdout trip -- not `recommend`'s persisted
    `results/recommendations.json`, computed fresh here so this phase never depends
    on `poi_rank.cli recommend` having been run first) supplies the FINAL top-K
    recommendation lists (`eval.personalization`/`eval.coverage`/`eval.longtail`/
    `eval.constraints` all consume this same `payload`), plus the calibration/
    confidence-decile/beta-sensitivity/lambda-sweep numbers Phase 6 already computes
    but had never persisted to `results/metrics.json` before this phase (module
    docstring: "not yet in metrics.json" is a documented, resolved gap, not new
    computation).
  - `personalization`, `coverage`, `longtail`, `constraint_compatibility`,
    `diversity` (category entropy + intra-list distance, reusing the lambda-sweep
    row already computed above), `cold_start` (interaction-count buckets, reusing
    `eval.new_poi_cohort`'s existing cohort numbers by cross-reference), and
    `ablations` (9 rows, `eval.ablations`) are all built from the SAME data this
    module already loads/computes -- no separate re-derivation.

Determinism: every stochastic step (baseline 1's per-trip RNG, the logistic
regression fit, the bootstrap resampling, the scoring pipeline's own calibration/
confidence-ensemble fits, every ablation retrain) is seeded from `configs/
model.yaml`'s / `configs/eval.yaml`'s / `configs/scoring.yaml`'s own `seed`s; two
runs of `poi_rank.cli evaluate` (given the same already-trained `artifacts/*.txt`)
produce a byte-identical `results/metrics.json` (`tests/test_determinism.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig, GeoChannelConfig
from poi_rank.candidates.recall_metrics import overall_and_longtail_recall
from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.oracle_export import oracle_dir_from_output
from poi_rank.eval import ablations as abl
from poi_rank.eval import cold_start as cs
from poi_rank.eval import constraints as cons
from poi_rank.eval import coverage as cov
from poi_rank.eval import longtail as lt
from poi_rank.eval import new_poi_cohort
from poi_rank.eval import oracle as oracle_reader
from poi_rank.eval import personalization as pers
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.metrics import (
    SystemMetrics,
    WilcoxonResult,
    aggregate_metric,
    compute_all_trip_metrics,
    evaluate_system,
    paired_wilcoxon,
)
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.traveler_features import assign_traveler_segments
from poi_rank.models import baselines as bl
from poi_rank.models import lambdamart as lm
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import (
    load_holdout_biased_evaluation_frame,
    load_holdout_evaluation_frame,
    load_train_ranking_frame,
)
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import run_scoring_pipeline

METRICS_FILENAME = "metrics.json"

BASELINE_SYSTEM_NAMES: tuple[str, ...] = (
    "random",
    "popularity",
    "popularity_geo_filter",
    "content_cosine",
    "item_knn_cf",
    "logistic_regression",
)
LAMBDAMART_SYSTEM_NAMES: tuple[str, ...] = ("lambdamart", "lambdamart_ips")
ORACLE_SYSTEM_NAME = "oracle"
ALL_SYSTEM_NAMES: tuple[str, ...] = (
    *BASELINE_SYSTEM_NAMES,
    *LAMBDAMART_SYSTEM_NAMES,
    ORACLE_SYSTEM_NAME,
)

WILCOXON_METRIC = "ndcg@10"
# spec.md section 8: paired Wilcoxon of (8) vs (2) and (8) vs (4) -- (8) is
# LambdaMART+IPS. Every system also gets a vs-popularity comparison (this harness's
# existing pattern from Phase 4b, kept for every system rather than special-cased).
POPULARITY_COMPARISON_SYSTEMS: tuple[str, ...] = (
    "content_cosine",
    "item_knn_cf",
    "logistic_regression",
    "lambdamart",
    "lambdamart_ips",
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
    """`scores` covers every non-oracle system (baselines 1-6 AND lambdamart/
    lambdamart_ips, wrapped as `BaselineResult` -- see `run_evaluate`) -- the oracle
    ceiling is evaluated separately since it has no `BaselineResult`/diagnostics."""
    system_metrics: dict[str, SystemMetrics] = {}
    for name in (*BASELINE_SYSTEM_NAMES, *LAMBDAMART_SYSTEM_NAMES):
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


def _candidate_recall_payload(
    data_dir: Path, pois_df: pd.DataFrame, long_tail_pop_pct_cutoff: float
) -> dict[str, Any]:
    """`candidate_recall@250` (spec.md section 11.10's own success-criteria row,
    already computed by `candidates/recall_metrics.py`/`poi_rank.cli candidates` but
    never previously persisted to `results/metrics.json` -- re-run here (cheap, pure
    recall computation, no candidate regeneration) so the success-criteria table has
    a real JSON source, per this phase's own "no hand-typed numbers" rule."""
    candidates_df = pd.read_parquet(data_dir / "candidates.parquet")
    holdout_random = pd.read_parquet(data_dir / "interactions_holdout_random.parquet")
    recall = overall_and_longtail_recall(
        pois_df, candidates_df, holdout_random, long_tail_pop_pct_cutoff
    )
    return {
        name: {
            "recall_mean": r.recall_mean,
            "n_trips_evaluated": r.n_trips_evaluated,
            "n_trips_excluded_no_relevant": r.n_trips_excluded_no_relevant,
        }
        for name, r in recall.items()
    }


def _bias_gap_payload(
    biased_frame: pd.DataFrame,
    all_scores: dict[str, pd.Series],
    system_metrics: dict[str, SystemMetrics],
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    """spec.md section 11.1's bias-gap table: every system's NDCG@10 on the
    SECONDARY biased holdout vs the already-computed unbiased-holdout NDCG@10.
    `all_scores` reuses each system's ALREADY-COMPUTED score series (row-order-
    aligned to `biased_frame` by construction, module docstring) -- no rescoring."""
    out: dict[str, Any] = {}
    for name in ALL_SYSTEM_NAMES:
        per_trip_biased = compute_all_trip_metrics(biased_frame, all_scores[name], (10,), (), ())[
            "ndcg@10"
        ]
        agg_biased = aggregate_metric(
            per_trip_biased,
            eval_cfg.bootstrap.n_resamples,
            eval_cfg.seed,
            eval_cfg.bootstrap.ci_low_pct,
            eval_cfg.bootstrap.ci_high_pct,
        )
        unbiased_mean = system_metrics[name].metrics[WILCOXON_METRIC].mean
        out[name] = {
            "ndcg@10_unbiased": unbiased_mean,
            "ndcg@10_biased": agg_biased.mean,
            "gap": agg_biased.mean - unbiased_mean,
            "biased_ci_low": agg_biased.ci_low,
            "biased_ci_high": agg_biased.ci_high,
        }
    return out


def _personalization_payload(
    scoring_result: dict[str, Any],
    trips_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    feature_cfg: FeatureBuildConfig,
) -> dict[str, Any]:
    """spec.md section 11.2: mean pairwise Jaccard@10, within- vs cross-archetype-
    proxy Jaccard@10, mean pairwise RBO(p=0.9). Archetype proxy = the SAME
    observable K-Means traveler-segment K-Means already established in Phase 3
    (`features.traveler_features.assign_traveler_segments`), never the oracle-only
    latent archetype mixture (`eval/personalization.py`'s own module docstring)."""
    lists_by_trip = pers.top10_lists_from_payload(scoring_result["payload"])
    mean_jaccard, n_pairs = pers.mean_pairwise_jaccard(lists_by_trip)
    mean_rbo, n_pairs_rbo = pers.mean_pairwise_rbo(lists_by_trip)

    segments = assign_traveler_segments(
        travelers_df,
        n_clusters=feature_cfg.poi_features.traveler_segment_clusters,
        seed=feature_cfg.seed,
    )
    trip_traveler = dict(zip(trips_df["trip_id"], trips_df["traveler_id"], strict=True))
    trip_segment = {
        trip_id: int(segments.loc[trip_traveler[trip_id]])
        for trip_id in lists_by_trip
        if trip_id in trip_traveler and trip_traveler[trip_id] in segments.index
    }
    archetype_result = pers.within_cross_archetype_jaccard(lists_by_trip, trip_segment)

    return {
        "mean_pairwise_jaccard_at_10": mean_jaccard,
        "n_pairs": n_pairs,
        "mean_pairwise_rbo": mean_rbo,
        "n_pairs_rbo": n_pairs_rbo,
        "archetype": archetype_result.to_dict(),
    }


def _popularity_top10_lists(
    holdout_frame: pd.DataFrame, popularity_score: pd.Series, k: int = 10
) -> dict[str, list[str]]:
    """Top-`k`-by-popularity-score `poi_id` list per trip -- the comparison
    population `eval.coverage`'s report is run against a second time (spec.md
    section 11.3: "compared against popularity baseline")."""
    working = holdout_frame[["trip_id", "poi_id"]].copy()
    working["score"] = popularity_score.to_numpy(dtype=np.float64)
    out: dict[str, list[str]] = {}
    for trip_id, group in working.groupby("trip_id", sort=True):
        top = group.sort_values(["score", "poi_id"], ascending=[False, True]).head(k)
        out[str(trip_id)] = top["poi_id"].astype(str).tolist()
    return out


def _coverage_payload(
    lists_by_trip: dict[str, list[str]],
    popularity_lists_by_trip: dict[str, list[str]],
    pois_df: pd.DataFrame,
    trips_df: pd.DataFrame,
) -> dict[str, Any]:
    catalog_poi_ids = set(pois_df["poi_id"])
    trip_destination = dict(zip(trips_df["trip_id"], trips_df["destination"], strict=True))
    poi_destination = dict(zip(pois_df["poi_id"], pois_df["destination"], strict=True))
    primary = cov.coverage_report(lists_by_trip, catalog_poi_ids, trip_destination, poi_destination)
    popularity = cov.coverage_report(
        popularity_lists_by_trip, catalog_poi_ids, trip_destination, poi_destination
    )
    return {"primary_system": primary.to_dict(), "popularity_baseline": popularity.to_dict()}


def _longtail_payload(
    lists_by_trip: dict[str, list[str]],
    pois_df: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    long_tail_pop_pct_cutoff: float,
) -> dict[str, Any]:
    pop_pct_by_poi = dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True))
    label_by_trip_poi = {
        (str(t), str(p)): int(lbl)
        for t, p, lbl in zip(
            holdout_frame["trip_id"], holdout_frame["poi_id"], holdout_frame["label"], strict=True
        )
    }
    result = lt.longtail_share_and_precision(
        lists_by_trip, pop_pct_by_poi, long_tail_pop_pct_cutoff, label_by_trip_poi
    )
    return result.to_dict()


def _constraint_compatibility_payload(scoring_result: dict[str, Any]) -> dict[str, Any]:
    values = [
        float(rec["context_compatibility"])
        for entry in scoring_result["payload"].values()
        for rec in entry["recommendations"]
    ]
    # Hard-constraint violation rate (spec.md section 11.5, build-blocking per
    # `tests/test_hard_constraints.py` -- this is a REAL count off the actual
    # assembled output, not asserted/hand-typed as "0"; `hard_constraints_ok` is
    # `bool(row["hard_gate"] == 1.0)` per `scoring.output.assemble_output_payload`,
    # so a non-zero count here would mean that module's own defensive assertion had
    # already failed before this ever ran).
    n_violations = sum(
        1
        for entry in scoring_result["payload"].values()
        for rec in entry["recommendations"]
        if not rec["hard_constraints_ok"]
    )
    out = cons.share_with_compatibility_above_target(values).to_dict()
    out["n_hard_constraint_violations"] = n_violations
    return out


def _diversity_payload(
    scoring_result: dict[str, Any], pois_df: pd.DataFrame, lambda_default: float
) -> dict[str, Any]:
    """Category entropy@10 (spec.md section 11.6) over every recommended `(trip,
    poi)` instance's canonical category, plus intra-list mean cosine DISTANCE
    (`1 - mean_intra_list_similarity`) read off the already-computed lambda-sweep
    row CLOSEST to the configured default lambda -- reused, not recomputed."""
    category_by_poi = dict(zip(pois_df["poi_id"], pois_df["category"], strict=True))
    counts: dict[str, int] = {}
    for entry in scoring_result["payload"].values():
        for rec in entry["recommendations"]:
            cat = category_by_poi.get(rec["poi_id"], "unknown")
            counts[cat] = counts.get(cat, 0) + 1
    freq = np.array(list(counts.values()), dtype=np.float64)
    entropy = cov.shannon_entropy(freq, base=2.0)

    default_row = min(
        scoring_result["lambda_sweep"], key=lambda r: abs(r["lambda"] - lambda_default)
    )
    return {
        "category_entropy_at_10_bits": entropy,
        "intra_list_mean_distance": 1.0 - default_row["mean_intra_list_similarity"],
        "lambda_sweep": scoring_result["lambda_sweep"],
    }


def _cold_start_payload(
    holdout_frame: pd.DataFrame, lambdamart_ips_score: pd.Series, eval_cfg: EvalConfig
) -> dict[str, Any]:
    by_bucket = cs.ndcg10_by_interaction_bucket(
        holdout_frame,
        lambdamart_ips_score,
        eval_cfg.bootstrap.n_resamples,
        eval_cfg.seed,
        eval_cfg.bootstrap.ci_low_pct,
        eval_cfg.bootstrap.ci_high_pct,
    )
    return {
        "ndcg@10_by_interaction_count_bucket": by_bucket,
        "note": (
            "New-POI cohort NDCG@10 is reported separately under the top-level "
            "'new_poi_cohort' key (eval/new_poi_cohort.py, Phase 5) -- not "
            "duplicated here. Leave-one-destination-out (LODO) is reported under "
            "the top-level 'lodo' key ONLY after `poi_rank.cli lodo` has been run "
            "(a separate command, not part of `make reproduce`'s default chain --"
            " see eval/cold_start.py's module docstring)."
        ),
    }


def _ablations_payload(
    system_metrics: dict[str, SystemMetrics],
    scoring_result: dict[str, Any],
    holdout_frame: pd.DataFrame,
    candidates_df: pd.DataFrame,
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    lambdamart_ips_score: pd.Series,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    eval_cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """The 9-row ablation table (spec.md section 11.9, module docstring). Every row
    is measured -- see `eval/ablations.py`'s module docstring for the cost-tiered
    reuse strategy (5 cheap rows reuse already-computed scores/frames, 4 require one
    full LightGBM retrain each)."""
    b = eval_cfg.bootstrap
    rows: list[abl.AblationRow] = []

    rows.append(
        abl.ips_weighting_row(
            system_metrics["lambdamart_ips"].per_trip[WILCOXON_METRIC],
            system_metrics["lambdamart"].per_trip[WILCOXON_METRIC],
            system_metrics["lambdamart_ips"].metrics[WILCOXON_METRIC],
            system_metrics["lambdamart"].metrics[WILCOXON_METRIC],
        )
    )

    rows.append(
        abl.calibration_row(
            scoring_result["full_frame"],
            scoring_result["raw_score"],
            scoring_result["relevance"],
            b.n_resamples,
            eval_cfg.seed,
            b.ci_low_pct,
            b.ci_high_pct,
        )
    )

    survivors = scoring_result["full_frame"].loc[scoring_result["full_frame"]["hard_gate"] == 1.0]
    rows.append(
        abl.mmr_row(
            survivors,
            scoring_cfg.diversity,
            scoring_cfg.diversity.lambda_default,
            scoring_cfg.output.top_k,
            b.n_resamples,
            eval_cfg.seed,
            b.ci_low_pct,
            b.ci_high_pct,
        )
    )

    rows.append(
        abl.leave_one_channel_out_row(
            "-CF_channel",
            "channel_cf",
            holdout_frame,
            candidates_df,
            lambdamart_ips_score,
            b.n_resamples,
            eval_cfg.seed,
            b.ci_low_pct,
            b.ci_high_pct,
        )
    )
    rows.append(
        abl.leave_one_channel_out_row(
            "-long_tail_quota",
            "channel_longtail",
            holdout_frame,
            candidates_df,
            lambdamart_ips_score,
            b.n_resamples,
            eval_cfg.seed,
            b.ci_low_pct,
            b.ci_high_pct,
        )
    )

    for label, prefix in abl.FEATURE_BLOCK_PREFIXES.items():
        rows.append(
            abl.feature_block_row(
                label,
                prefix,
                train_frame,
                interactions_train,
                pois_df,
                holdout_frame,
                lambdamart_ips_score,
                model_cfg,
                b.n_resamples,
                eval_cfg.seed,
                b.ci_low_pct,
                b.ci_high_pct,
            )
        )

    return [r.to_dict() for r in rows]


def _build_payload(
    system_metrics: dict[str, SystemMetrics],
    wilcoxon: dict[str, WilcoxonResult],
    model_cfg: ModelConfig,
    eval_cfg: EvalConfig,
    n_holdout_trips: int,
    new_poi_cohort_payload: dict[str, Any],
    bias_gap_payload: dict[str, Any],
    personalization_payload: dict[str, Any],
    coverage_payload: dict[str, Any],
    longtail_payload: dict[str, Any],
    constraint_compatibility_payload: dict[str, Any],
    diversity_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
    confidence_decile_payload: dict[str, Any],
    beta_sensitivity_payload: list[dict[str, Any]],
    cold_start_payload: dict[str, Any],
    ablations_payload: list[dict[str, Any]],
    candidate_recall_payload: dict[str, Any],
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
        "new_poi_cohort": new_poi_cohort_payload,
        "bias_gap": bias_gap_payload,
        "personalization": personalization_payload,
        "coverage": coverage_payload,
        "longtail": longtail_payload,
        "constraint_compatibility": constraint_compatibility_payload,
        "diversity": diversity_payload,
        "calibration": calibration_payload,
        "confidence_decile_validation": confidence_decile_payload,
        "beta_sensitivity": beta_sensitivity_payload,
        "cold_start": cold_start_payload,
        "ablations": ablations_payload,
        "candidate_recall": candidate_recall_payload,
        "meta": {
            "phase": "8",
            "n_holdout_trips": n_holdout_trips,
            "model_seed": model_cfg.seed,
            "eval_seed": eval_cfg.seed,
            "bootstrap_n_resamples": eval_cfg.bootstrap.n_resamples,
            "bootstrap_resample_unit": eval_cfg.bootstrap.resample_unit,
            "wilcoxon_metric": WILCOXON_METRIC,
            "note": CANDIDATE_RECALL_NOTE,
        },
    }


def _new_poi_cohort_payload(
    data_dir: Path,
    artifacts_dir: Path,
    holdout_frame: pd.DataFrame,
    lambdamart_scores: dict[str, pd.Series],
    datagen_cfg: DatagenConfig,
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    """New-POI robustness (spec.md section 8): NDCG@10 on the new-POI cohort for
    system 8 (dropout ON, the primary system's own recipe) vs the dedicated
    dropout-OFF ablation booster (`lambdamart_ips_no_dropout.txt`, `models
    .lambdamart.save_boosters`) -- same `eval.metrics` harness, restricted to the
    cohort's own candidate rows (`eval.new_poi_cohort`)."""
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    cohort_ids = new_poi_cohort.new_poi_ids(pois_df, datagen_cfg)
    cohort = new_poi_cohort.cohort_frame(holdout_frame, cohort_ids)
    summary = new_poi_cohort.cohort_summary(pois_df, holdout_frame, datagen_cfg)

    boosters = lm.load_boosters(artifacts_dir)
    numeric_columns = bl.numeric_feature_columns(cohort)
    categorical_columns = bl.categorical_feature_columns(cohort)
    score_no_dropout = lm.score_booster(
        boosters["lambdamart_ips_no_dropout"], cohort, numeric_columns, categorical_columns
    )
    score_with_dropout = lambdamart_scores["lambdamart_ips"].loc[cohort.index]

    ndcg_with_dropout = new_poi_cohort.evaluate_cohort_ndcg10(
        cohort,
        score_with_dropout,
        eval_cfg.bootstrap.n_resamples,
        eval_cfg.seed,
        eval_cfg.bootstrap.ci_low_pct,
        eval_cfg.bootstrap.ci_high_pct,
    )
    ndcg_without_dropout = new_poi_cohort.evaluate_cohort_ndcg10(
        cohort,
        score_no_dropout,
        eval_cfg.bootstrap.n_resamples,
        eval_cfg.seed,
        eval_cfg.bootstrap.ci_low_pct,
        eval_cfg.bootstrap.ci_high_pct,
    )
    per_trip_with = new_poi_cohort.evaluate_cohort_ndcg10_per_trip(cohort, score_with_dropout)
    per_trip_without = new_poi_cohort.evaluate_cohort_ndcg10_per_trip(cohort, score_no_dropout)
    dropout_wilcoxon = paired_wilcoxon(per_trip_with, per_trip_without)
    return {
        **summary,
        "ndcg@10_lambdamart_ips_with_dropout": ndcg_with_dropout.to_dict(),
        "ndcg@10_lambdamart_ips_no_dropout": ndcg_without_dropout.to_dict(),
        "wilcoxon_with_vs_without_dropout": {
            "metric": "ndcg@10",
            "statistic": dropout_wilcoxon.statistic,
            "p_value": dropout_wilcoxon.p_value,
            "n_pairs": dropout_wilcoxon.n_pairs,
        },
    }


def run_evaluate(
    data_dir: Path,
    results_dir: Path,
    artifacts_dir: Path,
    model_cfg: ModelConfig,
    eval_cfg: EvalConfig,
    feature_cfg: FeatureBuildConfig,
    candidates_cfg: CandidatesConfig,
    datagen_cfg: DatagenConfig,
    scoring_cfg: ScoringConfig,
) -> dict[str, Any]:
    """Run the full evaluation harness and write `results/<results_dir>/metrics.json`.
    Returns a summary dict (`output_path`, `payload`) for the CLI report and tests.
    Requires `poi_rank.cli train` to have already written `artifacts/*.txt` --
    systems 7/8 are LOADED here, never retrained (module docstring).
    """
    geo_cfg = candidates_cfg.geo
    budget_target_price_level = feature_cfg.traveler_features.budget_target_price_level
    holdout_frame = load_holdout_evaluation_frame(data_dir, budget_target_price_level)
    train_frame = load_train_ranking_frame(data_dir, budget_target_price_level)
    biased_frame = load_holdout_biased_evaluation_frame(data_dir, budget_target_price_level)

    baseline_scores = _compute_all_baseline_scores(
        holdout_frame, train_frame, data_dir, model_cfg, geo_cfg
    )
    lambdamart_scores = lm.load_and_score_holdout(artifacts_dir, holdout_frame)
    scores: dict[str, bl.BaselineResult] = {
        **baseline_scores,
        **{name: bl.BaselineResult(score=series) for name, series in lambdamart_scores.items()},
    }

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
    # Best-performing baseline (Phase 4b): lambdamart_ips (the primary proposed
    # system) vs content_cosine.
    wilcoxon["lambdamart_ips_vs_content_cosine"] = paired_wilcoxon(
        system_metrics["lambdamart_ips"].per_trip[WILCOXON_METRIC],
        system_metrics["content_cosine"].per_trip[WILCOXON_METRIC],
    )
    # The IPS ablation itself (spec.md section 8: "Report NDCG on the unbiased
    # holdout with and without IPS as a measured ablation").
    wilcoxon["lambdamart_vs_lambdamart_ips"] = paired_wilcoxon(
        system_metrics["lambdamart"].per_trip[WILCOXON_METRIC],
        system_metrics["lambdamart_ips"].per_trip[WILCOXON_METRIC],
    )

    new_poi_payload = _new_poi_cohort_payload(
        data_dir, artifacts_dir, holdout_frame, lambdamart_scores, datagen_cfg, eval_cfg
    )

    all_scores = {name: r.score for name, r in scores.items()}
    all_scores[ORACLE_SYSTEM_NAME] = oracle_score
    bias_gap_payload = _bias_gap_payload(biased_frame, all_scores, system_metrics, eval_cfg)

    # One full scoring-pipeline pass (spec.md section 9) over EVERY primary-holdout
    # trip -- supplies the final top-K lists (personalization/coverage/longtail/
    # constraints) plus the calibration/confidence-decile/beta-sensitivity/lambda-
    # sweep numbers (module docstring).
    scoring_result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        geo_cfg,
        candidates_cfg.longtail.pop_pct_cutoff,
    )

    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    candidates_df = pd.read_parquet(data_dir / "candidates.parquet")
    interactions_train = pd.read_parquet(data_dir / "interactions_train.parquet")

    lists_by_trip = pers.top10_lists_from_payload(scoring_result["payload"])
    personalization_payload = _personalization_payload(
        scoring_result, trips_df, travelers_df, feature_cfg
    )
    popularity_lists_by_trip = _popularity_top10_lists(
        holdout_frame, baseline_scores["popularity"].score
    )
    coverage_payload = _coverage_payload(lists_by_trip, popularity_lists_by_trip, pois_df, trips_df)
    longtail_payload = _longtail_payload(
        lists_by_trip, pois_df, holdout_frame, eval_cfg.long_tail_pop_pct_cutoff
    )
    constraint_compatibility_payload = _constraint_compatibility_payload(scoring_result)
    diversity_payload = _diversity_payload(
        scoring_result, pois_df, scoring_cfg.diversity.lambda_default
    )
    cold_start_payload = _cold_start_payload(
        holdout_frame, lambdamart_scores["lambdamart_ips"], eval_cfg
    )

    ablations_payload = _ablations_payload(
        system_metrics,
        scoring_result,
        holdout_frame,
        candidates_df,
        train_frame,
        interactions_train,
        pois_df,
        lambdamart_scores["lambdamart_ips"],
        model_cfg,
        scoring_cfg,
        eval_cfg,
    )

    candidate_recall_payload = _candidate_recall_payload(
        data_dir, pois_df, eval_cfg.long_tail_pop_pct_cutoff
    )

    n_holdout_trips = int(holdout_frame["trip_id"].nunique())
    payload = _build_payload(
        system_metrics,
        wilcoxon,
        model_cfg,
        eval_cfg,
        n_holdout_trips,
        new_poi_payload,
        bias_gap_payload,
        personalization_payload,
        coverage_payload,
        longtail_payload,
        constraint_compatibility_payload,
        diversity_payload,
        scoring_result["calibration"],
        scoring_result["confidence_decile_validation"],
        scoring_result["beta_sensitivity"],
        cold_start_payload,
        ablations_payload,
        candidate_recall_payload,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / METRICS_FILENAME
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"output_path": output_path, "payload": payload}
