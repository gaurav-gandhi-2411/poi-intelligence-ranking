"""Ablation table (spec.md section 11.9): `-text embeddings`, `-implicit taste`,
`-explicit interests`, `-behavioral block`, `-CF channel`, `-long-tail quota`,
`-IPS weighting`, `-calibration`, `-MMR` -- each a single row with delta NDCG@10 and
a CI, measured against the primary system (lambdamart_ips) on the unbiased holdout.

**Cost-tiered reuse, per this phase's own task brief:**
  - `-IPS weighting`: FREE -- system 7 (no IPS) vs system 8 (IPS) IS this ablation;
    `eval/run.py` builds this row directly from the already-computed
    `SystemMetrics` for "lambdamart"/"lambdamart_ips", never recomputed here.
  - `-calibration`, `-MMR`: cheap, reuse `scoring.output.run_scoring_pipeline`'s
    already-computed raw score / calibrated relevance score, and
    `scoring.diversity.mmr_rerank_all_trips` at lambda=1.0 (pure-utility ranking,
    "MMR off") vs the configured default lambda -- no retraining.
  - `-CF channel`, `-long-tail quota`: cheap, leave-one-channel-out candidate-set
    re-evaluation (`candidates/recall_metrics.py`'s own construction, reused
    against NDCG instead of recall) against the ALREADY-TRAINED lambdamart_ips
    score -- no retraining, no new candidate generation.
  - `-text embeddings`, `-implicit taste`, `-explicit interests`, `-behavioral
    block`: each requires ONE full LightGBM retrain with that feature block's
    columns dropped from the design matrix (`text_emb_*`, `implicit_*`,
    `explicit_*`, `behav_*` -- the SAME block-prefix convention
    `models/baselines.py::numeric_feature_columns` already establishes, reused
    directly here, never a duplicated prefix list).

Every row's `ndcg@10_full` baseline is the ALREADY-TRAINED system 8 (lambdamart_ips,
loaded from `artifacts/model.txt`, never retrained) scored identically for every row
-- so all 9 deltas are directly comparable against the same reference point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from poi_rank.candidates.channels import CHANNEL_NAMES
from poi_rank.eval.metrics import (
    MetricAggregate,
    aggregate_metric,
    compute_all_trip_metrics,
    paired_wilcoxon,
)
from poi_rank.models import baselines as bl
from poi_rank.models import lambdamart as lm
from poi_rank.models.config import ModelConfig
from poi_rank.scoring import diversity as div
from poi_rank.scoring.config import DiversityConfig

FEATURE_BLOCK_PREFIXES: dict[str, str] = {
    "-text_embeddings": "text_emb_",
    "-implicit_taste": "implicit_",
    "-explicit_interests": "explicit_",
    "-behavioral_block": "behav_",
}


@dataclass(frozen=True)
class AblationRow:
    ablation: str
    status: str  # "measured" or "skipped"
    note: str
    ndcg10_full: MetricAggregate | None = None
    ndcg10_ablated: MetricAggregate | None = None
    delta_ndcg10: float | None = None
    wilcoxon_p_value: float | None = None
    wilcoxon_n_pairs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation,
            "status": self.status,
            "note": self.note,
            "ndcg@10_full": self.ndcg10_full.to_dict() if self.ndcg10_full is not None else None,
            "ndcg@10_ablated": (
                self.ndcg10_ablated.to_dict() if self.ndcg10_ablated is not None else None
            ),
            "delta_ndcg@10": self.delta_ndcg10,
            "wilcoxon_p_value": self.wilcoxon_p_value,
            "wilcoxon_n_pairs": self.wilcoxon_n_pairs,
        }


def _compare_ndcg10(
    ablation: str,
    note: str,
    frame_full: pd.DataFrame,
    score_full: pd.Series,
    frame_ablated: pd.DataFrame,
    score_ablated: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> AblationRow:
    """Shared builder: per-trip NDCG@10 for the full system vs the ablated variant
    (possibly over different frames -- e.g. a leave-one-channel-out candidate set
    has fewer rows per trip), aggregated with bootstrap CI, paired Wilcoxon over the
    trips present in both (`eval.metrics.paired_wilcoxon`'s own index-alignment)."""
    per_trip_full = compute_all_trip_metrics(frame_full, score_full, (10,), (), ())["ndcg@10"]
    per_trip_ablated = compute_all_trip_metrics(frame_ablated, score_ablated, (10,), (), ())[
        "ndcg@10"
    ]
    agg_full = aggregate_metric(per_trip_full, n_resamples, seed, ci_low_pct, ci_high_pct)
    agg_ablated = aggregate_metric(per_trip_ablated, n_resamples, seed, ci_low_pct, ci_high_pct)
    wilcoxon = paired_wilcoxon(per_trip_ablated, per_trip_full)
    return AblationRow(
        ablation=ablation,
        status="measured",
        note=note,
        ndcg10_full=agg_full,
        ndcg10_ablated=agg_ablated,
        delta_ndcg10=agg_ablated.mean - agg_full.mean,
        wilcoxon_p_value=wilcoxon.p_value,
        wilcoxon_n_pairs=wilcoxon.n_pairs,
    )


# -----------------------------------------------------------------------------------
# -IPS weighting: free, built directly from the already-computed systems 7/8
# -----------------------------------------------------------------------------------


def ips_weighting_row(
    full_per_trip_ndcg10: pd.Series,
    ablated_per_trip_ndcg10: pd.Series,
    full_agg: MetricAggregate,
    ablated_agg: MetricAggregate,
) -> AblationRow:
    """`-IPS weighting`: system 7 (lambdamart, no IPS) is exactly the "IPS-ablated"
    variant of system 8 -- both already computed by `eval/run.py`'s main systems
    table, so this function only assembles the row, it computes nothing new."""
    wilcoxon = paired_wilcoxon(ablated_per_trip_ndcg10, full_per_trip_ndcg10)
    return AblationRow(
        ablation="-IPS_weighting",
        status="measured",
        note="Free: system 7 (lambdamart, uniform weight) IS this ablation of system 8.",
        ndcg10_full=full_agg,
        ndcg10_ablated=ablated_agg,
        delta_ndcg10=ablated_agg.mean - full_agg.mean,
        wilcoxon_p_value=wilcoxon.p_value,
        wilcoxon_n_pairs=wilcoxon.n_pairs,
    )


# -----------------------------------------------------------------------------------
# -calibration: isotonic calibration is monotonic non-decreasing, so it CANNOT
# change within-trip ranking except at tie-broken raw-score ties -- delta is
# expected to be at/near zero BY CONSTRUCTION, not a modeling failure if so.
# -----------------------------------------------------------------------------------


def calibration_row(
    frame: pd.DataFrame,
    raw_score: pd.Series,
    calibrated_score: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> AblationRow:
    return _compare_ndcg10(
        "-calibration",
        "Isotonic calibration is a monotonic non-decreasing transform of the raw "
        "score, so it cannot change within-trip RANKING (and therefore NDCG@10) "
        "except through tie-break reordering at flat isotonic segments -- an "
        "at/near-zero delta here is the mathematically EXPECTED result, not a "
        "modeling failure. Calibration's actual purpose (cross-traveler score "
        "comparability for planner_weight) is not something NDCG@10 measures.",
        frame,
        raw_score,
        frame,
        calibrated_score,
        n_resamples,
        seed,
        ci_low_pct,
        ci_high_pct,
    )


# -----------------------------------------------------------------------------------
# -MMR: lambda=1.0 (pure utility, no diversity penalty) vs the configured default
# -----------------------------------------------------------------------------------


def _mmr_scores_for_lambda(
    survivors: pd.DataFrame, cfg: DiversityConfig, lam: float, k: int
) -> pd.Series:
    """A per-row score reflecting each trip's MMR rank at this lambda (`-mmr_rank`
    for selected rows, `-inf` for the rest -- same convention
    `scoring.diversity.lambda_sweep_report` already uses internally, so NDCG@k over
    this score reproduces exactly what that report's own numbers measure)."""
    reranked = div.mmr_rerank_all_trips(survivors, cfg, lam, k)
    rank_by_trip_poi = dict(
        zip(
            zip(reranked["trip_id"], reranked["poi_id"], strict=True),
            reranked["mmr_rank"],
            strict=True,
        )
    )
    scores = []
    for trip_id, poi_id in zip(survivors["trip_id"], survivors["poi_id"], strict=True):
        rank = rank_by_trip_poi.get((trip_id, poi_id))
        scores.append(-float(rank) if rank is not None else float("-inf"))
    return pd.Series(scores, index=survivors.index)


def mmr_row(
    survivors: pd.DataFrame,
    diversity_cfg: DiversityConfig,
    lambda_default: float,
    top_k: int,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> AblationRow:
    """`-MMR`: lambda=1.0 reduces `argmax[lambda*utility - (1-lambda)*max_sim]` to
    plain utility-ranked top-K (spec.md section 9.4's own formula) -- "MMR off"
    without a separate code path, reusing `mmr_rerank_all_trips` directly."""
    score_default = _mmr_scores_for_lambda(survivors, diversity_cfg, lambda_default, top_k)
    score_no_mmr = _mmr_scores_for_lambda(survivors, diversity_cfg, 1.0, top_k)
    return _compare_ndcg10(
        "-MMR",
        f"lambda=1.0 (pure utility ranking, no diversity penalty) vs the configured "
        f"default lambda={lambda_default} -- reuses scoring.diversity.mmr_rerank_all_trips "
        "directly, no retraining.",
        survivors,
        score_default,
        survivors,
        score_no_mmr,
        n_resamples,
        seed,
        ci_low_pct,
        ci_high_pct,
    )


# -----------------------------------------------------------------------------------
# -CF channel / -long-tail quota: leave-one-channel-out candidate set, same
# already-trained lambdamart_ips score, no retraining, no new candidate generation.
# -----------------------------------------------------------------------------------


def _candidate_keys_excluding_channel(
    candidates_df: pd.DataFrame, exclude_channel: str
) -> set[tuple[str, str]]:
    """`(trip_id, poi_id)` pairs surviving the candidate union with `exclude_channel`
    removed -- same leave-one-channel-out construction as
    `candidates.recall_metrics._candidate_set_by_trip`, reimplemented locally (that
    helper is private, single extra caller here does not justify promoting it to a
    public API)."""
    channels = [c for c in CHANNEL_NAMES if c != exclude_channel]
    membership = candidates_df[channels].any(axis=1)
    kept = candidates_df.loc[membership, ["trip_id", "poi_id"]]
    return set(zip(kept["trip_id"].astype(str), kept["poi_id"].astype(str), strict=True))


def leave_one_channel_out_row(
    ablation_label: str,
    channel_name: str,
    holdout_frame: pd.DataFrame,
    candidates_df: pd.DataFrame,
    lambdamart_ips_score: pd.Series,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
) -> AblationRow:
    keep_keys = _candidate_keys_excluding_channel(candidates_df, channel_name)
    row_keys = list(
        zip(holdout_frame["trip_id"].astype(str), holdout_frame["poi_id"].astype(str), strict=True)
    )
    mask = pd.Series([k in keep_keys for k in row_keys], index=holdout_frame.index)
    ablated_frame = holdout_frame.loc[mask]
    ablated_score = lambdamart_ips_score.loc[ablated_frame.index]
    return _compare_ndcg10(
        ablation_label,
        f"Leave-one-channel-out ('{channel_name}' removed from the candidate "
        "union): re-evaluates the ALREADY-TRAINED lambdamart_ips score over the "
        "smaller candidate set, no retraining, no new candidate generation.",
        holdout_frame,
        lambdamart_ips_score,
        ablated_frame,
        ablated_score,
        n_resamples,
        seed,
        ci_low_pct,
        ci_high_pct,
    )


# -----------------------------------------------------------------------------------
# Feature-block ablations: -text embeddings / -implicit taste / -explicit interests
# / -behavioral block -- each requires ONE full LightGBM retrain.
# -----------------------------------------------------------------------------------


def train_block_dropped_booster(
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    model_cfg: ModelConfig,
    drop_prefix: str,
    num_boost_round: int | None = None,
) -> tuple[Any, list[str], list[str]]:
    """Train ONE LambdaMART+IPS booster (system 8's exact recipe) with every
    numeric feature column starting with `drop_prefix` removed from the design
    matrix entirely (module docstring's block-prefix reuse). Categorical columns
    are never prefixed with any of the 4 ablated blocks (`_NUMERIC_FEATURE_PREFIXES`
    vs `cat_*`, `models/baselines.py`), so only the numeric column list is
    filtered."""
    lm_cfg = model_cfg.lambdamart
    numeric_columns = [
        c for c in bl.numeric_feature_columns(train_frame) if not c.startswith(drop_prefix)
    ]
    categorical_columns = bl.categorical_feature_columns(train_frame)

    fit_frame, val_frame = lm.train_val_split_by_trip(
        train_frame, lm_cfg.val_fraction, lm_cfg.val_split_seed
    )
    p_expose_fit = lm.train_frame_p_expose(fit_frame, interactions_train, pois_df)
    ips_weight_fit = lm.compute_ips_weights(
        fit_frame["trip_id"], p_expose_fit, lm_cfg.ips_clip_low, lm_cfg.ips_clip_high
    )
    fit_frame_dropout, _mask = lm.apply_behavioral_dropout(
        fit_frame, lm_cfg.behavioral_dropout_rate, lm_cfg.behavioral_dropout_seed
    )
    booster = lm.fit_lambdamart_booster(
        fit_frame_dropout,
        val_frame,
        numeric_columns,
        categorical_columns,
        lm_cfg,
        model_cfg.seed,
        sample_weight=ips_weight_fit,
        num_boost_round=num_boost_round,
    )
    return booster, numeric_columns, categorical_columns


def feature_block_row(
    ablation_label: str,
    drop_prefix: str,
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    lambdamart_ips_score: pd.Series,
    model_cfg: ModelConfig,
    n_resamples: int,
    seed: int,
    ci_low_pct: float,
    ci_high_pct: float,
    num_boost_round: int | None = None,
) -> AblationRow:
    booster, numeric_columns, categorical_columns = train_block_dropped_booster(
        train_frame, interactions_train, pois_df, model_cfg, drop_prefix, num_boost_round
    )
    ablated_score = lm.score_booster(booster, holdout_frame, numeric_columns, categorical_columns)
    return _compare_ndcg10(
        ablation_label,
        f"Full retrain with every '{drop_prefix}*' numeric feature column dropped "
        "from the design matrix (models.baselines.numeric_feature_columns filtered "
        "before training) -- same IPS + behavioral-dropout recipe as system 8 "
        "otherwise.",
        holdout_frame,
        lambdamart_ips_score,
        holdout_frame,
        ablated_score,
        n_resamples,
        seed,
        ci_low_pct,
        ci_high_pct,
    )
