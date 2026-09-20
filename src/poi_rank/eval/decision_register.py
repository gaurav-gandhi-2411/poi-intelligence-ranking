"""Decision Register experiments (spec-v2 section 5, TECHNICAL.md section 12).

Every design choice not dictated by the assignment carries a MEASURED number or an explicit
NOT RUN -- prose justification is not acceptable. Each experiment writes
`results/parts/dr/<ID>.json` (`{id, decision, alternative, experiment, metric, results,
verdict, status}`); `compose_register` gathers them (plus NOT RUN stubs for the rest) into
`results/parts/decision_register.json`, which `compose` folds into metrics.json.

Evaluation discipline: every ranker below is fit on TRAIN trips only and scored on the primary
unbiased (random-exposure) holdout; the oracle is not read here. Verdict strings are built from
the measured numbers by code, never typed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.metrics import aggregate_metric, compute_all_trip_metrics
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame
from poi_rank.scoring._ranking_metrics import ndcg_at_k
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import run_scoring_pipeline

DR_DIRNAME = "dr"
SEED = 42

DR_CATALOG: dict[str, dict[str, str]] = {
    "DR1": {
        "decision": "Multiplicative utility rel^a * compat^b with a hard gate",
        "alternative": "The brief's additive formula a*rel + b*compat",
    },
    "DR2": {
        "decision": "LightGBM LambdaMART ranker",
        "alternative": "Two-tower neural ranker (shared-space dot product)",
    },
    "DR3": {
        "decision": "Listwise LambdaRank objective",
        "alternative": "Pointwise binary / graded regression, listwise rank_xendcg",
    },
    "DR4": {
        "decision": "TF-IDF -> SVD-64 POI text embedding",
        "alternative": "all-MiniLM-L6-v2 sentence embeddings (-> SVD-64)",
    },
    "DR5": {
        "decision": "Brute-force cosine retrieval",
        "alternative": "ANN index (faiss / hnswlib)",
    },
    "DR6": {
        "decision": "Geometric-mean compatibility aggregation",
        "alternative": "min(), plain product, arithmetic mean",
    },
    "DR7": {"decision": "IPS clip = 20", "alternative": "clip in {5, 10, 50, none}"},
    "DR8": {
        "decision": "180-day taste half-life and spec interaction weights",
        "alternative": "+-2x half-life; uniform interaction weights",
    },
    "DR9": {
        "decision": "Long-tail candidate quota = 50",
        "alternative": "quota in {0, 25, 100}",
    },
    "DR10": {"decision": "alpha = 1.0, beta = 0.7", "alternative": "alpha x beta grid"},
    "DR11": {
        "decision": "Candidate generation = learned retriever + long-tail + interest",
        "alternative": "The original 6 heuristic channels, and subsets of them",
    },
}


def dr_dir(results_dir: Path) -> Path:
    return results_dir / "parts" / DR_DIRNAME


def write_dr(
    results_dir: Path, dr_id: str, experiment: str, metric: str, results: Any, verdict: str
) -> Path:
    directory = dr_dir(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": dr_id,
        **DR_CATALOG[dr_id],
        "experiment": experiment,
        "metric": metric,
        "results": results,
        "verdict": verdict,
        "status": "MEASURED",
    }
    path = directory / f"{dr_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compose_register(results_dir: Path) -> Path:
    """Gather every DR row; an experiment with no result file is an explicit NOT RUN row."""
    rows: list[dict[str, Any]] = []
    for dr_id, meta in DR_CATALOG.items():
        path = dr_dir(results_dir) / f"{dr_id}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    "id": dr_id,
                    **meta,
                    "experiment": "NOT RUN",
                    "metric": "NOT RUN",
                    "results": None,
                    "verdict": "NOT RUN (cut for time; no evidence either way)",
                    "status": "NOT RUN",
                }
            )
    out = results_dir / "parts" / "decision_register.json"
    out.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


# -----------------------------------------------------------------------------------
# Shared ranking lab
# -----------------------------------------------------------------------------------


@dataclasses.dataclass
class Lab:
    train_frame: pd.DataFrame
    holdout_frame: pd.DataFrame
    interactions_train: pd.DataFrame
    pois_df: pd.DataFrame
    model_cfg: ModelConfig
    eval_cfg: EvalConfig


def load_lab(
    data_dir: Path, feature_cfg: FeatureBuildConfig, model_cfg: ModelConfig, eval_cfg: EvalConfig
) -> Lab:
    budget = feature_cfg.traveler_features.budget_target_price_level
    return Lab(
        train_frame=load_train_ranking_frame(data_dir, budget),
        holdout_frame=load_holdout_evaluation_frame(data_dir, budget),
        interactions_train=pd.read_parquet(data_dir / "interactions_train.parquet"),
        pois_df=pd.read_parquet(data_dir / "pois_prepared.parquet"),
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
    )


def per_trip_ndcg10(frame: pd.DataFrame, score: pd.Series) -> pd.Series:
    return compute_all_trip_metrics(frame, score, (10,), (), ())["ndcg@10"]


def summarize_ndcg(per_trip: pd.Series, eval_cfg: EvalConfig) -> dict[str, float]:
    b = eval_cfg.bootstrap
    agg = aggregate_metric(per_trip, b.n_resamples, eval_cfg.seed, b.ci_low_pct, b.ci_high_pct)
    return {"mean": agg.mean, "ci_low": agg.ci_low, "ci_high": agg.ci_high, "n": agg.n_included}


def sample_train_trips(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame
    trips = np.sort(frame["trip_id"].unique())
    keep = np.random.default_rng(seed).choice(
        trips, size=int(round(fraction * len(trips))), replace=False
    )
    return frame.loc[frame["trip_id"].isin(set(keep.tolist()))]


def fit_lgb(
    lab: Lab,
    *,
    objective: str = "lambdarank",
    ips: bool = True,
    ips_clip_high: float | None = None,
    trip_fraction: float = 1.0,
    seed: int = SEED,
    train_frame: pd.DataFrame | None = None,
    drop_prefixes: tuple[str, ...] = (),
) -> tuple[lgb.Booster, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    """System-8 recipe (IPS + behavioural dropout, early stopping on a train-carved val split),
    with the DR-varied knob swapped in. Returns (booster, numeric, categorical, fit, val)."""
    cfg = lab.model_cfg.lambdamart
    base = lab.train_frame if train_frame is None else train_frame
    sub = sample_train_trips(base, trip_fraction, seed)
    numeric = [c for c in numeric_feature_columns(sub) if not c.startswith(drop_prefixes)]
    categorical = categorical_feature_columns(sub)
    fit, val = lm.train_val_split_by_trip(sub, cfg.val_fraction, cfg.val_split_seed)
    weights = None
    if ips:
        p = lm.train_frame_p_expose(fit, lab.interactions_train, lab.pois_df)
        clip = cfg.ips_clip_high if ips_clip_high is None else ips_clip_high
        weights = lm.compute_ips_weights(fit["trip_id"], p, cfg.ips_clip_low, clip)
    fit_d, _ = lm.apply_behavioral_dropout(
        fit, cfg.behavioral_dropout_rate, cfg.behavioral_dropout_seed
    )

    params = lm._lgb_params(cfg, seed)
    params["objective"] = objective
    label_fit = fit_d["label"].to_numpy(dtype=np.float64)
    label_val = val["label"].to_numpy(dtype=np.float64)
    if objective == "binary":
        label_fit, label_val = (label_fit >= 1).astype(float), (label_val >= 1).astype(float)
    x_fit = lm._feature_matrix(fit_d, numeric, categorical)
    x_val = lm._feature_matrix(val, numeric, categorical)
    train_set = lgb.Dataset(
        x_fit,
        label=label_fit,
        group=lm._group_sizes(fit_d),
        weight=None if weights is None else weights.to_numpy(dtype=np.float64),
        categorical_feature=categorical,
        free_raw_data=False,
    )
    # Validation is always scored as NDCG@10 on the GRADED labels (early-stopping metric), so
    # objectives are compared on the metric that is reported, whatever they optimise.
    val_set = lgb.Dataset(
        x_val,
        label=val["label"].to_numpy(dtype=np.float64),
        group=lm._group_sizes(val),
        reference=train_set,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    del label_val
    params["metric"] = "ndcg"
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return booster, numeric, categorical, fit, val


def score_lgb(
    booster: lgb.Booster, numeric: list[str], categorical: list[str], frame: pd.DataFrame
) -> pd.Series:
    return lm.score_booster(booster, frame, numeric, categorical)


# -----------------------------------------------------------------------------------
# DR1 / DR6 / DR10: scoring-layer experiments on one scoring-pipeline pass
# -----------------------------------------------------------------------------------

_SUB_SCORES = (
    "budget_fit",
    "mobility_fit",
    "hours_fit",
    "reservation_fit",
    "party_fit",
    "duration_fit",
)


def _top_k_stats(
    full: pd.DataFrame, score: npt.NDArray[np.float64], k: int = 10
) -> dict[str, float]:
    """Top-`k` per trip by `score` over ALL candidates in `full`: hard-constraint violations
    (rows with hard_gate == 0), trips with >= 1 violation, mean compatibility@k, NDCG@k."""
    work = full[["trip_id", "poi_id", "label", "hard_gate", "compatibility"]].assign(score=score)
    work = work.sort_values(["trip_id", "score", "poi_id"], ascending=[True, False, True])
    top = work.groupby("trip_id", sort=True).head(k)
    # A -inf score means "filtered out before ranking" (the production hard gate): such a row is
    # never RETURNED, so it can neither count as a violation nor enter compat@k.
    top = top.loc[np.isfinite(top["score"])]
    violations = int((top["hard_gate"] == 0).sum())
    trips_with = int((top.assign(v=top["hard_gate"] == 0).groupby("trip_id")["v"].any()).sum())
    ndcgs = []
    for _t, g in work.groupby("trip_id", sort=True):
        v = ndcg_at_k(
            g["label"].to_numpy(dtype=np.int64),
            g["score"].to_numpy(dtype=np.float64),
            g["poi_id"].to_numpy(dtype=object),
            k,
        )
        if v is not None:
            ndcgs.append(v)
    return {
        "hard_constraint_violations_in_top10": violations,
        "trips_with_violation": trips_with,
        "n_trips": int(work["trip_id"].nunique()),
        "mean_compat_at_10": float(top["compatibility"].mean()),
        "ndcg_at_10": float(np.mean(ndcgs)),
    }


def run_dr_scoring(
    data_dir: Path,
    artifacts_dir: Path,
    results_dir: Path,
    feature_cfg: FeatureBuildConfig,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
) -> None:
    result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        candidates_cfg.geo,
        candidates_cfg.longtail.pop_pct_cutoff,
    )
    full: pd.DataFrame = result["full_frame"]
    a, b = scoring_cfg.utility.alpha, scoring_cfg.utility.beta
    rel = full["relevance"].to_numpy(dtype=np.float64)
    comp = full["compatibility"].to_numpy(dtype=np.float64)
    gate = full["hard_gate"].to_numpy(dtype=np.float64)

    # DR1: same candidates, same relevance/compatibility; only the combination rule differs.
    variants = {
        "multiplicative_gated (production)": np.where(gate == 1.0, rel**a * comp**b, -np.inf),
        "multiplicative_no_gate": rel**a * comp**b,
        "additive_no_gate (brief)": a * rel + b * comp,
        "additive_with_gate_filter": np.where(gate == 1.0, a * rel + b * comp, -np.inf),
    }
    dr1 = {name: _top_k_stats(full, s) for name, s in variants.items()}
    prod = dr1["multiplicative_gated (production)"]
    add = dr1["additive_no_gate (brief)"]
    write_dr(
        results_dir,
        "DR1",
        "Same holdout candidates/relevance/compatibility; rank by each combination rule; count "
        "hard-constraint violations (hard_gate == 0) among the top-10 of every holdout trip.",
        "hard-constraint violations in top-10; mean compat@10; NDCG@10",
        dr1,
        f"Additive scoring puts {add['hard_constraint_violations_in_top10']} hard-constraint "
        f"violations into {add['trips_with_violation']}/{add['n_trips']} trips' top-10; the "
        f"multiplicative gated rule puts {prod['hard_constraint_violations_in_top10']}. NDCG@10 "
        f"{add['ndcg_at_10']:.4f} (additive) vs {prod['ndcg_at_10']:.4f} (multiplicative gated); "
        f"mean compat@10 {add['mean_compat_at_10']:.4f} vs {prod['mean_compat_at_10']:.4f}.",
    )

    # DR6: compatibility aggregation (utility gated, and ungated to expose the aggregator).
    stacked = np.stack([full[c].to_numpy(dtype=np.float64) for c in _SUB_SCORES], axis=1)
    aggs = {
        "geometric_mean (production)": np.power(np.prod(stacked, axis=1), 1.0 / stacked.shape[1]),
        "min": stacked.min(axis=1),
        "product": np.prod(stacked, axis=1),
        "arithmetic_mean": stacked.mean(axis=1),
    }
    dr6: dict[str, Any] = {}
    for name, c in aggs.items():
        f2 = full.assign(compatibility=c)
        dr6[name] = {
            "gated": _top_k_stats(f2, np.where(gate == 1.0, rel**a * c**b, -np.inf)),
            "ungated": _top_k_stats(f2, rel**a * c**b),
        }
    g = dr6["geometric_mean (production)"]["ungated"]
    write_dr(
        results_dir,
        "DR6",
        "Recompute compatibility from the 6 sub-scores with each aggregator; rank gated and "
        "ungated multiplicative utility.",
        "NDCG@10; hard-constraint violations in top-10; mean compat@10 (of the aggregator used)",
        dr6,
        "Ungated hard-violation counts: "
        + ", ".join(
            f"{k}={v['ungated']['hard_constraint_violations_in_top10']}" for k, v in dr6.items()
        )
        + "; NDCG@10 gated: "
        + ", ".join(f"{k}={v['gated']['ndcg_at_10']:.4f}" for k, v in dr6.items())
        + f". Geometric mean ungated NDCG {g['ndcg_at_10']:.4f}.",
    )

    # DR10: alpha x beta grid (gated multiplicative).
    grid = []
    for alpha in (0.5, 1.0, 2.0):
        for beta in (0.0, 0.3, 0.7, 1.0, 1.5):
            s = _top_k_stats(full, np.where(gate == 1.0, rel**alpha * comp**beta, -np.inf))
            grid.append({"alpha": alpha, "beta": beta, **s})
    best = max(grid, key=lambda r: r["ndcg_at_10"])
    cur = next(r for r in grid if r["alpha"] == a and r["beta"] == b)
    write_dr(
        results_dir,
        "DR10",
        "Gated multiplicative utility over an alpha x beta grid on the same holdout scoring pass.",
        "NDCG@10 and mean compat@10 per (alpha, beta)",
        grid,
        f"Shipped (alpha={a}, beta={b}): NDCG@10 {cur['ndcg_at_10']:.4f}, compat@10 "
        f"{cur['mean_compat_at_10']:.4f}. NDCG-best cell (alpha={best['alpha']}, "
        f"beta={best['beta']}): {best['ndcg_at_10']:.4f}, compat@10 "
        f"{best['mean_compat_at_10']:.4f}.",
    )
