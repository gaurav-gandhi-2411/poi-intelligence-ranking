"""Experiment H harness: augment the ranking frame with explicit cross features and score
configurations on the train-carved VALIDATION split only (same protocol and metric as
`ranker_sweep.py`: IPS-weighted NDCG@10, seeds 42/7/11/13). See
`docs/experiments/H-ranker-cross-features.md` for the pre-registered protocol and adoption bar.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from poi_rank.data.geo_prep import haversine_km
from poi_rank.eval.decision_register import Lab
from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.features.cross_features import CrossFeatureContext, add_cross_features
from poi_rank.scoring import compatibility as compat
from poi_rank.scoring.config import ScoringConfig

SEEDS = (42, 7, 11, 13)
# ADOPTION_BAR = shipped 4-seed validation mean (0.3167, results/parts/ranker_sweep.json) + 0.010.
ADOPTION_BAR = 0.3267

COMPAT_FEATURES = (
    "xf_budget_fit",
    "xf_mobility_fit",
    "xf_hours_fit",
    "xf_reservation_fit",
    "xf_party_fit",
    "xf_duration_fit",
    "xf_travel_min",
    "xf_travel_ratio",
)


def compat_features(
    frame: pd.DataFrame,
    trips_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    budget_target: BudgetTargetPriceLevel,
    scoring_cfg: ScoringConfig,
) -> pd.DataFrame:
    """The scoring layer's six compatibility sub-scores (+ travel time and its ratio to the
    mobility mode's half-life) as RANKER features -- party fit etc. are stated-attribute x POI
    crosses the model previously only met after ranking."""
    keys = frame[["trip_id", "poi_id"]]
    ctx = compat.build_compatibility_context(
        keys, set(keys["trip_id"]), trips_df, travelers_df, pois_df
    )
    ctx = keys.merge(ctx, on=["trip_id", "poi_id"], how="left")
    cfg = scoring_cfg.compatibility
    dist_km = haversine_km(
        ctx["poi_lat"].to_numpy(dtype=float),
        ctx["poi_lon"].to_numpy(dtype=float),
        ctx["stay_lat"].to_numpy(dtype=float),
        ctx["stay_lon"].to_numpy(dtype=float),
    )
    speed = ctx["mobility"].map(cfg.mobility_fit.speed_for).to_numpy(dtype=float)
    half_life = ctx["mobility"].map(cfg.mobility_fit.half_life_for).to_numpy(dtype=float)
    travel_min = dist_km / speed * 60.0
    return pd.DataFrame(
        {
            "xf_budget_fit": compat.budget_fit_score(ctx, budget_target, cfg.budget_fit),
            "xf_mobility_fit": compat.mobility_fit_score(ctx, cfg.mobility_fit),
            "xf_hours_fit": compat.hours_fit_score(ctx, cfg.hours_fit),
            "xf_reservation_fit": compat.reservation_fit_score(ctx, cfg.reservation_fit),
            "xf_party_fit": compat.party_fit_score(ctx, cfg.party_fit),
            "xf_duration_fit": compat.duration_fit_score(ctx, cfg.duration_fit),
            "xf_travel_min": travel_min,
            "xf_travel_ratio": travel_min / half_life,
        },
        index=frame.index,
    )


def augment_frame(
    frame: pd.DataFrame,
    data_dir: Path,
    budget_target: BudgetTargetPriceLevel,
    scoring_cfg: ScoringConfig,
) -> pd.DataFrame:
    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    poi_feat = pd.read_parquet(data_dir / "poi_features.parquet")
    history = pd.concat(
        [
            pd.read_parquet(data_dir / "interactions_train.parquet"),
            pd.read_parquet(data_dir / "interactions_pretrip.parquet"),
        ],
        ignore_index=True,
    )[["traveler_id", "poi_id", "interaction_type", "label", "timestamp"]]
    train_ix = pd.read_parquet(data_dir / "interactions_train.parquet")
    ctx = CrossFeatureContext(
        trips_df=trips_df,
        session_start=train_ix.groupby("trip_id")["timestamp"].min(),
        poi_localness_reference=poi_feat["num_localness"].to_numpy(dtype=float),
        history=history,
        budget_target_price_level=budget_target,
    )
    out = add_cross_features(frame, ctx)
    comp = compat_features(out, trips_df, travelers_df, pois_df, budget_target, scoring_cfg)
    for col in comp.columns:
        out[col] = comp[col].to_numpy()
    return out


def lab_with_frame(lab: Lab, train_frame: pd.DataFrame) -> Lab:
    return replace(lab, train_frame=train_frame)


def paired_seed_stats(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired comparison of two per-seed series (a - b): mean gap, sd, paired t and Wilcoxon."""
    from scipy import stats

    d = np.asarray(a) - np.asarray(b)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b)
        w_p = float(w.pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "mean_gap": float(d.mean()),
        "sd_gap": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "n_seeds": float(len(d)),
        "n_a_above_b": float((d > 0).sum()),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": w_p,
    }
