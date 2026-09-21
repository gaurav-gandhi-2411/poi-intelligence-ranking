"""Experiment L1+L2 harness: select the preference-alignment exponent (gamma) and the MMR lambda on
the train-carved VALIDATION split (`docs/experiments/L-final.md`, pre-registered before any run).

Everything here is re-scoring of the shipped booster: nothing is refit. A configuration is
served through
the whole scoring path (calibrated relevance -> hard gate -> compatibility -> utility with
`pref_align^gamma` -> MMR over the top pool -> top-k) and scored on validation trips only. The
flip-overlap (top-k Jaccard between a trip and the same trip with its stated touristiness preference
negated, every dependent quantity recomputed) uses no label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from poi_rank.eval import longtail as lt
from poi_rank.eval.coverage import shannon_entropy
from poi_rank.eval.ranker_sweep import ips_weighted_ndcg10
from poi_rank.features.traveler_features import localness_preference_gap
from poi_rank.scoring import diversity as div
from poi_rank.scoring.config import DiversityConfig
from poi_rank.scoring.utility import compute_utility, pref_align

FloatArray = npt.NDArray[np.float64]

GAMMAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDAS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0)
SHIPPED_GAMMA = 0.0
SHIPPED_LAMBDA = 0.8
# Pre-registered (docs/experiments/L-final.md): under a third of the holdout NDCG@10 CI half-width
# (0.0104), and a floor on category entropy relative to the shipped configuration.
NDCG_TOLERANCE = 0.003
ENTROPY_FLOOR_RATIO = 0.9


def pref_align_constants(train_frame: pd.DataFrame) -> dict[str, float]:
    """Label-free constants of `pref_align` from the TRAIN ranking frame: the mean observable
    localness index and the population sd of `(localness - mean) * (-touristiness_pref)`."""
    loc = train_frame["num_localness"].to_numpy(dtype=np.float64)
    pref = train_frame["explicit_touristiness_pref"].to_numpy(dtype=np.float64)
    center = float(loc.mean())
    numerator = (loc - center) * (-pref)
    return {"center": center, "scale": float(numerator.std(ddof=0))}


def flip_touristiness(frame: pd.DataFrame, data_dir: Any, budget: Any) -> pd.DataFrame:
    """The frame with every trip's `touristiness_pref` negated and every dependent column
    recomputed (`explicit_touristiness_pref`, `interact_localness_gap`, all `xf_*`)."""
    from poi_rank.models.cross_ranking import attach_cross_features

    flipped = frame.copy()
    flipped["explicit_touristiness_pref"] = -flipped["explicit_touristiness_pref"]
    flipped["interact_localness_gap"] = localness_preference_gap(
        flipped["num_localness"].to_numpy(dtype=np.float64),
        flipped["explicit_touristiness_pref"].to_numpy(dtype=np.float64),
    )
    return attach_cross_features(flipped, data_dir, budget)


@dataclass
class ServingVariant:
    """Validation rows for one variant (original or preference-flipped), sorted by
    `(trip_id, poi_id)` so the two variants are row-aligned."""

    full: pd.DataFrame


def utilities(
    full: pd.DataFrame, gamma: float, alpha: float, beta: float, center: float, scale: float
) -> FloatArray:
    factor = pref_align(
        full["num_localness"].to_numpy(dtype=np.float64),
        full["explicit_touristiness_pref"].to_numpy(dtype=np.float64),
        center,
        scale,
    )
    return compute_utility(
        full["hard_gate"].to_numpy(dtype=np.float64),
        full["relevance"].to_numpy(dtype=np.float64),
        full["compatibility"].to_numpy(dtype=np.float64),
        alpha,
        beta,
        gamma,
        factor,
    )


def served_pools(
    full: pd.DataFrame, util: FloatArray, div_cfg: DiversityConfig
) -> tuple[pd.DataFrame, dict[str, div.MmrPool]]:
    """Hard-gate survivors carrying `utility`, and the lambda-independent MMR pools."""
    scored = full.assign(utility=util)
    survivors = scored.loc[scored["hard_gate"] == 1.0].reset_index(drop=True)
    return survivors, div.build_mmr_pools(survivors, div_cfg)


def lists_at_lambda(
    survivors: pd.DataFrame,
    pools: dict[str, div.MmrPool],
    div_cfg: DiversityConfig,
    lam: float,
    k: int,
) -> dict[str, list[str]]:
    reranked = div.mmr_rerank_all_trips(survivors, div_cfg, lam, k, pools)
    ordered = reranked.sort_values(["trip_id", "mmr_rank"])
    return {str(t): g["poi_id"].tolist() for t, g in ordered.groupby("trip_id", sort=True)}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def list_score_column(full: pd.DataFrame, lists: dict[str, list[str]]) -> FloatArray:
    """A per-row score that ranks each trip's served list first (best first) and everything else
    below it, for metrics defined on a score over all candidate rows."""
    position = {(t, p): len(ps) - i for t, ps in lists.items() for i, p in enumerate(ps)}
    keys = zip(full["trip_id"].astype(str), full["poi_id"].astype(str), strict=True)
    return np.array([float(position.get(key, 0)) for key in keys], dtype=np.float64)


def category_entropy(lists: dict[str, list[str]], category_by_poi: dict[str, str]) -> float:
    counts: dict[str, int] = {}
    for recs in lists.values():
        for poi in recs:
            cat = category_by_poi.get(poi, "unknown")
            counts[cat] = counts.get(cat, 0) + 1
    return shannon_entropy(np.array(list(counts.values()), dtype=np.float64), base=2.0)


def evaluate_config(
    base: pd.DataFrame,
    flipped: pd.DataFrame,
    gamma: float,
    lambdas: tuple[float, ...],
    ctx: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per lambda at this gamma: V-NDCG@10, V-entropy, V-LT-precision and flip-overlap
    (overall / cold-start / with-history)."""
    k = ctx["k"]
    div_cfg: DiversityConfig = ctx["div_cfg"]
    args = (gamma, ctx["alpha"], ctx["beta"], ctx["center"], ctx["scale"])
    surv_b, pools_b = served_pools(base, utilities(base, *args), div_cfg)
    surv_f, pools_f = served_pools(flipped, utilities(flipped, *args), div_cfg)
    trip_cold: dict[str, bool] = ctx["trip_cold"]
    out: list[dict[str, Any]] = []
    for lam in lambdas:
        lists_b = lists_at_lambda(surv_b, pools_b, div_cfg, lam, k)
        lists_f = lists_at_lambda(surv_f, pools_f, div_cfg, lam, k)
        overlap = {t: jaccard(lists_b[t], lists_f[t]) for t in lists_b}
        cold = [v for t, v in overlap.items() if trip_cold[t]]
        warm = [v for t, v in overlap.items() if not trip_cold[t]]
        score = list_score_column(base, lists_b)
        ndcg = ips_weighted_ndcg10(base, score, ctx["p_expose"])
        lt_rep = lt.longtail_share_and_precision(
            lists_b, ctx["pop_pct_by_poi"], ctx["cutoff"], ctx["label_by_trip_poi"]
        )
        vals = np.array(list(overlap.values()))
        out.append(
            {
                "gamma": gamma,
                "lambda": lam,
                "v_ndcg10_ips": ndcg,
                "v_category_entropy_bits": category_entropy(lists_b, ctx["category_by_poi"]),
                "v_longtail_precision": lt_rep.precision,
                "v_longtail_share": lt_rep.share,
                "flip_overlap": float(vals.mean()),
                "flip_overlap_se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                "flip_overlap_cold_start": float(np.mean(cold)) if cold else float("nan"),
                "flip_overlap_with_history": float(np.mean(warm)) if warm else float("nan"),
                "n_trips": len(vals),
                "n_cold_start_trips": len(cold),
            }
        )
    return out


def select_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The pre-registered rule (docs/experiments/L-final.md). `rows` must contain the shipped
    configuration (gamma 0, lambda 0.8)."""
    shipped = next(r for r in rows if r["gamma"] == SHIPPED_GAMMA and r["lambda"] == SHIPPED_LAMBDA)
    ndcg_floor = shipped["v_ndcg10_ips"] - NDCG_TOLERANCE
    entropy_floor = ENTROPY_FLOOR_RATIO * shipped["v_category_entropy_bits"]
    feasible = [
        r
        for r in rows
        if r["v_ndcg10_ips"] >= ndcg_floor and r["v_category_entropy_bits"] >= entropy_floor
    ]
    if not feasible:
        return {
            "winner": None,
            "reason": "no configuration satisfies both constraints; gamma=0, lambda=0.8 stay",
            "ndcg_floor": ndcg_floor,
            "entropy_floor": entropy_floor,
            "n_feasible": 0,
        }
    best = min(feasible, key=lambda r: r["flip_overlap"])
    band = best["flip_overlap_se"]
    tied = [r for r in feasible if r["flip_overlap"] <= best["flip_overlap"] + band]
    winner = max(tied, key=lambda r: (r["v_longtail_precision"] or 0.0, -r["flip_overlap"]))
    return {
        "winner": {"gamma": winner["gamma"], "lambda": winner["lambda"]},
        "reason": "min flip-overlap among feasible; ties within one SE: long-tail precision",
        "ndcg_floor": ndcg_floor,
        "entropy_floor": entropy_floor,
        "n_feasible": len(feasible),
        "tie_band_se": band,
        "n_tied": len(tied),
        "min_flip_overlap": best["flip_overlap"],
    }


def gradient_reproduction(frame: pd.DataFrame, score: FloatArray) -> dict[str, Any]:
    """K1's statistic on any per-row score: per-trip Spearman(observable localness, x)
    regressed on the trip's stated touristiness preference, for the label and for the score,
    split by cold-start / with
    history; `score_over_label` is the share of the label gradient the score reproduces."""
    work = frame[
        [
            "trip_id",
            "num_localness",
            "label",
            "explicit_touristiness_pref",
            "implicit_interaction_count",
        ]
    ].assign(_score=score)
    rows = []
    for _, g in work.groupby("trip_id", sort=True):
        if g["label"].nunique() < 2:
            continue
        rows.append(
            (
                float(g["explicit_touristiness_pref"].iloc[0]),
                float(g["implicit_interaction_count"].iloc[0]),
                stats.spearmanr(g["num_localness"], g["label"]).statistic,
                stats.spearmanr(g["num_localness"], g["_score"]).statistic,
            )
        )
    t = pd.DataFrame(rows, columns=["pref", "n_hist", "rho_label", "rho_score"]).dropna()

    def one(x: pd.DataFrame) -> dict[str, float]:
        lab = stats.linregress(x["pref"], x["rho_label"])
        sc = stats.linregress(x["pref"], x["rho_score"])
        return {
            "n_trips": float(len(x)),
            "label_slope": float(lab.slope),
            "score_slope": float(sc.slope),
            "score_over_label": float(sc.slope / lab.slope),
            "score_pref_spearman": float(stats.spearmanr(x["pref"], x["rho_score"]).statistic),
        }

    return {
        "all": one(t),
        "cold_start": one(t[t["n_hist"] == 0]),
        "with_history": one(t[t["n_hist"] > 0]),
    }
