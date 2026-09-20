"""K1: is the touristiness-preference axis learnable, and does the final ranker use it?

Diagnosis only -- nothing is re-tuned or re-selected; the shipped booster is only scored. Writes
`results/parts/touristiness_axis.json` (composed into metrics.json; the docs cite it by key).

  a. Distribution of `touristiness_pref` over the travelers.
  b. Are the preference-dependent columns (the localness x touristiness crosses among them)
     non-degenerate in the ranker's training frame, and does the shipped booster split on them?
  c. Does the simulator encode the axis in behaviour? On holdout trips (random-exposure labels,
     observable localness index only -- no oracle read): per-trip rank correlation between a
     candidate's localness and its label, regressed on the trip's stated preference.
  d. Does the shipped ranker reproduce that gradient? Same statistic on the raw ranker score,
     split by cold-start (no history) versus warm trips.
  e. The counterfactual flip of J2 (raw top-10 Jaccard) stratified by |pref| and by history.

    uv run python scripts/diagnose_touristiness_axis.py
"""

from __future__ import annotations

import torch  # noqa: F401, I001  (must import before pandas on this machine)

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.traveler_features import localness_preference_gap
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.cross_ranking import attach_cross_features
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
ART = ROOT / "artifacts"
OUT = ROOT / "results" / "parts" / "touristiness_axis.json"
PREF_COLUMNS = (
    "explicit_touristiness_pref",
    "interact_localness_gap",
    "xf_loc_align",
    "xf_loc_gap",
    "xf_loc_x_pref",
    "xf_pop_x_pref",
)


def _distribution() -> dict[str, Any]:
    travelers = pd.read_parquet(DATA / "travelers.parquet")
    pref = travelers["touristiness_pref"].to_numpy(dtype=float)
    hist, edges = np.histogram(pref, bins=np.linspace(-1.0, 1.0, 11))
    q = np.quantile(pref, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n_travelers": int(len(pref)),
        "mean": float(pref.mean()),
        "sd": float(pref.std(ddof=1)),
        "quantiles": dict(zip(["q05", "q25", "q50", "q75", "q95"], map(float, q), strict=True)),
        "share_abs_pref_at_least_half": float((np.abs(pref) >= 0.5).mean()),
        "share_abs_pref_below_fifth": float((np.abs(pref) < 0.2).mean()),
        "hist_bin_edges": [round(float(e), 2) for e in edges],
        "hist_counts": [int(h) for h in hist],
    }


def _training_frame_and_booster(budget: Any, booster: Any) -> dict[str, Any]:
    train = load_train_ranking_frame(DATA, budget)
    names = booster.feature_name()
    gain = booster.feature_importance("gain")
    split = booster.feature_importance("split")
    share = gain / gain.sum()
    per_col: dict[str, Any] = {}
    for c in PREF_COLUMNS:
        col = train[c].astype(float)
        row: dict[str, Any] = {
            "nan_rate": float(col.isna().mean()),
            "sd": float(col.std()),
            "n_unique": int(col.nunique()),
            "in_booster": c in names,
        }
        if c in names:
            i = names.index(c)
            row.update(
                splits=int(split[i]),
                gain_share=float(share[i]),
                gain_rank=int((gain > gain[i]).sum() + 1),
            )
        per_col[c] = row
    return {
        "n_train_rows": int(len(train)),
        "n_booster_features": len(names),
        "columns": per_col,
        "pref_dependent_gain_share_total": float(
            sum(share[names.index(c)] for c in PREF_COLUMNS if c in names)
        ),
        "xf_loc_x_pref_within_trip_sd_mean": float(
            train.groupby("trip_id")["xf_loc_x_pref"].std().mean()
        ),
    }


def _gradient(frame: pd.DataFrame) -> dict[str, Any]:
    """Across trips: does the localness-vs-outcome rank correlation shift with stated pref?"""
    rows = []
    for _, g in frame.groupby("trip_id"):
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
    t = pd.DataFrame(rows, columns=["pref", "n_hist", "rho_label", "rho_model"]).dropna()

    def one(x: pd.DataFrame) -> dict[str, Any]:
        lab = stats.linregress(x["pref"], x["rho_label"])
        mod = stats.linregress(x["pref"], x["rho_model"])
        return {
            "n_trips": int(len(x)),
            "label_pref_vs_rho_spearman": float(
                stats.spearmanr(x["pref"], x["rho_label"]).statistic
            ),
            "label_slope": float(lab.slope),
            "model_pref_vs_rho_spearman": float(
                stats.spearmanr(x["pref"], x["rho_model"]).statistic
            ),
            "model_slope": float(mod.slope),
            "model_over_label_slope": float(mod.slope / lab.slope),
        }

    return {
        "all": one(t),
        "warm": one(t[t["n_hist"] > 0]),
        "cold_start": one(t[t["n_hist"] == 0]),
        "note": (
            "per trip: Spearman(observable localness index, outcome) over the candidate rows, "
            "then regressed on the trip's stated touristiness_pref; label = random-exposure "
            "holdout label, model = raw ranker score"
        ),
    }


def _flip(frame: pd.DataFrame, booster: Any, budget: Any) -> dict[str, Any]:
    num, cat = numeric_feature_columns(frame), categorical_feature_columns(frame)
    flipped = frame.copy()
    flipped["explicit_touristiness_pref"] = -flipped["explicit_touristiness_pref"]
    flipped["interact_localness_gap"] = localness_preference_gap(
        flipped["num_localness"].to_numpy(dtype=np.float64),
        flipped["explicit_touristiness_pref"].to_numpy(dtype=np.float64),
    )
    flipped = attach_cross_features(flipped, DATA, budget)
    flipped["_alt"] = np.asarray(lm.score_booster(booster, flipped, num, cat))
    alt = flipped["_alt"]
    rows = []
    for tid, g in frame.groupby("trip_id"):
        a = set(g.nlargest(10, "_score")["poi_id"])
        b = set(frame.loc[g.index].assign(_alt=alt.loc[g.index]).nlargest(10, "_alt")["poi_id"])
        rows.append(
            (
                tid,
                len(a & b) / len(a | b),
                float(g["implicit_interaction_count"].iloc[0]),
                abs(float(g["explicit_touristiness_pref"].iloc[0])),
            )
        )
    d = pd.DataFrame(rows, columns=["trip_id", "jac", "n_hist", "abs_pref"])
    d["cold"] = d["n_hist"] == 0
    d["abs_pref_tercile"] = pd.qcut(d["abs_pref"], 3, labels=["low", "mid", "high"])
    by_tercile = d.groupby("abs_pref_tercile", observed=True).agg(
        n=("jac", "size"), mean_abs_pref=("abs_pref", "mean"), jaccard=("jac", "mean")
    )
    cross = d.groupby(["abs_pref_tercile", "cold"], observed=True)["jac"].agg(["size", "mean"])
    warm = d[~d["cold"]].copy()
    x_all = np.column_stack([np.ones(len(d)), d["abs_pref"], d["cold"].astype(float)])
    beta_all = np.linalg.lstsq(x_all, d["jac"].to_numpy(), rcond=None)[0]
    x_w = np.column_stack([np.ones(len(warm)), warm["abs_pref"], np.log1p(warm["n_hist"])])
    beta_w = np.linalg.lstsq(x_w, warm["jac"].to_numpy(), rcond=None)[0]
    warm["hist_tercile"] = pd.qcut(warm["n_hist"], 3, labels=["low", "mid", "high"])
    return {
        "n_trips": int(len(d)),
        "mean_jaccard": float(d["jac"].mean()),
        "mean_abs_pref_cold": float(d.loc[d["cold"], "abs_pref"].mean()),
        "mean_abs_pref_warm": float(d.loc[~d["cold"], "abs_pref"].mean()),
        "spearman_abs_pref_vs_jaccard": float(stats.spearmanr(d["abs_pref"], d["jac"]).statistic),
        "jaccard_by_abs_pref_tercile": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in by_tercile.to_dict("index").items()
        },
        "jaccard_by_abs_pref_tercile_and_history": {
            f"{k[0]}_{'cold' if k[1] else 'warm'}": {
                "n": int(v["size"]),
                "jaccard": float(v["mean"]),
            }
            for k, v in cross.to_dict("index").items()
        },
        "ols_all_jaccard_on_abs_pref_and_cold": {
            "intercept": float(beta_all[0]),
            "abs_pref": float(beta_all[1]),
            "cold_start": float(beta_all[2]),
        },
        "ols_warm_jaccard_on_abs_pref_and_log1p_history": {
            "intercept": float(beta_w[0]),
            "abs_pref": float(beta_w[1]),
            "log1p_history": float(beta_w[2]),
        },
        "spearman_history_vs_jaccard_warm": float(
            stats.spearmanr(warm["n_hist"], warm["jac"]).statistic
        ),
        "jaccard_by_history_tercile_warm": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in warm.groupby("hist_tercile", observed=True)
            .agg(n=("jac", "size"), mean_history=("n_hist", "mean"), jaccard=("jac", "mean"))
            .to_dict("index")
            .items()
        },
    }


def main() -> None:
    cfg = FeatureBuildConfig.from_yaml(ROOT / "configs" / "features.yaml")
    budget = cfg.traveler_features.budget_target_price_level
    booster = lm.load_boosters(ART)["lambdamart_ips"]
    hold = load_holdout_evaluation_frame(DATA, budget)
    num, cat = numeric_feature_columns(hold), categorical_feature_columns(hold)
    hold["_score"] = np.asarray(lm.score_booster(booster, hold, num, cat))
    result = {
        "method": (
            "scripts/diagnose_touristiness_axis.py: diagnosis of the shipped booster "
            "(lambdamart_ips) on the committed frames; no retraining, no selection"
        ),
        "a_preference_distribution": _distribution(),
        "b_training_frame_and_booster": _training_frame_and_booster(budget, booster),
        "c_d_localness_gradient_by_stated_pref": _gradient(hold),
        "e_flip_stratified": _flip(hold, booster, budget),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
