"""Experiment H runner (validation only; see docs/experiments/H-ranker-cross-features.md).

    uv run python scripts/h_run.py prep      # sanity: load the frames (they carry the xf_ columns)
    uv run python scripts/h_run.py h0a       # grouped SHAP on the shipped model
    uv run python scripts/h_run.py h0b       # cross-features-only probe vs shipped
    uv run python scripts/h_run.py h1        # full model + cross features (and group ablations)
    uv run python scripts/h_run.py h2        # init_score = content cosine
    uv run python scripts/h_run.py h3        # objective / group tuning on the best config so far

Results go to results/experiments/h/<stage>.json. The holdout is never read here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch  # noqa: F401, I001  (must import before pandas on this machine)

from poi_rank.eval.config import EvalConfig
from poi_rank.eval.decision_register import load_lab
from poi_rank.eval.ranker_h import (
    ADOPTION_BAR,
    SEEDS,
)
from poi_rank.eval.ranker_sweep import _Cache
from poi_rank.explain.shap_groups import compute_grouped_shap
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import ModelConfig
from poi_rank.scoring.config import ScoringConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
OUT = ROOT / "results" / "experiments" / "h"
SHIPPED = {"objective": "lambdarank", "clip": 20.0}
PROBE = (
    "interact_cos_taste_poi",
    "interact_category_affinity",
    "interact_interest_match",
    "xf_loc_align",
    "xf_price_signed",
    "xf_party_fit",
    "xf_mobility_fit",
    "xf_interest_cover",
    "xf_pop_x_pref",
    "xf_prior_poi_engaged",
)
GROUPS = {
    "loc": ("xf_loc_align", "xf_loc_gap", "xf_loc_x_pref", "xf_pop_x_pref"),
    "price": ("xf_price_signed", "xf_price_over", "xf_price_under"),
    "interest": (
        "xf_interest_tag_hits",
        "xf_interest_cat_hit",
        "xf_interest_cover",
        "xf_interest_tag_ratio",
    ),
    "compat": (
        "xf_budget_fit",
        "xf_mobility_fit",
        "xf_hours_fit",
        "xf_party_fit",
        "xf_duration_fit",
        "xf_travel_min",
        "xf_travel_ratio",
    ),
    "history": ("xf_prior_poi_engaged", "xf_cos_dismissed"),
}


def _cfgs() -> tuple[FeatureBuildConfig, ModelConfig, EvalConfig, ScoringConfig]:
    c = ROOT / "configs"
    return (
        FeatureBuildConfig.from_yaml(c / "features.yaml"),
        ModelConfig.from_yaml(c / "model.yaml"),
        EvalConfig.from_yaml(c / "eval.yaml"),
        ScoringConfig.from_yaml(c / "scoring.yaml"),
    )


def _lab() -> Any:
    feat, model, ev, _scoring = _cfgs()
    return load_lab(DATA, feat, model, ev)  # frames already carry the xf_ cross features


def _seeds(cache: _Cache, **kw: Any) -> dict[str, Any]:
    per = [
        cache.score(
            kw.get("objective", "lambdarank"),
            kw.get("clip", 20.0),
            kw.get("drop", ()),
            s,
            only=kw.get("only"),
            params=kw.get("params"),
            init_scale=kw.get("init_scale"),
        )[0]
        for s in SEEDS
    ]
    return {"per_seed": per, "mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1))}


def _save(name: str, payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def h0a() -> None:
    lab = _lab()
    cache = _Cache(lab)
    numeric = numeric_feature_columns(lab.train_frame)  # the shipped model uses the xf_ columns too
    cat = categorical_feature_columns(lab.train_frame)
    booster = lgb.Booster(model_file=str(ROOT / "artifacts" / "model.txt"))
    val = cache.val
    trips = np.sort(val["trip_id"].unique())
    keep = set(np.random.default_rng(42).choice(trips, size=min(300, len(trips)), replace=False))
    sample = val.loc[val["trip_id"].isin(keep)]
    res = compute_grouped_shap(booster, sample, numeric, cat)
    contrib = res.group_contributions
    groups = [c for c in contrib.columns if c not in ("trip_id", "poi_id")]
    mean_abs = {g: float(contrib[g].abs().mean()) for g in groups}
    total = sum(mean_abs.values())
    share = {g: v / total for g, v in mean_abs.items()}
    _save(
        "h0a_shap_groups",
        {
            "n_rows": int(len(sample)),
            "n_trips": len(keep),
            "mean_abs_shap": mean_abs,
            "share": share,
        },
    )
    for g, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"{g:16s} {v:6.3f}  mean|shap|={mean_abs[g]:.4f}")


def h0b() -> None:
    cache = _Cache(_lab())
    shipped = _seeds(cache, drop=("xf_",))
    probe = _seeds(cache, only=PROBE)
    _save(
        "h0b_probe",
        {"shipped_237": shipped, "probe_cross_only": probe, "probe_features": list(PROBE)},
    )
    print("shipped", shipped)
    print("probe  ", probe)


def h1() -> None:
    cache = _Cache(_lab())
    out: dict[str, Any] = {"full_plus_cross": _seeds(cache)}
    print("full+cross", out["full_plus_cross"], flush=True)
    for g in GROUPS:
        drop_others = tuple(c for gg, cc in GROUPS.items() if gg != g for c in cc)
        out[f"plus_{g}_only"] = _seeds(cache, drop=drop_others)
        print("plus", g, out[f"plus_{g}_only"], flush=True)
    _save("h1_cross_features", out)


def h2() -> None:
    cache = _Cache(_lab())
    out: dict[str, Any] = {}
    for label, drop in (("shipped_features", ("xf_",)), ("full_plus_cross", ())):
        for scale in (1.0, 2.0):
            out[f"{label}_init{scale:g}"] = _seeds(cache, drop=drop, init_scale=scale)
            print(label, scale, out[f"{label}_init{scale:g}"], flush=True)
    _save("h2_init_score", out)


def h3() -> None:
    best = json.loads((OUT / "h3_base.json").read_text()) if (OUT / "h3_base.json").exists() else {}
    cache = _Cache(_lab())
    init = best.get("init_scale")
    grid = {
        "trunc10": {"lambdarank_truncation_level": 10},
        "trunc20": {"lambdarank_truncation_level": 20},
        "trunc40": {"lambdarank_truncation_level": 40},
        "gain_linear": {"label_gain": [0, 1, 2, 3]},
        "gain_steep": {"label_gain": [0, 1, 5, 15]},
        "leaves15": {"num_leaves": 15},
        "leaves63": {"num_leaves": 63},
        "mdl50": {"min_data_in_leaf": 50},
        "mdl200": {"min_data_in_leaf": 200},
    }
    out = {k: _seeds(cache, params=v, init_scale=init) for k, v in grid.items()}
    for k, v in out.items():
        print(k, v["mean"], flush=True)
    _save("h3_group_tuning", {"init_scale": init, "grid": out, "bar": ADOPTION_BAR})


def h3b() -> None:
    """Combine the H3 single-parameter winners (declared before running): attribution run on the
    shipped features plus the joint configurations on full+cross. Highest 4-seed mean wins; a
    joint configuration is preferred over a simpler one only if it is >= 0.001 better."""
    cache = _Cache(_lab())
    leaves = {"num_leaves": 15}
    gain = {"label_gain": [0, 1, 2, 3]}
    trunc = {"lambdarank_truncation_level": 40}
    runs = {
        "shipped_features_leaves15": {"drop": ("xf_",), "params": leaves},
        "cross_leaves15": {"params": leaves},
        "cross_leaves15_gainlinear": {"params": {**leaves, **gain}},
        "cross_leaves15_trunc40": {"params": {**leaves, **trunc}},
        "cross_leaves15_gainlinear_trunc40": {"params": {**leaves, **gain, **trunc}},
    }
    out = {k: _seeds(cache, **v) for k, v in runs.items()}
    for k, v in out.items():
        print(k, v["mean"], v["per_seed"], flush=True)
    _save("h3b_combinations", {"runs": out, "bar": ADOPTION_BAR})


def prep() -> None:
    _lab()
    print("ok")


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.perf_counter()
    {"prep": prep, "h0a": h0a, "h0b": h0b, "h1": h1, "h2": h2, "h3": h3, "h3b": h3b}[stage]()
    print(f"[{stage}] {time.perf_counter() - t0:.0f}s")
