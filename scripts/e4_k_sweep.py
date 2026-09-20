"""E4 (pre-registered): full K sweep and the blind rule "smallest K with IPS-weighted VALIDATION
recall >= 0.93". Derived from the A3 prototype (same fit/val protocol).

A3 prototype: does a learned (IPS-weighted, oracle-free) first-stage retriever over the FULL
destination catalog fix candidate recall? Selection uses train-carved validation only; the
holdout number is printed once at the end, reporting-only."""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from poi_rank.candidates.recall_metrics import _canonical_holdout
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.models.ranking_data import _load_common, build_ranking_frame

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "synthetic"
t0 = time.perf_counter()
cfg = FeatureBuildConfig.from_yaml(ROOT / "configs" / "features.yaml")
bt = cfg.traveler_features.budget_target_price_level
d = _load_common(D)
trips, pois = d["trips_df"], d["pois_df"]
train_ix = pd.read_parquet(D / "interactions_train.parquet")
cmap = build_poi_id_canonical_map(pois)
tr = remap_interaction_poi_ids(train_ix, cmap)

rng = np.random.default_rng(42)
train_trips = trips.loc[~trips["is_holdout"], "trip_id"].to_numpy()
val_trips = set(rng.choice(train_trips, size=int(0.2 * len(train_trips)), replace=False))
fit_trips = set(train_trips) - val_trips

# training rows = exposed (trip, poi) impressions only, IPS-weighted
imp = tr.groupby(["trip_id", "poi_id"], as_index=False).agg(
    label=("label", "max"), p_expose=("p_expose", "min")
)
imp_all = imp
imp = imp.loc[imp["trip_id"].isin(fit_trips)]
fit = build_ranking_frame(
    imp[["trip_id", "poi_id"]],
    fit_trips,
    train_ix,
    trips,
    d["travelers_df"],
    pois,
    d["poi_features_df"],
    d["traveler_features_df"],
    bt,
)
w = imp.set_index(["trip_id", "poi_id"])["p_expose"]
fit_w = 1.0 / np.clip(
    w.reindex(pd.MultiIndex.from_frame(fit[["trip_id", "poi_id"]])).to_numpy(), 0.05, 1.0
)
fit_w = fit_w / fit_w.mean()
print("fit frame", fit.shape, f"{time.perf_counter()-t0:.0f}s", flush=True)

drop_prefix = ("text_emb_", "implicit_taste_")
feat_cols = [
    c
    for c in fit.columns
    if c not in ("label", "trip_id", "poi_id", "traveler_id")
    and not c.startswith(drop_prefix)
    and (pd.api.types.is_numeric_dtype(fit[c]) or pd.api.types.is_bool_dtype(fit[c]))
]
print(len(feat_cols), "features")
X = fit[feat_cols].astype(np.float32)
y = (fit["label"] >= 1).astype(int)
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=12,
    verbose=-1,
)
model.fit(X, y, sample_weight=fit_w)
print("fit done", f"{time.perf_counter()-t0:.0f}s", flush=True)


def full_catalog(trip_ids: set[str], ix: pd.DataFrame) -> pd.DataFrame:
    tt = trips.loc[trips["trip_id"].isin(trip_ids), ["trip_id", "destination"]]
    allp = pois[["poi_id", "destination"]]
    pairs = tt.merge(allp, on="destination")[["trip_id", "poi_id"]]
    return build_ranking_frame(
        pairs,
        trip_ids,
        ix,
        trips,
        d["travelers_df"],
        pois,
        d["poi_features_df"],
        d["traveler_features_df"],
        bt,
    )


def recall_at(fr: pd.DataFrame, K: int, exclude_seen: bool = False) -> dict[str, float]:
    fr = fr.assign(score=model.predict_proba(fr[feat_cols].astype(np.float32))[:, 1])
    pop = fr["num_pop_pct"]
    out = {}
    for name, lo, hi in (("overall", 0, 1.01), ("long_tail", 0, 0.5)):
        rs = []
        for _, g in fr.groupby("trip_id", sort=False):
            pos = g["label"] >= 1
            pos &= (g["num_pop_pct"] >= lo) & (g["num_pop_pct"] < hi)
            if pos.sum() == 0:
                continue
            top = g["score"].nlargest(K).index
            rs.append(pos.loc[top].sum() / pos.sum())
        out[name] = float(np.mean(rs))
    return out


cand = pd.read_parquet(D / "candidates.parquet")  # channel flags: longtail + interest
n_dest = pois.groupby("destination").size().to_dict()
dest_of = dict(zip(trips["trip_id"], trips["destination"], strict=True))
CH = [
    "channel_geo",
    "channel_interest",
    "channel_semantic",
    "channel_cf",
    "channel_longtail",
    "channel_archetype",
]
SUBSETS = {"longtail+interest": ["channel_longtail", "channel_interest"]}
sets = {}
for name, chans in SUBSETS.items():
    m = cand[chans].any(axis=1) if chans else pd.Series(False, index=cand.index)
    sets[name] = {t: set(g["poi_id"]) for t, g in cand.loc[m].groupby("trip_id")}


def union_recall(fr, K, name, ips=None):
    fr = fr.assign(score=model.predict_proba(fr[feat_cols].astype(np.float32))[:, 1])
    if ips is not None:
        fr = fr.assign(w=ips.reindex(pd.MultiIndex.from_frame(fr[["trip_id", "poi_id"]])).to_numpy())
    else:
        fr = fr.assign(w=1.0)
    res = {"overall": [], "long_tail": []}
    sizes, chances = [], []
    for t, g in fr.groupby("trip_id", sort=False):
        u = set(g.nlargest(K, "score")["poi_id"]) | sets[name].get(t, set())
        sizes.append(len(u))
        chances.append(len(u) / n_dest[dest_of[t]])
        pos = g.loc[g["label"] >= 1]
        if len(pos):
            res["overall"].append(np.average(pos["poi_id"].isin(u), weights=pos["w"]))
        lt = pos.loc[pos["num_pop_pct"] < 0.5]
        if len(lt):
            res["long_tail"].append(np.average(lt["poi_id"].isin(u), weights=lt["w"]))
    o = float(np.mean(res["overall"]))
    c = float(np.mean(chances))
    return dict(
        recall=round(o, 3),
        lt=round(float(np.mean(res["long_tail"])), 3),
        size=round(float(np.mean(sizes)), 0),
        lift=round(o - c, 3),
    )


val_fr = full_catalog(val_trips, train_ix)
hold_trips = set(trips.loc[trips["is_holdout"], "trip_id"])
hr = pd.read_parquet(D / "interactions_holdout_random.parquet")
h_fr = full_catalog(hold_trips, hr)
import json

ips_val = (1.0 / np.clip(imp_all.set_index(["trip_id", "poi_id"])["p_expose"], 0.05, 1.0))
grid = []
RULE = 0.93
for K in range(90, 361, 15):
    val = union_recall(val_fr, K, "longtail+interest", ips_val)
    hold = union_recall(h_fr, K, "longtail+interest")
    grid.append({"learned_K": K, "val_ips_weighted_selection": val, "holdout_reporting_only": hold})
    print(K, val, hold, flush=True)
rule_k = min(g["learned_K"] for g in grid if g["val_ips_weighted_selection"]["recall"] >= RULE)
# Amendment 2 (docs/experiments/H-ranker-cross-features.md): on the corrected features the recall
# rule K breaks the BLOCKING Gate-B lift row (>= 0.35) already on validation; the shipped K is the
# largest grid K whose VALIDATION lift is >= gate + margin.
GATE_LIFT, LIFT_MARGIN = 0.35, 0.01
feasible = [
    g["learned_K"] for g in grid if g["val_ips_weighted_selection"]["lift"] >= GATE_LIFT + LIFT_MARGIN
]
gate_feasible_k = max(feasible)
shipped_k = int(
    yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))["candidates"][
        "learned"
    ]["quota"]
)
by_k = {g["learned_K"]: g for g in grid}
out = {
    "gate_b_lift_row_target": GATE_LIFT,
    "gate_feasible_rule": f"largest K whose validation lift >= {GATE_LIFT} + {LIFT_MARGIN} margin",
    "gate_feasible_k": gate_feasible_k,
    "shipped_k": shipped_k,
    "at_rule_k": by_k[rule_k],
    "at_shipped_k": by_k[shipped_k],
    "pre_registered_rule": f"smallest K with IPS-weighted validation overall recall >= {RULE} "
    "(val = 20% of train trips vs logged positives, weighted 1/clip(p_expose)); applied once, "
    "blind to the holdout column, which is reporting-only",
    "rule_k": rule_k,
    "holdout_at_rule_k_reporting_only": next(
        g["holdout_reporting_only"] for g in grid if g["learned_K"] == rule_k
    ),
    "grid": grid,
}
(ROOT / "results" / "parts" / "e4_k_sweep.json").write_text(json.dumps(out, indent=2))
print("RULE_K", rule_k, out["holdout_at_rule_k_reporting_only"])
print("DONE")
