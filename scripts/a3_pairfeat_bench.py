"""A3 (oracle-free): do centred taste cosine / category-affinity pair features raise validation
NDCG@10? Selection is on the train-carved validation split only; nothing here reads the oracle
or the holdout."""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_train_ranking_frame

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "synthetic"
mcfg = ModelConfig.from_yaml(ROOT / "configs" / "model.yaml")
fcfg = FeatureBuildConfig.from_yaml(ROOT / "configs" / "features.yaml")
train = load_train_ranking_frame(D, fcfg.traveler_features.budget_target_price_level)

taste_cols = sorted(c for c in train.columns if c.startswith("implicit_taste_"))
emb_cols = sorted(c for c in train.columns if c.startswith("text_emb_"))
pf = pd.read_parquet(D / "poi_features.parquet")
mean_emb = pf[emb_cols].to_numpy(np.float64).mean(axis=0)  # catalog mean (POI side)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.where(d > 0, d, 1.0)


taste = train[taste_cols].to_numpy(np.float64)
emb = train[emb_cols].to_numpy(np.float64)
cold = np.linalg.norm(taste, axis=1) == 0
# Centering (spec-v3 3.2): remove the shared catalog-mean direction from BOTH sides before the
# cosine. The taste vector is a mean of L2-normalised POI embeddings, so its own mean component
# is the same catalog-mean direction; subtract it, keep cold-start (zero) vectors at zero.
taste_c = np.where(cold[:, None], 0.0, taste - mean_emb / np.linalg.norm(mean_emb) * (taste @ mean_emb / np.linalg.norm(mean_emb))[:, None])
train["x_cos_taste_centered"] = cos_rows(taste_c, emb - mean_emb)
cat_cols = {c.removeprefix("implicit_category_dist_"): c for c in train.columns if c.startswith("implicit_category_dist_")}
cat_vals = train["poi_category_raw"].astype(str).to_numpy()
aff = np.zeros(len(train))
for cat, col in cat_cols.items():
    m = cat_vals == cat
    aff[m] = train.loc[m, col].to_numpy(float)
train["x_category_affinity"] = aff

fit, val = lm.train_val_split_by_trip(train, mcfg.lambdamart.val_fraction, mcfg.lambdamart.val_split_seed)
# The pair frame now ships `interact_category_affinity`; the experiment measures its value, so the
# baseline must EXCLUDE it (the `x_category_affinity` variants re-add it explicitly).
base_num = [c for c in numeric_feature_columns(train) if c != "interact_category_affinity"]
cat = categorical_feature_columns(train)


ROWS: list[dict] = []


def run(name: str, extra: list[str], seed: int = 42) -> None:
    num = base_num + extra
    x_fit, x_val = lm._feature_matrix(fit, num, cat), lm._feature_matrix(val, num, cat)
    params = lm._lgb_params(mcfg.lambdamart, seed)
    ds = lgb.Dataset(
        x_fit, label=fit["label"].to_numpy(float), group=lm._group_sizes(fit),
        categorical_feature=cat, free_raw_data=False,
    )
    dv = lgb.Dataset(
        x_val, label=val["label"].to_numpy(float), group=lm._group_sizes(val), reference=ds,
        categorical_feature=cat, free_raw_data=False,
    )
    t = time.perf_counter()
    b = lgb.train(
        params, ds, num_boost_round=mcfg.lambdamart.n_estimators, valid_sets=[dv], valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    v = float(b.best_score["val"]["ndcg@10"])
    ROWS.append({"variant": name, "seed": seed, "val_ndcg@10": v, "iters": b.best_iteration})
    print(f"{name:34s} seed={seed} val_ndcg@10={v:.4f} iters={b.best_iteration} {time.perf_counter()-t:.0f}s", flush=True)


for seed in (42, 7):
    run("baseline", [], seed)
    run("+cos_taste_centered", ["x_cos_taste_centered"], seed)
    run("+category_affinity", ["x_category_affinity"], seed)
    run("+both", ["x_cos_taste_centered", "x_category_affinity"], seed)
import json

base = {r["seed"]: r["val_ndcg@10"] for r in ROWS if r["variant"] == "baseline"}
summary = {}
for name in ("+cos_taste_centered", "+category_affinity", "+both"):
    deltas = [r["val_ndcg@10"] - base[r["seed"]] for r in ROWS if r["variant"] == name]
    summary[name] = {"mean_delta_val_ndcg@10": float(np.mean(deltas)), "per_seed": deltas}
out = {
    "rule": "adopt a pair feature iff its mean val NDCG@10 delta over 2 seeds is > 0 with the "
    "same sign on both seeds (oracle-free train-carved validation; holdout untouched)",
    "rows": ROWS,
    "summary": summary,
}
(ROOT / "results" / "parts" / "a3_pairfeat.json").write_text(json.dumps(out, indent=2))
