"""A4: LightGBM lambdarank speed/quality micro-benchmark (histogram layout, max_bin, lr)."""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_train_ranking_frame

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "synthetic"
cfgs = ModelConfig.from_yaml(ROOT / "configs" / "model.yaml")
fcfg = FeatureBuildConfig.from_yaml(ROOT / "configs" / "features.yaml")
train = load_train_ranking_frame(D, fcfg.traveler_features.budget_target_price_level)
num, cat = numeric_feature_columns(train), categorical_feature_columns(train)
fit, val = lm.train_val_split_by_trip(
    train, cfgs.lambdamart.val_fraction, cfgs.lambdamart.val_split_seed
)
x_fit = lm._feature_matrix(fit, num, cat)
x_val = lm._feature_matrix(val, num, cat)
print("fit rows", len(fit), "features", x_fit.shape[1])


def run(name: str, **overrides: object) -> None:
    params = lm._lgb_params(cfgs.lambdamart, 42)
    params.update(overrides)
    ds = lgb.Dataset(
        x_fit,
        label=fit["label"].to_numpy(float),
        group=lm._group_sizes(fit),
        categorical_feature=cat,
        free_raw_data=False,
        params={"max_bin": params.get("max_bin", 255)},
    )
    dv = lgb.Dataset(
        x_val,
        label=val["label"].to_numpy(float),
        group=lm._group_sizes(val),
        reference=ds,
        categorical_feature=cat,
        free_raw_data=False,
    )
    t = time.perf_counter()
    b = lgb.train(
        params,
        ds,
        num_boost_round=cfgs.lambdamart.n_estimators,
        valid_sets=[dv],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    dt = time.perf_counter() - t
    ndcg = b.best_score["val"]["ndcg@10"]
    print(f"{name:28s} {dt:6.1f}s  iters={b.best_iteration:4d}  val_ndcg@10={ndcg:.4f}", flush=True)


for seed in (42, 7):
    run(f"baseline seed={seed}", seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    run(f"lr.1 bin63 seed={seed}", learning_rate=0.1, max_bin=63, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    run(f"lr.15 bin63 seed={seed}", learning_rate=0.15, max_bin=63, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
