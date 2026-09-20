"""E2: ONE joint hyper-choice sweep for the ranker, scored on VALIDATION ONLY.

Axes: objective {lambdarank, binary, rank_xendcg} x IPS clip-high {5, 10, 20, 50, none} x feature
blocks {all, no raw text embedding, no raw taste vector, neither} = 60 configs. Every config is fit
on the train-carved FIT split and scored on the train-carved VALIDATION trips; the holdout is not
read.

Validation metric (fixed in advance): IPS-weighted NDCG@10. The validation labels come from the
popularity-biased training log, so plain NDCG@10 would reward whatever fits that bias (it would
favour NOT using IPS). Each exposed validation row's gain is weighted by 1/clip(p_expose, 1/20, 1)
-- an inverse-propensity estimate of DCG under uniform exposure -- with the SAME clip for every
config (not the training clip being swept).

Protocol (pre-registered, successive halving): stage 1 fits all 60 configs once (seed 42); stage 2
re-fits the top 5 with 3 further seeds (7, 11, 13) and the winner is the best mean over the 4
seeds. All rows are reported so the choice is auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.eval.compose import write_part
from poi_rank.eval.decision_register import Lab
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns

OBJECTIVES = ("lambdarank", "binary", "rank_xendcg")
CLIPS: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 1e9)
BLOCKS: dict[str, tuple[str, ...]] = {
    "all_features": (),
    "no_raw_text_emb": ("text_emb_",),
    "no_raw_taste": ("implicit_taste_",),
    "no_raw_text_no_taste": ("text_emb_", "implicit_taste_"),
}
STAGE1_SEED = 42
STAGE2_SEEDS = (7, 11, 13)
TOP_N = 5
EVAL_CLIP = 20.0


def ips_weighted_ndcg10(
    frame: pd.DataFrame, score: npt.NDArray[np.float64], p_expose: pd.Series
) -> float:
    w = 1.0 / np.clip(p_expose.fillna(1.0).to_numpy(dtype=float), 1.0 / EVAL_CLIP, 1.0)
    work = frame[["trip_id", "poi_id"]].assign(
        score=score, gain=(np.power(2.0, frame["label"].to_numpy()) - 1.0) * w
    )
    disc = 1.0 / np.log2(np.arange(2, 12))
    vals: list[float] = []
    for _, g in work.groupby("trip_id", sort=True):
        gains = g["gain"].to_numpy()
        ideal = np.sort(gains)[::-1][:10]
        if ideal.sum() <= 0:
            continue
        order = np.lexsort((g["poi_id"].to_numpy(dtype=object), -g["score"].to_numpy()))
        top = gains[order][:10]
        vals.append(float((top * disc[: len(top)]).sum() / (ideal * disc[: len(ideal)]).sum()))
    return float(np.mean(vals))


class _Cache:
    """Everything that does not depend on the swept knobs, prepared once."""

    def __init__(self, lab: Lab) -> None:
        self.lab = lab
        cfg = lab.model_cfg.lambdamart
        self.cfg = cfg
        frame = lab.train_frame
        self.numeric = numeric_feature_columns(frame)
        self.categorical = categorical_feature_columns(frame)
        fit, val = lm.train_val_split_by_trip(frame, cfg.val_fraction, cfg.val_split_seed)
        self.p_fit = lm.train_frame_p_expose(fit, lab.interactions_train, lab.pois_df)
        self.fit_d, _ = lm.apply_behavioral_dropout(
            fit, cfg.behavioral_dropout_rate, cfg.behavioral_dropout_seed
        )
        self.val = val
        self.p_val = lm.train_frame_p_expose(val, lab.interactions_train, lab.pois_df)
        self.x_fit = lm._feature_matrix(self.fit_d, self.numeric, self.categorical)
        self.x_val = lm._feature_matrix(val, self.numeric, self.categorical)
        self.g_fit, self.g_val = lm._group_sizes(self.fit_d), lm._group_sizes(val)
        self.y_fit = self.fit_d["label"].to_numpy(dtype=np.float64)
        self.y_val = val["label"].to_numpy(dtype=np.float64)

    def cosine_init(self, scale: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-trip z-scored content-cosine baseline score (interest match + price fit, the same
        blend as `baseline_content_cosine`) times `scale`: the boosting starting point for the
        residual-learning variant (H2)."""

        def one(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
            price_fit = 1.0 - frame["interact_price_gap"].to_numpy(dtype=float) / 3.0
            raw = 0.7 * frame["interact_interest_match"].to_numpy(dtype=float) + 0.3 * price_fit
            s = pd.Series(raw, index=frame.index)
            g = s.groupby(frame["trip_id"].to_numpy())
            z = (s - g.transform("mean")) / g.transform("std").fillna(1.0).replace(0.0, 1.0)
            return np.asarray(scale * z.to_numpy(dtype=float), dtype=np.float64)

        return one(self.fit_d), one(self.val)

    def score(
        self,
        objective: str,
        clip: float,
        drop: tuple[str, ...],
        seed: int,
        *,
        only: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        init_scale: float | None = None,
    ) -> tuple[float, float]:
        cfg = self.cfg
        keep = [c for c in self.x_fit.columns if not c.startswith(drop)]
        if only is not None:
            keep = [c for c in keep if c in set(only)]
        cat = [c for c in self.categorical if c in keep]
        weights = lm.compute_ips_weights(
            self.fit_d["trip_id"], self.p_fit, cfg.ips_clip_low, clip
        ).to_numpy(dtype=np.float64)
        lgb_params = lm._lgb_params(cfg, seed)
        lgb_params["objective"] = objective
        lgb_params["metric"] = "ndcg"
        lgb_params.update(params or {})
        init_fit = init_val = None
        if init_scale is not None:
            init_fit, init_val = self.cosine_init(init_scale)
        y_fit = (self.y_fit >= 1).astype(float) if objective == "binary" else self.y_fit
        train_set = lgb.Dataset(
            self.x_fit[keep],
            label=y_fit,
            group=self.g_fit,
            weight=weights,
            init_score=init_fit,
            categorical_feature=cat,
            free_raw_data=False,
        )
        val_set = lgb.Dataset(
            self.x_val[keep],
            label=self.y_val,
            group=self.g_val,
            init_score=init_val,
            reference=train_set,
            categorical_feature=cat,
            free_raw_data=False,
        )
        booster = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=cfg.n_estimators,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred = np.asarray(booster.predict(self.x_val[keep]), dtype=np.float64)
        if init_val is not None:
            pred = pred + init_val
        return (
            ips_weighted_ndcg10(self.val, pred, self.p_val),
            float(booster.best_score["val"]["ndcg@10"]),
        )


def run_sweep(lab: Lab, results_dir: Path) -> dict[str, Any]:
    cache = _Cache(lab)
    rows: list[dict[str, Any]] = []
    for objective, clip, (block, drop) in product(OBJECTIVES, CLIPS, BLOCKS.items()):
        ips_ndcg, plain = cache.score(objective, clip, drop, STAGE1_SEED)
        rows.append(
            {
                "objective": objective,
                "ips_clip_high": "none" if clip >= 1e9 else clip,
                "blocks": block,
                "stage1_val_ips_weighted_ndcg10": ips_ndcg,
                "stage1_val_plain_ndcg10_biased_labels": plain,
                "_clip": clip,
                "_drop": drop,
            }
        )
        print(objective, rows[-1]["ips_clip_high"], block, round(ips_ndcg, 4), flush=True)
    finalists = sorted(rows, key=lambda r: -r["stage1_val_ips_weighted_ndcg10"])[:TOP_N]
    for r in finalists:
        per_seed = [r["stage1_val_ips_weighted_ndcg10"]] + [
            cache.score(r["objective"], r["_clip"], r["_drop"], s)[0] for s in STAGE2_SEEDS
        ]
        r["stage2_per_seed_val_ips_weighted_ndcg10"] = per_seed
        r["stage2_mean_val_ips_weighted_ndcg10"] = float(np.mean(per_seed))
        print(
            "finalist",
            r["objective"],
            r["ips_clip_high"],
            r["blocks"],
            round(np.mean(per_seed), 4),
            flush=True,
        )
    winner = max(finalists, key=lambda r: r["stage2_mean_val_ips_weighted_ndcg10"])
    shipped = next(
        r
        for r in rows
        if r["objective"] == "lambdarank" and r["_clip"] == 20.0 and r["blocks"] == "all_features"
    )
    shipped_seeds = [shipped["stage1_val_ips_weighted_ndcg10"]] + [
        cache.score("lambdarank", 20.0, (), s)[0] for s in STAGE2_SEEDS
    ]
    for r in rows:
        r.pop("_clip"), r.pop("_drop")
    payload = {
        "protocol": "validation only; holdout not read; metric = IPS-weighted NDCG@10 (eval clip "
        f"{EVAL_CLIP:g}); stage 1 = all 60 configs at seed {STAGE1_SEED}; stage 2 = top {TOP_N} "
        f"re-fit over seeds {list(STAGE2_SEEDS)} added",
        "n_configs": len(rows),
        "winner": winner,
        "previous_shipped_config_4_seed": {
            "per_seed": shipped_seeds,
            "mean": float(np.mean(shipped_seeds)),
        },
        "rows": rows,
    }
    write_part(results_dir, "ranker_sweep", payload)
    return payload
