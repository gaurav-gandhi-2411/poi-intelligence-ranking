"""Learned first-stage retriever (`channel_learned`).

Why this exists (A3 step 0, `results/parts/a3_step0*.json`, `results/parts/a3_retriever_grid.json`):
with the 6 heuristic channels the union recalled 0.596 of exposed holdout positives at 42% of
the catalog (chance 0.394, lift +0.20), while ranking by the true utility would recall 0.99 at
the same budget -- the ceiling is the retrieval design, not the data. The text representation
was NOT the bottleneck (D11 ridge R^2 = 0.865); the hand-built channels each read one slice of
the signal. A LightGBM scorer over the same observable pair features the ranker uses, fit on
the exposure-weighted (IPS) impression log, reads all of them.

Oracle-free by construction: fit only on `interactions_train` impressions, IPS-weighted by the
logged `p_expose`; never reads the oracle export (`tests/test_firewall_candidates.py`).

Leakage control: a retriever that scored the trips it was fit on would hand the ranker
in-sample-recall candidate sets at train time and out-of-sample ones at serve time. Train
trips are therefore scored CROSS-FITTED (K-fold by trip -- each fold scored by a model that
never saw its trips); holdout trips are scored by the full-train model.

Determinism: LightGBM `deterministic=True` + `force_row_wise=True`, all randomness from one
seed, output re-sorted; thread count changes scheduling only (asserted byte-identical by
`tests/test_retriever.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.candidates.config import LearnedChannelConfig
from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.features.pair_frame import build_ranking_frame
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids

# Raw embedding/taste blocks are dropped: their information reaches the model through the
# engineered `interact_cos_taste_poi` cross feature, and 96 raw dims would dominate a small
# retriever without adding recoverable signal (D11 says the 64-d text block is linear in
# poi_semantic already).
_DROP_PREFIXES = ("text_emb_", "implicit_taste_")
_NON_FEATURES = frozenset({"label", "trip_id", "poi_id", "traveler_id"})
# Trips per scoring chunk: bounds peak memory of the full-catalog pair frame (~480 rows/trip).
_SCORE_CHUNK_TRIPS = 250


@dataclass(frozen=True)
class Retriever:
    """A fitted booster plus the exact feature-column order it was trained on."""

    booster: lgb.Booster
    feature_columns: list[str]

    def score(self, frame: pd.DataFrame, num_threads: int) -> npt.NDArray[np.float32]:
        x = frame[self.feature_columns].astype(np.float32)
        pred = self.booster.predict(x, num_threads=num_threads)
        return np.asarray(pred, dtype=np.float32)


def retriever_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        c
        for c in frame.columns
        if c not in _NON_FEATURES
        and not c.startswith(_DROP_PREFIXES)
        and (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]))
    )


def _ips_weights(p_expose: npt.NDArray[np.float64], clip_min: float) -> npt.NDArray[np.float64]:
    """Clipped inverse-propensity weights normalised to mean 1 (same clipping idea as the
    ranker's IPS; the clip bounds variance from rarely-exposed, i.e. unpopular, POIs)."""
    w = 1.0 / np.clip(p_expose, clip_min, 1.0)
    out: npt.NDArray[np.float64] = w / w.mean()
    return out


def impression_table(
    interactions_train: pd.DataFrame, canonical_map: dict[str, str]
) -> pd.DataFrame:
    """One row per exposed `(trip_id, poi_id)`: max label, min logged `p_expose`."""
    remapped = remap_interaction_poi_ids(interactions_train, canonical_map)
    return remapped.groupby(["trip_id", "poi_id"], as_index=False).agg(
        label=("label", "max"), p_expose=("p_expose", "min")
    )


@dataclass(frozen=True)
class RetrieverInputs:
    pois_df: pd.DataFrame
    travelers_df: pd.DataFrame
    trips_df: pd.DataFrame
    poi_features_df: pd.DataFrame
    traveler_features_df: pd.DataFrame
    interactions_train: pd.DataFrame
    budget_target_price_level: BudgetTargetPriceLevel


def _pair_frame(inp: RetrieverInputs, pairs: pd.DataFrame, trip_ids: set[str]) -> pd.DataFrame:
    return build_ranking_frame(
        pairs[["trip_id", "poi_id"]],
        trip_ids,
        inp.interactions_train.iloc[0:0],
        inp.trips_df,
        inp.travelers_df,
        inp.pois_df,
        inp.poi_features_df,
        inp.traveler_features_df,
        inp.budget_target_price_level,
    )


def fit_retriever(
    inp: RetrieverInputs,
    impressions: pd.DataFrame,
    fit_trip_ids: set[str],
    cfg: LearnedChannelConfig,
    seed: int,
    num_threads: int,
) -> Retriever:
    imp = impressions.loc[impressions["trip_id"].isin(fit_trip_ids)]
    # Impression labels come from the log itself, not from a join on an emptied interactions
    # frame, so build the frame on pairs and attach labels/weights by key.
    frame = _pair_frame(inp, imp, fit_trip_ids)
    keyed = imp.set_index(["trip_id", "poi_id"])
    idx = pd.MultiIndex.from_frame(frame[["trip_id", "poi_id"]])
    label = keyed["label"].reindex(idx).to_numpy()
    weights = _ips_weights(
        keyed["p_expose"].reindex(idx).to_numpy(dtype=np.float64), cfg.ips_clip_min
    )
    cols = retriever_feature_columns(frame)
    params: dict[str, Any] = {
        "objective": "binary",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": num_threads,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(
        frame[cols].astype(np.float32), label=(label >= 1).astype(int), weight=weights
    )
    booster = lgb.train(params, dtrain, num_boost_round=cfg.n_estimators)
    return Retriever(booster=booster, feature_columns=cols)


def score_full_catalog(
    inp: RetrieverInputs, retriever: Retriever, trip_ids: list[str], num_threads: int
) -> pd.DataFrame:
    """Score every POI of each trip's destination. Chunked by trip to bound memory; rows
    come back keyed `(trip_id, poi_id)` with a float32 `score`, sorted."""
    trips = inp.trips_df.loc[inp.trips_df["trip_id"].isin(trip_ids), ["trip_id", "destination"]]
    catalog = inp.pois_df[["poi_id", "destination"]]
    ordered = sorted(trip_ids)
    parts: list[pd.DataFrame] = []
    for start in range(0, len(ordered), _SCORE_CHUNK_TRIPS):
        chunk = set(ordered[start : start + _SCORE_CHUNK_TRIPS])
        pairs = trips.loc[trips["trip_id"].isin(chunk)].merge(catalog, on="destination")
        frame = _pair_frame(inp, pairs, chunk)
        out = frame[["trip_id", "poi_id"]].copy()
        out["score"] = retriever.score(frame, num_threads)
        parts.append(out)
    return pd.concat(parts, ignore_index=True).sort_values(["trip_id", "poi_id"], ignore_index=True)


def crossfit_retriever_scores(
    inp: RetrieverInputs, cfg: LearnedChannelConfig, seed: int, num_threads: int
) -> pd.DataFrame:
    """Scores for EVERY trip: train trips cross-fitted K-fold by trip, holdout trips from the
    full-train model (module docstring)."""
    canonical_map = build_poi_id_canonical_map(inp.pois_df)
    impressions = impression_table(inp.interactions_train, canonical_map)
    is_holdout = inp.trips_df.set_index("trip_id")["is_holdout"]
    train_ids = np.array(sorted(is_holdout.index[~is_holdout]))
    holdout_ids = sorted(is_holdout.index[is_holdout])

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(train_ids), cfg.n_folds)
    parts: list[pd.DataFrame] = []
    for k, fold in enumerate(folds):
        fit_ids = {t for j, f in enumerate(folds) if j != k for t in f}
        model = fit_retriever(inp, impressions, fit_ids, cfg, seed + k, num_threads)
        parts.append(score_full_catalog(inp, model, [str(t) for t in fold], num_threads))
    full = fit_retriever(inp, impressions, {str(t) for t in train_ids}, cfg, seed, num_threads)
    parts.append(score_full_catalog(inp, full, [str(t) for t in holdout_ids], num_threads))
    return pd.concat(parts, ignore_index=True).sort_values(["trip_id", "poi_id"], ignore_index=True)


def top_k_by_trip(scores: pd.DataFrame, quota: int) -> dict[str, list[str]]:
    """Top-`quota` POIs per trip by score (ties broken by poi_id ascending -> deterministic)."""
    ranked = scores.sort_values(["trip_id", "score", "poi_id"], ascending=[True, False, True])
    top = ranked.groupby("trip_id", sort=True).head(quota)
    return {str(t): g["poi_id"].tolist() for t, g in top.groupby("trip_id", sort=True)}
