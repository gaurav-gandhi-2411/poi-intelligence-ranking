"""Representation-quality diagnostics (spec-v3 section 3.4, GG's A3 rulings): how much of the
DGP's latent `poi_semantic` space is recoverable from what the model can see, and which link in
the taste-match chain (text channel vs taste estimator) loses it.

REPORTING ONLY. Every function here reads the oracle export (eval/ is the sole permitted
reader) and is computed AFTER any representation choice is frozen; no representation
hyperparameter is ever selected on these numbers (selection uses oracle-free validation
NDCG). TECHNICAL.md states this: the oracle never touched a decision, it only ever scored
the result.

Statistics:
  * D11 -- out-of-fold ridge R^2 and first canonical correlation (CCA) from `poi_emb` to
    `poi_semantic`, 5-fold, seed 42.
  * D9 -- WITHIN-TRIP Spearman between the DGP's noise-free `cos(taste_t, poi_semantic_p)` and
    the observable `cos(taste_feature_t, poi_emb_p)`, averaged over trips (pooled Spearman
    mixes between-trip scale differences into the statistic; ruling: within-trip).
  * Ablations settling the D9 = (text fidelity) x (estimator fidelity) decomposition:
      - taste-estimator-alone: TRUE poi_semantic as the item representation, ESTIMATED taste
        (the production estimator run over the true-semantic vectors).
      - text-alone: TRUE taste against the observable text representation, mapped into
        semantic space by the out-of-fold D11 ridge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from poi_rank.eval.oracle import (
    load_holdout_utility_true,
    load_poi_latent,
    load_traveler_taste,
    oracle_dir_for,
)

FloatArray = npt.NDArray[np.float64]

SEED = 42
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5


def _unit(x: FloatArray) -> FloatArray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    out: FloatArray = x / np.where(norm > 0, norm, 1.0)
    return out


def ridge_recoverability(emb: FloatArray, semantic: FloatArray, seed: int = SEED) -> dict[str, Any]:
    """Out-of-fold ridge R^2 (variance-weighted over the semantic dims) and first canonical
    correlation, `emb -> semantic`. Also returns the out-of-fold predictions."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    pred = np.zeros_like(semantic)
    for train_idx, test_idx in kf.split(emb):
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(emb[train_idx], semantic[train_idx])
        pred[test_idx] = model.predict(emb[test_idx])
    ss_res = float(((semantic - pred) ** 2).sum())
    ss_tot = float(((semantic - semantic.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    cca_corrs: list[float] = []
    for train_idx, test_idx in kf.split(emb):
        cca = CCA(n_components=1, max_iter=1000).fit(emb[train_idx], semantic[train_idx])
        x_c, y_c = cca.transform(emb[test_idx], semantic[test_idx])
        cca_corrs.append(float(np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]))
    return {
        "ridge_r2_oof": r2,
        "cca_first_corr_oof": float(np.mean(cca_corrs)),
        "n_pois": int(len(emb)),
        "emb_dim": int(emb.shape[1]),
        "semantic_dim": int(semantic.shape[1]),
        "_pred": pred,
    }


def within_trip_spearman(
    trip_poi_index: dict[str, npt.NDArray[np.intp]],
    trip_reference: dict[str, FloatArray],
    trip_observable: dict[str, FloatArray],
) -> dict[str, Any]:
    """Mean per-trip Spearman between two score vectors over the same POI set. Trips whose
    observable (or reference) scores are constant (e.g. a zero cold-start taste vector) have
    no defined rank correlation: counted and excluded, never silently dropped."""
    rhos: list[float] = []
    n_undefined = 0
    for trip_id in trip_poi_index:
        a, b = trip_reference[trip_id], trip_observable[trip_id]
        if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
            n_undefined += 1
            continue
        rho = stats.spearmanr(a, b)[0]
        rhos.append(float(rho))
    arr = np.asarray(rhos)
    return {
        "mean_within_trip_spearman": float(arr.mean()) if len(arr) else float("nan"),
        "median_within_trip_spearman": float(np.median(arr)) if len(arr) else float("nan"),
        "n_trips": int(len(arr)),
        "n_trips_undefined_constant_scores": int(n_undefined),
    }


def load_reference_inputs(data_dir: Path) -> dict[str, Any]:
    """Canonical-POI true semantic matrix + traveler true taste, aligned to `pois_prepared`."""
    oracle_dir = oracle_dir_for(data_dir)
    pois = pd.read_parquet(data_dir / "pois_prepared.parquet")
    latent = load_poi_latent(oracle_dir).set_index("poi_id")
    semantic = np.stack(
        [np.asarray(latent.loc[p, "poi_semantic"], dtype=np.float64) for p in pois["poi_id"]]
    )
    taste_df = load_traveler_taste(oracle_dir)
    taste = {
        str(t): np.asarray(v, dtype=np.float64)
        for t, v in zip(taste_df["traveler_id"], taste_df["taste_vector"], strict=True)
    }
    return {"pois": pois, "semantic": semantic, "taste_true": taste}


def _holdout_trip_pois(data_dir: Path, pois: pd.DataFrame) -> dict[str, npt.NDArray[np.intp]]:
    """`trip_id -> row indices (into pois_prepared)` of the canonical POIs the oracle scored for
    that holdout trip (~all POIs of the trip's destination)."""
    util = load_holdout_utility_true(oracle_dir_for(data_dir))
    row_of = {p: i for i, p in enumerate(pois["poi_id"])}
    util = util.loc[util["poi_id"].isin(row_of)]
    out: dict[str, npt.NDArray[np.intp]] = {}
    for trip_id, g in util.groupby("trip_id"):
        out[str(trip_id)] = np.array([row_of[p] for p in g["poi_id"]], dtype=np.intp)
    return out


def _scores(
    trip_poi_index: dict[str, npt.NDArray[np.intp]],
    trip_vec: dict[str, FloatArray],
    poi_matrix: FloatArray,
) -> dict[str, FloatArray]:
    unit_pois = _unit(poi_matrix)
    out: dict[str, FloatArray] = {}
    for trip_id, idx in trip_poi_index.items():
        v = trip_vec[trip_id]
        n = np.linalg.norm(v)
        out[trip_id] = unit_pois[idx] @ (v / n) if n > 0 else np.zeros(len(idx))
    return out


def d9_and_ablations(
    data_dir: Path,
    traveler_features: pd.DataFrame,
    emb: FloatArray,
    est_taste_semantic: pd.DataFrame,
    holdout_trip_ids: set[str],
    trip_to_traveler: dict[str, str],
    inputs: dict[str, Any],
    oof_semantic_pred: FloatArray,
) -> dict[str, Any]:
    """D9 (within-trip) for the shipped chain plus both ablations.

    `est_taste_semantic` is a `traveler_features`-shaped frame built by running the production
    taste estimator over the TRUE semantic vectors (taste-estimator-alone ablation)."""
    pois, semantic, taste_true = inputs["pois"], inputs["semantic"], inputs["taste_true"]
    trip_idx = {
        t: i for t, i in _holdout_trip_pois(data_dir, pois).items() if t in holdout_trip_ids
    }
    true_vec = {t: taste_true[trip_to_traveler[t]] for t in trip_idx}
    reference = _scores(trip_idx, true_vec, semantic)

    def est_vecs(frame: pd.DataFrame, dim: int) -> dict[str, FloatArray]:
        cols = [f"implicit_taste_{i:02d}" for i in range(dim)]
        mat = frame[cols].to_numpy(dtype=np.float64)
        lookup = {str(t): mat[i] for i, t in enumerate(frame["trip_id"])}
        return {t: lookup[t] for t in trip_idx}

    shipped_obs = _scores(trip_idx, est_vecs(traveler_features, emb.shape[1]), emb)
    estimator_alone = _scores(trip_idx, est_vecs(est_taste_semantic, semantic.shape[1]), semantic)
    text_alone = _scores(trip_idx, true_vec, oof_semantic_pred)
    return {
        "shipped_chain_d9": within_trip_spearman(trip_idx, reference, shipped_obs),
        "taste_estimator_alone": within_trip_spearman(trip_idx, reference, estimator_alone),
        "text_alone": within_trip_spearman(trip_idx, reference, text_alone),
    }


ORACLE_RELEVANCE_TOP_FRACTION = 0.05  # "relevant" = top 5% of the destination catalog by true u


def oracle_relevance_recall(
    data_dir: Path, candidates: pd.DataFrame, pois: pd.DataFrame, holdout_trip_ids: set[str]
) -> dict[str, float]:
    """Recall of the candidate set against the oracle-relevance set (top 5% of the trip's
    destination catalog by TRUE utility) -- not conditioned on exposure at all, so it answers
    "did candidate generation find what actually matters". Reporting only."""
    util = load_holdout_utility_true(oracle_dir_for(data_dir))
    util = util.loc[
        util["poi_id"].isin(set(pois["poi_id"])) & util["trip_id"].isin(holdout_trip_ids)
    ]
    pop = dict(zip(pois["poi_id"], pois["pop_pct"], strict=True))
    cand = {str(t): set(g["poi_id"]) for t, g in candidates.groupby("trip_id")}
    overall: list[float] = []
    long_tail: list[float] = []
    for trip_id, g in util.groupby("trip_id"):
        k = max(1, int(round(ORACLE_RELEVANCE_TOP_FRACTION * len(g))))
        relevant = g.nlargest(k, "utility_true")["poi_id"]
        got = cand.get(str(trip_id), set())
        overall.append(float(relevant.isin(got).mean()))
        rel_lt = relevant[relevant.map(pop) < 0.5]
        if len(rel_lt):
            long_tail.append(float(rel_lt.isin(got).mean()))
    return {
        "overall": float(np.mean(overall)),
        "long_tail": float(np.mean(long_tail)),
        "top_fraction": ORACLE_RELEVANCE_TOP_FRACTION,
        "n_trips": float(len(overall)),
    }


def run_representation_report(data_dir: Path, feature_cfg: Any) -> dict[str, Any]:
    """D11 + within-trip D9 + both ablations + oracle-relevance recall, on the CURRENT
    features/candidates. REPORTING ONLY (module docstring): computed after every
    representation choice is frozen; the oracle never touched a decision."""
    from poi_rank.features.traveler_features import assemble_traveler_features

    inputs = load_reference_inputs(data_dir)
    pois, semantic = inputs["pois"], inputs["semantic"]
    pf = pd.read_parquet(data_dir / "poi_features.parquet").set_index("poi_id").loc[pois["poi_id"]]
    dim = feature_cfg.text_embedding.svd_dim
    emb = pf[[f"text_emb_{i:02d}" for i in range(dim)]].to_numpy(dtype=np.float64)

    recov = ridge_recoverability(emb, semantic)
    pred = recov.pop("_pred")

    trips = pd.read_parquet(data_dir / "trips.parquet")
    # Only holdout trips are scored, and a trip's taste estimate depends only on its traveler's
    # history strictly before its own start date -- so the estimator can be run on them alone.
    est_semantic = assemble_traveler_features(
        pd.read_parquet(data_dir / "travelers.parquet"),
        trips.loc[trips["is_holdout"]],
        pd.read_parquet(data_dir / "interactions_train.parquet"),
        pois,
        semantic.astype(np.float32),
        feature_cfg,
        pd.read_parquet(data_dir / "interactions_pretrip.parquet"),
    )
    holdout = set(trips.loc[trips["is_holdout"], "trip_id"].astype(str))
    trip_to_traveler = dict(
        zip(trips["trip_id"].astype(str), trips["traveler_id"].astype(str), strict=True)
    )
    d9 = d9_and_ablations(
        data_dir,
        pd.read_parquet(data_dir / "traveler_features.parquet"),
        emb,
        est_semantic,
        holdout,
        trip_to_traveler,
        inputs,
        pred,
    )
    candidates = pd.read_parquet(data_dir / "candidates.parquet")
    return {
        "d11_text_only": recov,
        # No behavioral block was built (A3 step 0: text D11 already >> 0.45), so the
        # concatenated representation IS the text representation.
        "d11_concatenated_equals_text_only": True,
        "d9_within_trip": d9,
        "oracle_relevance_recall": oracle_relevance_recall(data_dir, candidates, pois, holdout),
        "n_holdout_trips": len(holdout),
    }
