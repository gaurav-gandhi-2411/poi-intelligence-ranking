"""Decision Register experiments that need model refits or pipeline swaps (DR2, DR3, DR4, DR7,
DR9, DR11). Shared plumbing (`Lab`, `fit_lgb`, `write_dr`, ...) lives in
`eval/decision_register.py`; the scoring-layer experiments (DR1/DR6/DR10) are there too."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.eval.decision_register import (
    Lab,
    fit_lgb,
    per_trip_ndcg10,
    sample_train_trips,
    score_lgb,
    summarize_ndcg,
    write_dr,
)
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import lambdamart as lm
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns

# -----------------------------------------------------------------------------------
# DR2: two-tower learning curve; DR3: objectives; DR7: IPS clip
# -----------------------------------------------------------------------------------


def _fit_two_tower_ndcg(lab: Lab, fraction: float, seed: int) -> pd.Series:
    from poi_rank.eval.two_tower import TwoTowerEncoder, score_frame, train_two_tower

    cfg = lab.model_cfg.lambdamart
    sub = sample_train_trips(lab.train_frame, fraction, seed)
    fit, val = lm.train_val_split_by_trip(sub, cfg.val_fraction, cfg.val_split_seed)
    p = lm.train_frame_p_expose(fit, lab.interactions_train, lab.pois_df)
    w = lm.compute_ips_weights(fit["trip_id"], p, cfg.ips_clip_low, cfg.ips_clip_high)
    encoder = TwoTowerEncoder(fit)

    def val_ndcg(scores: Any) -> float:
        return float(per_trip_ndcg10(val, pd.Series(scores, index=val.index)).mean())

    model, encoder, _ = train_two_tower(encoder, fit, w.to_numpy(np.float64), val, val_ndcg, seed)
    return per_trip_ndcg10(lab.holdout_frame, score_frame(model, encoder, lab.holdout_frame))


def run_dr2(
    lab: Lab,
    results_dir: Path,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    seeds: tuple[int, ...] = (42, 43, 44),
) -> None:
    n_train_trips = int(lab.train_frame["trip_id"].nunique())
    curve: list[dict[str, Any]] = []
    for f in fractions:
        lgb_per: list[pd.Series] = []
        tt_per: list[pd.Series] = []
        for s in seeds:
            booster, num, cat, _, _ = fit_lgb(lab, trip_fraction=f, seed=s)
            lgb_per.append(
                per_trip_ndcg10(lab.holdout_frame, score_lgb(booster, num, cat, lab.holdout_frame))
            )
            tt_per.append(_fit_two_tower_ndcg(lab, f, s))
        row: dict[str, Any] = {"fraction": f, "n_train_trips": int(round(f * n_train_trips))}
        for name, per in (("lambdamart_ips", lgb_per), ("two_tower", tt_per)):
            row[name] = {
                **summarize_ndcg(pd.concat(per, axis=1).mean(axis=1), lab.eval_cfg),
                "seed_means": [float(p.mean()) for p in per],
            }
        curve.append(row)
        print(
            f"DR2 fraction {f}: lgb {row['lambdamart_ips']['mean']:.4f} "
            f"tt {row['two_tower']['mean']:.4f}",
            flush=True,
        )

    # Log-linear fit NDCG ~ a + b ln(n_trips) per model (4 points: an illustration of scale, not
    # a forecast).
    ln_n = np.log([r["n_train_trips"] for r in curve])
    fits: dict[str, tuple[float, float]] = {}
    for name in ("lambdamart_ips", "two_tower"):
        slope, intercept = np.polyfit(ln_n, [r[name]["mean"] for r in curve], 1)
        fits[name] = (float(intercept), float(slope))
    (a_l, b_l), (a_t, b_t) = fits["lambdamart_ips"], fits["two_tower"]
    crossover = float(np.exp((a_l - a_t) / (b_t - b_l))) if b_t > b_l else None
    any_cross = any(r["two_tower"]["mean"] >= r["lambdamart_ips"]["mean"] for r in curve)
    last = curve[-1]
    ci_overlap = last["two_tower"]["ci_high"] >= last["lambdamart_ips"]["ci_low"]
    tt_txt = ", ".join(f"{r['two_tower']['mean']:.4f}" for r in curve)
    lg_txt = ", ".join(f"{r['lambdamart_ips']['mean']:.4f}" for r in curve)
    if crossover is not None:
        extrap = (
            f"Log-linear extrapolation puts a crossover at ~{crossover:,.0f} training trips "
            f"({crossover / n_train_trips:.1f}x the {n_train_trips} used; 4-point fit, low "
            "confidence)."
        )
    else:
        extrap = (
            "The two-tower's fitted slope does not exceed LambdaMART's: no crossover extrapolated."
        )
    write_dr(
        results_dir,
        "DR2",
        f"Learning curve: both rankers fit on {list(fractions)} of TRAIN trips x seeds "
        f"{list(seeds)} (same IPS weights, same train-carved early stopping), scored on the full "
        "unbiased holdout; per-trip NDCG@10 averaged over seeds, 2000-resample trip bootstrap CIs.",
        "holdout NDCG@10 vs training-set fraction (mean, 95% CI)",
        {
            "curve": curve,
            "loglinear_fit": {k: {"a": v[0], "b": v[1]} for k, v in fits.items()},
            "extrapolated_crossover_train_trips": crossover,
            "n_train_trips_full": n_train_trips,
        },
        f"Two-tower NDCG@10 {tt_txt} vs LambdaMART-IPS {lg_txt} at fractions {list(fractions)}. "
        f"Crossover at any measured point: {any_cross}. {extrap} "
        + ("CIs overlap at 100%." if ci_overlap else "CIs separated at 100%."),
    )


def run_dr3(lab: Lab, results_dir: Path) -> None:
    rows: dict[str, Any] = {}
    for objective in ("lambdarank", "rank_xendcg", "binary", "regression"):
        booster, num, cat, _, _ = fit_lgb(lab, objective=objective)
        per = per_trip_ndcg10(lab.holdout_frame, score_lgb(booster, num, cat, lab.holdout_frame))
        rows[objective] = {
            **summarize_ndcg(per, lab.eval_cfg),
            "best_iteration": int(booster.best_iteration),
        }
    write_dr(
        results_dir,
        "DR3",
        "Same features, IPS weights, dropout, split and early-stopping metric (NDCG@10); only "
        "the LightGBM objective varies.",
        "holdout NDCG@10 (mean, 95% CI)",
        rows,
        "; ".join(
            f"{k} {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]" for k, v in rows.items()
        ),
    )


def run_dr7(lab: Lab, results_dir: Path) -> None:
    rows: dict[str, Any] = {}
    for clip in (5.0, 10.0, 20.0, 50.0, 1e9):
        booster, num, cat, _, _ = fit_lgb(lab, ips_clip_high=clip)
        per = per_trip_ndcg10(lab.holdout_frame, score_lgb(booster, num, cat, lab.holdout_frame))
        rows["none" if clip >= 1e9 else f"{clip:g}"] = summarize_ndcg(per, lab.eval_cfg)
    booster, num, cat, _, _ = fit_lgb(lab, ips=False)
    per = per_trip_ndcg10(lab.holdout_frame, score_lgb(booster, num, cat, lab.holdout_frame))
    rows["no_ips"] = summarize_ndcg(per, lab.eval_cfg)
    write_dr(
        results_dir,
        "DR7",
        "IPS clip-high swept with everything else fixed (weights renormalised per trip).",
        "holdout NDCG@10 (mean, 95% CI)",
        rows,
        "; ".join(
            f"clip {k}: {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
            for k, v in rows.items()
        ),
    )


# -----------------------------------------------------------------------------------
# DR4: text encoder (TF-IDF vs MiniLM) -- full ranking-pipeline swap, candidates fixed
# -----------------------------------------------------------------------------------


def run_dr4(data_dir: Path, results_dir: Path, feature_cfg: FeatureBuildConfig, lab: Lab) -> None:
    from poi_rank.eval.dgp_diagnostics import _run_minilm_encode_in_subprocess
    from poi_rank.eval.representation import (
        _holdout_trip_pois,
        _scores,
        load_reference_inputs,
        ridge_recoverability,
        within_trip_spearman,
    )
    from poi_rank.features.pair_frame import build_ranking_frame
    from poi_rank.features.poi_features import assemble_poi_features
    from poi_rank.features.text_embed import build_poi_corpus
    from poi_rank.features.traveler_features import assemble_traveler_features

    pois = lab.pois_df
    travelers = pd.read_parquet(data_dir / "travelers.parquet")
    trips = pd.read_parquet(data_dir / "trips.parquet")
    pretrip = pd.read_parquet(data_dir / "interactions_pretrip.parquet")
    holdout_random = pd.read_parquet(data_dir / "interactions_holdout_random.parquet")
    candidates = pd.read_parquet(data_dir / "candidates.parquet")
    budget = feature_cfg.traveler_features.budget_target_price_level

    minilm_path = results_dir / "parts" / "poi_emb_minilm_diagnostic.npy"
    if not minilm_path.exists():
        _run_minilm_encode_in_subprocess(
            build_poi_corpus(pois),
            feature_cfg.text_embedding.sentence_transformer_model,
            feature_cfg.text_embedding.svd_dim,
            feature_cfg.seed,
            minilm_path,
        )
    embeddings = {
        "tfidf_svd64": np.load(data_dir.parent.parent / "artifacts" / "poi_emb.npy"),
        "minilm_svd64": np.load(minilm_path),
    }

    inputs = load_reference_inputs(data_dir)
    holdout_ids = set(trips.loc[trips["is_holdout"], "trip_id"].astype(str))
    t2t = dict(zip(trips["trip_id"].astype(str), trips["traveler_id"].astype(str), strict=True))
    trip_idx = {t: i for t, i in _holdout_trip_pois(data_dir, pois).items() if t in holdout_ids}
    reference = _scores(
        trip_idx, {t: inputs["taste_true"][t2t[t]] for t in trip_idx}, inputs["semantic"]
    )
    train_ids = set(trips.loc[~trips["is_holdout"], "trip_id"])

    results: dict[str, Any] = {}
    for name, emb in embeddings.items():
        emb32 = emb.astype(np.float32)
        tf_all = assemble_traveler_features(
            travelers, trips, lab.interactions_train, pois, emb32, feature_cfg, pretrip
        )
        pf = assemble_poi_features(
            pois,
            lab.interactions_train,
            travelers,
            emb32,
            feature_cfg.poi_features,
            feature_cfg.seed,
        )
        cols = [f"implicit_taste_{i:02d}" for i in range(emb.shape[1])]
        lookup = {
            str(t): v
            for t, v in zip(tf_all["trip_id"], tf_all[cols].to_numpy(np.float64), strict=True)
        }
        d9 = within_trip_spearman(
            trip_idx,
            reference,
            _scores(trip_idx, {t: lookup[t] for t in trip_idx}, emb.astype(np.float64)),
        )
        d11 = ridge_recoverability(emb.astype(np.float64), inputs["semantic"])
        d11.pop("_pred")
        train_frame = build_ranking_frame(
            candidates,
            train_ids,
            lab.interactions_train,
            trips,
            travelers,
            pois,
            pf,
            tf_all,
            budget,
        )
        hold_frame = build_ranking_frame(
            candidates, holdout_ids, holdout_random, trips, travelers, pois, pf, tf_all, budget
        )
        booster, num, cat, _, _ = fit_lgb(lab, train_frame=train_frame)
        per = per_trip_ndcg10(hold_frame, score_lgb(booster, num, cat, hold_frame))
        results[name] = {
            "ndcg10": summarize_ndcg(per, lab.eval_cfg),
            "d9_within_trip": d9,
            "d11_ridge_r2": d11["ridge_r2_oof"],
            "d11_cca": d11["cca_first_corr_oof"],
        }
    a, b = results["tfidf_svd64"], results["minilm_svd64"]
    overlap = b["ndcg10"]["ci_low"] <= a["ndcg10"]["ci_high"]
    write_dr(
        results_dir,
        "DR4",
        "Swap ONLY the POI text embedding (and the taste vectors and POI features built from it) "
        "and refit the ranker on the fixed candidate sets; measure representation fidelity (D9 "
        "within-trip, D11) and holdout NDCG@10. THIS SYNTHETIC CORPUS is generated from "
        "anchored, synonym-rich phrase pools over latent dimensions, so its vocabulary design "
        "favours lexical overlap: the outcome is a statement about this dataset, not a verdict "
        "on sentence encoders.",
        "D11 ridge R2; within-trip D9; holdout NDCG@10",
        results,
        f"TF-IDF: D11 {a['d11_ridge_r2']:.3f}, "
        f"D9 {a['d9_within_trip']['mean_within_trip_spearman']:.3f}, "
        f"NDCG@10 {a['ndcg10']['mean']:.4f}. MiniLM: D11 {b['d11_ridge_r2']:.3f}, "
        f"D9 {b['d9_within_trip']['mean_within_trip_spearman']:.3f}, "
        f"NDCG@10 {b['ndcg10']['mean']:.4f} "
        f"(CIs {'overlap' if overlap else 'are separated'}). "
        "Dataset-specific (templated synonym-pool text); transfer to real POI text is untested.",
    )


# -----------------------------------------------------------------------------------
# DR9: long-tail quota; DR11: candidate channel sets
# -----------------------------------------------------------------------------------


def run_dr9(
    data_dir: Path,
    artifacts_dir: Path,
    results_dir: Path,
    feature_cfg: FeatureBuildConfig,
    candidates_cfg: CandidatesConfig,
    lab: Lab,
    quotas: tuple[int, ...] = (0, 25, 50, 100),
) -> None:
    from poi_rank.candidates.recall_metrics import overall_and_longtail_recall
    from poi_rank.candidates.retriever import (
        Retriever,
        RetrieverInputs,
        score_full_catalog,
        top_k_by_trip,
    )
    from poi_rank.candidates.union import generate_candidates
    from poi_rank.eval.run import _longtail_payload, _top_k_lists
    from poi_rank.features.pair_frame import build_ranking_frame

    pois = lab.pois_df
    travelers = pd.read_parquet(data_dir / "travelers.parquet")
    trips = pd.read_parquet(data_dir / "trips.parquet")
    pf = pd.read_parquet(data_dir / "poi_features.parquet")
    tf_all = pd.read_parquet(data_dir / "traveler_features.parquet")
    holdout_random = pd.read_parquet(data_dir / "interactions_holdout_random.parquet")
    budget = feature_cfg.traveler_features.budget_target_price_level
    hold_trips = trips.loc[trips["is_holdout"]]
    hold_ids = set(hold_trips["trip_id"])

    learned = candidates_cfg.learned
    assert learned is not None
    scores = score_full_catalog(
        RetrieverInputs(pois, travelers, hold_trips, pf, tf_all, lab.interactions_train, budget),
        Retriever.load(artifacts_dir),
        sorted(hold_ids),
        learned.num_threads,
    )
    top_k = top_k_by_trip(scores, learned.quota)
    booster = lm.load_boosters(artifacts_dir)["lambdamart_ips"]
    num = numeric_feature_columns(lab.holdout_frame)
    cat = categorical_feature_columns(lab.holdout_frame)

    rows: dict[str, Any] = {}
    for q in quotas:
        cfg_q = dataclasses.replace(
            candidates_cfg, longtail=dataclasses.replace(candidates_cfg.longtail, quota=q)
        )
        cand = generate_candidates(
            pois,
            travelers,
            hold_trips,
            pf,
            tf_all,
            lab.interactions_train,
            cfg_q,
            learned_top_k=top_k,
        )
        frame = build_ranking_frame(
            cand, hold_ids, holdout_random, trips, travelers, pois, pf, tf_all, budget
        )
        score = score_lgb(booster, num, cat, frame)
        per = per_trip_ndcg10(frame, score)
        lt_payload = _longtail_payload(_top_k_lists(frame, score), pois, frame, 0.4)
        recall = overall_and_longtail_recall(pois, cand, holdout_random)
        rows[str(q)] = {
            "ndcg10": summarize_ndcg(per, lab.eval_cfg),
            "longtail_at_10": lt_payload,
            "candidate_recall_overall": recall["overall"].recall_mean,
            "candidate_recall_long_tail": recall["long_tail"].recall_mean,
            "mean_candidates_per_trip": float(cand.groupby("trip_id").size().mean()),
        }
    first = next(iter(rows.values()))["longtail_at_10"]
    share_key = next(k for k in first if "share" in k)
    write_dr(
        results_dir,
        "DR9",
        "Regenerate the holdout candidate sets with a different long-tail hard-floor quota "
        "(retriever scores fixed); score with the SHIPPED booster (trained at quota 50, not "
        "refit per quota); raw ranker top-10, no MMR/gates.",
        "NDCG@10; long-tail share and precision @10; candidate recall",
        rows,
        "; ".join(
            f"quota {q}: NDCG@10 {v['ndcg10']['mean']:.4f}, long-tail {share_key} "
            f"{v['longtail_at_10'][share_key]:.3f}, candidate recall "
            f"{v['candidate_recall_overall']:.3f}"
            for q, v in rows.items()
        ),
    )


def run_dr11(results_dir: Path) -> None:
    grid = json.loads((results_dir / "parts" / "a3_retriever_grid.json").read_text("utf-8"))["grid"]
    legacy = json.loads((results_dir / "parts" / "a3_step0_recall.json").read_text("utf-8"))
    rows = [
        {
            "channels": g["channels"],
            "learned_K": g["learned_K"],
            "val_ips_recall": g["val_ips_weighted_selection"]["recall"],
            "val_ips_lift": g["val_ips_weighted_selection"]["lift"],
            "size": g["holdout_reporting_only"]["size"],
            "holdout_recall": g["holdout_reporting_only"]["recall"],
            "holdout_long_tail": g["holdout_reporting_only"]["lt"],
            "holdout_lift": g["holdout_reporting_only"]["lift"],
        }
        for g in grid
        if g["learned_K"] == 210
    ]
    shipped = next(r for r in rows if r["channels"] == "longtail+interest")
    base = legacy["recall_by_stratum"]["overall"]
    write_dr(
        results_dir,
        "DR11",
        "Channel-subset grid at learned K=210 on a train-carved validation split (IPS-weighted "
        "logged positives); holdout columns are reporting-only. `none` = the learned retriever "
        "alone. Baseline = the original 6-channel heuristic union.",
        "recall vs exposed positives, lift over chance, candidate-set size",
        {
            "K210_rows": rows,
            "legacy_six_channel_union": base,
            "true_utility_top_k_recall_diagnostic": legacy["oracle_topK_recall_ceiling_diagnostic"],
        },
        f"Legacy 6-channel union: recall {base['recall']:.3f} (lift {base['lift_abs']:+.3f}). "
        f"Shipped learned+long-tail+interest: recall {shipped['holdout_recall']:.3f} "
        f"(lift {shipped['holdout_lift']:+.3f}, {shipped['size']:.0f} candidates/trip). "
        "Ranking by true utility would recall "
        f"{legacy['oracle_topK_recall_ceiling_diagnostic']:.3f} "
        "at the same budget (diagnostic).",
    )


# -----------------------------------------------------------------------------------
# DR8: taste half-life and interaction weights -- rebuild traveler features, refit, score
# -----------------------------------------------------------------------------------


def run_dr8(data_dir: Path, results_dir: Path, feature_cfg: FeatureBuildConfig, lab: Lab) -> None:
    from poi_rank.eval.representation import (
        _holdout_trip_pois,
        _scores,
        load_reference_inputs,
        within_trip_spearman,
    )
    from poi_rank.features.config import TasteWeights
    from poi_rank.features.pair_frame import build_ranking_frame
    from poi_rank.features.traveler_features import assemble_traveler_features

    pois = lab.pois_df
    travelers = pd.read_parquet(data_dir / "travelers.parquet")
    trips = pd.read_parquet(data_dir / "trips.parquet")
    pretrip = pd.read_parquet(data_dir / "interactions_pretrip.parquet")
    holdout_random = pd.read_parquet(data_dir / "interactions_holdout_random.parquet")
    candidates = pd.read_parquet(data_dir / "candidates.parquet")
    pf = pd.read_parquet(data_dir / "poi_features.parquet")
    budget = feature_cfg.traveler_features.budget_target_price_level
    emb = np.load(data_dir.parent.parent / "artifacts" / "poi_emb.npy").astype(np.float32)

    inputs = load_reference_inputs(data_dir)
    holdout_ids = set(trips.loc[trips["is_holdout"], "trip_id"].astype(str))
    train_ids = set(trips.loc[~trips["is_holdout"], "trip_id"])
    t2t = dict(zip(trips["trip_id"].astype(str), trips["traveler_id"].astype(str), strict=True))
    trip_idx = {t: i for t, i in _holdout_trip_pois(data_dir, pois).items() if t in holdout_ids}
    reference = _scores(
        trip_idx, {t: inputs["taste_true"][t2t[t]] for t in trip_idx}, inputs["semantic"]
    )

    base = feature_cfg.traveler_features
    uniform = TasteWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    variants = {
        "halflife_180d (shipped)": base,
        "halflife_90d": dataclasses.replace(base, taste_halflife_days=base.taste_halflife_days / 2),
        "halflife_360d": dataclasses.replace(
            base, taste_halflife_days=base.taste_halflife_days * 2
        ),
        "uniform_weights": dataclasses.replace(base, taste_weights=uniform),
    }
    results: dict[str, Any] = {}
    for name, tf_cfg in variants.items():
        cfg_v = dataclasses.replace(feature_cfg, traveler_features=tf_cfg)
        tf_all = assemble_traveler_features(
            travelers, trips, lab.interactions_train, pois, emb, cfg_v, pretrip
        )
        cols = [f"implicit_taste_{i:02d}" for i in range(emb.shape[1])]
        lookup = {
            str(t): v
            for t, v in zip(tf_all["trip_id"], tf_all[cols].to_numpy(np.float64), strict=True)
        }
        d9 = within_trip_spearman(
            trip_idx,
            reference,
            _scores(trip_idx, {t: lookup[t] for t in trip_idx}, emb.astype(np.float64)),
        )
        train_frame = build_ranking_frame(
            candidates,
            train_ids,
            lab.interactions_train,
            trips,
            travelers,
            pois,
            pf,
            tf_all,
            budget,
        )
        hold_frame = build_ranking_frame(
            candidates, holdout_ids, holdout_random, trips, travelers, pois, pf, tf_all, budget
        )
        booster, num, cat, _, _ = fit_lgb(lab, train_frame=train_frame)
        per = per_trip_ndcg10(hold_frame, score_lgb(booster, num, cat, hold_frame))
        results[name] = {
            "ndcg10": summarize_ndcg(per, lab.eval_cfg),
            "d9_within_trip": d9,
        }
        print(f"DR8 {name}: {results[name]['ndcg10']['mean']:.4f}", flush=True)
    write_dr(
        results_dir,
        "DR8",
        "Rebuild the traveler features with a different taste half-life / interaction weights, "
        "refit the ranker on the fixed candidate sets, and score the unbiased holdout; D9 "
        "(within-trip, reporting-only) shows the effect on estimator fidelity.",
        "within-trip D9; holdout NDCG@10",
        results,
        "; ".join(
            f"{k}: NDCG@10 {v['ndcg10']['mean']:.4f}, D9 "
            f"{v['d9_within_trip']['mean_within_trip_spearman']:.3f}"
            for k, v in results.items()
        ),
    )
