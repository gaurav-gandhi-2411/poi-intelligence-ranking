"""Full scoring-pipeline orchestration + output JSON assembly (spec.md section 9.5).
`poi_rank.cli recommend`'s entry point (`run_recommend`).

Pipeline, for the primary unbiased holdout trip set (spec.md section 11.1's own
evaluation population -- "real trips" this phase is asked to score, consistent with
every prior phase's evaluation convention):

1. Load `booster_lambdamart_ips` (system 8, `artifacts/model.txt`) -- LOADED, never
   retrained here (same discipline as `eval/run.py`).
2. Fit the isotonic calibrator on a calibration split carved from TRAIN
   (`scoring/calibration.py`), persist it to `artifacts/calibrator.pkl` (spec.md
   section 14 lists this as a committed artifact), apply it to the holdout raw
   scores to get `preference_score` (= `relevance`).
3. Compute `hard_gate` + the 6 compatibility sub-scores + `compatibility`
   (`scoring/compatibility.py`) for every holdout candidate.
4. `utility = hard_gate * relevance^alpha * compatibility^beta`
   (`scoring/utility.py`), plus the beta-sensitivity table.
5. Compute `confidence` (`scoring/confidence.py`, including the 5-seed ensemble),
   plus the confidence-decile NDCG validation + Spearman rho.
6. **Filter out every `hard_gate == 0` candidate before any ranking step** (spec.md
   section 11.5's build-blocking "0 violations in top-10" requirement: filtered at
   the source, not just relying on `utility == 0` to sort it out of the top-K) --
   the hard-constraint test (`tests/test_hard_constraints.py`) asserts this
   pipeline never produces a violation, but this module ALSO asserts it internally
   (defense in depth) before ever serializing output.
7. MMR-rerank each trip's top-50-by-utility pool to a top-K list
   (`scoring/diversity.py`), plus the lambda-sweep NDCG-vs-diversity curve.
8. Assemble the spec.md section 9.5 JSON schema, one entry per trip -- `top_signals`/
   `explanation` are assembled here as documented PLACEHOLDERS (module-level
   constants below); `run_recommend`'s `payload_enricher` hook is where a caller
   (`poi_rank.cli recommend`, via `poi_rank.explain.output_enrichment
   .build_payload_enricher`) replaces them with real grouped-TreeSHAP-driven content
   (spec.md section 10, Phase 7) -- `scoring/` itself never computes or imports that
   content directly (firewall: `tests/test_firewall_scoring.py
   ::test_scoring_never_imports_explain`).
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from poi_rank.candidates.config import GeoChannelConfig
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models import baselines as bl
from poi_rank.models import lambdamart as lm
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import load_holdout_evaluation_frame, load_train_ranking_frame
from poi_rank.scoring import calibration as cal
from poi_rank.scoring import compatibility as compat
from poi_rank.scoring import confidence as conf
from poi_rank.scoring import diversity as div
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.utility import (
    beta_sensitivity_payload,
    beta_sensitivity_table,
    compute_utility,
    pref_align,
)

FloatArray = npt.NDArray[np.float64]

RECOMMENDATIONS_FILENAME = "recommendations.json"
CALIBRATOR_FILENAME = "calibrator.pkl"
RELIABILITY_FIGURE_FILENAME = "calibration_reliability.png"
LAMBDA_SWEEP_FIGURE_FILENAME = "mmr_lambda_sweep.png"

# `pref_align` (experiment L1) is reported beside the six compatibility sub-scores so a downstream
# planner sees every factor of the utility; it is NOT one of the six geometric-mean terms.
COMPATIBILITY_BREAKDOWN_KEYS: tuple[str, ...] = (
    *compat.COMPATIBILITY_SUB_SCORE_NAMES,
    "pref_align",
)

_PLACEHOLDER_TOP_SIGNALS: list[dict[str, Any]] = [
    {
        "feature_group": "placeholder_pending_explainability_phase",
        "contribution": 0.0,
    }
]
_PLACEHOLDER_EXPLANATION: list[str] = [
    "Explanation generation (grouped TreeSHAP + template layer, spec.md section 10) "
    "is implemented in a later phase (src/poi_rank/explain/) -- this is a structural "
    "placeholder with the correct output schema, not a real explanation."
]


# -----------------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------------


def fit_and_apply_calibration(
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    raw_score_holdout: pd.Series,
    booster: Any,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
) -> dict[str, Any]:
    """Fit the isotonic calibrator on a calibration split carved from TRAIN (module
    docstring), apply it to the holdout raw scores, and report ECE/Brier
    before-vs-after on the HOLDOUT (never the calibration split itself -- see
    `scoring/calibration.py`'s own module docstring)."""
    fit_frame, _val_frame = lm.train_val_split_by_trip(
        train_frame, model_cfg.lambdamart.val_fraction, model_cfg.lambdamart.val_split_seed
    )
    calib_cfg = scoring_cfg.calibration
    _remaining_fit, calib_frame = cal.carve_calibration_split(
        fit_frame, calib_cfg.calibration_fraction, calib_cfg.calibration_split_seed
    )
    calib_raw = lm.score_booster(booster, calib_frame, numeric_columns, categorical_columns)
    ir = cal.fit_isotonic_calibrator(calib_raw, calib_frame["label"])

    relevance = cal.apply_calibrator(ir, raw_score_holdout)
    naive = cal.naive_probability_from_raw_score(raw_score_holdout)
    holdout_binary_label = (holdout_frame["label"].to_numpy(dtype=np.int64) >= 1).astype(np.float64)

    ece_before = cal.expected_calibration_error(
        naive.to_numpy(dtype=np.float64), holdout_binary_label, scoring_cfg.calibration.n_bins
    )
    ece_after = cal.expected_calibration_error(
        relevance.to_numpy(dtype=np.float64), holdout_binary_label, scoring_cfg.calibration.n_bins
    )
    brier_before = cal.brier_score(naive.to_numpy(dtype=np.float64), holdout_binary_label)
    brier_after = cal.brier_score(relevance.to_numpy(dtype=np.float64), holdout_binary_label)

    return {
        "calibrator": ir,
        "relevance": relevance,
        "naive_probability": naive,
        "n_calibration_rows": int(len(calib_frame)),
        "n_calibration_trips": int(calib_frame["trip_id"].nunique()),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "holdout_binary_label": holdout_binary_label,
    }


# -----------------------------------------------------------------------------------
# Confidence + decile validation
# -----------------------------------------------------------------------------------


def confidence_decile_validation(frame: pd.DataFrame, k: int) -> dict[str, Any]:
    """Bin TRIPS (not individual candidate rows) into confidence deciles, using
    each trip's mean confidence over its own top-`k`-by-utility candidates as a
    single representative scalar -- spec.md section 9.3 says "bin recommendations
    into confidence deciles and report NDCG@10 per decile"; read here as per-trip
    RECOMMENDATION LISTS (NDCG is inherently a per-list metric, not a per-item one),
    a resolved ambiguity documented in docs/DATA_CARD.md. Reports mean NDCG@k per
    decile plus the Spearman rho of (decile rank, decile mean NDCG@k) -- spec.md's
    own monotonicity target is rho >= 0.7.
    """
    from poi_rank.scoring._ranking_metrics import ndcg_at_k

    per_trip_rows: list[dict[str, Any]] = []
    for trip_id, group in frame.groupby("trip_id", sort=True):
        top = group.sort_values(["utility", "poi_id"], ascending=[False, True]).head(k)
        trip_confidence = float(top["confidence"].mean())

        labels = group["label"].to_numpy(dtype=np.int64)
        scores = group["utility"].to_numpy(dtype=np.float64)
        poi_ids = group["poi_id"].to_numpy(dtype=object)
        ndcg = ndcg_at_k(labels, scores, poi_ids, k)
        per_trip_rows.append({"trip_id": trip_id, "confidence": trip_confidence, "ndcg": ndcg})

    per_trip = pd.DataFrame(per_trip_rows).dropna(subset=["ndcg"])
    if len(per_trip) < 10:
        return {
            "n_trips_included": int(len(per_trip)),
            "deciles": [],
            "spearman_rho": None,
            "spearman_p_value": None,
            "target_met": False,
            "note": "fewer than 10 trips with a defined NDCG@k -- deciles undefined",
        }

    per_trip = per_trip.sort_values("confidence").reset_index(drop=True)
    per_trip["decile"] = pd.qcut(per_trip["confidence"], q=10, labels=False, duplicates="drop") + 1

    decile_rows = []
    for decile, group in per_trip.groupby("decile", sort=True):
        decile_rows.append(
            {
                "decile": int(decile),
                "mean_confidence": float(group["confidence"].mean()),
                "ndcg@k_mean": float(group["ndcg"].mean()),
                "n_trips": int(len(group)),
            }
        )

    deciles_arr = np.array([r["decile"] for r in decile_rows], dtype=np.float64)
    ndcg_arr = np.array([r["ndcg@k_mean"] for r in decile_rows], dtype=np.float64)
    if len(decile_rows) >= 2:
        rho, p_value = stats.spearmanr(deciles_arr, ndcg_arr)
    else:
        rho, p_value = float("nan"), float("nan")

    rho_val = float(rho) if not np.isnan(rho) else None
    return {
        "n_trips_included": int(len(per_trip)),
        "deciles": decile_rows,
        "spearman_rho": rho_val,
        "spearman_p_value": float(p_value) if not np.isnan(p_value) else None,
        "target_met": bool(rho_val is not None and rho_val >= 0.7),
    }


# -----------------------------------------------------------------------------------
# Output JSON assembly (spec.md section 9.5)
# -----------------------------------------------------------------------------------


def assemble_output_payload(
    survivors: pd.DataFrame,
    reranked: pd.DataFrame,
    trips_df: pd.DataFrame,
    scoring_cfg: ScoringConfig,
    longtail_pop_pct_cutoff: float,
) -> dict[str, Any]:
    """Assemble the spec.md section 9.5 JSON schema, one entry per trip. `survivors`
    must be `hard_gate == 1` ONLY (module docstring step 6); this function asserts
    that invariant again defensively before ever building a recommendation entry --
    a violation here raises, it is never silently dropped or logged-and-ignored.

    `diversity_group` and `top_signals`/`explanation` are documented placeholders
    (module-level constants) here -- grouped TreeSHAP (spec.md section 10,
    `src/poi_rank/explain/`) lives outside this module by firewall construction
    (`scoring/` must never import `explain/`); `run_recommend`'s `payload_enricher`
    hook is where a caller replaces `top_signals`/`explanation` with real content
    AFTER this function returns (module docstring). `diversity_group` uses a simple
    category + local/touristy heuristic (`num_pop_pct` vs
    `longtail_pop_pct_cutoff`, reused from `candidates/config.py`'s own long-tail
    threshold for consistency with the rest of this project's "local discovery"
    framing) rather than the eventual TreeSHAP-grouped signal. See docs/DATA_CARD.md.
    """
    lookup = survivors.set_index(["trip_id", "poi_id"])
    trip_ctx = trips_df.set_index("trip_id")[["traveler_id", "destination"]]
    generated_at = datetime.now(UTC).isoformat()

    payload: dict[str, Any] = {}
    for trip_id, group in reranked.groupby("trip_id", sort=True):
        ordered = group.sort_values("mmr_rank")
        rows = [lookup.loc[(trip_id, pid)] for pid in ordered["poi_id"]]
        utilities = np.array([float(r["utility"]) for r in rows], dtype=np.float64)
        weight_denom = float(utilities.sum())
        weights = (
            utilities / weight_denom
            if weight_denom > 0
            else np.full(len(utilities), 1.0 / len(utilities) if len(utilities) else 0.0)
        )

        recs: list[dict[str, Any]] = []
        for pos, (poi_id, row, weight) in enumerate(
            zip(ordered["poi_id"], rows, weights, strict=True), start=1
        ):
            assert row["hard_gate"] == 1.0, (
                f"hard-constraint violation: {trip_id}/{poi_id} has hard_gate=0 but "
                "reached output assembly -- this must never happen (spec.md section 11.5)"
            )
            pop_pct = float(row["num_pop_pct"])
            bucket = "local" if pop_pct < longtail_pop_pct_cutoff else "touristy"
            recs.append(
                {
                    "poi_id": str(poi_id),
                    "rank": pos,
                    "utility": float(row["utility"]),
                    "planner_weight": float(weight),
                    "preference_score": float(row["relevance"]),
                    "context_compatibility": float(row["compatibility"]),
                    "pref_align": float(row["pref_align"]),
                    "compatibility_breakdown": {
                        name: float(row[name]) for name in COMPATIBILITY_BREAKDOWN_KEYS
                    },
                    "confidence": float(row["confidence"]),
                    "hard_constraints_ok": bool(row["hard_gate"] == 1.0),
                    "expected_duration_min": float(row["num_expected_duration_min"]),
                    "diversity_group": f"{row['poi_category_raw']}_{bucket}",
                    "popularity_percentile": float(pop_pct * 100.0),
                    "localness_index": float(row["num_localness"]),
                    "top_signals": _PLACEHOLDER_TOP_SIGNALS,
                    "explanation": _PLACEHOLDER_EXPLANATION,
                }
            )

        ctx = trip_ctx.loc[trip_id]
        payload[str(trip_id)] = {
            "trip_id": str(trip_id),
            "traveler_id": str(ctx["traveler_id"]),
            "destination": str(ctx["destination"]),
            "generated_at": generated_at,
            "model_version": scoring_cfg.output.model_version,
            "recommendations": recs,
        }
    return payload


# -----------------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------------


def run_scoring_pipeline(
    data_dir: Path,
    artifacts_dir: Path,
    feature_cfg: FeatureBuildConfig,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    geo_cfg: GeoChannelConfig,
    longtail_pop_pct_cutoff: float,
    trip_id_filter: set[str] | None = None,
    holdout_frame_override: pd.DataFrame | None = None,
    trips_df_override: pd.DataFrame | None = None,
    travelers_df_override: pd.DataFrame | None = None,
    candidates_df_override: pd.DataFrame | None = None,
    calibrator_override: Any | None = None,
) -> dict[str, Any]:
    """Run the full spec.md section 9 scoring pipeline over the primary holdout
    trip set (optionally restricted to `trip_id_filter`), returning every
    intermediate diagnostic AND the final assembled recommendation JSON payload.
    Does not write any files -- `run_recommend` (below) handles persistence.

    The 4 `*_override` parameters, all `None` by default (preserving this
    function's original from-disk-only behavior exactly for `poi_rank.cli
    recommend` and every existing test), let a caller substitute an
    already-assembled `(trip_id, poi_id)` ranking frame / trips / travelers /
    candidates population instead of loading the real primary holdout from
    `data_dir` -- `eval/scenarios.py`'s hand-built synthetic traveler/trip
    profiles run through this EXACT SAME pipeline this way (module docstring's
    "1-8" steps unchanged), rather than a second, parallel scoring
    implementation that could silently drift from this one. `train_frame`
    (calibration fitting, confidence ensemble training) and `pois_df` (the real
    catalog every candidate POI is drawn from) are never overridden -- a
    synthetic traveler/trip is a new REQUEST against the same real system, not a
    new training population or a new catalog."""
    budget_target_price_level = feature_cfg.traveler_features.budget_target_price_level
    if holdout_frame_override is not None:
        holdout_frame = holdout_frame_override
    else:
        holdout_frame = load_holdout_evaluation_frame(data_dir, budget_target_price_level)

    if trip_id_filter is not None:
        keep = holdout_frame["trip_id"].isin(trip_id_filter)
        holdout_frame = holdout_frame.loc[keep].reset_index(drop=True)

    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")

    # `review_count` attached onto `holdout_frame` BEFORE any other computation --
    # `poi_id` is unique in `pois_df`, so this LEFT merge is a pure column-add: same
    # row count and row order as `holdout_frame`, never a fan-out. Every subsequent
    # per-row array (raw score, relevance, confidence, ...) is computed against
    # THIS SAME frame object and assigned back as a plain column, so alignment is
    # guaranteed correct by construction -- never reconstructed after the fact via
    # `.reindex`/positional-array mixing across two independently-merged frames
    # (a real bug class: a later `trip_id, poi_id` merge is free to reorder rows,
    # which would silently desynchronize any array computed against the
    # pre-merge frame and assigned back by raw numpy-array position).
    holdout_frame = holdout_frame.merge(
        pois_df[["poi_id", "review_count", "name"]], on="poi_id", how="left"
    )

    numeric_columns = bl.numeric_feature_columns(holdout_frame)
    categorical_columns = bl.categorical_feature_columns(holdout_frame)
    boosters = lm.load_boosters(artifacts_dir)
    booster_ips = boosters["lambdamart_ips"]
    raw_score = lm.score_booster(booster_ips, holdout_frame, numeric_columns, categorical_columns)
    raw_score_by_key: dict[tuple[str, str], float] = {
        (str(t), str(p)): float(s)
        for t, p, s in zip(
            holdout_frame["trip_id"], holdout_frame["poi_id"], raw_score, strict=True
        )
    }

    if calibrator_override is None:
        train_frame = load_train_ranking_frame(data_dir, budget_target_price_level)
        calib_result = fit_and_apply_calibration(
            train_frame,
            holdout_frame,
            raw_score,
            booster_ips,
            numeric_columns,
            categorical_columns,
            model_cfg,
            scoring_cfg,
        )
    else:
        # Serving path (`poi_rank.cli demo`): apply the calibrator persisted by `recommend`
        # instead of re-fitting it on the whole train frame; calibration diagnostics need
        # labelled holdout rows and are not computed here.
        calib_result = {
            "calibrator": calibrator_override,
            "relevance": cal.apply_calibrator(calibrator_override, raw_score),
            "naive_probability": cal.naive_probability_from_raw_score(raw_score),
            "n_calibration_rows": 0,
            "n_calibration_trips": 0,
            "ece_before": float("nan"),
            "ece_after": float("nan"),
            "brier_before": float("nan"),
            "brier_after": float("nan"),
            "holdout_binary_label": (holdout_frame["label"].to_numpy(dtype=np.int64) >= 1).astype(
                np.float64
            ),
        }
    holdout_frame["relevance"] = calib_result["relevance"]

    # Confidence: ensemble std (5 seeds), trained/scored against this SAME
    # numeric/categorical column layout.
    ensemble_boosters = conf.load_ensemble(scoring_cfg.confidence.ensemble_seeds, artifacts_dir)
    ensemble_std = conf.ensemble_std_scores(
        ensemble_boosters, holdout_frame, numeric_columns, categorical_columns
    )
    bin_width = cal.calibration_bin_width(
        calib_result["calibrator"], raw_score.to_numpy(dtype=np.float64)
    )
    holdout_frame["confidence"] = conf.compute_confidence(
        holdout_frame["implicit_interaction_count"].to_numpy(dtype=np.float64),
        holdout_frame["behav_impressions"].to_numpy(dtype=np.float64),
        holdout_frame["review_count"].to_numpy(dtype=np.float64),
        ensemble_std,
        bin_width,
        scoring_cfg.confidence,
    )

    trips_df = (
        trips_df_override
        if trips_df_override is not None
        else pd.read_parquet(data_dir / "trips.parquet")
    )
    travelers_df = (
        travelers_df_override
        if travelers_df_override is not None
        else pd.read_parquet(data_dir / "travelers.parquet")
    )
    candidates_df = (
        candidates_df_override
        if candidates_df_override is not None
        else pd.read_parquet(data_dir / "candidates.parquet")
    )

    trip_ids = set(holdout_frame["trip_id"].unique())
    compat_frame = compat.compute_compatibility_frame(
        candidates_df,
        trip_ids,
        trips_df,
        travelers_df,
        pois_df,
        budget_target_price_level,
        geo_cfg,
        scoring_cfg.compatibility,
    )

    # `compat_frame` is keyed by the SAME (trip_id, poi_id) candidate universe
    # (both ultimately restrict `candidates_df` to the same `trip_ids`) -- an inner
    # join here is expected to be lossless, but `how="inner"` (not `"left"`) is kept
    # anyway as a defensive check: a candidate present in one frame but not the
    # other would silently vanish from the OUTPUT rather than crash, which is the
    # conservative failure mode for a recommendation pipeline (never recommend a
    # candidate this phase could not fully score).
    full = holdout_frame.merge(compat_frame, on=["trip_id", "poi_id"], how="inner")

    ucfg = scoring_cfg.utility
    full["pref_align"] = pref_align(
        full["num_localness"].to_numpy(dtype=np.float64),
        full["explicit_touristiness_pref"].to_numpy(dtype=np.float64),
        ucfg.pref_align_center,
        ucfg.pref_align_scale,
    )
    full["utility"] = compute_utility(
        full["hard_gate"].to_numpy(dtype=np.float64),
        full["relevance"].to_numpy(dtype=np.float64),
        full["compatibility"].to_numpy(dtype=np.float64),
        ucfg.alpha,
        ucfg.beta,
        ucfg.gamma,
        full["pref_align"].to_numpy(dtype=np.float64),
    )

    # Beta sensitivity: the preference factor rides along inside the relevance term
    # (relevance_eff^alpha == relevance^alpha * pref_align^gamma), so each beta row is the utility
    # actually served at that beta.
    relevance_eff = full["relevance"].to_numpy(dtype=np.float64) * np.power(
        full["pref_align"].to_numpy(dtype=np.float64), ucfg.gamma / ucfg.alpha
    )
    beta_rows = beta_sensitivity_table(
        full,
        full["hard_gate"].to_numpy(dtype=np.float64),
        relevance_eff,
        full["compatibility"].to_numpy(dtype=np.float64),
        scoring_cfg.utility,
        k=scoring_cfg.output.top_k,
    )

    decile_validation = confidence_decile_validation(full, scoring_cfg.output.top_k)

    # Hard-constraint enforcement (module docstring step 6): filter BEFORE ranking,
    # never rely on utility==0 alone to keep a violation out of the top-K.
    survivors = full.loc[full["hard_gate"] == 1.0].reset_index(drop=True)

    top_k = scoring_cfg.output.top_k
    lambda_rows = div.lambda_sweep_report(
        survivors, scoring_cfg.diversity, scoring_cfg.diversity.lambda_sweep, top_k
    )

    reranked = div.mmr_rerank_all_trips(
        survivors, scoring_cfg.diversity, scoring_cfg.diversity.lambda_default, top_k
    )

    payload = assemble_output_payload(
        survivors, reranked, trips_df, scoring_cfg, longtail_pop_pct_cutoff
    )

    return {
        "payload": payload,
        "calibration": {
            "n_calibration_rows": calib_result["n_calibration_rows"],
            "n_calibration_trips": calib_result["n_calibration_trips"],
            "ece_before": calib_result["ece_before"],
            "ece_after": calib_result["ece_after"],
            "brier_before": calib_result["brier_before"],
            "brier_after": calib_result["brier_after"],
        },
        "calibrator": calib_result["calibrator"],
        "naive_probability": calib_result["naive_probability"],
        "holdout_binary_label": calib_result["holdout_binary_label"],
        "relevance": calib_result["relevance"],
        "beta_sensitivity": beta_sensitivity_payload(beta_rows),
        "confidence_decile_validation": decile_validation,
        "lambda_sweep": div.lambda_sweep_payload(lambda_rows),
        "lambda_sweep_rows": lambda_rows,
        "full_frame": full,
        "n_holdout_trips": int(holdout_frame["trip_id"].nunique()),
        # Raw (pre-calibration) LambdaMART+IPS score, re-aligned to `full`'s row
        # order via an explicit `(trip_id, poi_id)` dict lookup (never a positional/
        # `.loc[full.index]` alignment -- `full`'s index is a fresh RangeIndex from
        # the `holdout_frame.merge(compat_frame, ...)` above and must not be assumed
        # to share label semantics with `raw_score`'s own pre-merge index, same
        # "never rely on merge-induced row order" discipline this module's own
        # docstring already states for every other per-row array here) -- exposed
        # for `eval/ablations.py`'s `-calibration` ablation (raw vs
        # `relevance`/calibrated NDCG@10 over the identical row population).
        "raw_score": pd.Series(
            [
                raw_score_by_key[(str(t), str(p))]
                for t, p in zip(full["trip_id"], full["poi_id"], strict=True)
            ],
            index=full.index,
            name="score_lambdamart_ips_raw",
        ),
    }


# -----------------------------------------------------------------------------------
# `poi_rank.cli recommend` entry point
# -----------------------------------------------------------------------------------


def run_recommend(
    data_dir: Path,
    artifacts_dir: Path,
    results_dir: Path,
    feature_cfg: FeatureBuildConfig,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    geo_cfg: GeoChannelConfig,
    longtail_pop_pct_cutoff: float,
    trip_id_filter: set[str] | None = None,
    payload_enricher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """`poi_rank.cli recommend`'s entry point: run the full scoring pipeline,
    persist `results/recommendations.json`, `artifacts/calibrator.pkl` (spec.md
    section 14), and the calibration-reliability / MMR-lambda-sweep figures
    (`results/figures/`). Requires `poi_rank.cli train` to have already written
    `artifacts/model.txt` (system 8, LOADED here, never retrained). Returns a
    summary dict for the CLI report and tests.

    `payload_enricher`, if given, is called with the FULL `run_scoring_pipeline`
    result dict (has `full_frame`, `payload`, everything) and must return a
    replacement payload -- the hook `poi_rank.explain.output_enrichment
    .build_payload_enricher` plugs into (module docstring: `scoring/` must never
    import `poi_rank.explain` per `tests/test_firewall_scoring.py
    ::test_scoring_never_imports_explain`, so this parameter is a plain, generic
    `Callable` -- `poi_rank.cli`'s `recommend` command is the only place that ever
    constructs a real one). `None` (the default) leaves `result["payload"]`
    untouched, i.e. `top_signals`/`explanation` stay `scoring/output.py`'s own
    documented placeholders -- unchanged default behavior for any caller (a direct
    test, a future scoring-only script) that doesn't wire in `explain/`.
    """
    result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        geo_cfg,
        longtail_pop_pct_cutoff,
        trip_id_filter=trip_id_filter,
    )
    payload = payload_enricher(result) if payload_enricher is not None else result["payload"]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = artifacts_dir / CALIBRATOR_FILENAME
    with calibrator_path.open("wb") as f:
        pickle.dump(result["calibrator"], f)

    results_dir.mkdir(parents=True, exist_ok=True)
    recommendations_path = results_dir / RECOMMENDATIONS_FILENAME
    recommendations_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figures_dir = results_dir / "figures"
    reliability_path = figures_dir / RELIABILITY_FIGURE_FILENAME
    cal.reliability_diagram(
        result["naive_probability"].to_numpy(dtype=np.float64),
        result["relevance"].to_numpy(dtype=np.float64),
        result["holdout_binary_label"],
        scoring_cfg.calibration.n_bins,
        reliability_path,
    )

    lambda_sweep_path = figures_dir / LAMBDA_SWEEP_FIGURE_FILENAME
    div.plot_lambda_sweep(result["lambda_sweep_rows"], lambda_sweep_path)

    return {
        "recommendations_path": recommendations_path,
        "calibrator_path": calibrator_path,
        "reliability_figure_path": reliability_path,
        "lambda_sweep_figure_path": lambda_sweep_path,
        "n_holdout_trips": result["n_holdout_trips"],
        "n_trips_output": len(payload),
        "calibration": result["calibration"],
        "beta_sensitivity": result["beta_sensitivity"],
        "confidence_decile_validation": result["confidence_decile_validation"],
        "lambda_sweep": result["lambda_sweep"],
    }
