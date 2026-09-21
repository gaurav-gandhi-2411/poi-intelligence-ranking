"""Orchestration (spec.md section 10): the LAST stage before output. Computes
grouped TreeSHAP against `scoring/output.py`'s own fully-joined `full` frame, builds
the per-recommendation `top_signals`/`explanation` (+ counterfactual line where
applicable), and returns a NEW payload with `scoring/output.py`'s placeholders
(`_PLACEHOLDER_TOP_SIGNALS`/`_PLACEHOLDER_EXPLANATION`) REPLACED by real content.

**Firewall direction** (task's explicit architectural instruction, matching Phase
6's already-established rule that `scoring/` must never import `explain/` --
`tests/test_firewall_scoring.py::test_scoring_never_imports_explain`): this module
cannot be called FROM inside `scoring/output.py`. Instead `poi_rank.cli`'s
`recommend` command wires the two together: `scoring.output.run_recommend` accepts a
generic `payload_enricher: Callable[[dict[str, Any]], dict[str, Any]] | None`
parameter (a plain `Callable` type hint -- zero import of `poi_rank.explain` inside
`scoring/output.py`), and `build_payload_enricher` (below) is the closure `cli.py`
constructs from THIS module and passes in. `explain/` legitimately imports from
`scoring/` (compatibility, utility are upstream of explainability) -- the opposite
direction from what `scoring/` is allowed.

**`diversity_group` is left as Phase 6's existing heuristic, deliberately** (task
item 6, "optional polish, don't over-invest"): `f"{category}_{'local'/'touristy'}"`
already reads as a clean, genuinely-observable semantic label and grouped TreeSHAP
does not obviously produce a cleaner one (the dominant SHAP group for most
recommendations is `interest_match`/`implicit_taste`, which would just relabel most
groups by category anyway, adding complexity for no measured improvement) --
documented as a considered decision in docs/DATA_CARD.md, not an oversight.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.data.geo_prep import haversine_km
from poi_rank.explain.counterfactual import compute_counterfactual_line
from poi_rank.explain.shap_groups import FEATURE_GROUPS, compute_grouped_shap
from poi_rank.explain.templates import ExplanationContext, build_explanation, top_signals
from poi_rank.features.config import BudgetTargetPriceLevel, FeatureBuildConfig
from poi_rank.models import baselines as bl
from poi_rank.models import lambdamart as lm
from poi_rank.scoring.config import MobilityFitConfig, ScoringConfig

FloatArray = npt.NDArray[np.float64]

FEATURE_GROUP_IMPORTANCE_FIGURE_FILENAME = "feature_group_importance.png"

TRAVELERS_FILENAME = "travelers.parquet"
POIS_PREPARED_FILENAME = "pois_prepared.parquet"


# -----------------------------------------------------------------------------------
# Travel time (genuinely recomputed, never fabricated -- module docstring)
# -----------------------------------------------------------------------------------


def compute_travel_minutes(
    frame: pd.DataFrame, pois_df: pd.DataFrame, mobility_cfg: MobilityFitConfig
) -> pd.Series:
    """Travel time in minutes from each row's trip stay point to its candidate POI,
    under the traveler's own mobility mode -- the EXACT formula
    `scoring/compatibility.py::mobility_fit_score` uses internally
    (`haversine_km(...) / speed_kmh[mode] * 60`), recomputed here (rather than
    imported) because `mobility_fit_score` only returns the decayed [0,1] SCORE, not
    the underlying minutes the explanation template needs to show. `.map()` (not a
    `merge`) preserves `frame`'s own row order/index exactly -- the same
    reorder-risk `scoring/output.py`'s own module docstring already documents for
    merges, avoided here the same way.
    """
    poi_lat = frame["poi_id"].map(pois_df.set_index("poi_id")["lat"])
    poi_lon = frame["poi_id"].map(pois_df.set_index("poi_id")["lon"])
    dist_km = haversine_km(
        poi_lat.to_numpy(dtype=np.float64),
        poi_lon.to_numpy(dtype=np.float64),
        frame["stay_lat"].to_numpy(dtype=np.float64),
        frame["stay_lon"].to_numpy(dtype=np.float64),
    )
    speed = frame["mobility"].map(mobility_cfg.speed_for).to_numpy(dtype=np.float64)
    travel_min = dist_km / speed * 60.0
    return pd.Series(travel_min, index=frame.index, name="travel_min")


# -----------------------------------------------------------------------------------
# Per-row explanation context
# -----------------------------------------------------------------------------------


def build_context_frame(
    full: pd.DataFrame,
    pois_df: pd.DataFrame,
    travelers_df: pd.DataFrame,
    mobility_cfg: MobilityFitConfig,
    budget_target_price_level: BudgetTargetPriceLevel,
    pref_align_active: bool = False,
) -> list[ExplanationContext]:
    """One `ExplanationContext` per row of `full`, in row order -- `party_type`
    (dropped by both `models.ranking_data.build_ranking_frame` and
    `scoring.compatibility.compute_compatibility_frame`'s own returned columns) is
    the only field this function pulls from a fresh `travelers_df` merge (`.map()`,
    same index-preserving discipline as `compute_travel_minutes`); every other field
    is already present on `full` itself.

    `price_level`/`budget_target_price_level` are pulled the same `.map()` way for
    the same reason `travel_min` is: `scoring.compatibility.compute_compatibility_frame`
    only returns the decayed [0,1] `budget_fit` SCORE, never the raw price/target the
    explanation template needs to state the correct DIRECTION of a weak budget fit
    (see `explain/templates.py::_weak_compat_line`'s docstring for the bug this fixes
    -- an earlier version always said "priced above", which was factually wrong for
    535/766, 70%, of the real "weak budget fit" lines this pipeline generated on the
    committed dataset, since a low `budget_fit` can equally come from the POI being
    CHEAPER than the traveler's target)."""
    travel_min = compute_travel_minutes(full, pois_df, mobility_cfg)
    party_type = full["traveler_id"].map(travelers_df.set_index("traveler_id")["party_type"])
    price_level = full["poi_id"].map(pois_df.set_index("poi_id")["price_level_imputed"])
    budget_target = full["budget"].map(budget_target_price_level.get)

    contexts: list[ExplanationContext] = []
    for i, (_idx, row) in enumerate(full.iterrows()):
        poi_terms = frozenset({str(row["poi_category_raw"])}) | frozenset(row["poi_tags_raw"])
        contexts.append(
            ExplanationContext(
                poi_category=str(row["poi_category_raw"]),
                destination=str(row["destination"]),
                budget=str(row["budget"]),
                party_type=str(party_type.iloc[i]),
                mobility=str(row["mobility"]),
                pop_pct=float(row["num_pop_pct"]),
                rating_shrunk=float(row["num_rating_shrunk"]),
                review_count=float(row["review_count"]),
                budget_fit=float(row["budget_fit"]),
                mobility_fit=float(row["mobility_fit"]),
                hours_fit=float(row["hours_fit"]),
                party_fit=float(row["party_fit"]),
                travel_min=float(travel_min.iloc[i]),
                implicit_interaction_count=float(row["implicit_interaction_count"]),
                interests=frozenset(row["interests"]),
                poi_terms=poi_terms,
                price_level=float(price_level.iloc[i]),
                budget_target_price_level=float(budget_target.iloc[i]),
                pref_align=float(row["pref_align"]) if pref_align_active else 0.5,
                touristiness_pref=float(row["explicit_touristiness_pref"]),
            )
        )
    return contexts


# -----------------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------------


def plot_feature_group_importance(group_contributions: pd.DataFrame, output_path: Path) -> None:
    """Global feature-group importance (spec.md section 10): mean |SHAP| per group
    across every RETURNED recommendation row (`group_contributions`, one row per
    `(trip_id, poi_id)`, `explain/shap_groups.py`'s ~10 group columns)."""
    mean_abs = group_contributions.abs().mean().reindex(FEATURE_GROUPS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(FEATURE_GROUPS)[::-1], mean_abs.to_numpy()[::-1])
    ax.set_xlabel("Mean |SHAP contribution|")
    ax.set_title("Feature-group importance (grouped TreeSHAP, returned recommendations)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------------
# Payload enrichment
# -----------------------------------------------------------------------------------


def enrich_recommend_result(
    result: dict[str, Any],
    data_dir: Path,
    artifacts_dir: Path,
    figures_dir: Path,
    scoring_cfg: ScoringConfig,
    feature_cfg: FeatureBuildConfig,
    pois_df_override: pd.DataFrame | None = None,
    travelers_df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """`result`: `scoring.output.run_scoring_pipeline`'s full return dict (has
    `full_frame` -- every candidate scored, hard-gate/compatibility/relevance/
    utility all attached -- and `payload`, the placeholder-carrying assembled JSON).
    Returns a NEW payload (deep copy, `result["payload"]` itself is left untouched)
    with `top_signals`/`explanation` replaced by real grouped-TreeSHAP-driven
    content, and writes `results/figures/feature_group_importance.png`.

    `pois_df_override`/`travelers_df_override`, both `None` by default (preserving
    this function's original from-disk-only behavior for `poi_rank.cli recommend`),
    let a caller substitute the POI/traveler population `build_context_frame` maps
    against -- `eval/scenarios.py` passes a `travelers_df` that includes its
    hand-built synthetic travelers (absent from the real, on-disk
    `travelers.parquet`, so `full["traveler_id"].map(...)` would otherwise return
    NaN `party_type` for every synthetic row's explanation context)."""
    full: pd.DataFrame = result["full_frame"]

    pois_df = (
        pois_df_override
        if pois_df_override is not None
        else pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)
    )
    travelers_df = (
        travelers_df_override
        if travelers_df_override is not None
        else pd.read_parquet(data_dir / TRAVELERS_FILENAME)
    )

    # TreeSHAP only for the (trip, POI) rows that are actually RETURNED: we never explain a
    # POI we do not recommend, and exact TreeSHAP over every scored candidate (~200/trip vs
    # 10 returned) was the dominant cost of `recommend`.
    full_position = {
        (str(tid), str(pid)): i
        for i, (tid, pid) in enumerate(zip(full["trip_id"], full["poi_id"], strict=True))
    }
    returned_rows = sorted(
        {
            full_position[(str(trip_id), rec["poi_id"])]
            for trip_id, trip_payload in result["payload"].items()
            for rec in trip_payload["recommendations"]
        }
    )
    explained = full.iloc[returned_rows]

    numeric_columns = bl.numeric_feature_columns(explained)
    categorical_columns = bl.categorical_feature_columns(explained)
    booster = lm.load_boosters(artifacts_dir)["lambdamart_ips"]
    shap_result = compute_grouped_shap(booster, explained, numeric_columns, categorical_columns)

    plot_feature_group_importance(
        shap_result.group_contributions, figures_dir / FEATURE_GROUP_IMPORTANCE_FIGURE_FILENAME
    )

    contexts = build_context_frame(
        explained,
        pois_df,
        travelers_df,
        scoring_cfg.compatibility.mobility_fit,
        feature_cfg.traveler_features.budget_target_price_level,
        pref_align_active=scoring_cfg.utility.gamma != 0.0,
    )
    group_contrib_records = shap_result.group_contributions.to_dict("records")

    # Position of each returned (trip, POI) within `explained` (SHAP / context row order).
    key_to_position = {
        (str(tid), str(pid)): i
        for i, (tid, pid) in enumerate(zip(explained["trip_id"], explained["poi_id"], strict=True))
    }

    survivors = full.loc[full["hard_gate"] == 1.0]
    survivors_by_trip = {
        str(trip_id): group for trip_id, group in survivors.groupby("trip_id", sort=False)
    }

    payload: dict[str, Any] = copy.deepcopy(result["payload"])
    for trip_id, trip_payload in payload.items():
        trip_survivors = survivors_by_trip.get(trip_id)
        for rec in trip_payload["recommendations"]:
            poi_id = rec["poi_id"]
            pos = key_to_position[(trip_id, poi_id)]
            group_contributions = group_contrib_records[pos]
            ctx = contexts[pos]

            rec["top_signals"] = top_signals(group_contributions)
            explanation = build_explanation(
                group_contributions, rec["compatibility_breakdown"], ctx
            )
            if trip_survivors is not None:
                cf_line = compute_counterfactual_line(
                    trip_survivors,
                    poi_id,
                    scoring_cfg.utility.alpha,
                    scoring_cfg.utility.beta,
                    scoring_cfg.utility.gamma,
                )
                if cf_line is not None:
                    explanation.append(cf_line)
            rec["explanation"] = explanation

    return payload


def build_payload_enricher(
    data_dir: Path,
    artifacts_dir: Path,
    figures_dir: Path,
    scoring_cfg: ScoringConfig,
    feature_cfg: FeatureBuildConfig,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Closure `poi_rank.cli`'s `recommend` command passes as `scoring.output
    .run_recommend`'s generic `payload_enricher` parameter (module docstring)."""

    def _enrich(result: dict[str, Any]) -> dict[str, Any]:
        return enrich_recommend_result(
            result, data_dir, artifacts_dir, figures_dir, scoring_cfg, feature_cfg
        )

    return _enrich
