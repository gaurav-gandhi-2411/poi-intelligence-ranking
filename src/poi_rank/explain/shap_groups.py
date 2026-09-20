"""Grouped TreeSHAP (spec.md section 10): raw per-feature SHAP values for the
primary LambdaMART+IPS booster (`artifacts/model.txt`), aggregated into ~10
semantic feature groups.

**Why `shap.TreeExplainer`, not KernelSHAP**: LightGBM is an exact tree model --
`TreeExplainer` computes *exact* Shapley values in polynomial time by walking the
tree structure directly (Lundberg et al. 2018), not the model-agnostic sampling
approximation KernelSHAP needs for a black-box model. Verified directly against this
project's own booster (not assumed from the library's docs): `shap_values.sum(axis=1)
+ expected_value` reproduces `booster.predict(X)` to float roundoff (~1e-15) on a
500-row sample of the real holdout frame -- see `tests/test_shap_groups.py
::test_shap_values_sum_to_raw_margin_output`.

**The ~10 groups, spec.md section 10's own named list**: interest_match,
implicit_taste, localness_fit, popularity, price_fit, geo, hours, party_fit,
quality, novelty. Every one of the 237 real feature columns this project's
LambdaMART booster actually consumes (233 numeric + 4 categorical,
`models.baselines.numeric_feature_columns`/`categorical_feature_columns`) is
assigned to EXACTLY one group -- `tests/test_shap_groups.py
::test_every_real_feature_column_is_assigned_to_exactly_one_group` verifies this
against the real ranking frame, not a hand-maintained column list that could drift.

**`quality`**: spec.md section 2.2's own DGP documents `latent_quality_p` as
"observed only *noisily* through `rating` and `review_count`" -- so this group maps
to the observable noisy-proxy columns `num_rating_shrunk`/`num_log_review_count`,
per the task's explicit instruction, NOT to any oracle-only latent value (this
module never references the oracle export directory).

**`novelty`**: until the cross-feature work (experiment H) no column in this project's feature
tables was a defensible observable proxy for the DGP's per-trip `novelty_t` term, so the group
shipped GENUINELY EMPTY (contribution exactly 0.0). It now holds `xf_prior_poi_engaged`, the
time-decayed count of the traveler's OWN earlier engagements with this POI, strictly before the
trip's own session (an observable repeat-visit signal, not the latent `novelty_t`). The group is
still excluded from `top_signals`/explanations (`explain/templates.py`): no renderer exists for it.

See docs/DATA_CARD.md's Phase 7 section for the full column -> group table.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap

FloatArray = npt.NDArray[np.float64]

# -----------------------------------------------------------------------------------
# The 10 groups, in spec.md section 10's own listed order.
# -----------------------------------------------------------------------------------

INTEREST_MATCH = "interest_match"
IMPLICIT_TASTE = "implicit_taste"
LOCALNESS_FIT = "localness_fit"
POPULARITY = "popularity"
PRICE_FIT = "price_fit"
GEO = "geo"
HOURS = "hours"
PARTY_FIT = "party_fit"
QUALITY = "quality"
NOVELTY = "novelty"

FEATURE_GROUPS: tuple[str, ...] = (
    INTEREST_MATCH,
    IMPLICIT_TASTE,
    LOCALNESS_FIT,
    POPULARITY,
    PRICE_FIT,
    GEO,
    HOURS,
    PARTY_FIT,
    QUALITY,
    NOVELTY,
)

# -----------------------------------------------------------------------------------
# Column -> group assignment. Two mechanisms, checked in order: (1) an exact-name
# lookup for columns whose group membership doesn't follow a clean prefix rule, (2)
# a prefix lookup for columns that do. Every real column must hit exactly one of
# these -- `classify_feature_column` raises (fails CLOSED, per rule 98a) on anything
# neither table recognizes, rather than silently dropping it from every group's sum.
# -----------------------------------------------------------------------------------

_EXACT_GROUP: dict[str, str] = {
    # interest_match: explicit stated-interest matching + the POI's own
    # category/subcategory (what the traveler would compare their stated interests
    # against).
    "interact_interest_match": INTEREST_MATCH,
    "cat_category": INTEREST_MATCH,
    "cat_subcategory": INTEREST_MATCH,
    # implicit_taste: traveler-history-derived signals (taste-vector cosine, engaged-
    # interaction summary stats) -- `explicit_days_remaining` is included here (not
    # under `hours`) because `features/traveler_features.py`'s own module docstring
    # documents it as reducing to the exact same underlying quantity as
    # `implicit_days_since_last_interaction` in this implementation; grouping them
    # together keeps two columns carrying identical information in one group rather
    # than double-representing one true signal across two groups.
    # Explicit traveler x POI cross features (features/cross_features.py, models/cross_ranking.py).
    "xf_loc_align": LOCALNESS_FIT,
    "xf_loc_gap": LOCALNESS_FIT,
    "xf_loc_x_pref": LOCALNESS_FIT,
    "xf_pop_x_pref": LOCALNESS_FIT,
    "xf_price_signed": PRICE_FIT,
    "xf_price_over": PRICE_FIT,
    "xf_price_under": PRICE_FIT,
    "xf_budget_fit": PRICE_FIT,
    "xf_interest_tag_hits": INTEREST_MATCH,
    "xf_interest_cat_hit": INTEREST_MATCH,
    "xf_interest_cover": INTEREST_MATCH,
    "xf_interest_tag_ratio": INTEREST_MATCH,
    "xf_mobility_fit": GEO,
    "xf_travel_min": GEO,
    "xf_travel_ratio": GEO,
    "xf_hours_fit": HOURS,
    "xf_duration_fit": HOURS,
    "xf_party_fit": PARTY_FIT,
    "xf_prior_poi_engaged": NOVELTY,
    "xf_cos_dismissed": IMPLICIT_TASTE,
    "interact_cos_taste_poi": IMPLICIT_TASTE,
    "implicit_interaction_count": IMPLICIT_TASTE,
    "implicit_days_since_last_interaction": IMPLICIT_TASTE,
    "implicit_days_since_last_interaction_was_missing": IMPLICIT_TASTE,
    "explicit_days_remaining": IMPLICIT_TASTE,
    "implicit_breadth_categories": IMPLICIT_TASTE,
    "implicit_mean_price_level": IMPLICIT_TASTE,
    "implicit_mean_price_level_was_missing": IMPLICIT_TASTE,
    "implicit_mean_pop_pct": IMPLICIT_TASTE,
    "implicit_mean_pop_pct_was_missing": IMPLICIT_TASTE,
    "implicit_mean_localness": IMPLICIT_TASTE,
    "implicit_mean_localness_was_missing": IMPLICIT_TASTE,
    # POI-level *rate* behavioral columns (how people who saw this POI responded) are
    # implicit_taste (a collective-engagement taste signal); raw *volume* columns
    # (impressions/unique_travelers) are popularity instead -- see POPULARITY below.
    "behav_ctr_smoothed": IMPLICIT_TASTE,
    "behav_save_rate": IMPLICIT_TASTE,
    "behav_visit_rate": IMPLICIT_TASTE,
    "behav_dismiss_rate": IMPLICIT_TASTE,
    # localness_fit: the candidate POI's own localness/touristiness signals, plus the
    # traveler's stated touristiness preference these are measured against.
    "num_localness": LOCALNESS_FIT,
    "interact_localness_gap": LOCALNESS_FIT,
    # Share of the traveler's engaged history in THIS POI's category (history-derived taste).
    "interact_category_affinity": IMPLICIT_TASTE,
    "geo_dist_to_tourist_centroid_km": LOCALNESS_FIT,
    "explicit_touristiness_pref": LOCALNESS_FIT,
    # popularity: raw visitation/exposure volume -- distinct from implicit_taste's
    # engagement-rate columns above, and from quality's rating/review columns below
    # (per the task's explicit rating/review_count -> quality instruction).
    "num_pop_pct": POPULARITY,
    "behav_impressions": POPULARITY,
    "behav_unique_travelers": POPULARITY,
    "num_crowd_index": POPULARITY,
    # price_fit
    "num_price_level": PRICE_FIT,
    "cat_price_level": PRICE_FIT,
    "interact_price_gap": PRICE_FIT,
    "explicit_budget_ordinal": PRICE_FIT,
    # geo: physical/spatial attributes of the POI and the traveler's mobility mode
    # (mobility interacts with geography -- how far is "close" depends on mode).
    "geo_lat": GEO,
    "geo_lon": GEO,
    "geo_dist_to_transit_km": GEO,
    "geo_density_500m": GEO,
    "cat_indoor_outdoor": GEO,
    # hours: broadened to "temporal/scheduling fit" (spec.md section 10's list has no
    # separate duration/reservation/season bucket) -- open-hours, reservation lead
    # time, trip duration/pace/season all describe WHEN and how long a visit fits,
    # not WHERE or WHAT.
    "num_open_hours_per_week": HOURS,
    "num_reservation_lead_days": HOURS,
    "explicit_pace_ordinal": HOURS,
    "num_expected_duration_min": HOURS,
    "explicit_trip_duration_days": HOURS,
    # quality: the observable noisy proxy for the DGP's latent_quality_p (module
    # docstring, spec.md section 2.2).
    "num_rating_shrunk": QUALITY,
    "num_log_review_count": QUALITY,
}

_PREFIX_GROUP: tuple[tuple[str, str], ...] = (
    ("explicit_interest_", INTEREST_MATCH),
    ("text_emb_", INTEREST_MATCH),
    ("implicit_taste_", IMPLICIT_TASTE),
    ("implicit_category_dist_", IMPLICIT_TASTE),
    ("behav_archetype_affinity_", IMPLICIT_TASTE),
    ("explicit_mobility_", GEO),
    ("explicit_season_", HOURS),
    ("explicit_party_", PARTY_FIT),
    ("explicit_accessibility_", PARTY_FIT),
)


def classify_feature_column(column: str) -> str:
    """Map one real feature-column name to exactly one of `FEATURE_GROUPS`. Fails
    CLOSED (raises `ValueError`, rule 98a) on any column neither `_EXACT_GROUP` nor
    `_PREFIX_GROUP` recognizes -- an unmapped column must never silently vanish from
    every group's SHAP sum."""
    if column in _EXACT_GROUP:
        return _EXACT_GROUP[column]
    for prefix, group in _PREFIX_GROUP:
        if column.startswith(prefix):
            return group
    raise ValueError(
        f"feature column {column!r} is not assigned to any of the {len(FEATURE_GROUPS)} "
        "explain/shap_groups.py TreeSHAP groups -- add it to _EXACT_GROUP or "
        "_PREFIX_GROUP (see docs/DATA_CARD.md Phase 7 mapping table)"
    )


def build_group_membership(
    numeric_columns: list[str], categorical_columns: list[str]
) -> dict[str, list[str]]:
    """`{group_name: [member_column, ...]}` for every one of `FEATURE_GROUPS` --
    always includes every group name as a key, even `novelty` (module docstring:
    genuinely empty, not merely unpopulated for this particular column set)."""
    membership: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    for column in [*numeric_columns, *categorical_columns]:
        membership[classify_feature_column(column)].append(column)
    return {g: sorted(cols) for g, cols in membership.items()}


# -----------------------------------------------------------------------------------
# TreeSHAP computation + group aggregation
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedShapResult:
    """`group_contributions`: one row per input row (aligned to `frame.index`), one
    column per `FEATURE_GROUPS` entry, summed raw SHAP values. `expected_value`: the
    booster's base value (SHAP's `explainer.expected_value`) -- `group_contributions
    .sum(axis=1) + expected_value` reproduces the booster's raw margin prediction
    (verified in `tests/test_shap_groups.py`)."""

    group_contributions: pd.DataFrame
    expected_value: float
    raw_shap_values: FloatArray
    feature_columns: list[str]


def compute_grouped_shap(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> GroupedShapResult:
    """Exact TreeSHAP for `booster` over `frame`'s feature matrix (the SAME
    `_feature_matrix` layout `models.lambdamart.score_booster` scores against --
    numeric block cast to float64, categorical block passed through as pandas
    `category` dtype), aggregated into the ~10 semantic groups (module docstring)."""
    feature_columns = [*numeric_columns, *categorical_columns]
    numeric = frame[numeric_columns].astype(np.float32)  # same dtype as lambdamart._feature_matrix
    categorical = frame[categorical_columns]
    x = pd.concat([numeric, categorical], axis=1)

    explainer = shap.TreeExplainer(booster)
    raw_shap_values = np.asarray(explainer.shap_values(x), dtype=np.float64)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    membership = build_group_membership(numeric_columns, categorical_columns)
    col_index = {c: i for i, c in enumerate(feature_columns)}

    group_data: dict[str, FloatArray] = {}
    for group in FEATURE_GROUPS:
        member_idx = [col_index[c] for c in membership[group]]
        if member_idx:
            group_data[group] = raw_shap_values[:, member_idx].sum(axis=1)
        else:
            group_data[group] = np.zeros(len(frame), dtype=np.float64)

    group_contributions = pd.DataFrame(group_data, index=frame.index)[list(FEATURE_GROUPS)]
    return GroupedShapResult(
        group_contributions=group_contributions,
        expected_value=expected_value,
        raw_shap_values=raw_shap_values,
        feature_columns=feature_columns,
    )
