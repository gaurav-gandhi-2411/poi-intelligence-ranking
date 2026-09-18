"""`explain/shap_groups.py`: grouped TreeSHAP correctness (spec.md section 10).

Uses the real session-scoped fixture chain (`explain_grouped_shap`,
`scoring_pipeline_result`) for the two invariants that must hold against REAL data
(SHAP-sum correctness, exhaustive/non-overlapping column coverage); everything else
is a pure unit test against `classify_feature_column`/`build_group_membership`
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from poi_rank.explain.shap_groups import (
    FEATURE_GROUPS,
    NOVELTY,
    build_group_membership,
    classify_feature_column,
)
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns

# -----------------------------------------------------------------------------------
# classify_feature_column / build_group_membership -- pure unit tests
# -----------------------------------------------------------------------------------


def test_every_feature_group_name_is_covered_by_the_membership_dict() -> None:
    membership = build_group_membership(["num_rating_shrunk"], ["cat_category"])
    assert set(membership.keys()) == set(FEATURE_GROUPS)


def test_novelty_group_is_genuinely_empty_by_construction() -> None:
    """docs/DATA_CARD.md: `novelty` has no observable proxy in this project's
    feature tables -- verified here that NO real column, from a representative
    sample of every prefix family, classifies into it."""
    membership = build_group_membership(
        numeric_columns=[
            "num_rating_shrunk",
            "text_emb_00",
            "implicit_taste_00",
            "behav_impressions",
            "geo_lat",
            "explicit_interest_foodie",
        ],
        categorical_columns=["cat_category"],
    )
    assert membership[NOVELTY] == []


def test_unmapped_column_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not assigned"):
        classify_feature_column("this_column_does_not_exist_anywhere")


def test_quality_group_maps_to_rating_and_review_count_columns() -> None:
    """Task instruction / spec.md section 2.2: `latent_quality_p` is observed only
    noisily through `rating` and `review_count` -- `quality` must map to exactly
    those two observable proxies, not to any oracle-only latent value."""
    membership = build_group_membership(["num_rating_shrunk", "num_log_review_count"], [])
    assert set(membership["quality"]) == {"num_rating_shrunk", "num_log_review_count"}


@pytest.mark.parametrize(
    ("column", "expected_group"),
    [
        ("explicit_interest_foodie", "interest_match"),
        ("text_emb_00", "interest_match"),
        ("cat_category", "interest_match"),
        ("implicit_taste_00", "implicit_taste"),
        ("implicit_category_dist_museum", "implicit_taste"),
        ("behav_archetype_affinity_00", "implicit_taste"),
        ("behav_ctr_smoothed", "implicit_taste"),
        ("num_localness", "localness_fit"),
        ("interact_localness_gap", "localness_fit"),
        ("num_pop_pct", "popularity"),
        ("behav_impressions", "popularity"),
        ("num_price_level", "price_fit"),
        ("cat_price_level", "price_fit"),
        ("geo_lat", "geo"),
        ("explicit_mobility_walk", "geo"),
        ("num_open_hours_per_week", "hours"),
        ("explicit_season_winter", "hours"),
        ("explicit_party_solo", "party_fit"),
        ("explicit_accessibility_wheelchair", "party_fit"),
        ("num_rating_shrunk", "quality"),
    ],
)
def test_classify_feature_column_representative_cases(column: str, expected_group: str) -> None:
    assert classify_feature_column(column) == expected_group


# -----------------------------------------------------------------------------------
# Real-data invariants
# -----------------------------------------------------------------------------------


def test_every_real_feature_column_is_assigned_to_exactly_one_group(
    scoring_pipeline_result: dict[str, Any],
) -> None:
    """Every numeric/categorical column the real ranking frame actually carries
    (`models.baselines.numeric_feature_columns`/`categorical_feature_columns`, the
    SAME functions `models/lambdamart.py` uses to build the LightGBM design matrix)
    must classify into exactly one group -- not a hand-maintained column list that
    could silently drift from the real feature table."""
    full = scoring_pipeline_result["full_frame"]
    numeric_columns = numeric_feature_columns(full)
    categorical_columns = categorical_feature_columns(full)
    all_columns = [*numeric_columns, *categorical_columns]

    membership = build_group_membership(numeric_columns, categorical_columns)
    assigned = [c for cols in membership.values() for c in cols]

    msg = "every real column must be assigned exactly once"
    assert sorted(assigned) == sorted(all_columns), msg
    assert len(assigned) == len(set(assigned)), "no column may be double-counted across groups"


def test_shap_values_sum_to_raw_margin_output(
    scoring_pipeline_result: dict[str, Any], explain_grouped_shap: Any, explain_ips_booster: Any
) -> None:
    """Core SHAP correctness invariant: `sum(SHAP values) + expected_value ==
    raw model margin output`, verified AFTER the group-aggregation step -- proves no
    feature's SHAP value was silently dropped or double-counted when summing into
    the ~10 semantic groups. The comparison target is the SAME booster scoring the
    SAME frame via `models.lambdamart.score_booster`'s own production
    `_feature_matrix` layout."""
    from poi_rank.models.lambdamart import score_booster

    full = scoring_pipeline_result["full_frame"]
    numeric_columns = numeric_feature_columns(full)
    categorical_columns = categorical_feature_columns(full)

    raw_predictions = (
        explain_grouped_shap.group_contributions.sum(axis=1).to_numpy(dtype=np.float64)
        + explain_grouped_shap.expected_value
    )
    expected = score_booster(explain_ips_booster, full, numeric_columns, categorical_columns)

    assert np.allclose(raw_predictions, expected.to_numpy(dtype=np.float64), atol=1e-6)
