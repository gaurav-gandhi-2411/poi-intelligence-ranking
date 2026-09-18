"""Phase 2 data-prep tests (spec.md section 4): dedup, category canonicalization,
popularity percentile, opening hours, imputation, and general prepared-output sanity.

Localness-vs-oracle Spearman validation lives in its own file
(tests/test_localness_oracle.py) since it is the single load-bearing correctness
signal of this phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.data.categories import (
    CANONICAL_CATEGORIES,
    OTHER_CATEGORY,
    canonicalize_categories,
    other_category_report,
)
from poi_rank.data.dedup import dedup_pois
from poi_rank.data.hours import HOURS_PER_WEEK, compute_hours, parse_hours_mask
from poi_rank.data.impute import impute_numeric_fields
from poi_rank.data.popularity import compute_popularity_percentile
from poi_rank.data.prepare import PREPARED_POIS_FILENAME

# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_dedup_merge_rate_close_to_injected(prepared_data: dict[str, Any]) -> None:
    """The recovered merge rate should be close to (not wildly above or below)
    Phase 1's actual injected near-duplicate rate -- the checkable correctness
    signal spec.md calls for."""
    pois_raw: pd.DataFrame = prepared_data["pois_raw"]
    report = prepared_data["report"]

    injected_rate = float(pois_raw["is_duplicate"].mean())
    recovered_rate = report.dedup.merge_rate

    assert recovered_rate > 0, "dedup found zero merges; injected duplicates should be findable"
    # Allow the recovered rate to undershoot the injected rate (a similarity/distance
    # threshold will always miss some genuine near-duplicates) but not overshoot it
    # by more than a small margin (over-merging unrelated POIs would be a bug).
    assert recovered_rate <= injected_rate + 0.01, (
        f"recovered merge_rate={recovered_rate:.4f} overshoots injected "
        f"rate={injected_rate:.4f} by more than the tolerance -- possible over-merging"
    )
    assert recovered_rate >= injected_rate * 0.75, (
        f"recovered merge_rate={recovered_rate:.4f} recovers less than 75% of the "
        f"injected rate={injected_rate:.4f} -- dedup may be too conservative"
    )


def test_dedup_recovers_injected_duplicate_pairs(prepared_data: dict[str, Any]) -> None:
    """Spot-check: for each row Phase 1 flagged `is_duplicate=True`, its `duplicate_of`
    base POI should usually end up merged into the same output cluster."""
    pois_raw: pd.DataFrame = prepared_data["pois_raw"]
    prepared: pd.DataFrame = prepared_data["prepared"]

    merged_into: dict[str, str] = {}
    for row in prepared.itertuples(index=False):
        for original_id in row.merged_poi_ids:
            merged_into[original_id] = row.poi_id

    dup_rows = pois_raw.loc[pois_raw["is_duplicate"]]
    assert len(dup_rows) > 0, "expected Phase 1 to inject at least one near-duplicate"

    n_recovered = sum(
        1
        for r in dup_rows.itertuples(index=False)
        if merged_into.get(r.poi_id) == merged_into.get(r.duplicate_of)
    )
    recovery_rate = n_recovered / len(dup_rows)
    assert recovery_rate >= 0.85, (
        f"only {n_recovered}/{len(dup_rows)} ({recovery_rate:.1%}) injected duplicates "
        "ended up merged with their base POI"
    )


def test_merged_record_fields_are_plausible(prepared_data: dict[str, Any]) -> None:
    """A merged row's `review_count` should equal the sum of its constituent rows'
    original review counts, and `rating` should fall within their min/max range."""
    pois_raw: pd.DataFrame = prepared_data["pois_raw"]
    prepared: pd.DataFrame = prepared_data["prepared"]

    merged_rows = prepared.loc[prepared["n_merged"] > 1]
    assert len(merged_rows) > 0, "expected at least one multi-row merge cluster"

    for row in merged_rows.head(10).itertuples(index=False):
        originals = pois_raw.loc[pois_raw["poi_id"].isin(row.merged_poi_ids)]
        assert row.review_count == originals["review_count"].sum()
        assert originals["rating"].min() - 1e-6 <= row.rating <= originals["rating"].max() + 1e-6
        # tags union: every original tag must survive into the merged tag list.
        all_original_tags = {t for tags in originals["tags"] for t in tags}
        assert all_original_tags <= set(row.tags)


def test_dedup_on_synthetic_block() -> None:
    """Unit test on a hand-built 4-row catalog: two near-identical rows <50m apart
    should merge; one far-away same-name row and one differently-named nearby row
    should not."""
    df = pd.DataFrame(
        [
            {
                "poi_id": "A1",
                "destination": "seoul",
                "name": "Cozy Hongdae Kitchen",
                "lat": 37.5665,
                "lon": 126.9780,
                "description": "short",
                "tags": ["local"],
                "review_count": 100,
                "rating": 4.0,
            },
            {
                "poi_id": "A2",
                "destination": "seoul",
                "name": "Cozy Hongdae Kitchen Annex",
                "lat": 37.56653,  # ~3m north
                "lon": 126.97802,
                "description": "a slightly longer description of the same place",
                "tags": ["hidden-gem"],
                "review_count": 10,
                "rating": 4.2,
            },
            {
                "poi_id": "B1",
                "destination": "seoul",
                "name": "Cozy Hongdae Kitchen",
                "lat": 37.62,  # far away, same name
                "lon": 126.98,
                "description": "unrelated",
                "tags": [],
                "review_count": 50,
                "rating": 3.5,
            },
            {
                "poi_id": "C1",
                "destination": "seoul",
                "name": "Rustic Gion Teahouse",
                "lat": 37.56651,  # close to A1 but a different POI
                "lon": 126.97798,
                "description": "different place",
                "tags": [],
                "review_count": 20,
                "rating": 3.0,
            },
        ]
    )
    out, report = dedup_pois(df, h3_resolution=8, name_threshold=88.0, max_distance_m=50.0)

    assert report.n_input_rows == 4
    assert report.n_merged_away == 1  # exactly A1+A2 merge
    assert len(out) == 3

    merged_row = out.loc[out["poi_id"] == "A1"].iloc[0]
    assert merged_row["review_count"] == 110
    assert set(merged_row["merged_poi_ids"]) == {"A1", "A2"}
    assert "B1" in out["poi_id"].to_numpy()
    assert "C1" in out["poi_id"].to_numpy()


# ---------------------------------------------------------------------------
# Category canonicalization
# ---------------------------------------------------------------------------


def test_category_canonicalization_no_dirty_strings_leak(prepared_data: dict[str, Any]) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    allowed = set(CANONICAL_CATEGORIES) | {OTHER_CATEGORY}
    assert set(prepared["category"].unique()) <= allowed


def test_category_other_rate_is_reported_and_small(prepared_data: dict[str, Any]) -> None:
    report = prepared_data["report"]
    other = report.other_category
    assert other["n_total"] == len(prepared_data["prepared"])
    # Reported honestly either way; for this DGP's exhaustive variant table the rate
    # is expected to be at or near zero, but the report itself (not a specific
    # nonzero value) is the spec requirement.
    assert 0.0 <= other["other_rate"] <= 0.05, f"unexpectedly high other_rate={other['other_rate']}"


def test_canonicalize_categories_unmapped_string_becomes_other() -> None:
    raw = pd.Series(["Restaurant", "restaurants", "totally-unknown-string", "museum"])
    result = canonicalize_categories(raw)
    assert result["category"].tolist() == ["restaurant", "restaurant", "other", "museum"]
    report = other_category_report(result["category"])
    assert report["n_other"] == 1
    assert report["n_total"] == 4


# ---------------------------------------------------------------------------
# Popularity percentile
# ---------------------------------------------------------------------------


def test_pop_pct_is_valid_percentile(prepared_data: dict[str, Any]) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    assert prepared["pop_pct"].between(0.0, 1.0).all()
    # every destination's max should be close to 1.0 (the top-ranked POI)
    max_by_dest = prepared.groupby("destination")["pop_pct"].max()
    assert (max_by_dest > 0.99).all()


def test_pop_pct_is_within_destination_not_global() -> None:
    """Swapping which destination a POI 'belongs to' must change its percentile --
    proof `pop_pct` is not a global rank."""
    df = pd.DataFrame(
        {
            "destination": ["seoul"] * 3 + ["kyoto"] * 3,
            "review_count": [1000, 500, 10, 5, 3, 1],
            "rating_shrunk": [4.5, 4.0, 3.0, 4.5, 4.0, 3.0],
        }
    )
    pop_pct = compute_popularity_percentile(df)

    # The 3rd seoul row (review_count=10) ranks last within seoul (pct ~0.33).
    assert pop_pct.iloc[2] < 0.5

    # Re-labeling that same row into kyoto's group changes its rank: kyoto's smallest
    # review_count is 1, so a review_count=10 row would rank *first* in kyoto.
    df_swapped = df.copy()
    df_swapped.loc[2, "destination"] = "kyoto"
    pop_pct_swapped = compute_popularity_percentile(df_swapped)
    assert pop_pct_swapped.iloc[2] > pop_pct.iloc[2]


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------


def test_hours_mask_well_formed_shape(prepared_data: dict[str, Any]) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    lengths = prepared["hours_mask"].apply(len)
    assert (lengths == HOURS_PER_WEEK).all()

    def _all_bool(mask: list[bool]) -> bool:
        return all(isinstance(bit, bool | np.bool_) for bit in mask)

    assert prepared["hours_mask"].apply(_all_bool).all()


def test_parse_hours_mask_known_good_regular_hours() -> None:
    import json

    raw = json.dumps(
        {
            "mon": {"open": "09:00", "close": "17:00"},
            "tue": {"open": "09:00", "close": "17:00"},
            "wed": {"open": "09:00", "close": "17:00"},
            "thu": {"open": "09:00", "close": "17:00"},
            "fri": {"open": "09:00", "close": "17:00"},
            "sat": None,
            "sun": None,
        }
    )
    mask = parse_hours_mask(raw)
    assert mask.shape == (HOURS_PER_WEEK,)
    # Monday (day_idx=0): hours 9..16 open, hour 8 and 17 closed.
    assert not mask[8]
    assert mask[9]
    assert mask[16]
    assert not mask[17]
    # Saturday (day_idx=5) entirely closed.
    assert not mask[5 * 24 : 6 * 24].any()


def test_parse_hours_mask_overnight_wrap() -> None:
    import json

    raw = json.dumps({"mon": {"open": "18:00", "close": "02:00"}})
    mask = parse_hours_mask(raw)
    day_mask = mask[0:24]
    assert day_mask[18] and day_mask[23]
    assert day_mask[0] and day_mask[1]
    assert not day_mask[10]


def test_parse_hours_mask_missing_field_is_all_closed() -> None:
    mask = parse_hours_mask(None)
    assert mask.shape == (HOURS_PER_WEEK,)
    assert not mask.any()


def test_compute_hours_imputes_missing_with_category_pattern() -> None:
    import json

    good_hours = json.dumps(
        {
            d: {"open": "09:00", "close": "17:00"}
            for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        }
    )
    df = pd.DataFrame(
        {
            "category": ["museum", "museum", "museum", "cafe"],
            "opening_hours": [good_hours, good_hours, None, good_hours],
        }
    )
    result = compute_hours(df)
    assert result["hours_missing"].tolist() == [False, False, True, False]
    # the missing museum row should get exactly the museum modal pattern (both valid
    # museum rows are identical, so the modal pattern equals that exact mask).
    assert result.loc[2, "hours_mask"] == result.loc[0, "hours_mask"]


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def test_impute_numeric_fields_no_nan_remains(prepared_data: dict[str, Any]) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    assert not prepared["price_level_imputed"].isna().any()
    assert not prepared["expected_duration_min_imputed"].isna().any()


def test_impute_numeric_fields_was_missing_matches_original_nan(
    prepared_data: dict[str, Any],
) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    price_na = pd.to_numeric(prepared["price_level"], errors="coerce").isna()
    assert (prepared["price_level_was_missing"] == price_na).all()


def test_impute_numeric_fields_group_median_basic() -> None:
    df = pd.DataFrame(
        {
            "destination": ["seoul", "seoul", "seoul", "seoul"],
            "category": ["cafe", "cafe", "cafe", "cafe"],
            "price_level": [1, 3, np.nan, 5],
        }
    )
    out = impute_numeric_fields(df, fields=("price_level",))
    assert out["price_level_was_missing"].tolist() == [False, False, True, False]
    # median of [1, 3, 5] = 3
    assert out.loc[2, "price_level_imputed"] == 3


# ---------------------------------------------------------------------------
# General sanity: no hard-constraint-adjacent NaN leakage
# ---------------------------------------------------------------------------


def test_no_nan_in_hard_constraint_adjacent_fields(prepared_data: dict[str, Any]) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    critical_cols = ["poi_id", "destination", "category", "lat", "lon", "accessibility"]
    for col in critical_cols:
        assert not prepared[col].isna().any(), f"unexpected NaN in critical column '{col}'"
    assert (
        prepared["accessibility"]
        .apply(lambda a: {"wheelchair", "stroller", "kid_friendly"} <= set(a.keys()))
        .all()
    )


def test_prepared_output_filename_constant_matches_convention() -> None:
    assert PREPARED_POIS_FILENAME == "pois_prepared.parquet"


@pytest.mark.parametrize(
    "column",
    ["rating_shrunk", "pop_pct", "localness", "dist_to_transit_km", "h3_cell"],
)
def test_prepared_output_has_expected_new_columns(
    prepared_data: dict[str, Any], column: str
) -> None:
    prepared: pd.DataFrame = prepared_data["prepared"]
    assert column in prepared.columns
    if column != "h3_cell":
        assert not prepared[column].isna().any()
