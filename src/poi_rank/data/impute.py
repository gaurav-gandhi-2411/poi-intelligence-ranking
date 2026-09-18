"""Median-by-(destination, category) imputation for remaining numeric gaps (spec.md
section 4): `price_level`, `expected_duration_min`.

**Resolving spec.md's two-part sentence** ("median-by-(destination,category) + explicit
`_was_missing` indicator... LightGBM handles NaN natively, but indicators are kept for
explainability"): this module never overwrites the original column's real NaNs -- a
later model phase is expected to consume the raw column directly, letting LightGBM's
native NaN handling do its job. Instead, for each field it adds *two* new columns:
`<field>_imputed` (the median-filled value, for any explicit/explainability use that
wants an always-present number) and `<field>_was_missing` (the indicator). Three
columns per field total (raw + imputed + indicator), so neither half of spec.md's
sentence is silently dropped. Applied post-dedup, using the canonical `category`.
"""

from __future__ import annotations

import pandas as pd

NUMERIC_IMPUTE_FIELDS: tuple[str, ...] = ("price_level", "expected_duration_min")
# price_level is an ordinal 1-4 scale; its imputed value is rounded to stay on-scale.
_ROUND_TO_INT: frozenset[str] = frozenset({"price_level"})


def _group_median_fill(series: pd.Series, destination: pd.Series, category: pd.Series) -> pd.Series:
    """Fill NaNs with the (destination, category) group median; falls back to the
    destination median, then the global median, for any group with zero valid rows."""
    group_median = series.groupby([destination, category]).transform("median")
    filled = series.fillna(group_median)

    if filled.isna().any():
        dest_median = series.groupby(destination).transform("median")
        filled = filled.fillna(dest_median)
    if filled.isna().any():
        filled = filled.fillna(series.median())
    return filled


def impute_numeric_fields(
    df: pd.DataFrame, fields: tuple[str, ...] = NUMERIC_IMPUTE_FIELDS
) -> pd.DataFrame:
    """Returns a DataFrame (aligned to `df.index`) with `<field>_imputed` and
    `<field>_was_missing` columns for every field in `fields`. Does not modify or
    return the original columns -- the caller concatenates this onto the main
    prepared frame alongside the untouched raw columns."""
    out: dict[str, pd.Series] = {}
    for field in fields:
        # `pd.to_numeric` (not `.astype(float)`) because after dedup's merge, a
        # pandas-nullable `Int64` column with `pd.NA` entries (`price_level`) can end
        # up object-dtype, and `.astype(float)` cannot convert `pd.NA` directly.
        series = pd.to_numeric(df[field], errors="coerce")
        was_missing = series.isna()
        filled = _group_median_fill(series, df["destination"], df["category"])
        if field in _ROUND_TO_INT:
            filled = filled.round()
        out[f"{field}_imputed"] = filled
        out[f"{field}_was_missing"] = was_missing
    return pd.DataFrame(out, index=df.index)
