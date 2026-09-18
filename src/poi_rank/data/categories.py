"""Category-string canonicalization (spec.md section 4).

Maps the dirty category strings Phase 1 injects (~6% of rows,
`dirtiness.inconsistent_category_rate` in `configs/datagen.yaml`) plus the 12 clean
canonical strings themselves back to one of 12 canonical categories. Anything else
maps to `"other"` and is counted/reported rather than silently absorbed (spec.md
requirement).

The variant table below is hand-authored by inspecting
`datagen/taxonomy.py::CATEGORY_STRING_VARIANTS` directly -- not imported from
`datagen/`. A real production category-cleaning pipeline would not import its
training-data generator's internals, and duplicating this small (36-entry) table
keeps `data/` fully self-contained.
"""

from __future__ import annotations

import pandas as pd

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "restaurant",
    "cafe",
    "museum",
    "historic_site",
    "nature_park",
    "nightlife",
    "shopping",
    "family_activity",
    "wellness_spa",
    "religious_site",
    "viewpoint",
    "entertainment",
)
OTHER_CATEGORY = "other"

# Dirty variant strings Phase 1 injects (inspected from
# datagen/taxonomy.py::CATEGORY_STRING_VARIANTS).
_CATEGORY_VARIANTS: dict[str, tuple[str, ...]] = {
    "restaurant": ("Restaurant", "restaurants", "Food & Dining"),
    "cafe": ("Cafe", "cafes", "Coffee Shop"),
    "museum": ("Museum", "museums", "Museum / Gallery"),
    "historic_site": ("Historic Site", "historic sites", "Historical Landmark"),
    "nature_park": ("Park", "parks & nature", "Nature / Park"),
    "nightlife": ("Nightlife", "bars & clubs", "Night Life"),
    "shopping": ("Shopping", "shops", "Retail / Shopping"),
    "family_activity": ("Family Activity", "family activities", "Kids & Family"),
    "wellness_spa": ("Wellness", "spa & wellness", "Spa / Wellness"),
    "religious_site": ("Religious Site", "religious sites", "Temple / Shrine"),
    "viewpoint": ("Viewpoint", "viewpoints", "Scenic View"),
    "entertainment": ("Entertainment", "entertainment venues", "Fun & Entertainment"),
}


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical in CANONICAL_CATEGORIES:
        lookup[canonical.strip().lower()] = canonical
        for variant in _CATEGORY_VARIANTS[canonical]:
            lookup[variant.strip().lower()] = canonical
    return lookup


_LOOKUP = _build_lookup()


def canonicalize_categories(raw: pd.Series) -> pd.DataFrame:
    """Map a raw (possibly dirty) category string Series to canonical categories.

    Returns a DataFrame (aligned to `raw.index`) with:
      - `category`: canonical value, or `"other"` for anything unmapped.
      - `category_raw`: the untouched original string, kept for audit/provenance.
    """
    normalized = raw.astype(str).str.strip().str.lower()
    canonical = normalized.map(_LOOKUP).fillna(OTHER_CATEGORY)
    return pd.DataFrame({"category": canonical, "category_raw": raw.to_numpy()}, index=raw.index)


def other_category_report(canonical: pd.Series) -> dict[str, float | int]:
    """Count/rate of rows that fell into `"other"` -- spec.md requires this be logged,
    never silently absorbed."""
    n_other = int((canonical == OTHER_CATEGORY).sum())
    n_total = int(len(canonical))
    return {
        "n_other": n_other,
        "n_total": n_total,
        "other_rate": n_other / n_total if n_total else 0.0,
    }
