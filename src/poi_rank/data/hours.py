"""Opening-hours parsing (spec.md section 4).

Raw `opening_hours` is a JSON-encoded 7-day open/close map, or `None` when Phase 1
injects the whole-field "missing/irregular hours" dirtiness
(`missing_opening_hours_rate` in `configs/datagen.yaml`). Parsed into a 168-bit
weekly boolean mask (24 hourly bins x 7 days) plus an `hours_missing` flag. Missing
rows are imputed with their category's median (modal) weekly pattern, computed from
that category's own valid (non-missing) rows.

Note: a day entry of `null` *within* a present `opening_hours` record (e.g. a museum
closed on Monday) represents a real closure, not missingness -- it contributes an
all-closed segment to the mask and is not counted in `hours_missing`.
"""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pandas as pd

DAYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
HOURS_PER_WEEK = 24 * len(DAYS)


def _time_to_hour(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60.0


def _day_hour_mask(open_h: float, close_h: float) -> npt.NDArray[np.bool_]:
    """Hour-resolution (24-bin) open mask for one day. `close_h <= open_h` is treated
    as an overnight wrap (e.g. nightlife 18:00-02:00): the late-night hours are
    attributed to the day the session *started*, the standard simplification for a
    day-keyed weekly-hours mask."""
    hours = np.arange(24, dtype=np.float64)
    if close_h > open_h:
        return (hours + 1.0 > open_h) & (hours < close_h)
    return (hours + 1.0 > open_h) | (hours < close_h)


def parse_hours_mask(raw: str | None) -> npt.NDArray[np.bool_]:
    """Parse one POI's raw `opening_hours` JSON string into a 168-bit mask (day-major:
    index = `day_idx * 24 + hour`, in `DAYS` order). `raw is None` (whole-field
    missing) returns an all-False mask -- the caller is responsible for overwriting
    it with the category-imputed pattern."""
    if raw is None:
        return np.zeros(HOURS_PER_WEEK, dtype=bool)
    parsed: dict[str, dict[str, str] | None] = json.loads(raw)
    mask = np.zeros(HOURS_PER_WEEK, dtype=bool)
    for day_idx, day in enumerate(DAYS):
        entry = parsed.get(day)
        if entry is None:
            continue
        open_h = _time_to_hour(entry["open"])
        close_h = _time_to_hour(entry["close"])
        mask[day_idx * 24 : (day_idx + 1) * 24] = _day_hour_mask(open_h, close_h)
    return mask


def _category_modal_pattern(masks: pd.Series) -> npt.NDArray[np.bool_]:
    """Bin-wise majority vote (>=50% of valid POIs open) across a category's valid
    masks."""
    stacked = np.stack(masks.to_numpy())
    modal: npt.NDArray[np.bool_] = stacked.mean(axis=0) >= 0.5
    return modal


def compute_hours(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    """Compute `hours_mask` (168-bool list) and `hours_missing` for every POI.

    Rows with `opening_hours is None` are imputed with their category's modal weekly
    pattern, computed only from that category's own valid rows; falls back to the
    global modal pattern (across all valid rows) if a category has zero valid rows.
    """
    hours_missing = df["opening_hours"].isna()
    raw_masks = df["opening_hours"].apply(parse_hours_mask)

    valid = ~hours_missing
    global_pattern = (
        _category_modal_pattern(raw_masks.loc[valid])
        if valid.any()
        else np.zeros(HOURS_PER_WEEK, dtype=bool)
    )
    category_patterns: dict[str, npt.NDArray[np.bool_]] = {}
    for cat, idx in df.loc[valid].groupby(category_col).groups.items():
        category_patterns[str(cat)] = _category_modal_pattern(raw_masks.loc[idx])

    final_masks = raw_masks.copy()
    for i in df.index[hours_missing]:
        cat = str(df.at[i, category_col])
        final_masks.at[i] = category_patterns.get(cat, global_pattern)

    return pd.DataFrame(
        {
            "hours_mask": [m.tolist() for m in final_masks],
            "hours_missing": hours_missing,
        },
        index=df.index,
    )
