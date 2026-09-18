"""POI feature table assembly (spec.md section 5).

Columns are grouped by block prefix (`text_emb_*`, `num_*`, `cat_*`, `geo_*`,
`behav_*`) rather than emitted as one undifferentiated table -- required for the
ablation harness (spec.md section 11.9: `-text embeddings`, `-behavioral block`) to
cleanly drop a whole block by column-name prefix later, without touching this module
again.

**Behavioral block is computed from `interactions_train.parquet` ONLY** -- never
holdout logs (that would leak eval-window behavior into a feature available at
training time). POI ids are reconciled through
`reconcile.build_poi_id_canonical_map` first (see `reconcile.py`'s module docstring)
so interactions logged against a since-merged-away duplicate POI id correctly roll up
onto the surviving canonical row. A POI with zero train-window impressions (e.g. a
new POI created inside the holdout window) gets `behav_impressions=0` and every rate
column `0.0` -- never NaN silently inherited from a missing groupby key, and never a
value derived from holdout interactions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.data.geo_prep import haversine_km
from poi_rank.features.config import PoiFeaturesConfig
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.features.traveler_features import assign_traveler_segments

FloatArray = npt.NDArray[np.float32]

TEXT_PREFIX = "text_emb_"
NUM_PREFIX = "num_"
CAT_PREFIX = "cat_"
GEO_PREFIX = "geo_"
BEHAV_PREFIX = "behav_"

CATEGORICAL_COLUMNS: tuple[str, ...] = ("category", "subcategory", "indoor_outdoor", "price_level")


# -----------------------------------------------------------------------------------
# Text block
# -----------------------------------------------------------------------------------


def build_text_block(index: pd.Index, embeddings: FloatArray) -> pd.DataFrame:
    """One `text_emb_NN` column per embedding dimension, L2-normalized (guaranteed by
    `text_embed.py`)."""
    cols = {f"{TEXT_PREFIX}{i:02d}": embeddings[:, i] for i in range(embeddings.shape[1])}
    return pd.DataFrame(cols, index=index)


# -----------------------------------------------------------------------------------
# Numeric block
# -----------------------------------------------------------------------------------


def compute_crowd_index(avg_crowd_by_hour: pd.Series) -> pd.Series:
    """Peak-to-mean ratio of each POI's 24-hour crowd curve: higher = crowding
    concentrated in a narrow window (e.g. a dinner rush), lower = steady all-day
    traffic. `avg_crowd_by_hour` is already max-normalized by datagen (peak == 1.0
    for any non-degenerate curve, `datagen/catalog.py::_sample_avg_crowd_by_hour`),
    so this reduces to `1 / mean(curve)` in practice -- computed as an explicit
    peak/mean ratio rather than hardcoding that identity, so it stays correct if a
    future DGP revision changes the normalization.
    """

    def _ratio(curve: object) -> float:
        arr = np.asarray(curve, dtype=float)
        mean = arr.mean()
        return float(arr.max() / mean) if mean > 0 else 0.0

    return avg_crowd_by_hour.apply(_ratio)


def compute_open_hours_per_week(hours_mask: pd.Series) -> pd.Series:
    """Sum of the 168-bit hourly weekly mask -- each bit is one hour, so the sum is
    directly "hours open per week."""
    return hours_mask.apply(lambda mask: int(np.asarray(mask).sum()))


def build_numeric_block(pois_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric POI features (spec.md section 5). Uses the `_imputed` (always-present)
    columns from Phase 2 for `price_level`/`expected_duration_min` -- spec.md's
    section 5 table literally lists `price_level` in BOTH the Numeric and Categorical
    rows; resolved here by using the imputed numeric form for the numeric block and
    the raw (missingness-preserving) form for the categorical block, rather than
    literally duplicating one representation -- see docs/DATA_CARD.md.
    """
    return pd.DataFrame(
        {
            f"{NUM_PREFIX}rating_shrunk": pois_df["rating_shrunk"].astype(float),
            f"{NUM_PREFIX}log_review_count": np.log1p(pois_df["review_count"].astype(float)),
            f"{NUM_PREFIX}pop_pct": pois_df["pop_pct"].astype(float),
            f"{NUM_PREFIX}localness": pois_df["localness"].astype(float),
            f"{NUM_PREFIX}price_level": pois_df["price_level_imputed"].astype(float),
            f"{NUM_PREFIX}expected_duration_min": pois_df["expected_duration_min_imputed"].astype(
                float
            ),
            f"{NUM_PREFIX}open_hours_per_week": compute_open_hours_per_week(
                pois_df["hours_mask"]
            ).astype(float),
            f"{NUM_PREFIX}reservation_lead_days": pois_df["reservation_lead_days"].astype(float),
            f"{NUM_PREFIX}crowd_index": compute_crowd_index(pois_df["avg_crowd_by_hour"]),
        },
        index=pois_df.index,
    )


# -----------------------------------------------------------------------------------
# Categorical block (pandas `category` dtype -- LightGBM-native, never one-hot)
# -----------------------------------------------------------------------------------


def build_categorical_block(pois_df: pd.DataFrame) -> pd.DataFrame:
    """`category`, `subcategory`, `indoor_outdoor`, `price_level` as pandas `category`
    dtype (LightGBM's native categorical handling, spec.md section 8) -- never
    one-hot-encoded. `price_level` here is the RAW column (missingness preserved as a
    genuine NaN category, not the `_imputed` numeric form used in the numeric block)
    since categorical dtype natively supports missing values.

    `price_level`'s category *labels* are stringified (`"1"`..`"4"`, not the numeric
    `Int64` values themselves): a `category` dtype backed by nullable `Int64`
    categories silently round-trips through `to_parquet`/`read_parquet` as plain
    `float64` (a measured `pyarrow` quirk, not a pandas-in-memory behavior -- verified
    directly while building this module), which would silently undo the
    LightGBM-native-categorical intent the moment the feature table is written to
    disk. String-labeled categories round-trip as `category` correctly.
    """
    out: dict[str, pd.Categorical] = {}
    for col in CATEGORICAL_COLUMNS:
        if col == "price_level":
            series = (
                pois_df[col]
                .apply(lambda x: str(int(x)) if pd.notna(x) else None)
                .astype("category")
            )
        else:
            series = pois_df[col].astype("category")
        out[f"{CAT_PREFIX}{col}"] = series
    return pd.DataFrame(out, index=pois_df.index)


# -----------------------------------------------------------------------------------
# Geo block
# -----------------------------------------------------------------------------------


def compute_poi_density_500m(pois_df: pd.DataFrame, radius_km: float) -> pd.Series:
    """Count of *other* same-destination POIs within `radius_km` haversine distance
    of each POI."""
    result = pd.Series(0, index=pois_df.index, dtype=int)
    for _, group in pois_df.groupby("destination", sort=False):
        lat = group["lat"].to_numpy(dtype=float)
        lon = group["lon"].to_numpy(dtype=float)
        dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        within = (dist < radius_km).sum(axis=1) - 1  # exclude self
        result.loc[group.index] = within
    return result


def build_geo_block(pois_df: pd.DataFrame, density_radius_km: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{GEO_PREFIX}lat": pois_df["lat"].astype(float),
            f"{GEO_PREFIX}lon": pois_df["lon"].astype(float),
            f"{GEO_PREFIX}h3_cell": pois_df["h3_cell"],
            f"{GEO_PREFIX}dist_to_tourist_centroid_km": pois_df[
                "dist_to_tourist_centroid_km"
            ].astype(float),
            f"{GEO_PREFIX}dist_to_transit_km": pois_df["dist_to_transit_km"].astype(float),
            f"{GEO_PREFIX}density_{int(density_radius_km * 1000)}m": compute_poi_density_500m(
                pois_df, density_radius_km
            ),
        },
        index=pois_df.index,
    )


# -----------------------------------------------------------------------------------
# Behavioral block (train-window only)
# -----------------------------------------------------------------------------------


def build_behavioral_block(
    pois_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    traveler_segments: pd.Series,
    ctr_smoothing_alpha: float,
    n_segments: int,
) -> pd.DataFrame:
    """Behavioral POI features, computed from `interactions_train.parquet` ONLY
    (never holdout). See module docstring for the POI-id reconciliation step and the
    zero-impression-POI contract.
    """
    canonical_map = build_poi_id_canonical_map(pois_df)
    remapped = remap_interaction_poi_ids(interactions_train, canonical_map)
    remapped = remapped.assign(segment=remapped["traveler_id"].map(traveler_segments))

    poi_ids = pois_df["poi_id"]
    grouped = remapped.groupby("poi_id")

    impressions = grouped.size().reindex(poi_ids, fill_value=0).astype(float)
    positives = (
        remapped.loc[remapped["label"] >= 1].groupby("poi_id").size().reindex(poi_ids, fill_value=0)
    ).astype(float)
    global_ctr = float(positives.sum() / impressions.sum()) if impressions.sum() > 0 else 0.0

    def _type_count(itype: str) -> pd.Series:
        return (
            remapped.loc[remapped["interaction_type"] == itype]
            .groupby("poi_id")
            .size()
            .reindex(poi_ids, fill_value=0)
        ).astype(float)

    save_n = _type_count("save")
    visit_n = _type_count("visit")
    dismiss_n = _type_count("dismiss")
    unique_travelers = (grouped["traveler_id"].nunique().reindex(poi_ids, fill_value=0)).astype(
        float
    )

    safe_impressions = impressions.replace(0.0, np.nan)
    save_rate = (save_n / safe_impressions).fillna(0.0)
    visit_rate = (visit_n / safe_impressions).fillna(0.0)
    dismiss_rate = (dismiss_n / safe_impressions).fillna(0.0)
    ctr_smoothed = (positives + ctr_smoothing_alpha * global_ctr) / (
        impressions + ctr_smoothing_alpha
    )

    engaged = remapped.loc[remapped["label"] >= 1]
    seg_counts = (
        engaged.groupby(["poi_id", "segment"]).size().unstack(fill_value=0)
        if len(engaged) > 0
        else pd.DataFrame(index=[], columns=range(n_segments))
    )
    seg_counts = seg_counts.reindex(index=poi_ids, columns=range(n_segments), fill_value=0).astype(
        float
    )
    seg_totals = seg_counts.sum(axis=1)
    seg_share = seg_counts.div(seg_totals.replace(0.0, np.nan), axis=0).fillna(0.0)

    out = pd.DataFrame(
        {
            f"{BEHAV_PREFIX}impressions": impressions.to_numpy(),
            f"{BEHAV_PREFIX}ctr_smoothed": ctr_smoothed.to_numpy(),
            f"{BEHAV_PREFIX}save_rate": save_rate.to_numpy(),
            f"{BEHAV_PREFIX}visit_rate": visit_rate.to_numpy(),
            f"{BEHAV_PREFIX}dismiss_rate": dismiss_rate.to_numpy(),
            f"{BEHAV_PREFIX}unique_travelers": unique_travelers.to_numpy(),
        },
        index=pois_df.index,
    )
    for i in range(n_segments):
        out[f"{BEHAV_PREFIX}archetype_affinity_{i:02d}"] = seg_share[i].to_numpy()
    return out


# -----------------------------------------------------------------------------------
# Top-level assembly
# -----------------------------------------------------------------------------------


def assemble_poi_features(
    pois_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    travelers_df: pd.DataFrame,
    text_embeddings: FloatArray,
    cfg: PoiFeaturesConfig,
    seed: int,
) -> pd.DataFrame:
    """Assemble the full POI feature table (spec.md section 5), one row per POI in
    `pois_df` (post-dedup, `pois_prepared.parquet`)."""
    segments = assign_traveler_segments(
        travelers_df, n_clusters=cfg.traveler_segment_clusters, seed=seed
    )

    text_block = build_text_block(pois_df.index, text_embeddings)
    numeric_block = build_numeric_block(pois_df)
    categorical_block = build_categorical_block(pois_df)
    geo_block = build_geo_block(pois_df, cfg.density_radius_km)
    behavioral_block = build_behavioral_block(
        pois_df,
        interactions_train,
        segments,
        cfg.ctr_smoothing_alpha,
        cfg.traveler_segment_clusters,
    )

    keys = pois_df[["poi_id", "destination"]].reset_index(drop=True)
    return pd.concat(
        [
            keys,
            text_block.reset_index(drop=True),
            numeric_block.reset_index(drop=True),
            categorical_block.reset_index(drop=True),
            geo_block.reset_index(drop=True),
            behavioral_block.reset_index(drop=True),
        ],
        axis=1,
    )
