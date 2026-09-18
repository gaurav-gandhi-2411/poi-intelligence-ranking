"""Orchestrates the full Phase 2 data-prep pipeline end to end (spec.md section 4):
dedup -> category canonicalization -> rating shrinkage -> popularity percentile ->
localness index -> opening-hours mask -> numeric imputation -> geo (H3 cell +
synthetic transit graph). Writes `data/synthetic/pois_prepared.parquet`.

Deterministic: the only stochastic step (synthetic transit-node placement) is seeded
from `configs/features.yaml`'s `seed` via its own `np.random.Generator` -- data prep
never touches or re-derives datagen's seeded RNG stream, it only reads datagen's
already-written parquet output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poi_rank.data.categories import canonicalize_categories, other_category_report
from poi_rank.data.config import FeaturesConfig
from poi_rank.data.dedup import DedupReport, dedup_pois
from poi_rank.data.geo_prep import (
    distance_to_nearest_transit_km,
    h3_cell,
    synthesize_transit_nodes,
)
from poi_rank.data.hours import compute_hours
from poi_rank.data.impute import impute_numeric_fields
from poi_rank.data.localness import compute_localness
from poi_rank.data.popularity import compute_popularity_percentile
from poi_rank.data.shrinkage import compute_shrunk_rating

PREPARED_POIS_FILENAME = "pois_prepared.parquet"


@dataclass(frozen=True)
class PrepareReport:
    """Summary of one `prepare_pois` run, used for the CLI report and tests."""

    dedup: DedupReport
    other_category: dict[str, float | int]
    n_input_pois: int
    n_output_pois: int


def prepare_pois(pois_raw: pd.DataFrame, cfg: FeaturesConfig) -> tuple[pd.DataFrame, PrepareReport]:
    """Run the full POI prep pipeline on the raw (dirty) exported catalog."""
    deduped, dedup_report = dedup_pois(
        pois_raw,
        h3_resolution=cfg.dedup.h3_resolution,
        name_threshold=cfg.dedup.name_similarity_threshold,
        max_distance_m=cfg.dedup.max_distance_m,
    )
    deduped = deduped.reset_index(drop=True)

    cat_df = canonicalize_categories(deduped["category"])
    other_report = other_category_report(cat_df["category"])
    deduped["category_raw"] = cat_df["category_raw"]
    deduped["category"] = cat_df["category"]

    deduped["rating_shrunk"] = compute_shrunk_rating(deduped)
    deduped["pop_pct"] = compute_popularity_percentile(deduped)

    local_df = compute_localness(
        deduped,
        weight_popularity=cfg.localness.weight_popularity,
        weight_foreign=cfg.localness.weight_foreign,
        weight_geo=cfg.localness.weight_geo,
        weight_tag=cfg.localness.weight_tag,
        top_decile=cfg.localness.tourist_centroid_top_decile,
    )
    deduped = pd.concat([deduped, local_df], axis=1)

    hours_df = compute_hours(deduped)
    deduped = pd.concat([deduped, hours_df], axis=1)

    impute_df = impute_numeric_fields(deduped)
    deduped = pd.concat([deduped, impute_df], axis=1)

    deduped["h3_cell"] = h3_cell(
        deduped["lat"].to_numpy(dtype=float),
        deduped["lon"].to_numpy(dtype=float),
        cfg.dedup.h3_resolution,
    )

    transit_rng = np.random.default_rng(cfg.seed)
    transit_nodes = synthesize_transit_nodes(
        transit_rng,
        deduped,
        cfg.geo.transit_nodes_per_destination,
        cfg.geo.transit_node_spread_deg,
    )
    deduped["dist_to_transit_km"] = distance_to_nearest_transit_km(deduped, transit_nodes)

    report = PrepareReport(
        dedup=dedup_report,
        other_category=other_report,
        n_input_pois=len(pois_raw),
        n_output_pois=len(deduped),
    )
    return deduped, report


def run_prepare(features_cfg: FeaturesConfig, data_dir: Path) -> dict[str, Any]:
    """Load `data/synthetic/pois.parquet`, run the full prep pipeline, and write
    `data/synthetic/pois_prepared.parquet`. Returns a summary dict for the CLI report
    and tests."""
    pois_raw = pd.read_parquet(data_dir / "pois.parquet")
    prepared, report = prepare_pois(pois_raw, features_cfg)

    output_path = data_dir / PREPARED_POIS_FILENAME
    prepared.to_parquet(output_path, index=False)

    return {
        "output_path": output_path,
        "n_input_pois": report.n_input_pois,
        "n_output_pois": report.n_output_pois,
        "n_merged_away": report.dedup.n_merged_away,
        "dedup_merge_rate": report.dedup.merge_rate,
        "n_other_category": report.other_category["n_other"],
        "other_category_rate": report.other_category["other_rate"],
    }
