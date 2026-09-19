"""Orchestrates the Phase 3 feature-build pipeline end to end (spec.md sections 5-6):
loads `pois_prepared.parquet`, `travelers.parquet`, `trips.parquet`,
`interactions_train.parquet`; builds (or loads from cache) the POI text embedding;
assembles the POI and traveler feature tables; writes
`data/synthetic/poi_features.parquet` and `data/synthetic/traveler_features.parquet`.

File-naming convention mirrors Phase 2's `pois_prepared.parquet` (same
`data/synthetic/` directory, `<entity>_<stage>` naming) rather than introducing a new
top-level `data/features/` directory -- consistent with the existing flat layout.
Deterministic: every stochastic step downstream of datagen's own seeded RNG (TF-IDF/
SVD, the traveler-segment K-Means) is seeded from `configs/features.yaml`'s top-level
`seed`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.poi_features import assemble_poi_features
from poi_rank.features.text_embed import build_poi_text_embeddings
from poi_rank.features.traveler_features import assemble_traveler_features

POI_FEATURES_FILENAME = "poi_features.parquet"
TRAVELER_FEATURES_FILENAME = "traveler_features.parquet"


def run_features(cfg: FeatureBuildConfig, data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the full Phase 3 feature-build pipeline and write both feature tables.

    Returns a summary dict for the CLI report and tests.
    """
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    interactions_train = pd.read_parquet(data_dir / "interactions_train.parquet")
    # Block A RC3.2 (docs/DATA_CARD.md "DGP remediation, Block A"): pre-trip
    # history, optional for backward-compat with datasets generated before this
    # file existed.
    pretrip_path = data_dir / "interactions_pretrip.parquet"
    interactions_pretrip = pd.read_parquet(pretrip_path) if pretrip_path.exists() else None

    cache_path = artifacts_dir / "poi_emb.npy"
    text_embeddings = build_poi_text_embeddings(pois_df, cfg.text_embedding, cfg.seed, cache_path)

    poi_features = assemble_poi_features(
        pois_df, interactions_train, travelers_df, text_embeddings, cfg.poi_features, cfg.seed
    )
    traveler_features = assemble_traveler_features(
        travelers_df,
        trips_df,
        interactions_train,
        pois_df,
        text_embeddings,
        cfg,
        interactions_pretrip,
    )

    poi_features_path = data_dir / POI_FEATURES_FILENAME
    traveler_features_path = data_dir / TRAVELER_FEATURES_FILENAME
    poi_features.to_parquet(poi_features_path, index=False)
    traveler_features.to_parquet(traveler_features_path, index=False)

    return {
        "poi_features_path": poi_features_path,
        "traveler_features_path": traveler_features_path,
        "text_embedding_cache": cache_path,
        "text_embedding_method": cfg.text_embedding.method,
        "n_pois": len(poi_features),
        "n_poi_feature_columns": poi_features.shape[1],
        "n_traveler_trip_rows": len(traveler_features),
        "n_traveler_feature_columns": traveler_features.shape[1],
    }
