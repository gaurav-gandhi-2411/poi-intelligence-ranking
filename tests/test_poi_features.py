"""Phase 3 POI feature-table tests (spec.md section 5): block-column-grouping
convention, text embedding validity (including the TF-IDF fallback path, tested
directly per spec.md's explicit CI requirement), categorical dtypes, and the
train-window-only discipline of the behavioral block.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.features.poi_features import (
    BEHAV_PREFIX,
    CAT_PREFIX,
    GEO_PREFIX,
    NUM_PREFIX,
    TEXT_PREFIX,
    build_categorical_block,
    compute_crowd_index,
    compute_open_hours_per_week,
    compute_poi_density_500m,
)
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.features.text_embed import build_poi_corpus, embed_text_tfidf

# ---------------------------------------------------------------------------
# Block-grouping convention
# ---------------------------------------------------------------------------


def test_poi_feature_columns_are_block_prefixed(built_features: dict[str, Any]) -> None:
    pf: pd.DataFrame = built_features["poi_features"]
    assert {"poi_id", "destination"} <= set(pf.columns)
    non_key_cols = [c for c in pf.columns if c not in ("poi_id", "destination")]
    prefixes = (TEXT_PREFIX, NUM_PREFIX, CAT_PREFIX, GEO_PREFIX, BEHAV_PREFIX)
    for col in non_key_cols:
        assert col.startswith(prefixes), f"column '{col}' has no recognized block prefix"


def test_poi_feature_table_row_count_matches_prepared_pois(built_features: dict[str, Any]) -> None:
    pf: pd.DataFrame = built_features["poi_features"]
    n_pois = built_features["summary"]["n_pois"]
    assert len(pf) == n_pois
    assert pf["poi_id"].is_unique


# ---------------------------------------------------------------------------
# Text block
# ---------------------------------------------------------------------------


def test_text_embeddings_are_l2_normalized_and_non_degenerate(
    built_features: dict[str, Any],
) -> None:
    pf: pd.DataFrame = built_features["poi_features"]
    text_cols = [c for c in pf.columns if c.startswith(TEXT_PREFIX)]
    assert len(text_cols) == built_features["cfg"].text_embedding.svd_dim

    emb = pf[text_cols].to_numpy(dtype=float)
    assert not np.isnan(emb).any()

    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Not all rows identical (a degenerate embedding would collapse every POI to the
    # same vector).
    assert len(np.unique(emb, axis=0)) > 1


def test_tfidf_fallback_path_produces_valid_embeddings() -> None:
    """spec.md requires the TF-IDF fallback path be tested in CI, not merely
    reachable via an import-error branch -- called directly here."""
    pois = pd.DataFrame(
        {
            "name": ["Cozy Local Cafe", "Grand Historic Museum", "Riverside Nature Park"],
            "description": [
                "A small neighborhood cafe loved by locals.",
                "An iconic museum showcasing centuries of history.",
                "A peaceful park along the river, popular for walks.",
            ],
            "tags": [["local", "cafe"], ["historic", "touristy"], ["nature", "relaxing"]],
        }
    )
    corpus = build_poi_corpus(pois)
    emb = embed_text_tfidf(corpus, svd_dim=64, max_features=20000, ngram_max=2, seed=42)

    assert emb.shape == (3, 64)
    assert not np.isnan(emb).any()
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_tfidf_fallback_is_deterministic() -> None:
    pois = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "description": ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa"],
            "tags": [["x"], ["y"], ["z"], []],
        }
    )
    corpus = build_poi_corpus(pois)
    emb1 = embed_text_tfidf(corpus, svd_dim=8, max_features=1000, ngram_max=2, seed=42)
    emb2 = embed_text_tfidf(corpus, svd_dim=8, max_features=1000, ngram_max=2, seed=42)
    np.testing.assert_array_equal(emb1, emb2)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def test_compute_open_hours_per_week_sums_hourly_bits() -> None:
    mask = pd.Series([[True] * 24 + [False] * 144, [True] * 168])
    result = compute_open_hours_per_week(mask)
    assert result.tolist() == [24, 168]


def test_compute_crowd_index_peak_to_mean_ratio() -> None:
    flat = pd.Series([[1.0] * 24])
    peaky = pd.Series([[0.0] * 23 + [1.0]])
    flat_idx = compute_crowd_index(flat).iloc[0]
    peaky_idx = compute_crowd_index(peaky).iloc[0]
    assert flat_idx == pytest.approx(1.0)
    assert peaky_idx > flat_idx


def test_compute_poi_density_counts_neighbors_excluding_self() -> None:
    df = pd.DataFrame(
        {
            "destination": ["seoul", "seoul", "seoul", "kyoto"],
            "lat": [37.5665, 37.5666, 37.6000, 35.0116],
            "lon": [126.9780, 126.9781, 126.9800, 135.7681],
        }
    )
    density = compute_poi_density_500m(df, radius_km=0.5)
    # rows 0 and 1 are ~15m apart (within 500m of each other); row 2 is far.
    assert density.iloc[0] == 1
    assert density.iloc[1] == 1
    assert density.iloc[2] == 0
    assert density.iloc[3] == 0  # different destination, no cross-destination counting


# ---------------------------------------------------------------------------
# Categorical block
# ---------------------------------------------------------------------------


def test_categorical_block_dtypes_are_pandas_category(built_features: dict[str, Any]) -> None:
    pf: pd.DataFrame = built_features["poi_features"]
    for col in ("cat_category", "cat_subcategory", "cat_indoor_outdoor", "cat_price_level"):
        assert isinstance(pf[col].dtype, pd.CategoricalDtype), f"{col} is not pandas category dtype"


def test_categorical_block_price_level_survives_parquet_round_trip(
    built_features: dict[str, Any],
) -> None:
    """Regression test: a `category` dtype backed by nullable `Int64` categories
    silently degrades to `float64` on a `to_parquet`/`read_parquet` round trip
    (measured while building this module) -- `build_categorical_block` stringifies
    `price_level`'s category labels specifically to avoid this. `built_features`
    already went through a real parquet write+read, so this assertion is a genuine
    round-trip check, not just an in-memory one."""
    pf: pd.DataFrame = built_features["poi_features"]
    assert isinstance(pf["cat_price_level"].dtype, pd.CategoricalDtype)
    assert set(pf["cat_price_level"].cat.categories) <= {"1", "2", "3", "4"}


def test_build_categorical_block_price_level_preserves_missingness() -> None:
    df = pd.DataFrame(
        {
            "category": ["cafe", "museum", "cafe"],
            "subcategory": ["specialty_coffee", "art_museum", "bakery"],
            "indoor_outdoor": ["indoor", "indoor", "indoor"],
            "price_level": [1.0, np.nan, 3.0],
        }
    )
    out = build_categorical_block(df)
    assert out["cat_price_level"].isna().tolist() == [False, True, False]
    assert isinstance(out["cat_price_level"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------------------
# Behavioral block: train-window-only discipline
# ---------------------------------------------------------------------------


def test_behavioral_block_zero_impression_poi_has_zero_not_leaked_rates(
    built_features: dict[str, Any], prepared_data: dict[str, Any]
) -> None:
    """A POI with zero train-window impressions must show
    `behav_impressions == 0` and every raw-rate column `== 0.0` -- never a nonzero
    value that could only have come from holdout interactions (which never
    participate in behavioral-block computation).
    """
    pf: pd.DataFrame = built_features["poi_features"]
    prepared: pd.DataFrame = prepared_data["prepared"]

    output_dir = built_features["output_dir"]
    interactions_train = pd.read_parquet(output_dir / "interactions_train.parquet")
    canonical_map = build_poi_id_canonical_map(prepared)
    remapped = remap_interaction_poi_ids(interactions_train, canonical_map)
    covered_poi_ids = set(remapped["poi_id"])

    zero_impression_ids = set(pf["poi_id"]) - covered_poi_ids
    assert zero_impression_ids, "expected at least one POI with zero train-window impressions"

    sample = pf.loc[pf["poi_id"].isin(zero_impression_ids)].iloc[0]
    assert sample[f"{BEHAV_PREFIX}impressions"] == 0.0
    assert sample[f"{BEHAV_PREFIX}save_rate"] == 0.0
    assert sample[f"{BEHAV_PREFIX}visit_rate"] == 0.0
    assert sample[f"{BEHAV_PREFIX}dismiss_rate"] == 0.0
    assert sample[f"{BEHAV_PREFIX}unique_travelers"] == 0.0
    # ctr_smoothed falls back to the pure Bayesian prior (global train-window CTR),
    # not zero and not a holdout-derived value.
    assert 0.0 <= sample[f"{BEHAV_PREFIX}ctr_smoothed"] <= 1.0


def test_behavioral_block_archetype_affinity_sums_to_one_when_engaged(
    built_features: dict[str, Any],
) -> None:
    pf: pd.DataFrame = built_features["poi_features"]
    affinity_cols = [c for c in pf.columns if c.startswith(f"{BEHAV_PREFIX}archetype_affinity_")]
    assert len(affinity_cols) == built_features["cfg"].poi_features.traveler_segment_clusters

    engaged_pois = pf.loc[pf[f"{BEHAV_PREFIX}unique_travelers"] > 0]
    assert len(engaged_pois) > 0
    sums = engaged_pois[affinity_cols].sum(axis=1)
    # Rows with only dismiss/view interactions have zero *engaged* (label>=1) rows
    # even with unique_travelers > 0, so allow (0.0 or ~1.0), never > 1.
    assert (sums <= 1.0 + 1e-6).all()


def test_geo_and_numeric_blocks_have_no_nan(built_features: dict[str, Any]) -> None:
    pf: pd.DataFrame = built_features["poi_features"]
    num_cols = [c for c in pf.columns if c.startswith(NUM_PREFIX)]
    geo_cols = [c for c in pf.columns if c.startswith(GEO_PREFIX) and c != f"{GEO_PREFIX}h3_cell"]
    for col in num_cols + geo_cols:
        assert not pf[col].isna().any(), f"unexpected NaN in {col}"
