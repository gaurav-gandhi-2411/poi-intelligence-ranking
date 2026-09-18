"""Phase 4a candidate-generation tests (spec.md section 7): per-channel quota +
destination-scoping unit tests on hand-built `DestinationIndex` fixtures, the
long-tail hard-floor popularity check, the item-item CF co-interaction definition and
its leakage-safe seed, the archetype-prior cold-start path, union dedup, and
end-to-end determinism against the full-scale committed-shape pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.candidates.channels import (
    CHANNEL_NAMES,
    DestinationIndex,
    build_item_item_cf,
    channel_archetype,
    channel_cf,
    channel_geo,
    channel_interest,
    channel_longtail,
    channel_semantic,
    compute_semantic_similarity,
    trip_seed,
)
from poi_rank.candidates.config import (
    ArchetypeChannelConfig,
    CandidatesConfig,
    CollaborativeChannelConfig,
    GeoChannelConfig,
    InterestChannelConfig,
    LongtailChannelConfig,
    SemanticChannelConfig,
)
from poi_rank.candidates.union import cf_seed_poi_ids, generate_candidates
from poi_rank.features.traveler_features import (
    assign_traveler_segments,
    group_interactions_by_traveler,
)

# ---------------------------------------------------------------------------
# Hand-built DestinationIndex helper
# ---------------------------------------------------------------------------


def _index(
    poi_ids: list[str],
    lat: list[float],
    lon: list[float],
    h3_cell: list[str],
    pop_pct: list[float],
    rating_shrunk: list[float],
    poi_terms: list[frozenset[str]],
    embeddings: list[list[float]],
    archetype_top_by_segment: dict[int, list[str]] | None = None,
) -> DestinationIndex:
    return DestinationIndex(
        poi_ids=np.array(poi_ids, dtype=object),
        lat=np.array(lat, dtype=float),
        lon=np.array(lon, dtype=float),
        h3_cell=np.array(h3_cell, dtype=object),
        pop_pct=np.array(pop_pct, dtype=float),
        rating_shrunk=np.array(rating_shrunk, dtype=float),
        poi_terms=poi_terms,
        embeddings=np.array(embeddings, dtype=np.float32),
        archetype_top_by_segment=archetype_top_by_segment or {},
    )


# ---------------------------------------------------------------------------
# Geo channel
# ---------------------------------------------------------------------------


def test_channel_geo_excludes_out_of_radius_and_caps_at_quota() -> None:
    import h3

    stay_lat, stay_lon = 37.5665, 126.9780
    res = 8
    near1 = (37.5670, 126.9785)  # a few hundred meters away
    near2 = (37.5700, 126.9800)  # ~1km away, still within 2km walk radius
    far = (38.5000, 127.9000)  # >100km away, outside any radius

    poi_ids = ["P_near1", "P_near2", "P_far"]
    coords = [near1, near2, far]
    h3_cells = [h3.geo_to_h3(la, lo, res) for la, lo in coords]
    idx = _index(
        poi_ids=poi_ids,
        lat=[c[0] for c in coords],
        lon=[c[1] for c in coords],
        h3_cell=h3_cells,
        pop_pct=[0.5, 0.5, 0.5],
        rating_shrunk=[3.0, 3.0, 3.0],
        poi_terms=[frozenset(), frozenset(), frozenset()],
        embeddings=[[0.0], [0.0], [0.0]],
    )
    cfg = GeoChannelConfig(
        quota=5,
        h3_resolution=res,
        radius_km_walk=2.0,
        radius_km_public_transport=8.0,
        radius_km_car=25.0,
        radius_km_mixed=25.0,
    )
    result = channel_geo(idx, stay_lat, stay_lon, "walk", cfg)
    assert set(result) == {"P_near1", "P_near2"}
    assert "P_far" not in result


def test_channel_geo_respects_quota_nearest_first() -> None:
    import h3

    stay_lat, stay_lon = 0.0, 0.0
    res = 8
    # 4 POIs at increasing distance from the stay point, all within car radius.
    offsets = [0.001, 0.002, 0.003, 0.004]
    coords = [(o, o) for o in offsets]
    poi_ids = [f"P{i}" for i in range(4)]
    h3_cells = [h3.geo_to_h3(la, lo, res) for la, lo in coords]
    idx = _index(
        poi_ids=poi_ids,
        lat=[c[0] for c in coords],
        lon=[c[1] for c in coords],
        h3_cell=h3_cells,
        pop_pct=[0.5] * 4,
        rating_shrunk=[3.0] * 4,
        poi_terms=[frozenset()] * 4,
        embeddings=[[0.0]] * 4,
    )
    cfg = GeoChannelConfig(
        quota=2,
        h3_resolution=res,
        radius_km_walk=2.0,
        radius_km_public_transport=8.0,
        radius_km_car=25.0,
        radius_km_mixed=25.0,
    )
    result = channel_geo(idx, stay_lat, stay_lon, "car", cfg)
    assert result == ["P0", "P1"]  # nearest 2, capped at quota=2


# ---------------------------------------------------------------------------
# Interest channel
# ---------------------------------------------------------------------------


def test_channel_interest_matches_category_or_tags_ranked_by_rating() -> None:
    idx = _index(
        poi_ids=["P0", "P1", "P2"],
        lat=[0.0, 0.0, 0.0],
        lon=[0.0, 0.0, 0.0],
        h3_cell=["a", "b", "c"],
        pop_pct=[0.5, 0.5, 0.5],
        rating_shrunk=[3.0, 5.0, 4.5],
        poi_terms=[
            frozenset({"cafe", "local"}),
            frozenset({"museum", "history"}),
            frozenset({"cafe"}),
        ],
        embeddings=[[0.0]] * 3,
    )
    result = channel_interest(idx, {"cafe"}, quota=5)
    assert result == ["P2", "P0"]  # both match "cafe", ranked by rating desc


def test_channel_interest_quota_and_empty_interests() -> None:
    idx = _index(
        poi_ids=["P0", "P1"],
        lat=[0.0, 0.0],
        lon=[0.0, 0.0],
        h3_cell=["a", "b"],
        pop_pct=[0.5, 0.5],
        rating_shrunk=[3.0, 5.0],
        poi_terms=[frozenset({"cafe"}), frozenset({"cafe"})],
        embeddings=[[0.0]] * 2,
    )
    assert channel_interest(idx, {"cafe"}, quota=1) == ["P1"]
    assert channel_interest(idx, set(), quota=5) == []


# ---------------------------------------------------------------------------
# Semantic channel
# ---------------------------------------------------------------------------


def test_compute_semantic_similarity_and_channel_semantic_top_k() -> None:
    idx = _index(
        poi_ids=["P0", "P1", "P2"],
        lat=[0.0, 0.0, 0.0],
        lon=[0.0, 0.0, 0.0],
        h3_cell=["a", "b", "c"],
        pop_pct=[0.5, 0.5, 0.5],
        rating_shrunk=[3.0, 3.0, 3.0],
        poi_terms=[frozenset()] * 3,
        embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )
    taste = np.array([1.0, 0.0])
    sims = compute_semantic_similarity(taste, idx.embeddings)
    np.testing.assert_allclose(sims, [1.0, 0.0, -1.0], atol=1e-6)

    result = channel_semantic(idx, sims, quota=2)
    assert result == ["P0", "P1"]


def test_compute_semantic_similarity_zero_taste_vector_is_all_zero() -> None:
    idx = _index(
        poi_ids=["P0", "P1"],
        lat=[0.0, 0.0],
        lon=[0.0, 0.0],
        h3_cell=["a", "b"],
        pop_pct=[0.5, 0.5],
        rating_shrunk=[3.0, 3.0],
        poi_terms=[frozenset()] * 2,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    sims = compute_semantic_similarity(np.zeros(2), idx.embeddings)
    np.testing.assert_array_equal(sims, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Long-tail channel: HARD FLOOR only ever draws from below the popularity cutoff
# ---------------------------------------------------------------------------


def test_channel_longtail_only_draws_from_below_pop_pct_cutoff() -> None:
    idx = _index(
        poi_ids=["P0", "P1", "P2", "P3"],
        lat=[0.0] * 4,
        lon=[0.0] * 4,
        h3_cell=["a", "b", "c", "d"],
        pop_pct=[0.1, 0.6, 0.2, 0.05],  # P1 fails the cutoff
        rating_shrunk=[3.0] * 4,
        poi_terms=[frozenset()] * 4,
        embeddings=[[0.0]] * 4,
    )
    sims = np.array([0.5, 0.9, 0.5, 0.9])  # all pass the sim threshold
    cfg = LongtailChannelConfig(
        quota=10, pop_pct_cutoff=0.4, semantic_sim_threshold=0.0, epsilon=0.0
    )
    rng = np.random.default_rng(0)
    result = channel_longtail(idx, sims, is_cold_start=False, cfg=cfg, rng=rng)
    assert "P1" not in result
    assert set(result) == {"P0", "P2", "P3"}
    for pid in result:
        pop = idx.pop_pct[list(idx.poi_ids).index(pid)]
        assert pop < cfg.pop_pct_cutoff


def test_channel_longtail_semantic_filter_excludes_low_similarity() -> None:
    idx = _index(
        poi_ids=["P0", "P1"],
        lat=[0.0, 0.0],
        lon=[0.0, 0.0],
        h3_cell=["a", "b"],
        pop_pct=[0.1, 0.1],  # both pass the popularity cutoff
        rating_shrunk=[3.0, 3.0],
        poi_terms=[frozenset()] * 2,
        embeddings=[[0.0]] * 2,
    )
    sims = np.array([0.5, -0.1])  # P1 fails the semantic-relevance floor
    cfg = LongtailChannelConfig(
        quota=10, pop_pct_cutoff=0.4, semantic_sim_threshold=0.0, epsilon=0.0
    )
    rng = np.random.default_rng(0)
    result = channel_longtail(idx, sims, is_cold_start=False, cfg=cfg, rng=rng)
    assert result == ["P0"]


def test_channel_longtail_cold_start_skips_semantic_filter() -> None:
    """A zero-taste-vector traveler's `sims` are identically 0.0 for every POI (see
    `compute_semantic_similarity`); a literal `sims > 0.0` filter would starve the
    HARD FLOOR for every cold-start trip -- the cold-start fallback must still
    return POIs purely on the popularity criterion."""
    idx = _index(
        poi_ids=["P0", "P1"],
        lat=[0.0, 0.0],
        lon=[0.0, 0.0],
        h3_cell=["a", "b"],
        pop_pct=[0.1, 0.1],
        rating_shrunk=[3.0, 3.0],
        poi_terms=[frozenset()] * 2,
        embeddings=[[0.0]] * 2,
    )
    sims = np.zeros(2)
    cfg = LongtailChannelConfig(
        quota=10, pop_pct_cutoff=0.4, semantic_sim_threshold=0.0, epsilon=0.0
    )
    rng = np.random.default_rng(0)
    result = channel_longtail(idx, sims, is_cold_start=True, cfg=cfg, rng=rng)
    assert set(result) == {"P0", "P1"}


def test_channel_longtail_epsilon_greedy_is_deterministic_and_no_duplicates() -> None:
    idx = _index(
        poi_ids=[f"P{i}" for i in range(6)],
        lat=[0.0] * 6,
        lon=[0.0] * 6,
        h3_cell=list("abcdef"),
        pop_pct=[0.1] * 6,
        rating_shrunk=[3.0] * 6,
        poi_terms=[frozenset()] * 6,
        embeddings=[[0.0]] * 6,
    )
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    cfg = LongtailChannelConfig(
        quota=4, pop_pct_cutoff=0.4, semantic_sim_threshold=0.0, epsilon=0.5
    )

    rng1 = np.random.default_rng(trip_seed(42, "T0001"))
    result1 = channel_longtail(idx, sims, is_cold_start=False, cfg=cfg, rng=rng1)
    rng2 = np.random.default_rng(trip_seed(42, "T0001"))
    result2 = channel_longtail(idx, sims, is_cold_start=False, cfg=cfg, rng=rng2)

    assert result1 == result2
    assert len(result1) == len(set(result1))  # no in-channel duplicates
    assert len(result1) == 4


def test_trip_seed_is_deterministic_and_varies_by_trip_id() -> None:
    assert trip_seed(42, "T0001") == trip_seed(42, "T0001")
    assert trip_seed(42, "T0001") != trip_seed(42, "T0002")
    assert trip_seed(42, "T0001") != trip_seed(7, "T0001")


# ---------------------------------------------------------------------------
# Archetype channel
# ---------------------------------------------------------------------------


def test_channel_archetype_quota_and_unknown_segment() -> None:
    idx = _index(
        poi_ids=["P0"],
        lat=[0.0],
        lon=[0.0],
        h3_cell=["a"],
        pop_pct=[0.5],
        rating_shrunk=[3.0],
        poi_terms=[frozenset()],
        embeddings=[[0.0]],
        archetype_top_by_segment={0: ["P1", "P2", "P3"], 1: ["P4", "P5"]},
    )
    assert channel_archetype(idx, segment=0, quota=2) == ["P1", "P2"]
    # unknown segment -> empty, not a KeyError.
    assert channel_archetype(idx, segment=99, quota=2) == []


def test_channel_archetype_cold_start_traveler_gets_nonempty_candidates() -> None:
    """spec.md section 12: the archetype-prior channel is the designated cold-start
    path. Segment assignment (`assign_traveler_segments`) reads only stated
    interests/budget/party_type/touristiness_pref -- never interaction history --
    so a traveler with ZERO interactions anywhere must still get a non-empty
    archetype channel."""
    travelers = pd.DataFrame(
        {
            "traveler_id": ["U_COLD", "U_OTHER"],
            "interests": [["local"], ["luxury"]],
            "budget": ["low", "high"],
            "party_type": ["solo", "couple"],
            "touristiness_pref": [-0.5, 0.8],
        }
    )
    segments = assign_traveler_segments(travelers, n_clusters=2, seed=42)
    cold_segment = int(segments["U_COLD"])

    idx = _index(
        poi_ids=["P0", "P1"],
        lat=[0.0, 0.0],
        lon=[0.0, 0.0],
        h3_cell=["a", "b"],
        pop_pct=[0.5, 0.5],
        rating_shrunk=[3.0, 3.0],
        poi_terms=[frozenset()] * 2,
        embeddings=[[0.0]] * 2,
        archetype_top_by_segment={cold_segment: ["P0", "P1"]},
    )
    result = channel_archetype(idx, cold_segment, quota=5)
    assert result == ["P0", "P1"]  # non-empty, even though U_COLD has zero interactions


# ---------------------------------------------------------------------------
# Collaborative (item-item kNN) channel: co-interaction definition + leakage-safe seed
# ---------------------------------------------------------------------------


def test_build_item_item_cf_co_interaction_definition() -> None:
    """Two POIs co-interact iff the SAME traveler positively engaged (label >= 1)
    with both, anywhere in the train window."""
    pois = pd.DataFrame({"poi_id": ["P1", "P2", "P3"], "merged_poi_ids": [["P1"], ["P2"], ["P3"]]})
    interactions = pd.DataFrame(
        {
            "traveler_id": ["U1", "U1", "U2", "U3"],
            "poi_id": ["P1", "P2", "P3", "P3"],
            "label": [3, 2, 1, 0],  # U3's P3 row is a non-engaged view -> excluded
        }
    )
    cf = build_item_item_cf(pois, interactions)
    p1, p2, p3 = cf.poi_index["P1"], cf.poi_index["P2"], cf.poi_index["P3"]
    assert cf.similarity[p1, p2] > 0.0  # co-engaged by U1
    assert cf.similarity[p1, p3] == pytest.approx(0.0)  # never co-engaged by the same traveler
    assert cf.similarity[p1, p1] == 0.0  # zero diagonal


def test_build_item_item_cf_empty_engagement_returns_zero_matrix() -> None:
    pois = pd.DataFrame({"poi_id": ["P1", "P2"], "merged_poi_ids": [["P1"], ["P2"]]})
    interactions = pd.DataFrame({"traveler_id": ["U1"], "poi_id": ["P1"], "label": [0]})
    cf = build_item_item_cf(pois, interactions)
    assert cf.similarity.sum() == 0.0


def test_channel_cf_excludes_seed_pois_and_uses_similarity() -> None:
    pois = pd.DataFrame({"poi_id": ["P1", "P2", "P3"], "merged_poi_ids": [["P1"], ["P2"], ["P3"]]})
    interactions = pd.DataFrame(
        {"traveler_id": ["U1", "U1"], "poi_id": ["P1", "P2"], "label": [3, 3]}
    )
    cf = build_item_item_cf(pois, interactions)

    idx = _index(
        poi_ids=["P1", "P2", "P3"],
        lat=[0.0] * 3,
        lon=[0.0] * 3,
        h3_cell=["a", "b", "c"],
        pop_pct=[0.5] * 3,
        rating_shrunk=[3.0] * 3,
        poi_terms=[frozenset()] * 3,
        embeddings=[[0.0]] * 3,
    )
    cfg = CollaborativeChannelConfig(quota=5, top_k_neighbors=5)
    result = channel_cf(idx, cf, seed_poi_ids=["P1"], cfg=cfg)
    assert result == ["P2"]  # similar to seed P1, excludes the seed itself, excludes unrelated P3


def test_channel_cf_empty_seed_history_returns_empty() -> None:
    """A traveler with no prior engaged history (their first trip) must get an
    EMPTY collaborative-filtering channel, not an error or an arbitrary fallback --
    the archetype-prior channel is the designated cold-start path, not this one."""
    pois = pd.DataFrame({"poi_id": ["P1"], "merged_poi_ids": [["P1"]]})
    interactions = pd.DataFrame({"traveler_id": [], "poi_id": [], "label": []})
    cf = build_item_item_cf(pois, interactions)
    idx = _index(
        poi_ids=["P1"],
        lat=[0.0],
        lon=[0.0],
        h3_cell=["a"],
        pop_pct=[0.5],
        rating_shrunk=[3.0],
        poi_terms=[frozenset()],
        embeddings=[[0.0]],
    )
    cfg = CollaborativeChannelConfig(quota=5, top_k_neighbors=5)
    assert channel_cf(idx, cf, seed_poi_ids=[], cfg=cfg) == []


def test_cf_seed_poi_ids_excludes_current_trip_and_future_interactions() -> None:
    """Direct leakage-safety check on the collaborative-filtering channel's seed
    construction (mirrors `test_traveler_features.py`'s equivalent proof for the
    taste vector): only the traveler's own EARLIER-trip, positively-engaged
    interactions may seed the channel."""
    interactions = pd.DataFrame(
        {
            "traveler_id": ["U1", "U1", "U1", "U1"],
            "trip_id": ["T1", "T1", "T2", "T2"],
            "poi_id": ["P1", "P2", "P3", "P4"],
            "label": [3, 0, 3, 3],  # P2 is a non-engaged view; P3/P4 belong to T2 itself
            "timestamp": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-06-01"),  # T2's own session -> must be excluded from T2's seed
                pd.Timestamp("2025-06-02"),
            ],
        }
    )
    by_traveler = group_interactions_by_traveler(interactions)

    as_of_t2 = pd.Timestamp("2025-05-01")
    seeds_for_t2 = cf_seed_poi_ids(by_traveler, "U1", "T2", as_of_t2, interactions)
    # only T1's engaged (label>=1) row, strictly before T2's as-of.
    assert seeds_for_t2 == ["P1"]

    as_of_t1 = pd.Timestamp("2024-12-01")
    seeds_for_t1 = cf_seed_poi_ids(by_traveler, "U1", "T1", as_of_t1, interactions)
    assert seeds_for_t1 == []  # first trip -> no prior history


# ---------------------------------------------------------------------------
# Union: dedup, destination-scoping, and channel-independent filtering
# (end-to-end via generate_candidates on a small hand-built dataset)
# ---------------------------------------------------------------------------


def _tiny_generate_candidates_inputs() -> dict[str, Any]:
    import h3

    res = 8
    stay_lat, stay_lon = 0.0, 0.0
    near = (0.001, 0.001)
    farther = (0.01, 0.01)
    far_away = (5.0, 5.0)

    pois = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "destination": ["testville"] * 4 + ["otherville"] * 2,
            "lat": [near[0], farther[0], near[0], far_away[0], 10.0, 10.0],
            "lon": [near[1], farther[1], near[1], far_away[1], 10.0, 10.0],
            "h3_cell": [
                h3.geo_to_h3(near[0], near[1], res),
                h3.geo_to_h3(farther[0], farther[1], res),
                h3.geo_to_h3(near[0], near[1], res),
                h3.geo_to_h3(far_away[0], far_away[1], res),
                h3.geo_to_h3(10.0, 10.0, res),
                h3.geo_to_h3(10.0, 10.0, res),
            ],
            "pop_pct": [0.1, 0.6, 0.2, 0.05, 0.3, 0.3],
            "rating_shrunk": [4.0, 3.0, 5.0, 2.0, 3.0, 3.0],
            "category": ["cafe", "museum", "cafe", "park", "cafe", "museum"],
            "tags": [["local"], ["history"], ["trendy"], ["nature"], ["local"], ["history"]],
            "merged_poi_ids": [[pid] for pid in ["P1", "P2", "P3", "P4", "P5", "P6"]],
        }
    )

    poi_features = pd.DataFrame(
        {
            "poi_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "text_emb_00": [1.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            "text_emb_01": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )

    travelers = pd.DataFrame(
        {
            "traveler_id": ["U1", "U2"],
            "interests": [["local"], ["nature"]],
            "budget": ["medium", "low"],
            "party_type": ["solo", "solo"],
            "touristiness_pref": [0.0, 0.0],
            "mobility": ["car", "car"],
        }
    )

    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "traveler_id": ["U1", "U2"],
            "destination": ["testville", "otherville"],
            "start_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-01")],
            "stay_lat": [stay_lat, 10.0],
            "stay_lon": [stay_lon, 10.0],
        }
    )

    interactions_train = pd.DataFrame(
        {
            "traveler_id": pd.Series([], dtype=str),
            "trip_id": pd.Series([], dtype=str),
            "poi_id": pd.Series([], dtype=str),
            "label": pd.Series([], dtype=int),
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
        }
    )

    # Discover U1's observable segment so the archetype-affinity fixture can steer
    # the archetype channel toward a specific, hand-verifiable POI (P2).
    segments = assign_traveler_segments(travelers, n_clusters=2, seed=42)
    u1_segment = int(segments["U1"])
    other_segment = 1 - u1_segment
    affinity_col = f"behav_archetype_affinity_{u1_segment:02d}"
    other_col = f"behav_archetype_affinity_{other_segment:02d}"
    poi_features[affinity_col] = [0.1, 0.6, 0.2, 0.1, 0.0, 0.0]
    poi_features[other_col] = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]

    cfg = CandidatesConfig(
        seed=42,
        traveler_segment_clusters=2,
        geo=GeoChannelConfig(
            quota=2,
            h3_resolution=res,
            radius_km_walk=50.0,
            radius_km_public_transport=50.0,
            radius_km_car=50.0,
            radius_km_mixed=50.0,
        ),
        interest=InterestChannelConfig(quota=2),
        semantic=SemanticChannelConfig(quota=2),
        collaborative=CollaborativeChannelConfig(quota=2, top_k_neighbors=5),
        longtail=LongtailChannelConfig(
            quota=2, pop_pct_cutoff=0.4, semantic_sim_threshold=0.0, epsilon=0.0
        ),
        archetype=ArchetypeChannelConfig(quota=2),
    )

    return {
        "pois": pois,
        "travelers": travelers,
        "trips": trips,
        "poi_features": poi_features,
        "interactions_train": interactions_train,
        "cfg": cfg,
    }


def _traveler_features_for() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "traveler_id": ["U1", "U2"],
            "trip_id": ["T1", "T2"],
            "implicit_taste_00": [1.0, 0.0],
            "implicit_taste_01": [0.0, 0.0],
        }
    )


def test_generate_candidates_union_dedup_and_destination_scoping() -> None:
    inputs = _tiny_generate_candidates_inputs()
    traveler_features = _traveler_features_for()

    result = generate_candidates(
        inputs["pois"],
        inputs["travelers"],
        inputs["trips"],
        inputs["poi_features"],
        traveler_features,
        inputs["interactions_train"],
        inputs["cfg"],
    )

    # Union dedup: exactly one row per (trip_id, poi_id), never two.
    assert not result[["trip_id", "poi_id"]].duplicated().any()

    t1 = result.loc[result["trip_id"] == "T1"].set_index("poi_id")
    # Destination scoping: T1 (testville) never contains otherville's P5/P6.
    assert set(t1.index) <= {"P1", "P2", "P3", "P4"}
    # P4 qualifies for NO channel (too far for geo, no interest-tag match, negative
    # semantic similarity excludes it from both semantic-quota ranking and the
    # long-tail semantic-relevance filter) -- proof the union is a genuine subset,
    # not "every POI in the destination automatically becomes a candidate."
    assert "P4" not in t1.index

    # P1: selected by geo, interest, semantic, AND long-tail -- counted once, all
    # four flags True (this is the union-dedup case the task explicitly asks for).
    assert bool(t1.loc["P1", "channel_geo"])
    assert bool(t1.loc["P1", "channel_interest"])
    assert bool(t1.loc["P1", "channel_semantic"])
    assert bool(t1.loc["P1", "channel_longtail"])
    assert not bool(t1.loc["P1", "channel_cf"])  # no train interactions at all
    assert not bool(t1.loc["P1", "channel_archetype"])  # P2 outranks P1 for U1's segment

    # P2: selected ONLY by the archetype-prior channel -- proof archetype finds a
    # POI none of the other 5 channels would have surfaced.
    assert bool(t1.loc["P2", "channel_archetype"])
    other_channels = (
        "channel_geo",
        "channel_interest",
        "channel_semantic",
        "channel_cf",
        "channel_longtail",
    )
    for ch in other_channels:
        assert not bool(t1.loc["P2", ch]), f"expected P2 to be archetype-only, but {ch} also fired"

    # T2 (otherville) never contains testville's P1-P4.
    t2 = result.loc[result["trip_id"] == "T2"]
    assert set(t2["poi_id"]) <= {"P5", "P6"}


def test_generate_candidates_is_deterministic_across_repeated_runs() -> None:
    inputs = _tiny_generate_candidates_inputs()
    traveler_features = _traveler_features_for()
    args = (
        inputs["pois"],
        inputs["travelers"],
        inputs["trips"],
        inputs["poi_features"],
        traveler_features,
        inputs["interactions_train"],
        inputs["cfg"],
    )
    result1 = generate_candidates(*args)
    result2 = generate_candidates(*args)
    pd.testing.assert_frame_equal(result1, result2)


def test_generate_candidates_output_columns() -> None:
    inputs = _tiny_generate_candidates_inputs()
    traveler_features = _traveler_features_for()
    result = generate_candidates(
        inputs["pois"],
        inputs["travelers"],
        inputs["trips"],
        inputs["poi_features"],
        traveler_features,
        inputs["interactions_train"],
        inputs["cfg"],
    )
    assert list(result.columns) == ["trip_id", "poi_id", *CHANNEL_NAMES]
    for ch in CHANNEL_NAMES:
        assert result[ch].dtype == bool


# ---------------------------------------------------------------------------
# Full-scale integration: destination scoping + determinism against
# `built_features`-shaped, real-scale data (session-scoped fixture).
# ---------------------------------------------------------------------------


def test_full_scale_candidates_never_cross_destination(
    generated_candidates: dict[str, Any],
) -> None:
    candidates_df: pd.DataFrame = generated_candidates["candidates"]
    pois: pd.DataFrame = generated_candidates["pois"]
    trips: pd.DataFrame = generated_candidates["trips"]

    dest_by_trip = trips.set_index("trip_id")["destination"]
    dest_by_poi = pois.set_index("poi_id")["destination"]

    merged = candidates_df.assign(
        trip_destination=candidates_df["trip_id"].map(dest_by_trip),
        poi_destination=candidates_df["poi_id"].map(dest_by_poi),
    )
    assert (merged["trip_destination"] == merged["poi_destination"]).all()


def test_full_scale_candidates_per_channel_respects_quota_and_no_row_dupes(
    generated_candidates: dict[str, Any],
) -> None:
    candidates_df: pd.DataFrame = generated_candidates["candidates"]
    cfg: CandidatesConfig = generated_candidates["cfg"]
    quotas = {
        "channel_geo": cfg.geo.quota,
        "channel_interest": cfg.interest.quota,
        "channel_semantic": cfg.semantic.quota,
        "channel_cf": cfg.collaborative.quota,
        "channel_longtail": cfg.longtail.quota,
        "channel_archetype": cfg.archetype.quota,
    }
    assert not candidates_df[["trip_id", "poi_id"]].duplicated().any()
    for ch, quota in quotas.items():
        per_trip_counts = candidates_df.groupby("trip_id")[ch].sum()
        assert (per_trip_counts <= quota).all(), f"{ch} exceeded its quota of {quota}"


def test_full_scale_longtail_never_exceeds_pop_pct_cutoff(
    generated_candidates: dict[str, Any],
) -> None:
    candidates_df: pd.DataFrame = generated_candidates["candidates"]
    pois: pd.DataFrame = generated_candidates["pois"]
    cfg: CandidatesConfig = generated_candidates["cfg"]

    pop_by_poi = pois.set_index("poi_id")["pop_pct"]
    longtail_rows = candidates_df.loc[candidates_df["channel_longtail"]]
    pop_values = longtail_rows["poi_id"].map(pop_by_poi)
    assert (pop_values < cfg.longtail.pop_pct_cutoff).all()
