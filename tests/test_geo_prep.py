"""Unit tests for `data/geo_prep.py`: haversine distance, H3 cell assignment, and the
synthetic transit-node graph."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from poi_rank.data.geo_prep import (
    distance_to_nearest_transit_km,
    h3_cell,
    haversine_km,
    synthesize_transit_nodes,
)


def test_haversine_km_zero_distance() -> None:
    assert haversine_km(37.5665, 126.9780, 37.5665, 126.9780) == pytest.approx(0.0, abs=1e-9)


def test_haversine_km_known_distance() -> None:
    # Seoul City Hall to Gyeongbokgung Palace is ~2.5 km apart.
    d = haversine_km(37.5665, 126.9780, 37.5796, 126.9770)
    assert 1.0 < float(d) < 4.0


def test_haversine_km_vectorized_broadcast() -> None:
    lat1 = np.array([[37.5665], [35.0116]])
    lon1 = np.array([[126.9780], [135.7681]])
    lat2 = np.array([[37.5665, 37.58]])
    lon2 = np.array([[126.9780, 126.98]])
    d = haversine_km(lat1, lon1, lat2, lon2)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_h3_cell_returns_resolution_8_index() -> None:
    import h3

    cells = h3_cell(np.array([37.5665]), np.array([126.9780]), resolution=8)
    assert len(cells) == 1
    assert h3.h3_get_resolution(cells[0]) == 8


def test_synthesize_transit_nodes_deterministic_and_scoped_per_destination() -> None:
    pois = pd.DataFrame(
        {
            "destination": ["seoul", "seoul", "kyoto"],
            "lat": [37.5, 37.6, 35.0],
            "lon": [127.0, 127.1, 135.7],
        }
    )
    rng1 = np.random.default_rng(42)
    nodes1 = synthesize_transit_nodes(rng1, pois, n_nodes_per_destination=5, spread_deg=0.02)
    rng2 = np.random.default_rng(42)
    nodes2 = synthesize_transit_nodes(rng2, pois, n_nodes_per_destination=5, spread_deg=0.02)

    assert len(nodes1) == 10  # 5 per destination x 2 destinations
    pd.testing.assert_frame_equal(nodes1, nodes2)  # deterministic given the same seed
    assert set(nodes1["destination"].unique()) == {"seoul", "kyoto"}


def test_distance_to_nearest_transit_km_picks_closest_same_destination_node() -> None:
    pois = pd.DataFrame(
        {"poi_id": ["P1"], "destination": ["seoul"], "lat": [37.5665], "lon": [126.9780]}
    )
    nodes = pd.DataFrame(
        {
            "transit_node_id": ["T1", "T2", "T3"],
            "destination": ["seoul", "seoul", "kyoto"],
            "lat": [37.5665, 37.60, 35.0],
            "lon": [126.9780, 127.00, 135.7],
        }
    )
    dist = distance_to_nearest_transit_km(pois, nodes)
    assert dist.iloc[0] == pytest.approx(0.0, abs=1e-6)  # T1 is an exact match


def test_dist_to_transit_km_is_nonnegative_for_generated_data(
    prepared_data: dict[str, Any],
) -> None:
    prepared = prepared_data["prepared"]
    assert (prepared["dist_to_transit_km"] >= 0).all()
    assert prepared["dist_to_transit_km"].notna().all()
