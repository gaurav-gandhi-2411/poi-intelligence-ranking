"""Geo helpers for data prep (spec.md section 4): haversine distance, H3 resolution-8
cell assignment, distance-to-stay, and a synthetic per-destination transit graph for
distance-to-nearest-transit-node.

**Design choice on haversine math:** duplicated here rather than imported from
`datagen/geo.py`, even though the latter is pure trig with zero downstream
dependencies and importing it would not technically violate the DGP firewall (the
firewall is one-directional -- `datagen/**` must never import downstream, but nothing
stops downstream importing `datagen/`). Duplicating ~10 lines removes even the
appearance of a `data/` <-> `datagen/` coupling for a trivial amount of code, keeping
the module boundary unambiguous. Documented in docs/DATA_CARD.md.
"""

from __future__ import annotations

import h3
import numpy as np
import numpy.typing as npt
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: npt.NDArray[np.float64] | float,
    lon1: npt.NDArray[np.float64] | float,
    lat2: npt.NDArray[np.float64] | float,
    lon2: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """Vectorized great-circle distance in km between two (broadcastable) points/arrays."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    result: npt.NDArray[np.float64] = EARTH_RADIUS_KM * c
    return result


def h3_cell(
    lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64], resolution: int = 8
) -> npt.NDArray[np.str_]:
    """Assign an H3 cell index (given `resolution`, default 8) to each (lat, lon) pair."""
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=float))
    cells = np.array(
        [
            h3.geo_to_h3(float(la), float(lo), resolution)
            for la, lo in zip(lat_arr, lon_arr, strict=True)
        ],
        dtype=object,
    )
    return cells


def distance_to_point_km(
    lat: npt.NDArray[np.float64] | float,
    lon: npt.NDArray[np.float64] | float,
    point_lat: float,
    point_lon: float,
) -> npt.NDArray[np.float64]:
    """Distance from each (lat, lon) to a single reference point.

    Used for "distance-to-stay" (spec.md section 4): a trip's `stay_lat`/`stay_lon`
    (from `trips.parquet` -- the field lives on the trip, not the traveler, contrary
    to a literal reading of the phase-2 task description; see docs/DATA_CARD.md).
    Trip-conditional, so it is never materialized as a static column on the prepared
    POI catalog -- it is applied per (trip, candidate POI) pair at the later
    candidate-generation/feature stage, which is why this stays a plain callable
    helper rather than a `prepare.py` pipeline step.
    """
    return haversine_km(lat, lon, point_lat, point_lon)


def synthesize_transit_nodes(
    rng: np.random.Generator,
    pois_df: pd.DataFrame,
    n_nodes_per_destination: int,
    spread_deg: float,
) -> pd.DataFrame:
    """Synthesize a small transit-node graph per destination.

    Phase 1's DGP does not generate a transit network, and spec.md section 4 asks for
    "distance-to-nearest-transit-node (synthetic transit graph per destination)" as a
    prep-stage concern. Nodes are scattered (Gaussian, std `spread_deg`) around each
    destination's *actual* POI centroid computed from the data at hand -- not a
    hardcoded city-center constant -- so this module never needs to reference
    `datagen/catalog.py::DEST_CENTERS`. This is observable infrastructure metadata,
    not a latent variable, so synthesizing it at the prep stage (rather than in
    `datagen/`) is a legitimate, documented choice (docs/DATA_CARD.md), not a silent
    invention.
    """
    rows: list[dict[str, object]] = []
    node_idx = 0
    for dest, group in pois_df.groupby("destination", sort=True):
        center_lat = float(group["lat"].mean())
        center_lon = float(group["lon"].mean())
        for _ in range(n_nodes_per_destination):
            node_idx += 1
            rows.append(
                {
                    "transit_node_id": f"TR{node_idx:04d}",
                    "destination": dest,
                    "lat": center_lat + float(rng.normal(0, spread_deg)),
                    "lon": center_lon + float(rng.normal(0, spread_deg)),
                }
            )
    return pd.DataFrame(rows)


def distance_to_nearest_transit_km(
    pois_df: pd.DataFrame, transit_nodes_df: pd.DataFrame
) -> pd.Series:
    """Per-POI haversine distance (km) to the nearest same-destination transit node."""
    result = pd.Series(np.nan, index=pois_df.index, dtype=float, name="dist_to_transit_km")
    for dest, group in pois_df.groupby("destination", sort=False):
        nodes = transit_nodes_df.loc[transit_nodes_df["destination"] == dest]
        node_lat = nodes["lat"].to_numpy(dtype=float)[np.newaxis, :]
        node_lon = nodes["lon"].to_numpy(dtype=float)[np.newaxis, :]
        poi_lat = group["lat"].to_numpy(dtype=float)[:, np.newaxis]
        poi_lon = group["lon"].to_numpy(dtype=float)[:, np.newaxis]
        dists = haversine_km(poi_lat, poi_lon, node_lat, node_lon)
        result.loc[group.index] = dists.min(axis=1)
    return result
