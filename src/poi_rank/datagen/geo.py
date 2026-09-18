"""Vectorized haversine distance helper used only by the DGP's exposure policy.

Kept self-contained inside datagen/ (no dependency on a later `data/` geo module) so
the DGP firewall test never has to reason about cross-phase geo utilities.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float, lon1: float, lat2: npt.NDArray[np.float64], lon2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Great-circle distance in km from a single point to an array of points."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    result: npt.NDArray[np.float64] = EARTH_RADIUS_KM * c
    return result
