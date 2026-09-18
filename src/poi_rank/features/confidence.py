"""Confidence shrinkage `alpha_t = n_t / (n_t + k)` (spec.md section 6).

**Two different mechanisms for two different jobs** (spec.md is explicit about this):
this module is the evidence-VOLUME shrinkage mechanism, used only in the cold-start
fallback path and the confidence score -- a monotone-increasing weight in `[0, 1)`
answering "how much do we trust this traveler's *own* evidence volume." It has
nothing to do with `traveler_features.py`'s implicit taste-vector TIME-DECAY
weighting (`exp(-delta_t / tau)`, which discounts *individual interactions by
recency*, not the traveler's overall evidence count). Never conflated, never shared
code -- see `traveler_features.py`'s module docstring for the mirror-image statement
of this same distinction.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatOrArray = npt.NDArray[np.float64] | float


def confidence_shrinkage_alpha(n_interactions: FloatOrArray, k: float = 5.0) -> FloatOrArray:
    """`alpha_t = n_t / (n_t + k)`.

    `alpha_t -> 0` for a brand-new traveler (`n_t = 0`), `alpha_t -> 1` as evidence
    accumulates. `k` is the evidence count at which `alpha_t = 0.5`.
    """
    return n_interactions / (n_interactions + k)
