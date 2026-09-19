"""Shared Plackett-Luce / softmax sampling primitives for the DGP.

Used by both `exposure.py` (which POIs are shown) and `interactions.py` (which shown
POIs get engaged with) so the stochastic-choice simulation is implemented once.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable softmax over the last axis."""
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0:
        return np.full_like(x, 1.0 / len(x))
    result: npt.NDArray[np.float64] = exp / total
    return result


def choice_index(rng: np.random.Generator, probs: npt.NDArray[np.float64]) -> int:
    """`int(rng.choice(len(probs), p=probs))`, bit-identical and several times cheaper.

    `Generator.choice(a, p=p)` draws ONE uniform double and inverts the normalised CDF with
    `searchsorted(..., side="right")`; doing that directly consumes the RNG stream exactly the
    same way while skipping choice()'s per-call validation. The DGP calls this ~340k times per
    dataset (A4: it was ~40% of `generate`). Equivalence to `rng.choice` is asserted in
    `tests/test_datagen_sampling.py`, and the regenerated dataset is byte-identical.
    """
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]
    return int(cdf.searchsorted(rng.random(), side="right"))


def pl_sample_without_replacement(
    rng: np.random.Generator, weights: npt.NDArray[np.float64], k: int
) -> list[int]:
    """Sequential Plackett-Luce draw of `k` distinct indices, proportional to `weights`.

    Standard Luce's-choice-axiom sampling: draw one index proportional to the
    remaining weight mass, remove it, renormalize, repeat. Returns indices into the
    original `weights` array in draw order (draw order == implied preference order).
    """
    n = len(weights)
    k = min(k, n)
    if k <= 0:
        return []
    remaining_idx = list(range(n))
    remaining_w = np.asarray(weights, dtype=np.float64).copy()
    chosen: list[int] = []
    for _ in range(k):
        total = remaining_w.sum()
        if total <= 0 or not np.isfinite(total):
            break
        probs = remaining_w / total
        pick = choice_index(rng, probs)
        chosen.append(remaining_idx[pick])
        del remaining_idx[pick]
        remaining_w = np.delete(remaining_w, pick)
    return chosen
