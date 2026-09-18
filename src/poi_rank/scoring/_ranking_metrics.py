"""Minimal internal duplicate of `eval.metrics.dcg_at_k`/`ndcg_at_k` (identical
`(2^rel - 1) / log2(i+1)` exponential graded-gain formula, consistent with
`configs/model.yaml`'s `label_gain=[0,1,3,7]`) -- used ONLY by this package's own
internal diagnostics: the beta-sensitivity table (`scoring/utility.py`) and the MMR
lambda-sweep NDCG-vs-diversity curve (`scoring/diversity.py`).

**Why duplicated instead of imported from `poi_rank.eval`**: the same reason
`data/geo_prep.py` duplicates `datagen/geo.py`'s haversine instead of importing it
(see that module's own docstring) -- `scoring/` sits upstream of `eval/` in this
project's dependency direction (a later phase's `eval/` naturally consumes
`scoring/`'s output the same way it already consumes `models/`'s output, per
`eval/run.py`), so an import the other way would invert that boundary. Duplicating
~25 lines of pure, stable, already-tested logic removes even the appearance of a
`scoring/` <-> `eval/` coupling. Documented in docs/DATA_CARD.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _rank_order(poi_ids: npt.NDArray[np.object_], scores: FloatArray) -> npt.NDArray[np.intp]:
    """Descending-score order, poi_id-ascending tie-break -- same `np.lexsort`
    determinism convention used throughout this project."""
    order: npt.NDArray[np.intp] = np.lexsort((poi_ids, -scores))
    return order


def dcg_at_k(labels_in_rank_order: npt.NDArray[np.int64], k: int) -> float:
    """`sum_{i=1}^k (2^rel_i - 1) / log2(i + 1)`, over `labels_in_rank_order`
    (already sorted by the ranking under evaluation)."""
    kk = min(k, len(labels_in_rank_order))
    if kk <= 0:
        return 0.0
    ranks = np.arange(1, kk + 1, dtype=np.float64)
    gains = np.power(2.0, labels_in_rank_order[:kk].astype(np.float64)) - 1.0
    discounts = np.log2(ranks + 1.0)
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    labels: npt.NDArray[np.int64], scores: FloatArray, poi_ids: npt.NDArray[np.object_], k: int
) -> float | None:
    """NDCG@k for one trip. `None` (undefined) when the ideal ranking's DCG is 0 --
    i.e. this candidate set contains no relevant POI at all."""
    order = _rank_order(poi_ids, scores)
    dcg = dcg_at_k(labels[order], k)
    ideal_order = np.argsort(-labels, kind="stable")
    idcg = dcg_at_k(labels[ideal_order], k)
    if idcg == 0.0:
        return None
    return dcg / idcg
