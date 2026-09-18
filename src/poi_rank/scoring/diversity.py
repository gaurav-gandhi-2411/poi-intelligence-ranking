"""MMR diversity re-rank (spec.md section 9.4):

```
argmax [ lambda * utility - (1 - lambda) * max_sim_to_selected ]
similarity = 0.6 * cosine(emb) + 0.4 * same-category indicator
```

Standard greedy MMR over each trip's top-50-by-utility pool. Sweep
`lambda in {0.5..1.0}` and report the NDCG-vs-diversity trade-off curve (this
package's own internal `_ranking_metrics.ndcg_at_k` duplicate -- see that module's
docstring for why it is not imported from `poi_rank.eval`). Default `lambda = 0.8`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.scoring._ranking_metrics import ndcg_at_k
from poi_rank.scoring.config import DiversityConfig

FloatArray = npt.NDArray[np.float64]

TEXT_EMB_PREFIX = "text_emb_"


def pairwise_similarity(
    embeddings: FloatArray,
    categories: npt.NDArray[np.object_],
    cosine_weight: float,
    category_weight: float,
) -> FloatArray:
    """`0.6 * cosine(emb) + 0.4 * same-category indicator` (spec.md section 9.4,
    verbatim), for every pair within one pool. `embeddings` are already
    L2-normalized (guaranteed by `features/text_embed.py`), so `emb @ emb.T` is
    already the cosine-similarity matrix."""
    cos = embeddings @ embeddings.T
    same_category = categories[:, None] == categories[None, :]
    result: FloatArray = cosine_weight * cos + category_weight * same_category.astype(np.float64)
    return result


def mmr_select(
    utility: FloatArray, sim_matrix: FloatArray, poi_ids: list[str], lam: float, k: int
) -> list[int]:
    """Greedy MMR (spec.md section 9.4): repeatedly pick
    `argmax[lambda*utility - (1-lambda)*max_sim_to_already_selected]` until `k`
    items are selected (or the pool is exhausted). Deterministic poi_id-ascending
    tie-break, matching this project's established convention throughout."""
    n = len(utility)
    kk = min(k, n)
    selected: list[int] = []
    remaining = list(range(n))
    for _ in range(kk):
        best_i: int | None = None
        best_score = -np.inf
        for i in remaining:
            max_sim = max((sim_matrix[i, j] for j in selected), default=0.0)
            score = lam * utility[i] - (1.0 - lam) * max_sim
            if (
                best_i is None
                or score > best_score
                or (score == best_score and poi_ids[i] < poi_ids[best_i])
            ):
                best_score = score
                best_i = i
        assert best_i is not None  # `remaining` is non-empty on every iteration here
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def mmr_rerank_trip(group: pd.DataFrame, cfg: DiversityConfig, lam: float, k: int) -> list[str]:
    """MMR-reranked `poi_id` list (best-first) for ONE trip's candidate rows.
    `group` must already be restricted to `hard_gate == 1` rows and carry `utility`,
    `poi_id`, `poi_category_raw`, and the `text_emb_*` columns."""
    pool = group.sort_values(["utility", "poi_id"], ascending=[False, True]).head(cfg.top_pool_size)
    emb_cols = sorted(c for c in pool.columns if c.startswith(TEXT_EMB_PREFIX))
    embeddings = pool[emb_cols].to_numpy(dtype=np.float64)
    categories = pool["poi_category_raw"].to_numpy(dtype=object)
    sim = pairwise_similarity(
        embeddings, categories, cfg.similarity_cosine_weight, cfg.similarity_category_weight
    )
    utility = pool["utility"].to_numpy(dtype=np.float64)
    poi_ids = pool["poi_id"].astype(str).tolist()
    order_idx = mmr_select(utility, sim, poi_ids, lam, k)
    return [poi_ids[i] for i in order_idx]


def mmr_rerank_all_trips(
    frame: pd.DataFrame, cfg: DiversityConfig, lam: float, k: int
) -> pd.DataFrame:
    """`mmr_rerank_trip` applied per `trip_id`, returned as a long
    `(trip_id, poi_id, mmr_rank)` frame, `mmr_rank` 1-indexed, best-first."""
    rows: list[dict[str, Any]] = []
    for trip_id, group in frame.groupby("trip_id", sort=True):
        ordered = mmr_rerank_trip(group, cfg, lam, k)
        for rank, poi_id in enumerate(ordered, start=1):
            rows.append({"trip_id": trip_id, "poi_id": poi_id, "mmr_rank": rank})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------------
# Lambda sweep: NDCG-vs-diversity trade-off (spec.md section 9.4)
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaSweepRow:
    lam: float
    ndcg_at_10_mean: float
    mean_intra_list_similarity: float
    n_trips_included_ndcg: int
    n_trips_included_diversity: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lambda": self.lam,
            "ndcg@10_mean": self.ndcg_at_10_mean,
            "mean_intra_list_similarity": self.mean_intra_list_similarity,
            "n_trips_included_ndcg": self.n_trips_included_ndcg,
            "n_trips_included_diversity": self.n_trips_included_diversity,
        }


def _mean_intra_list_similarity(
    group: pd.DataFrame, cosine_weight: float, category_weight: float
) -> float | None:
    if len(group) < 2:
        return None
    emb_cols = sorted(c for c in group.columns if c.startswith(TEXT_EMB_PREFIX))
    embeddings = group[emb_cols].to_numpy(dtype=np.float64)
    categories = group["poi_category_raw"].to_numpy(dtype=object)
    sim = pairwise_similarity(embeddings, categories, cosine_weight, category_weight)
    iu = np.triu_indices(len(group), k=1)
    return float(sim[iu].mean())


def lambda_sweep_report(
    frame: pd.DataFrame, cfg: DiversityConfig, lambdas: tuple[float, ...], k: int
) -> list[LambdaSweepRow]:
    """For each `lam` in `lambdas`: MMR-rerank every trip's top-`cfg.top_pool_size`
    pool to a top-`k` list, then report mean NDCG@k and mean intra-list similarity
    (spec.md section 9.4's own similarity formula) across trips -- the
    NDCG-vs-diversity trade-off curve. `frame` must carry `trip_id`, `poi_id`,
    `label`, `utility`, `poi_category_raw`, and `text_emb_*`, restricted to
    `hard_gate == 1` rows.

    **NDCG@k is computed against each trip's FULL candidate pool (`frame`), not
    just the k rows MMR selected** -- this project's own `ndcg_at_k` convention
    (`eval/metrics.py`, duplicated here as `_ranking_metrics.ndcg_at_k`) always
    derives the IDEAL ranking (IDCG) from the full candidate set a system is
    scored against, truncated to k; computing IDCG from only the already-selected
    k rows would silently inflate NDCG by comparing the achieved DCG against a much
    weaker "ideal" (the best possible ordering of a tiny, already-cherry-picked
    pool, not the trip's true best-possible top-k). Selected rows get a score
    reflecting their MMR rank (`-mmr_rank`, so rank 1 sorts highest); every other
    candidate in the trip's pool gets `-inf` (sorts below every selected row, same
    as `models/baselines.py::baseline_popularity_geo_filter`'s own
    outside-the-filter convention).
    """
    rows: list[LambdaSweepRow] = []
    for lam in lambdas:
        reranked = mmr_rerank_all_trips(frame, cfg, lam, k)
        rank_by_trip_poi = reranked.set_index(["trip_id", "poi_id"])["mmr_rank"]

        ndcg_vals: list[float] = []
        div_vals: list[float] = []
        for trip_id, group in frame.groupby("trip_id", sort=True):
            labels = group["label"].to_numpy(dtype=np.int64)
            poi_ids = group["poi_id"].to_numpy(dtype=object)
            mmr_rank = np.array(
                [rank_by_trip_poi.get((trip_id, pid), np.nan) for pid in poi_ids],
                dtype=np.float64,
            )
            scores = np.where(np.isfinite(mmr_rank), -mmr_rank, -np.inf)
            ndcg = ndcg_at_k(labels, scores, poi_ids, k)
            if ndcg is not None:
                ndcg_vals.append(ndcg)

            selected = group.loc[np.isfinite(mmr_rank)]
            div = _mean_intra_list_similarity(
                selected, cfg.similarity_cosine_weight, cfg.similarity_category_weight
            )
            if div is not None:
                div_vals.append(div)

        rows.append(
            LambdaSweepRow(
                lam=float(lam),
                ndcg_at_10_mean=float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
                mean_intra_list_similarity=float(np.mean(div_vals)) if div_vals else 0.0,
                n_trips_included_ndcg=len(ndcg_vals),
                n_trips_included_diversity=len(div_vals),
            )
        )
    return rows


def plot_lambda_sweep(rows: list[LambdaSweepRow], output_path: Path) -> None:
    """Save the NDCG-vs-diversity trade-off curve (spec.md section 9.4) to
    `output_path`."""
    lambdas = [r.lam for r in rows]
    ndcg = [r.ndcg_at_10_mean for r in rows]
    diversity = [1.0 - r.mean_intra_list_similarity for r in rows]  # higher = more diverse

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel("MMR lambda")
    ax1.set_ylabel("NDCG@10 (mean)", color="tab:blue")
    ax1.plot(lambdas, ndcg, marker="o", color="tab:blue", label="NDCG@10")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("diversity (1 - mean intra-list similarity)", color="tab:orange")
    ax2.plot(lambdas, diversity, marker="s", color="tab:orange", label="diversity")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("MMR lambda sweep: NDCG-vs-diversity trade-off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def lambda_sweep_payload(rows: list[LambdaSweepRow]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]
