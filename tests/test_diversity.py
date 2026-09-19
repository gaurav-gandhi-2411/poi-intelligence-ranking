"""`scoring/diversity.py` unit tests (spec.md section 9.4): the similarity formula
on a hand-checked pair, greedy MMR selection producing a valid top-K list, and the
lambda=1.0 degenerate case (MMR should reduce to pure utility ranking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poi_rank.scoring.config import DiversityConfig
from poi_rank.scoring.diversity import mmr_rerank_trip, mmr_select, pairwise_similarity


def _cfg(**overrides: float) -> DiversityConfig:
    base = dict(
        lambda_default=0.8,
        lambda_sweep=(0.5, 1.0),
        top_pool_size=50,
        similarity_cosine_weight=0.6,
        similarity_category_weight=0.4,
    )
    base.update(overrides)
    return DiversityConfig(**base)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------------
# pairwise_similarity: hand-checked pair
# -----------------------------------------------------------------------------------


def test_pairwise_similarity_hand_checked_pair() -> None:
    """Two orthogonal unit embeddings, DIFFERENT category -> cosine=0, same-cat=0
    -> similarity=0. Two IDENTICAL embeddings, SAME category -> cosine=1,
    same-cat=1 -> similarity = 0.6*1 + 0.4*1 = 1.0."""
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    categories = np.array(["food", "museum", "food"], dtype=object)
    sim = pairwise_similarity(embeddings, categories, cosine_weight=0.6, category_weight=0.4)

    assert sim[0, 1] == pytest.approx(0.0)  # orthogonal emb, different category
    assert sim[0, 2] == pytest.approx(1.0)  # identical emb, same category
    assert sim[1, 2] == pytest.approx(0.0)  # orthogonal emb, different category


def test_pairwise_similarity_same_category_different_embedding() -> None:
    """Orthogonal embeddings but SAME category: cosine=0, same-cat=1 ->
    similarity = 0.4."""
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    categories = np.array(["food", "food"], dtype=object)
    sim = pairwise_similarity(embeddings, categories, cosine_weight=0.6, category_weight=0.4)
    assert sim[0, 1] == pytest.approx(0.4)


# -----------------------------------------------------------------------------------
# mmr_select: valid top-K, deterministic tie-break
# -----------------------------------------------------------------------------------


def test_mmr_select_returns_k_distinct_indices() -> None:
    rng = np.random.default_rng(3)
    n = 20
    utility = rng.random(n)
    emb = rng.normal(size=(n, 4))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    categories = np.array([f"cat{i % 3}" for i in range(n)], dtype=object)
    sim = pairwise_similarity(emb, categories, 0.6, 0.4)
    poi_ids = [f"P{i:03d}" for i in range(n)]

    selected = mmr_select(utility, sim, poi_ids, lam=0.7, k=10)
    assert len(selected) == 10
    assert len(set(selected)) == 10


def test_mmr_select_lambda_one_reduces_to_pure_utility_ranking() -> None:
    """At lambda=1.0, the `-(1-lambda)*max_sim` term vanishes entirely -> MMR must
    select in EXACT descending-utility order, identical to a plain sort."""
    utility = np.array([0.9, 0.5, 0.95, 0.1, 0.7])
    sim = np.ones((5, 5))  # even with maximal similarity, lambda=1 ignores it
    poi_ids = [f"P{i}" for i in range(5)]
    selected = mmr_select(utility, sim, poi_ids, lam=1.0, k=5)
    expected_order = list(np.argsort(-utility))
    assert selected == expected_order


def test_mmr_select_deterministic_tie_break_by_poi_id() -> None:
    utility = np.array([0.5, 0.5])
    sim = np.zeros((2, 2))
    selected = mmr_select(utility, sim, ["P002", "P001"], lam=0.8, k=2)
    # Equal utility+score -> lower poi_id (P001, index 1) selected first.
    assert selected[0] == 1


def test_mmr_select_prefers_diverse_item_at_low_lambda() -> None:
    """3 items: A (utility=1.0), B (utility=0.99, near-identical to A), C
    (utility=0.5, dissimilar to A). At a low lambda favoring diversity, after
    picking A, C should be preferred over the near-duplicate B despite B's higher
    utility."""
    utility = np.array([1.0, 0.99, 0.5])
    sim = np.array(
        [
            [1.0, 0.98, 0.0],
            [0.98, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    poi_ids = ["A", "B", "C"]
    selected = mmr_select(utility, sim, poi_ids, lam=0.3, k=2)
    assert selected[0] == 0  # A first (highest raw utility, nothing selected yet)
    assert selected[1] == 2  # C preferred over near-duplicate B


# -----------------------------------------------------------------------------------
# mmr_rerank_trip: end-to-end on a small synthetic group
# -----------------------------------------------------------------------------------


def test_mmr_rerank_trip_produces_valid_ordered_list() -> None:
    rng = np.random.default_rng(5)
    n = 15
    emb = rng.normal(size=(n, 4))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    group = pd.DataFrame(
        {
            "poi_id": [f"P{i:03d}" for i in range(n)],
            "utility": rng.random(n),
            "poi_category_raw": [f"cat{i % 4}" for i in range(n)],
            **{f"text_emb_{j:02d}": emb[:, j] for j in range(4)},
        }
    )
    ordered = mmr_rerank_trip(group, _cfg(top_pool_size=15), lam=0.8, k=5)
    assert len(ordered) == 5
    assert len(set(ordered)) == 5
    assert all(pid in group["poi_id"].to_numpy() for pid in ordered)


def _reference_mmr_select(
    utility: np.ndarray, sim: np.ndarray, poi_ids: list[str], lam: float, k: int
) -> list[int]:
    """The original O(n^2 k) pure-Python greedy MMR, kept as an oracle for the vectorised
    `mmr_select` (behaviour-preserving refactor guard)."""
    n = len(utility)
    selected: list[int] = []
    remaining = list(range(n))
    for _ in range(min(k, n)):
        best_i: int | None = None
        best_score = -np.inf
        for i in remaining:
            max_sim = max((sim[i, j] for j in selected), default=0.0)
            score = lam * utility[i] - (1.0 - lam) * max_sim
            if (
                best_i is None
                or score > best_score
                or (score == best_score and poi_ids[i] < poi_ids[best_i])
            ):
                best_score, best_i = score, i
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


@pytest.mark.parametrize("lam", [0.5, 0.8, 1.0])
def test_mmr_select_matches_reference_implementation(lam: float) -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(3, 40))
        emb = rng.normal(size=(n, 6))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sim = 0.6 * (emb @ emb.T) + 0.4 * (rng.integers(0, 2, size=(n, n)) > 0)
        sim = (sim + sim.T) / 2
        utility = np.round(rng.normal(size=n), 1)  # coarse -> many exact ties
        poi_ids = [f"P{i:03d}" for i in rng.permutation(n)]
        assert mmr_select(utility, sim, poi_ids, lam, 10) == _reference_mmr_select(
            utility, sim, poi_ids, lam, 10
        )
