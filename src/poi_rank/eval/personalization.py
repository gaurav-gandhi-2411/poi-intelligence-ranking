"""Personalization metrics (spec.md section 11.2): mean pairwise Jaccard@10, the
within- vs cross-archetype Jaccard@10 comparison spec.md calls "the correct test",
and rank-biased overlap (RBO, p=0.9) as a rank-aware complement.

**Recommendation-set source**: spec.md says "top-10 recommendation sets from
`results/recommendations.json`, or recompute top-10 by utility from the full
evaluation frame if that's cleaner -- your call, document it." This module takes
neither literally -- it consumes the ALREADY-ASSEMBLED per-trip recommendation list
from `scoring.output.run_scoring_pipeline`'s own `payload` dict (the exact same
object `poi_rank.cli recommend` persists as `results/recommendations.json`, just
computed in-process by `eval/run.py` so this metric never depends on `recommend`
having been run first -- `recommend` is not part of `make reproduce`'s default chain,
Phase 6's own documented decision). This is the FINAL, post-MMR, post-hard-gate
top-K list -- the real output the system produces -- not a re-derived top-10-by-raw-
score shortcut. See `docs/DATA_CARD.md` for the resolved ambiguity.

**Archetype proxy**: spec.md section 11.2 wants "within-archetype vs cross-archetype"
groups but the DGP's latent archetype mixture is oracle-only and never exported. Per
this phase's task brief, this module reuses the SAME observable K-Means
traveler-segment proxy already established in Phase 3 (`features.traveler_features
.assign_traveler_segments`, reused again in Phase 4a's archetype-prior candidate
channel) as the "archetype" grouping -- never reads the oracle-only export
directory.

**Trip-pairs, not literally traveler-pairs**: each holdout trip has exactly one
traveler in this dataset's population, so "all traveler pairs" and "all pairs of
holdout trips' recommendation lists" are the same population here -- computed over
trip pairs directly (documented, not a silent reinterpretation).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

TARGET_CROSS_ARCHETYPE_JACCARD = 0.25
TARGET_WITHIN_CROSS_RATIO = 2.0
RBO_P = 0.9


def top10_lists_from_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    """`{trip_id: [poi_id, ...]}` in rank order, from a `run_scoring_pipeline`-shaped
    payload (module docstring) -- `payload[trip_id]["recommendations"]` is already
    sorted by `rank` ascending (`scoring.output.assemble_output_payload`)."""
    return {
        trip_id: [rec["poi_id"] for rec in entry["recommendations"]]
        for trip_id, entry in payload.items()
    }


def jaccard(a: set[str], b: set[str]) -> float:
    """`|a intersect b| / |a union b|`. Two empty sets are defined as maximally
    similar (`1.0`, trivially identical empty recommendation lists) -- documented
    convention, not expected to occur for any real top-10 list in this dataset."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _all_pairs(keys: Sequence[str]) -> Iterator[tuple[str, str]]:
    yield from combinations(sorted(keys), 2)


def mean_pairwise_jaccard(lists_by_trip: dict[str, list[str]]) -> tuple[float, int]:
    """Mean Jaccard@10 across ALL distinct pairs of trips' top-10 sets (spec.md
    section 11.2's "mean pairwise Jaccard@10 across all traveler pairs", module
    docstring). Returns `(mean, n_pairs)`."""
    trip_ids = list(lists_by_trip)
    values = [
        jaccard(set(lists_by_trip[a]), set(lists_by_trip[b])) for a, b in _all_pairs(trip_ids)
    ]
    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


def rank_biased_overlap(list_a: Sequence[str], list_b: Sequence[str], p: float = RBO_P) -> float:
    """Extrapolated rank-biased overlap (Webber, Moffat & Zobel 2010, eq. 21 --
    the EQUAL-DEPTH variant): for two rankings truncated to the SAME depth `k`,

        RBO = (X_k / k) * p^k + ((1 - p) / p) * sum_{d=1}^{k} (X_d / d) * p^d

    where `X_d = |set(list_a[:d]) intersect set(list_b[:d])|`. Both lists in this
    project are always top-K recommendation lists at the SAME configured `top_k`
    (spec.md section 9.5's `output.top_k`), so the equal-depth formula applies
    directly -- if the two inputs differ in length (e.g. one trip's hard-gate
    survivor pool was smaller than `top_k`), both are truncated to `k =
    min(len(list_a), len(list_b))` before computing `X_d`, a documented
    simplification of the paper's more general unequal-length "extrapolated" formula
    (`docs/DATA_CARD.md`). Identical lists (any order) -> `1.0`; lists with zero
    overlap at every depth -> `0.0`.
    """
    k = min(len(list_a), len(list_b))
    if k == 0:
        return 1.0 if len(list_a) == len(list_b) else 0.0
    a_trunc = list(list_a[:k])
    b_trunc = list(list_b[:k])

    overlap_sum = 0.0
    x_k = 0
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    for d in range(1, k + 1):
        seen_a.add(a_trunc[d - 1])
        seen_b.add(b_trunc[d - 1])
        x_d = len(seen_a & seen_b)
        overlap_sum += (x_d / d) * (p**d)
        if d == k:
            x_k = x_d

    first_term = (x_k / k) * (p**k)
    second_term = ((1.0 - p) / p) * overlap_sum
    return float(first_term + second_term)


def mean_pairwise_rbo(lists_by_trip: dict[str, list[str]], p: float = RBO_P) -> tuple[float, int]:
    """Mean RBO across ALL distinct pairs of trips' top-10 lists, same pairing
    population as `mean_pairwise_jaccard`. Returns `(mean, n_pairs)`."""
    trip_ids = list(lists_by_trip)
    values = [
        rank_biased_overlap(lists_by_trip[a], lists_by_trip[b], p) for a, b in _all_pairs(trip_ids)
    ]
    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


@dataclass(frozen=True)
class ArchetypeJaccardResult:
    """Within- vs cross-archetype-proxy Jaccard@10 (spec.md section 11.2's "the
    correct test": high within, low across). Targets: cross <= 0.25, ratio >= 2.0."""

    within_mean: float
    within_n_pairs: int
    cross_mean: float
    cross_n_pairs: int
    ratio: float

    def to_dict(self) -> dict[str, float | int | bool | None]:
        # `ratio` is `float("inf")` when `cross_mean == 0.0` (module docstring) --
        # JSON has no literal infinity, so this is serialized as `None` with the
        # boolean target-met flag still computed correctly off the raw (non-JSON)
        # `self.ratio` value, never silently truncated to a large-but-finite number.
        ratio_json: float | None = self.ratio if self.ratio != float("inf") else None
        return {
            "within_archetype_jaccard_mean": self.within_mean,
            "within_archetype_n_pairs": self.within_n_pairs,
            "cross_archetype_jaccard_mean": self.cross_mean,
            "cross_archetype_n_pairs": self.cross_n_pairs,
            "within_cross_ratio": ratio_json,
            "cross_archetype_target_met": bool(self.cross_mean <= TARGET_CROSS_ARCHETYPE_JACCARD),
            "ratio_target_met": bool(self.ratio >= TARGET_WITHIN_CROSS_RATIO),
        }


def within_cross_archetype_jaccard(
    lists_by_trip: dict[str, list[str]], trip_segment: dict[str, int]
) -> ArchetypeJaccardResult:
    """Partitions every distinct trip pair into "within" (same segment proxy) vs
    "cross" (different segment proxy) and reports mean Jaccard@10 for each, plus the
    within/cross ratio (`inf` if `cross_mean == 0`, per module convention of never
    silently coercing an undefined ratio to a misleadingly finite number -- reported
    as `float("inf")`, valid JSON-incompatible so callers must handle it explicitly,
    same discipline as every other "undefined metric" in this codebase)."""
    trip_ids = [t for t in lists_by_trip if t in trip_segment]
    within_vals: list[float] = []
    cross_vals: list[float] = []
    for a, b in _all_pairs(trip_ids):
        j = jaccard(set(lists_by_trip[a]), set(lists_by_trip[b]))
        if trip_segment[a] == trip_segment[b]:
            within_vals.append(j)
        else:
            cross_vals.append(j)

    within_mean = sum(within_vals) / len(within_vals) if within_vals else 0.0
    cross_mean = sum(cross_vals) / len(cross_vals) if cross_vals else 0.0
    ratio = (within_mean / cross_mean) if cross_mean > 0 else float("inf")
    return ArchetypeJaccardResult(
        within_mean=within_mean,
        within_n_pairs=len(within_vals),
        cross_mean=cross_mean,
        cross_n_pairs=len(cross_vals),
        ratio=ratio,
    )
