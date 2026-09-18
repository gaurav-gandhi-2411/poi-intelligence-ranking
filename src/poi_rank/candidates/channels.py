"""The 6 candidate-generation channels (spec.md section 7): pure functions over a
precomputed per-destination `DestinationIndex` (and, for the collaborative channel, a
precomputed global `ItemItemCF`), each respecting its own hard quota independently
before the union/dedup step in `union.py`.

**Firewall**: this module (like all of `candidates/`) may import from `poi_rank.data`
and `poi_rank.features` (candidates is downstream of and consumes their output) but
must never import from `poi_rank.models` or `poi_rank.scoring`, and must never
reference the oracle-only export directory -- enforced by
`tests/test_firewall_candidates.py`.

**Filtering to the trip's own destination**: every channel operates on a
`DestinationIndex` that is already scoped to exactly one destination (built once by
`build_destination_indices`, one index per destination) -- there is no code path here
that could accidentally rank a POI from a different destination into a trip's
candidate set.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import h3
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from poi_rank.candidates.config import (
    CollaborativeChannelConfig,
    GeoChannelConfig,
    LongtailChannelConfig,
)
from poi_rank.data.geo_prep import haversine_km
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids

FloatArray = npt.NDArray[np.float64]

TEXT_EMB_PREFIX = "text_emb_"

CHANNEL_NAMES: tuple[str, ...] = (
    "channel_geo",
    "channel_interest",
    "channel_semantic",
    "channel_cf",
    "channel_longtail",
    "channel_archetype",
)

# A generous +1-ring safety margin on top of the literal `ceil(radius/edge_length)`
# k-ring size: H3's k-ring covers an approximate, not exact, disk of the given radius
# (its true footprint varies with hex orientation relative to the query point), so an
# under-sized k could clip real in-radius POIs sitting just past the coarse hex
# boundary. The exact haversine cutoff applied immediately after (see
# `channel_geo`) is what actually enforces the literal km radius -- k-ring is only
# ever a coarse, cheap first filter over `pois_prepared.h3_cell`, never the final
# radius decision.
_GEO_KRING_SAFETY_MARGIN = 1


def _poi_id_list(arr: npt.NDArray[np.object_]) -> list[str]:
    """`.tolist()` on an object-dtype numpy array of poi_ids returns `list[Any]` to
    mypy's strict mode -- this narrows the return type explicitly at the one place
    every channel needs it, rather than repeating a `# type: ignore` at each call
    site."""
    return [str(x) for x in arr]


def trip_seed(base_seed: int, trip_id: str) -> int:
    """Deterministic per-trip RNG seed derived from `(base_seed, trip_id)` via
    SHA256 -- NEVER Python's built-in `hash()`, which is only reproducible across
    process runs when `PYTHONHASHSEED` happens to be externally fixed (the Makefile
    sets it, but a direct `uv run python -m poi_rank.cli candidates` invocation
    outside `make` would not). Used only by the long-tail channel's epsilon-greedy
    sampling."""
    digest = hashlib.sha256(f"{base_seed}:{trip_id}".encode()).hexdigest()
    return int(digest[:8], 16)


# -----------------------------------------------------------------------------------
# Per-destination precomputed index
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DestinationIndex:
    """Precomputed, destination-scoped arrays -- built once per destination
    (`build_destination_indices`) so the 800-trip x 6-channel loop in `union.py`
    never re-filters `pois_prepared.parquet`/`poi_features.parquet` from scratch."""

    poi_ids: npt.NDArray[np.object_]
    lat: FloatArray
    lon: FloatArray
    h3_cell: npt.NDArray[np.object_]
    pop_pct: FloatArray
    rating_shrunk: FloatArray
    poi_terms: list[frozenset[str]]
    embeddings: npt.NDArray[np.float32]
    archetype_top_by_segment: dict[int, list[str]]


def _poi_terms(tags: list[str], category: str) -> frozenset[str]:
    return frozenset({*tags, category})


def build_destination_indices(
    pois_df: pd.DataFrame, poi_features_df: pd.DataFrame, n_segments: int
) -> dict[str, DestinationIndex]:
    """One `DestinationIndex` per destination in `pois_df` (`pois_prepared.parquet`).

    `archetype_top_by_segment[seg]` is the FULL destination-ranked poi_id list for
    observable K-Means segment `seg` (ranked by `poi_features.parquet`'s
    `behav_archetype_affinity_{seg:02d}` desc, `rating_shrunk` desc, `poi_id` asc as
    tie-breaks) -- `channel_archetype` slices its own quota off the front, so this is
    computed once here rather than per trip.
    """
    emb_cols = sorted(c for c in poi_features_df.columns if c.startswith(TEXT_EMB_PREFIX))
    affinity_cols = {i: f"behav_archetype_affinity_{i:02d}" for i in range(n_segments)}
    pf_by_id = poi_features_df.set_index("poi_id")

    out: dict[str, DestinationIndex] = {}
    for dest, group in pois_df.groupby("destination", sort=True):
        group = group.reset_index(drop=True)
        poi_ids = group["poi_id"].to_numpy(dtype=object)
        pf_group = pf_by_id.loc[poi_ids]
        rating = group["rating_shrunk"].to_numpy(dtype=float)

        archetype_top: dict[int, list[str]] = {}
        for seg, col in affinity_cols.items():
            aff = pf_group[col].to_numpy(dtype=float)
            order = np.lexsort((poi_ids, -rating, -aff))
            archetype_top[seg] = poi_ids[order].tolist()

        out[str(dest)] = DestinationIndex(
            poi_ids=poi_ids,
            lat=group["lat"].to_numpy(dtype=float),
            lon=group["lon"].to_numpy(dtype=float),
            h3_cell=group["h3_cell"].to_numpy(dtype=object),
            pop_pct=group["pop_pct"].to_numpy(dtype=float),
            rating_shrunk=rating,
            poi_terms=[
                _poi_terms(list(t), str(c))
                for t, c in zip(group["tags"], group["category"], strict=True)
            ],
            embeddings=pf_group[emb_cols].to_numpy(dtype=np.float32),
            archetype_top_by_segment=archetype_top,
        )
    return out


# -----------------------------------------------------------------------------------
# Geo channel
# -----------------------------------------------------------------------------------


def channel_geo(
    idx: DestinationIndex,
    stay_lat: float,
    stay_lon: float,
    mobility: str,
    cfg: GeoChannelConfig,
) -> list[str]:
    """H3 k-ring from the stay point (coarse candidate gathering over
    `pois_prepared.h3_cell`) followed by an exact haversine cutoff at the literal
    mobility-conditioned radius (spec.md section 7) -- see module docstring for why
    the k-ring alone is not treated as the final radius decision. Nearest-first,
    truncated to `cfg.quota`."""
    radius_km = cfg.radius_km_for_mobility(mobility)
    edge_km = h3.edge_length(cfg.h3_resolution, unit="km")
    k = max(1, math.ceil(radius_km / edge_km)) + _GEO_KRING_SAFETY_MARGIN
    stay_cell = h3.geo_to_h3(stay_lat, stay_lon, cfg.h3_resolution)
    ring_cells = h3.k_ring(stay_cell, k)

    in_ring_mask = np.isin(idx.h3_cell, list(ring_cells))
    if not in_ring_mask.any():
        return []

    dist = haversine_km(idx.lat[in_ring_mask], idx.lon[in_ring_mask], stay_lat, stay_lon)
    within_radius = dist <= radius_km
    if not within_radius.any():
        return []

    cand_ids = idx.poi_ids[in_ring_mask][within_radius]
    cand_dist = dist[within_radius]
    order = np.lexsort((cand_ids, cand_dist))  # primary: distance asc, tie-break poi_id asc
    return _poi_id_list(cand_ids[order][: cfg.quota])


# -----------------------------------------------------------------------------------
# Interest/category channel
# -----------------------------------------------------------------------------------


def channel_interest(idx: DestinationIndex, interests: set[str], quota: int) -> list[str]:
    """POIs whose `{category} u {tags}` set intersects the traveler's stated
    interests (spec.md section 7), ranked by `rating_shrunk` desc (tie-break poi_id
    asc), truncated to `quota`."""
    if not interests:
        return []
    mask = np.array([bool(interests & terms) for terms in idx.poi_terms])
    if not mask.any():
        return []
    cand_ids = idx.poi_ids[mask]
    cand_rating = idx.rating_shrunk[mask]
    order = np.lexsort((cand_ids, -cand_rating))  # primary: rating desc, tie-break poi_id asc
    return _poi_id_list(cand_ids[order][:quota])


# -----------------------------------------------------------------------------------
# Semantic channel
# -----------------------------------------------------------------------------------


def compute_semantic_similarity(
    taste_vec: FloatArray, dest_embeddings: npt.NDArray[np.float32]
) -> FloatArray:
    """`cos(taste_t, poi_emb)` for every POI in one destination, brute-force (no
    faiss -- spec.md section 7 explicitly forbids ANN at this scale). POI embeddings
    are already L2-normalized (guaranteed by `text_embed.py`), so this reduces to a
    single matrix-vector dot product once `taste_vec` is unit-normalized. A
    zero-magnitude `taste_vec` (a traveler with no as-of-safe interaction history --
    ~79% of trips on the committed dataset, see docs/DATA_CARD.md) yields all-zero
    similarity for every POI -- a documented, graceful degeneracy, not an error; the
    archetype-prior channel is this system's designated cold-start path."""
    norm = float(np.linalg.norm(taste_vec))
    if norm == 0.0:
        return np.zeros(dest_embeddings.shape[0], dtype=np.float64)
    unit = taste_vec / norm
    result: FloatArray = dest_embeddings.astype(np.float64) @ unit
    return result


def channel_semantic(idx: DestinationIndex, sims: FloatArray, quota: int) -> list[str]:
    """Top-`quota` POIs by `sims` (already destination-scoped, row-aligned with
    `idx.poi_ids`), tie-broken by poi_id for determinism."""
    order = np.lexsort((idx.poi_ids, -sims))
    return _poi_id_list(idx.poi_ids[order][:quota])


# -----------------------------------------------------------------------------------
# Collaborative (item-item kNN) channel
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemItemCF:
    """Global item-item co-interaction cosine-similarity matrix, fit once over the
    full `interactions_train.parquet` window."""

    poi_index: dict[str, int]
    poi_ids: list[str]
    similarity: npt.NDArray[np.float32]  # (n_pois, n_pois), zero diagonal


def build_item_item_cf(pois_df: pd.DataFrame, interactions_train: pd.DataFrame) -> ItemItemCF:
    """Item-item collaborative-filtering similarity matrix.

    **"Co-interaction" defined**: two POIs co-interact if the SAME traveler
    positively engaged (`label >= 1`) with both, anywhere in the train window (not
    restricted to the same slate/session) -- the standard user-item -> item-item
    cosine-similarity construction (binary engagement matrix `R`, item similarity =
    column-cosine of `R`).

    **Leakage discipline**: this global similarity matrix is fit ONCE over the FULL
    `interactions_train` window, mirroring `poi_features.py::build_behavioral_block`'s
    established precedent (POI-level behavioral aggregates already use the full train
    window, not a per-trip as-of cutoff -- see docs/DATA_CARD.md resolved ambiguity
    #18's sibling reasoning). The temporal-leakage-safety obligation for THIS phase
    (spec.md's "reuse the same temporal as-of-cutoff discipline... do not leak")
    applies to the per-trip SEED passed into `channel_cf`, not to this matrix -- the
    seed is the traveler's own as-of-safe engaged history, built via
    `traveler_history_before` (reused directly from `features/traveler_features.py`,
    never reimplemented). `interactions_train.parquet` also structurally contains
    zero interactions for any holdout trip (verified: 0 of 121,360 rows reference a
    holdout `trip_id`), so for every trip actually evaluated by
    `candidates/recall_metrics.py` (which reads `interactions_holdout_random.parquet`
    -- holdout trips only) this matrix carries no information about that trip's own
    session at all.
    """
    canonical_map = build_poi_id_canonical_map(pois_df)
    remapped = remap_interaction_poi_ids(interactions_train, canonical_map)
    engaged = remapped.loc[remapped["label"] >= 1]

    poi_ids = pois_df["poi_id"].tolist()
    poi_index = {pid: i for i, pid in enumerate(poi_ids)}

    traveler_ids = sorted(engaged["traveler_id"].unique())
    if not traveler_ids:
        return ItemItemCF(
            poi_index=poi_index,
            poi_ids=poi_ids,
            similarity=np.zeros((len(poi_ids), len(poi_ids)), dtype=np.float32),
        )
    traveler_index = {tid: i for i, tid in enumerate(traveler_ids)}

    rows = engaged["traveler_id"].map(traveler_index).to_numpy()
    cols = engaged["poi_id"].map(poi_index).to_numpy()
    data = np.ones(len(rows), dtype=np.float32)
    r = csr_matrix((data, (rows, cols)), shape=(len(traveler_ids), len(poi_ids)))
    r.data[:] = 1.0  # binarize repeated engagements with the same POI

    co_occurrence = (r.T @ r).toarray().astype(np.float32)
    col_norms = np.sqrt(np.diag(co_occurrence))
    denom = np.outer(col_norms, col_norms)
    denom[denom == 0.0] = 1.0
    similarity = (co_occurrence / denom).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    return ItemItemCF(poi_index=poi_index, poi_ids=poi_ids, similarity=similarity)


def channel_cf(
    idx: DestinationIndex,
    cf: ItemItemCF,
    seed_poi_ids: list[str],
    cfg: CollaborativeChannelConfig,
) -> list[str]:
    """Item-item kNN: for each of the traveler's own as-of-safe, positively-engaged
    seed POIs, look up its `cfg.top_k_neighbors` nearest items by co-interaction
    cosine similarity, restrict to this trip's destination, exclude the seeds
    themselves, and rank by summed similarity across seeds. Correctly, and
    intentionally, returns an EMPTY list for a traveler with no prior engaged history
    (their first trip -- ~79% of trips on the committed dataset) -- the
    archetype-prior channel is this system's designated cold-start path per spec.md
    section 12, not this one; see docs/DATA_CARD.md."""
    seed_idx = [cf.poi_index[pid] for pid in seed_poi_ids if pid in cf.poi_index]
    if not seed_idx:
        return []

    dest_poi_set = set(idx.poi_ids.tolist())
    seed_set = set(seed_poi_ids)
    scores: dict[str, float] = {}
    for si in seed_idx:
        row = cf.similarity[si]
        nn_idx = np.argsort(-row, kind="stable")[: cfg.top_k_neighbors]
        for ni in nn_idx:
            sim = float(row[ni])
            if sim <= 0.0:
                continue
            pid = cf.poi_ids[ni]
            if pid in dest_poi_set and pid not in seed_set:
                scores[pid] = scores.get(pid, 0.0) + sim

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pid for pid, _ in ordered[: cfg.quota]]


# -----------------------------------------------------------------------------------
# Long-tail exploration channel (HARD FLOOR)
# -----------------------------------------------------------------------------------


def channel_longtail(
    idx: DestinationIndex,
    sims: FloatArray,
    is_cold_start: bool,
    cfg: LongtailChannelConfig,
    rng: np.random.Generator,
) -> list[str]:
    """`pop_pct < cfg.pop_pct_cutoff` AND `sims > cfg.semantic_sim_threshold`,
    epsilon-greedy sampled (spec.md section 7): `round(cfg.epsilon * n_target)` POIs
    drawn uniformly at random from the qualifying pool, the rest filled by semantic
    score rank -- disjoint by construction (the random draw pool excludes the
    already-selected top-ranked POIs), so no in-channel duplicates are possible.

    **Cold-start fallback**: for a zero-taste-vector traveler (`is_cold_start`),
    `compute_semantic_similarity` returns an identical 0.0 for every POI in the
    destination, so a literal `sims > 0.0` filter would exclude every POI and starve
    this channel's HARD FLOOR for ~79% of trips. Documented fix: the semantic filter
    is skipped entirely for cold-start travelers (only the popularity filter
    applies), and the "rest by semantic score" ranking degenerates to a stable
    poi_id-order tie-break (all scores equal) -- exploration is genuinely uniform for
    these travelers, which is arguably the more honest behavior for a traveler with
    no taste signal at all.
    """
    pop_mask = idx.pop_pct < cfg.pop_pct_cutoff
    qualifying_mask = pop_mask if is_cold_start else pop_mask & (sims > cfg.semantic_sim_threshold)

    qualifying_idx = np.where(qualifying_mask)[0]
    if len(qualifying_idx) == 0:
        return []

    order = qualifying_idx[np.lexsort((idx.poi_ids[qualifying_idx], -sims[qualifying_idx]))]
    n_target = min(cfg.quota, len(order))
    n_random = int(round(cfg.epsilon * n_target))
    n_top = n_target - n_random

    top_selected = order[:n_top]
    remaining_pool = order[n_top:]
    if n_random > 0 and len(remaining_pool) > 0:
        n_draw = min(n_random, len(remaining_pool))
        random_selected = rng.choice(remaining_pool, size=n_draw, replace=False)
    else:
        random_selected = np.array([], dtype=order.dtype)

    chosen = np.concatenate([top_selected, random_selected])
    return _poi_id_list(idx.poi_ids[chosen])


# -----------------------------------------------------------------------------------
# Archetype-prior channel (cold-start path, spec.md section 12)
# -----------------------------------------------------------------------------------


def channel_archetype(idx: DestinationIndex, segment: int, quota: int) -> list[str]:
    """Top-`quota` POIs for the traveler's observable K-Means segment (see
    `build_destination_indices`), by `behav_archetype_affinity_{segment:02d}`.
    Works unconditionally for any traveler with a stated `interests`/`budget`/
    `party_type`/`touristiness_pref` -- the segment assignment (reused directly from
    `features.traveler_features.assign_traveler_segments`) never reads interaction
    history, so this channel is non-empty even for a genuine zero-interaction
    cold-start traveler (spec.md section 12's designated cold-start path)."""
    return idx.archetype_top_by_segment.get(segment, [])[:quota]
