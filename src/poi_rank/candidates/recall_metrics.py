"""Candidate-recall metrics (spec.md sections 7 / 11.10) -- the measurable answer to
"how do we avoid killing the long tail." Ground truth is
`interactions_holdout_random.parquet`'s `label >= 1` rows: POIs a traveler was
independently, later, observed to positively engage with under a UNIFORM-RANDOM
exposure policy. This is standard, non-circular recall@k methodology -- checking
whether candidate generation would have included a POI the traveler was later
observed to like is not latent information and reads nothing from the oracle-only
export directory.

Three reported numbers (all against the full candidate union):

1. **Overall `candidate_recall@250`** -- mean per-trip recall over all holdout trips
   with >= 1 relevant POI.
2. **Long-tail-stratum `candidate_recall@250`** -- the same computation, but the
   "relevant" set for each trip is restricted to POIs in the bottom-50%
   within-destination popularity stratum (spec.md's exact requirement).
3. **Per-channel marginal recall** (leave-one-channel-out): `recall_full -
   recall_without_channel`, computed post-hoc from the ONE `candidates.parquet`
   artifact's boolean membership columns -- no channel needs to be regenerated to
   compute this.

`interactions_holdout_random.parquet`'s `poi_id`s reference the same pre-dedup raw
catalog as `interactions_train.parquet` (verified: 54 of 1,498 distinct holdout
poi_ids are not standalone rows in `pois_prepared.parquet` -- merged away by Phase
2's dedup) -- every function here canonical-remaps the holdout log before comparing
against `candidates.parquet`'s (already-canonical) poi_ids, via the same
`features.reconcile` machinery Phase 3 established, never a bespoke remap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from poi_rank.candidates.channels import CHANNEL_NAMES
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids


@dataclass(frozen=True)
class RecallResult:
    """Mean per-trip candidate recall, plus bookkeeping on how many trips had no
    ground-truth-relevant POI to check against (excluded from the mean, reported --
    never silently dropped)."""

    recall_mean: float
    n_trips_evaluated: int
    n_trips_excluded_no_relevant: int


def _relevant_by_trip(
    holdout_random: pd.DataFrame,
    pop_pct_by_poi: dict[str, float] | None = None,
    long_tail_cutoff: float | None = None,
) -> dict[str, set[str]]:
    """`{trip_id: {poi_id, ...}}` of `label >= 1` holdout POIs per trip (already
    canonical-remapped by the caller). If `pop_pct_by_poi`/`long_tail_cutoff` are
    given, each trip's relevant set is restricted to POIs at or below the popularity
    cutoff -- the "long-tail stratum" restriction."""
    positives = holdout_random.loc[holdout_random["label"] >= 1]
    out: dict[str, set[str]] = {}
    for trip_id, group in positives.groupby("trip_id"):
        poi_ids = set(group["poi_id"])
        if pop_pct_by_poi is not None and long_tail_cutoff is not None:
            poi_ids = {
                pid
                for pid in poi_ids
                if pop_pct_by_poi.get(pid) is not None and pop_pct_by_poi[pid] < long_tail_cutoff
            }
        out[str(trip_id)] = poi_ids
    return out


def _candidate_set_by_trip(
    candidates_df: pd.DataFrame, exclude_channel: str | None = None
) -> dict[str, set[str]]:
    """`{trip_id: {poi_id, ...}}` union candidate set. If `exclude_channel` is given,
    a POI stays in the set as long as ANY of the OTHER 5 channels still flags it
    (leave-one-channel-out) -- never regenerates a channel, only re-unions the
    existing boolean columns."""
    channels = [c for c in CHANNEL_NAMES if c != exclude_channel]
    membership = candidates_df[channels].any(axis=1)
    kept = candidates_df.loc[membership, ["trip_id", "poi_id"]]
    out: dict[str, set[str]] = {}
    for trip_id, group in kept.groupby("trip_id"):
        out[str(trip_id)] = set(group["poi_id"])
    return out


def candidate_recall_at_k(
    relevant_by_trip: dict[str, set[str]], candidate_set_by_trip: dict[str, set[str]]
) -> RecallResult:
    """Mean per-trip recall over trips with >= 1 relevant POI."""
    per_trip: list[float] = []
    n_excluded = 0
    for trip_id, relevant in relevant_by_trip.items():
        if not relevant:
            n_excluded += 1
            continue
        candidates = candidate_set_by_trip.get(trip_id, set())
        hit = len(relevant & candidates)
        per_trip.append(hit / len(relevant))
    mean = float(np.mean(per_trip)) if per_trip else 0.0
    return RecallResult(
        recall_mean=mean, n_trips_evaluated=len(per_trip), n_trips_excluded_no_relevant=n_excluded
    )


def _canonical_holdout(pois_df: pd.DataFrame, holdout_random: pd.DataFrame) -> pd.DataFrame:
    canonical_map = build_poi_id_canonical_map(pois_df)
    return remap_interaction_poi_ids(holdout_random, canonical_map)


def overall_and_longtail_recall(
    pois_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    holdout_random: pd.DataFrame,
    long_tail_cutoff: float = 0.5,
) -> dict[str, RecallResult]:
    """Overall `candidate_recall@250` and the bottom-50%-popularity-stratum recall
    (spec.md's exact requirement), both against the full candidate union."""
    holdout = _canonical_holdout(pois_df, holdout_random)
    pop_pct_by_poi = dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True))

    full_candidates = _candidate_set_by_trip(candidates_df)
    relevant_all = _relevant_by_trip(holdout)
    relevant_longtail = _relevant_by_trip(holdout, pop_pct_by_poi, long_tail_cutoff)

    return {
        "overall": candidate_recall_at_k(relevant_all, full_candidates),
        "long_tail": candidate_recall_at_k(relevant_longtail, full_candidates),
    }


def marginal_recall_per_channel(
    pois_df: pd.DataFrame, candidates_df: pd.DataFrame, holdout_random: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Leave-one-channel-out marginal OVERALL recall@250 for each enabled channel:
    `recall_full - recall_without_channel`. Reported for every channel regardless of
    the result -- spec.md section 7 says a channel with ~0 marginal recall "gets
    deleted"; this function's job is only to measure and report honestly, never to
    curve-fit by adjusting channel logic to inflate the number."""
    holdout = _canonical_holdout(pois_df, holdout_random)
    relevant_all = _relevant_by_trip(holdout)
    full = candidate_recall_at_k(relevant_all, _candidate_set_by_trip(candidates_df)).recall_mean

    out: dict[str, dict[str, float]] = {}
    for channel in CHANNEL_NAMES:
        without = candidate_recall_at_k(
            relevant_all, _candidate_set_by_trip(candidates_df, exclude_channel=channel)
        ).recall_mean
        out[channel] = {
            "recall_full": full,
            "recall_without_channel": without,
            "marginal_recall": full - without,
        }
    return out


# Popularity strata for the per-stratum chance-lift table (within-destination `pop_pct`).
# "long_tail" is the bottom-50% stratum used by the gate; the quartiles show where recall
# is lost. Cut-points are fixed constants, not tuned.
POPULARITY_STRATA: dict[str, tuple[float, float]] = {
    "q1_least_popular": (0.0, 0.25),
    "q2": (0.25, 0.5),
    "q3": (0.5, 0.75),
    "q4_most_popular": (0.75, 1.0000001),
    "long_tail": (0.0, 0.5),
    "overall": (0.0, 1.0000001),
}


def recall_with_chance_lift(
    pois_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    holdout_random: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Per-stratum candidate recall against EXPOSED holdout positives (uniform-random
    exposure, so `label >= 1` is an unbiased positive draw), each with its OWN chance
    baseline: for a trip, the chance recall of a stratum is the share of that stratum's
    destination POIs that the candidate set contains -- what a random candidate set of the
    same size would recall. `lift = recall - chance` (absolute), per-stratum, because a
    pooled chance baseline hides strata where the channels are near chance (the failure mode
    that survived four phases unnoticed).
    """
    holdout = _canonical_holdout(pois_df, holdout_random)
    positives = holdout.loc[holdout["label"] >= 1]
    pop = dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True))
    dest = dict(zip(pois_df["poi_id"], pois_df["destination"], strict=True))
    poi_ids = pois_df["poi_id"].to_numpy()
    poi_pop = pois_df["pop_pct"].to_numpy(dtype=float)
    poi_dest = pois_df["destination"].to_numpy()
    cand_sets = _candidate_set_by_trip(candidates_df)
    pos_by_trip = {str(t): set(g["poi_id"]) for t, g in positives.groupby("trip_id")}

    out: dict[str, dict[str, float]] = {}
    for name, (lo, hi) in POPULARITY_STRATA.items():
        recalls: list[float] = []
        chances: list[float] = []
        for trip_id, relevant in pos_by_trip.items():
            cands = cand_sets.get(trip_id, set())
            rel = {p for p in relevant if lo <= pop[p] < hi}
            if not rel:
                continue
            trip_dest = dest[next(iter(rel))]
            in_stratum = poi_ids[(poi_dest == trip_dest) & (poi_pop >= lo) & (poi_pop < hi)]
            chance = sum(1 for p in in_stratum if p in cands) / len(in_stratum)
            recalls.append(len(rel & cands) / len(rel))
            chances.append(chance)
        recall = float(np.mean(recalls))
        chance_mean = float(np.mean(chances))
        out[name] = {
            "recall": recall,
            "chance_recall": chance_mean,
            "lift_abs": recall - chance_mean,
            "n_trips": float(len(recalls)),
        }
    return out
