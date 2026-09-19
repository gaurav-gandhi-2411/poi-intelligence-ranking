"""Orchestrates the full synthetic data-generation pipeline end to end.

Single seeded `np.random.Generator` threaded through every stage in a fixed call
order (never scattered `np.random.seed()` calls) so two runs are byte-identical
(spec.md section 14, tested by tests/test_determinism.py).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.datagen.catalog import apply_catalog_dirtiness, generate_all_catalogs
from poi_rank.datagen.config import DatagenConfig, UtilityWeights
from poi_rank.datagen.exposure import build_biased_slate, build_random_slate
from poi_rank.datagen.interactions import INTERACTION_LABELS, simulate_slate_choices
from poi_rank.datagen.oracle_export import (
    oracle_dir_from_output,
    write_holdout_utility,
    write_poi_latent,
    write_term_standardization,
    write_traveler_taste,
)
from poi_rank.datagen.taxonomy import CATEGORY_INDEX
from poi_rank.datagen.timeline import build_timeline
from poi_rank.datagen.travelers import generate_travelers, generate_trips
from poi_rank.datagen.utility import (
    TermStandardization,
    compute_term_standardization,
    novelty_array,
    true_utility_noise_free,
)


@dataclass(frozen=True)
class CatalogArrays:
    """Numpy views over the true POI catalog, built once and reused per trip."""

    poi_id: npt.NDArray[np.str_]
    destination: npt.NDArray[np.str_]
    category: npt.NDArray[np.str_]
    subcategory: npt.NDArray[np.str_]
    lat: npt.NDArray[np.float64]
    lon: npt.NDArray[np.float64]
    popularity_raw: npt.NDArray[np.float64]
    created_at: npt.NDArray[np.datetime64]
    latent_localness: npt.NDArray[np.float64]
    latent_quality: npt.NDArray[np.float64]
    price_level_true: npt.NDArray[np.float64]
    wheelchair: npt.NDArray[np.bool_]
    stroller: npt.NDArray[np.bool_]
    kid_friendly: npt.NDArray[np.bool_]
    poi_semantic: npt.NDArray[np.float64]
    category_taste_idx: npt.NDArray[np.intp]


def build_catalog_arrays(poi_true_df: pd.DataFrame) -> CatalogArrays:
    return CatalogArrays(
        poi_id=poi_true_df["poi_id"].to_numpy(),
        destination=poi_true_df["destination"].to_numpy(),
        category=poi_true_df["category"].to_numpy(),
        subcategory=poi_true_df["subcategory"].to_numpy(),
        lat=poi_true_df["lat"].to_numpy(dtype=float),
        lon=poi_true_df["lon"].to_numpy(dtype=float),
        popularity_raw=poi_true_df["popularity_raw"].to_numpy(dtype=float),
        created_at=poi_true_df["created_at"].to_numpy(),
        latent_localness=poi_true_df["latent_localness"].to_numpy(dtype=float),
        latent_quality=poi_true_df["latent_quality"].to_numpy(dtype=float),
        price_level_true=poi_true_df["price_level"].to_numpy(dtype=float),
        wheelchair=poi_true_df["wheelchair"].to_numpy(dtype=bool),
        stroller=poi_true_df["stroller"].to_numpy(dtype=bool),
        kid_friendly=poi_true_df["kid_friendly"].to_numpy(dtype=bool),
        poi_semantic=np.stack(poi_true_df["poi_semantic"].to_numpy()),
        category_taste_idx=np.array(
            [CATEGORY_INDEX[c] for c in poi_true_df["category"]], dtype=np.intp
        ),
    )


@dataclass
class _TravelerHistory:
    seen_poi_ids: set[str] = field(default_factory=set)
    seen_category_subcategory: set[tuple[str, str]] = field(default_factory=set)


class _SlateCounter:
    def __init__(self) -> None:
        self._n = 0

    def next_id(self) -> str:
        self._n += 1
        return f"SLT{self._n:07d}"


def _eligible_indices(
    arrays: CatalogArrays, destination: str, as_of: pd.Timestamp
) -> npt.NDArray[np.intp]:
    mask = (arrays.destination == destination) & (arrays.created_at <= as_of)
    return np.where(mask)[0]


def _noise_free_utility_for_indices(
    weights: UtilityWeights,
    standardization: TermStandardization,
    destination: str,
    taste_vec: npt.NDArray[np.float64],
    touristiness_pref: float,
    party_type: str,
    budget: str,
    arrays: CatalogArrays,
    idx: npt.NDArray[np.intp],
    novelty: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return true_utility_noise_free(
        weights=weights,
        standardization=standardization,
        destination=destination,
        taste_vec=taste_vec,
        touristiness_pref=touristiness_pref,
        party_type=party_type,
        budget=budget,
        poi_semantic=arrays.poi_semantic[idx],
        poi_category_taste_idx=arrays.category_taste_idx[idx],
        poi_latent_localness=arrays.latent_localness[idx],
        poi_latent_quality=arrays.latent_quality[idx],
        poi_wheelchair=arrays.wheelchair[idx],
        poi_stroller=arrays.stroller[idx],
        poi_kid_friendly=arrays.kid_friendly[idx],
        poi_category=arrays.category[idx],
        poi_price_level_true=arrays.price_level_true[idx],
        novelty=novelty,
    )


def _sample_ordered_lead_days(
    rng: np.random.Generator, n_slates: int, lead_days_min: int, lead_days_max: int
) -> npt.NDArray[np.intp]:
    """Block A RC3.1 (docs/DATA_CARD.md "DGP remediation, Block A"): draw `n_slates`
    DISTINCT lead-day offsets (when the range allows it) and return them sorted
    DESCENDING -- i.e. slate generation order == real chronological order within a
    trip's own browsing session (earliest calendar date generated first, closest to
    the reference date generated last). This is what makes a per-impression as-of
    cutoff genuinely unambiguous: two slates of the SAME trip never tie on calendar
    day AND disagree with generation order, unlike the pre-remediation independent
    `rng.integers(0, 46)` draw per slate."""
    span = lead_days_max - lead_days_min + 1
    if n_slates <= span:
        offsets = rng.choice(span, size=n_slates, replace=False)
    else:
        offsets = rng.choice(span, size=n_slates, replace=True)
    lead_days = np.sort(offsets)[::-1] + lead_days_min
    result: npt.NDArray[np.intp] = lead_days.astype(np.intp)
    return result


def _generate_policy_slates(
    rng: np.random.Generator,
    policy: str,
    n_slates: int,
    slate_size: int,
    arrays: CatalogArrays,
    eligible_idx: npt.NDArray[np.intp],
    stay_lat: float,
    stay_lon: float,
    trip_id: str,
    traveler_id: str,
    session_base_ts: pd.Timestamp,
    cfg: DatagenConfig,
    utility_noise_free_eligible: npt.NDArray[np.float64],
    counter: _SlateCounter,
    lead_days_min: int = 0,
    lead_days_max: int = 45,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Generate `n_slates` slates of a given policy for one trip. Returns rows plus
    the set of POI ids that received a real engagement (used to update novelty).

    `slate_size` is now an explicit parameter (Block A RC1, docs/DATA_CARD.md "DGP
    remediation, Block A") -- callers pass `cfg.slate.slate_size` for biased slates
    and `cfg.slate.random_holdout_slate_size` for the primary random-holdout policy,
    rather than this function reading one fixed config field internally.

    `lead_days_min`/`lead_days_max` (Block A RC3.1/RC3.2) let a caller generate a
    PRE-TRIP history batch (dated well before the trip's own in-trip sessions,
    default `(0, 45)`) using this exact same machinery, never a parallel one.
    """
    rows: list[dict[str, Any]] = []
    engaged_poi_ids: set[str] = set()
    sigma = cfg.noise.sigma
    if policy == "random":
        tau = cfg.choice.random_holdout_tau
        engage_lambda = cfg.interaction_generation.random_holdout_engage_lambda
        dismiss_lambda = cfg.interaction_generation.random_holdout_dismiss_lambda
    else:
        tau = cfg.choice.tau
        engage_lambda = cfg.interaction_generation.engage_lambda
        dismiss_lambda = cfg.interaction_generation.dismiss_lambda
    rank_decay = cfg.interaction_generation.rank_decay
    lead_days_arr = _sample_ordered_lead_days(rng, n_slates, lead_days_min, lead_days_max)

    for slate_num in range(n_slates):
        if policy == "biased":
            slate = build_biased_slate(
                rng,
                arrays.popularity_raw[eligible_idx],
                arrays.lat[eligible_idx],
                arrays.lon[eligible_idx],
                stay_lat,
                stay_lon,
                slate_size,
                cfg,
            )
        else:
            slate = build_random_slate(rng, len(eligible_idx), slate_size)

        slate_global_idx = eligible_idx[slate.indices]
        u_true = utility_noise_free_eligible[slate.indices]
        u_choice = u_true + rng.normal(0, sigma, size=len(u_true))

        lead_days = int(lead_days_arr[slate_num])
        timestamp = max(session_base_ts - pd.Timedelta(days=lead_days), pd.Timestamp("2000-01-01"))
        slate_id = counter.next_id()

        outcomes = simulate_slate_choices(
            rng, u_choice, tau, engage_lambda, dismiss_lambda, rank_decay
        )
        outcome_map = dict(outcomes)

        for local_pos in range(len(slate_global_idx)):
            gidx = slate_global_idx[local_pos]
            poi_id = str(arrays.poi_id[gidx])
            itype = outcome_map.get(local_pos, "view")
            label = INTERACTION_LABELS[itype]
            if label >= 1:
                engaged_poi_ids.add(poi_id)
            rows.append(
                {
                    "traveler_id": traveler_id,
                    "trip_id": trip_id,
                    "poi_id": poi_id,
                    "slate_id": slate_id,
                    "interaction_type": itype,
                    "label": label,
                    "hard_negative": itype == "dismiss",
                    "timestamp": timestamp,
                    "position_in_slate": local_pos,
                    "p_expose": float(slate.p_expose[local_pos]),
                }
            )
    return rows, engaged_poi_ids


def run_generate(cfg: DatagenConfig, output_dir: Path) -> dict[str, Any]:
    """Run the full DGP and write all committed synthetic files + oracle export.

    Returns a summary dict with row counts and per-file SHA256 hashes, used both for
    the CLI's printed report and tests/test_determinism.py.
    """
    rng = np.random.default_rng(cfg.seed)
    timeline = build_timeline(cfg)

    poi_true_df = generate_all_catalogs(rng, cfg, timeline)
    poi_export_df = apply_catalog_dirtiness(rng, poi_true_df, cfg)
    arrays = build_catalog_arrays(poi_true_df)

    travelers_df, taste_vectors = generate_travelers(rng, cfg)
    trips_df = generate_trips(rng, travelers_df, cfg, timeline)
    trips_df["is_holdout"] = trips_df["start_date"] >= timeline.split

    # Block A RC2a: fit the utility-term standardization ONCE, before any
    # trip/slate simulation, over the full generation population (deterministic,
    # no interaction simulation needed -- docs/DATA_CARD.md "DGP remediation,
    # Block A").
    standardization = compute_term_standardization(poi_true_df, travelers_df, taste_vectors)

    n_train_trips = int((~trips_df["is_holdout"]).sum())
    n_holdout_trips = int(trips_df["is_holdout"].sum())
    train_slate_size = cfg.slate.slate_size
    # Block A RC1: the primary random holdout uses a WIDER slate size than the
    # biased train/secondary-logged-holdout slates (docs/DATA_CARD.md "DGP
    # remediation, Block A").
    random_holdout_slate_size = cfg.slate.random_holdout_slate_size
    logged_holdout_slate_size = cfg.slate.slate_size
    mean_slates_train = cfg.scale.target_train_impressions / max(
        n_train_trips * train_slate_size, 1
    )
    mean_slates_holdout_random = cfg.scale.target_holdout_random_impressions / max(
        n_holdout_trips * random_holdout_slate_size, 1
    )
    mean_slates_holdout_logged = cfg.scale.target_holdout_logged_impressions / max(
        n_holdout_trips * logged_holdout_slate_size, 1
    )

    traveler_row_by_id = travelers_df.set_index("traveler_id")
    counter = _SlateCounter()
    history: dict[str, _TravelerHistory] = {}

    train_rows: list[dict[str, Any]] = []
    holdout_random_rows: list[dict[str, Any]] = []
    holdout_logged_rows: list[dict[str, Any]] = []
    pretrip_rows: list[dict[str, Any]] = []
    oracle_utility_rows: list[dict[str, Any]] = []

    weights = cfg.utility_weights
    same_poi_pen = cfg.novelty.same_poi_repeat_penalty
    similar_pen = cfg.novelty.similar_category_repeat_penalty
    pretrip_cfg = cfg.pretrip_history

    for trip in trips_df.itertuples(index=False):
        traveler = traveler_row_by_id.loc[trip.traveler_id]
        taste_vec = taste_vectors[trip.traveler_id]
        hist = history.setdefault(trip.traveler_id, _TravelerHistory())

        eligible_idx = _eligible_indices(arrays, trip.destination, trip.start_date)
        novelty = novelty_array(
            arrays.poi_id[eligible_idx],
            arrays.category[eligible_idx],
            arrays.subcategory[eligible_idx],
            hist.seen_poi_ids,
            hist.seen_category_subcategory,
            same_poi_pen,
            similar_pen,
        )
        u_true_eligible = _noise_free_utility_for_indices(
            weights,
            standardization,
            trip.destination,
            taste_vec,
            traveler.touristiness_pref,
            traveler.party_type,
            traveler.budget,
            arrays,
            eligible_idx,
            novelty,
        )

        # Block A RC3.2: pre-trip synthetic interaction history, seeded once per
        # traveler on their FIRST trip, using the SAME exposure+choice machinery as
        # in-trip sessions (never a parallel mechanism) -- docs/DATA_CARD.md "DGP
        # remediation, Block A". Runs BEFORE this trip's own slates so `hist` (and
        # therefore this trip's own novelty, computed above from PRIOR trips only)
        # is unaffected -- pre-trip history informs LATER trips and, via the
        # per-impression as-of cutoff in `features/traveler_features.py`, this
        # same first trip's own feature row.
        if trip.trip_sequence == 1:
            is_cold_start_cohort = rng.random() < pretrip_cfg.cold_start_fraction
            if not is_cold_start_cohort:
                target_engaged = int(
                    rng.integers(pretrip_cfg.min_interactions, pretrip_cfg.max_interactions + 1)
                )
                n_pretrip_slates = max(
                    1, round(target_engaged / cfg.interaction_generation.engage_lambda)
                )
                pretrip_batch, _pretrip_engaged = _generate_policy_slates(
                    rng,
                    "biased",
                    n_pretrip_slates,
                    train_slate_size,
                    arrays,
                    eligible_idx,
                    trip.stay_lat,
                    trip.stay_lon,
                    trip.trip_id,
                    trip.traveler_id,
                    trip.start_date,
                    cfg,
                    u_true_eligible,
                    counter,
                    lead_days_min=pretrip_cfg.lead_days_min,
                    lead_days_max=pretrip_cfg.lead_days_max,
                )
                pretrip_rows.extend(pretrip_batch)

        newly_engaged: set[str] = set()

        if not trip.is_holdout:
            n_slates = rng.poisson(mean_slates_train)
            rows, engaged = _generate_policy_slates(
                rng,
                "biased",
                n_slates,
                train_slate_size,
                arrays,
                eligible_idx,
                trip.stay_lat,
                trip.stay_lon,
                trip.trip_id,
                trip.traveler_id,
                trip.start_date,
                cfg,
                u_true_eligible,
                counter,
            )
            train_rows.extend(rows)
            newly_engaged |= engaged
        else:
            n_slates_r = rng.poisson(mean_slates_holdout_random)
            rows_r, engaged_r = _generate_policy_slates(
                rng,
                "random",
                n_slates_r,
                random_holdout_slate_size,
                arrays,
                eligible_idx,
                trip.stay_lat,
                trip.stay_lon,
                trip.trip_id,
                trip.traveler_id,
                trip.start_date,
                cfg,
                u_true_eligible,
                counter,
            )
            holdout_random_rows.extend(rows_r)
            newly_engaged |= engaged_r

            n_slates_b = rng.poisson(mean_slates_holdout_logged)
            rows_b, engaged_b = _generate_policy_slates(
                rng,
                "biased",
                n_slates_b,
                logged_holdout_slate_size,
                arrays,
                eligible_idx,
                trip.stay_lat,
                trip.stay_lon,
                trip.trip_id,
                trip.traveler_id,
                trip.start_date,
                cfg,
                u_true_eligible,
                counter,
            )
            holdout_logged_rows.extend(rows_b)
            newly_engaged |= engaged_b

            for local_i, gidx in enumerate(eligible_idx):
                oracle_utility_rows.append(
                    {
                        "trip_id": trip.trip_id,
                        "traveler_id": trip.traveler_id,
                        "poi_id": str(arrays.poi_id[gidx]),
                        "utility_true": float(u_true_eligible[local_i]),
                    }
                )

        hist.seen_poi_ids |= newly_engaged
        for poi_id in newly_engaged:
            row_idx = np.where(arrays.poi_id == poi_id)[0][0]
            hist.seen_category_subcategory.add(
                (str(arrays.category[row_idx]), str(arrays.subcategory[row_idx]))
            )

    interactions_train = pd.DataFrame(train_rows)
    interactions_holdout_random = pd.DataFrame(holdout_random_rows)
    interactions_holdout_logged = pd.DataFrame(holdout_logged_rows)
    interactions_pretrip = pd.DataFrame(pretrip_rows)
    oracle_utility_df = pd.DataFrame(oracle_utility_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = oracle_dir_from_output(output_dir)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    def _write_parquet(df: pd.DataFrame, name: str) -> None:
        path = output_dir / name
        df.to_parquet(path, index=False)
        files[name] = path

    _write_parquet(poi_export_df, "pois.parquet")
    _write_parquet(travelers_df, "travelers.parquet")
    _write_parquet(trips_df, "trips.parquet")
    _write_parquet(interactions_train, "interactions_train.parquet")
    _write_parquet(interactions_holdout_random, "interactions_holdout_random.parquet")
    _write_parquet(interactions_holdout_logged, "interactions_holdout_logged.parquet")
    # Block A RC3.2: pre-trip history in its OWN file, never merged into
    # interactions_train.parquet -- keeps that file's existing "one row = one real
    # exposure event belonging to a TRAIN trip's own session" semantics and its
    # "zero holdout trip_id rows" invariant fully intact for downstream consumers
    # (docs/DATA_CARD.md "DGP remediation, Block A"). Read only by
    # `features/traveler_features.py`, for history purposes.
    _write_parquet(interactions_pretrip, "interactions_pretrip.parquet")

    destinations_df = pd.DataFrame(
        [{"destination": d} for d in sorted(trips_df["destination"].unique())]
    )
    _write_parquet(destinations_df, "destinations.parquet")

    oracle_paths = {
        "traveler_taste.parquet": write_traveler_taste(oracle_dir, taste_vectors),
        "poi_latent.parquet": write_poi_latent(oracle_dir, poi_true_df),
        "holdout_utility_true.parquet": write_holdout_utility(oracle_dir, oracle_utility_df),
        "term_standardization.json": write_term_standardization(oracle_dir, standardization),
    }
    for name, path in oracle_paths.items():
        files[f"{oracle_dir.name}/{name}"] = path

    sha256 = {name: _sha256_of_file(path) for name, path in files.items()}

    counts = {
        "n_destinations": int(trips_df["destination"].nunique()),
        "n_pois": len(poi_export_df),
        "n_travelers": len(travelers_df),
        "n_trips": len(trips_df),
        "n_train_trips": n_train_trips,
        "n_holdout_trips": n_holdout_trips,
        "n_train_impressions": len(interactions_train),
        "n_holdout_random_impressions": len(interactions_holdout_random),
        "n_holdout_logged_impressions": len(interactions_holdout_logged),
        "n_pretrip_impressions": len(interactions_pretrip),
        "n_pretrip_positives": int((interactions_pretrip["label"] > 0).sum())
        if len(interactions_pretrip)
        else 0,
        "n_train_positives": int((interactions_train["label"] > 0).sum())
        if len(interactions_train)
        else 0,
        "n_holdout_random_positives": (
            int((interactions_holdout_random["label"] > 0).sum())
            if len(interactions_holdout_random)
            else 0
        ),
        "n_holdout_logged_positives": (
            int((interactions_holdout_logged["label"] > 0).sum())
            if len(interactions_holdout_logged)
            else 0
        ),
    }
    counts["n_total_positives"] = (
        counts["n_train_positives"]
        + counts["n_holdout_random_positives"]
        + counts["n_holdout_logged_positives"]
    )

    return {"files": files, "sha256": sha256, "counts": counts}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
