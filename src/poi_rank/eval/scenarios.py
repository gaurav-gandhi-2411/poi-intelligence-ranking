"""The 3+1 required scenarios (spec.md section 15, brief section 17): `poi_rank.cli
scenarios` / `make scenarios`.

Constructs 4 hand-specified, SYNTHETIC traveler/trip profiles (not real travelers
from the committed dataset -- genuinely new inputs) and runs each through the
EXACT SAME live pipeline `poi_rank.cli recommend` runs for real holdout trips:
candidate generation (`candidates.union.generate_candidates`) -> LambdaMART+IPS
scoring + isotonic calibration + confidence (`scoring.output.run_scoring_pipeline`,
LOADING `artifacts/model.txt`, never retraining) -> compatibility/hard gates ->
multiplicative utility -> MMR diversity re-rank -> grouped-TreeSHAP explanations
(`explain.output_enrichment.enrich_recommend_result`). Writes
`results/scenarios/{1,2,3,4}.json` plus `results/scenarios/overlap_matrix.json`
(the pairwise top-10 Jaccard table spec.md section 15 asks for), and
`docs/RESULTS.md`'s scenarios section is generated from these files exactly the
same way `eval/report.py` already generates the rest of that document from
`results/metrics.json` -- no hand-typed numbers here either.

**Cold start by construction, not a bug to route around**: none of these 4
travelers/trips exist in `interactions_train.parquet`, so every implicit/
behavioral SIGNAL keyed by `(traveler_id, ...)` (implicit taste vector,
`implicit_interaction_count`, the collaborative-filtering channel's seed set) is
genuinely empty for all 4 -- exactly what a real new user of this system would
look like. `channel_archetype` (spec.md section 12's designated cold-start path)
and the stated-interest/geo/long-tail channels are this pipeline's real answer to
that cold start, not a special-cased fallback added here.

**Why 4 new small, backward-compatible parameters were added to 3 existing
modules, rather than a second parallel scoring implementation living only in
this file** (each documented at its own call site): `scoring.output
.run_scoring_pipeline` gained `holdout_frame_override`/`trips_df_override`/
`travelers_df_override`/`candidates_df_override` (all `None` by default,
zero behavior change for `poi_rank.cli recommend`); `explain.output_enrichment
.enrich_recommend_result` gained `pois_df_override`/`travelers_df_override`
(same discipline); `candidates.union.generate_candidates` gained
`segments_override` (needed so a synthetic traveler's K-Means segment ID means
the SAME thing as `poi_features.parquet`'s already-persisted
`behav_archetype_affinity_NN` columns -- see
`features.traveler_features.assign_traveler_segments_out_of_sample`'s own
docstring for the relabeling bug this avoids). Reusing the ONE real pipeline
this way means these 4 scenarios can never silently drift from what
`poi_rank.cli recommend` actually does for a real trip.

Destination: **seoul**, for all 4 scenarios (documented choice -- spec.md section
15 does not specify one; a single shared destination keeps the top-10 overlap
comparison across scenarios a clean apples-to-apples read, rather than
conflating "different traveler" with "different catalog"). Stay point: the
destination's own real POI centroid (mean lat/lon of every `pois_prepared
.parquet` row in that destination) -- the same "actual centroid, not a
hardcoded city-center constant" convention `data/geo_prep.py
::synthesize_transit_nodes` already establishes in this codebase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.candidates.retriever import (
    Retriever,
    RetrieverInputs,
    score_full_catalog,
    top_k_by_trip,
)
from poi_rank.candidates.union import generate_candidates
from poi_rank.eval.personalization import jaccard
from poi_rank.explain.output_enrichment import enrich_recommend_result
from poi_rank.features.config import BudgetTargetPriceLevel, FeatureBuildConfig
from poi_rank.features.text_embed import build_poi_text_embeddings
from poi_rank.features.traveler_features import (
    assemble_traveler_features,
    assign_traveler_segments_out_of_sample,
)
from poi_rank.models.config import ModelConfig
from poi_rank.models.ranking_data import build_ranking_frame
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import run_scoring_pipeline

SCENARIOS_DIRNAME = "scenarios"
OVERLAP_MATRIX_FILENAME = "overlap_matrix.json"

SCENARIO_DESTINATION = "seoul"
DIAGNOSTIC_BASE_SCENARIO = 1

# Trip logistics spec.md section 15 leaves unspecified for every scenario --
# applied uniformly, documented once here rather than per-scenario.
SCENARIO_TRIP_DURATION_DAYS = 4
SCENARIO_START_DATE_LEAD_DAYS = 30  # past the real dataset's own last observed trip


@dataclass(frozen=True)
class ScenarioProfile:
    """One spec.md section 15 traveler/trip profile, in exactly the field shape
    `datagen/travelers.py::generate_travelers`/`generate_trips` produce (so it
    flows through every downstream function with zero special-casing)."""

    number: int
    name: str
    interests: tuple[str, ...]
    touristiness_pref: float
    budget: str
    mobility: str
    party_type: str
    pace: str
    accessibility_needs: tuple[str, ...]
    notes: str


# spec.md section 15's interests are plain-English descriptions
# ("local food, neighborhoods"), not literal entries of the real stated-interest
# vocabulary (`datagen/taxonomy.py::INTEREST_LABELS = CATEGORIES + TAGS` -- the
# only vocabulary any real traveler's `interests` list is ever drawn from). Each
# is mapped onto the closest real label(s) below; every synthetic traveler's
# `interests` therefore only ever contains values a real traveler could also have
# (required for `explicit_interest_*`'s one-hot vocabulary, built from the real
# 31-label set, to line up with what `artifacts/model.txt` was trained on).
SCENARIO_PROFILES: tuple[ScenarioProfile, ...] = (
    ScenarioProfile(
        number=1,
        name="Local Experience",
        interests=("local", "foodie", "authentic"),
        touristiness_pref=-0.8,
        budget="medium",
        mobility="public_transport",
        party_type="solo",
        pace="moderate",
        accessibility_needs=(),
        notes=(
            "'local food' -> {local, foodie}; 'neighborhoods' -> {authentic} (shares "
            "the 'local' tag with the food interest -- an authentic-local-neighborhood "
            "preference is naturally the same underlying taste). `pace` is not "
            "specified by spec.md section 15 for this scenario -- defaulted to "
            "'moderate' (the middle of the 3-value PACE_ORDER scale)."
        ),
    ),
    ScenarioProfile(
        number=2,
        name="History & Architecture",
        interests=("historic_site", "historic", "cultural", "museum"),
        touristiness_pref=0.4,
        budget="high",
        mobility="public_transport",
        party_type="couple",
        pace="moderate",
        accessibility_needs=(),
        notes=(
            "'history' -> {historic_site, historic}; 'architecture' -> {cultural} "
            "(no literal 'architecture' tag exists in CATEGORIES+TAGS); "
            "'museums' -> {museum} (exact match). `pace` defaulted to 'moderate' "
            "(not specified)."
        ),
    ),
    ScenarioProfile(
        number=3,
        name="Family with young children",
        interests=("family_activity", "nature_park", "family-friendly"),
        touristiness_pref=0.0,
        budget="medium",
        mobility="car",
        party_type="family_young_kids",
        pace="relaxed",
        accessibility_needs=("stroller",),
        notes=(
            "'activities' -> {family_activity}; 'parks' -> {nature_park}; "
            "'interactive experiences' -> {family-friendly} (no literal 'interactive' "
            "tag exists). `touristiness_pref` and `mobility` are not specified by "
            "spec.md section 15 for this scenario -- defaulted to 0.0 (neutral) and "
            "'car' (a common real choice for a family with young children traveling "
            "with a stroller) respectively; `party_type`/accessibility "
            "(`stroller`)/`pace` (`relaxed`)/`budget` (`medium`) are exactly as "
            "spec.md states."
        ),
    ),
)

# spec.md section 15's 4th diagnostic scenario: "the same traveler profile" as one
# of the 3 above, with `touristiness_pref` flipped to the opposite extreme
# (+0.8 <-> -0.8), everything else held constant. Base chosen: Scenario 1 (Local
# Experience, touristiness_pref=-0.8) -- it already sits at the -0.8 extreme
# spec.md's own illustrative example uses, so flipping to +0.8 matches spec.md's
# literal "+0.8 -> -0.8" example values exactly, just reversed direction. Scenario
# 2 (+0.4) was not chosen: its value is not already at an extreme, so "flip to the
# opposite extreme" would require an extra magnitude jump beyond a pure sign flip,
# a less clean diagnostic than a direct extreme-to-extreme reversal.
DIAGNOSTIC_SCENARIO = ScenarioProfile(
    number=4,
    name="Diagnostic (Scenario 1 profile, touristiness_pref flipped to the opposite extreme)",
    interests=SCENARIO_PROFILES[0].interests,
    touristiness_pref=0.8,
    budget=SCENARIO_PROFILES[0].budget,
    mobility=SCENARIO_PROFILES[0].mobility,
    party_type=SCENARIO_PROFILES[0].party_type,
    pace=SCENARIO_PROFILES[0].pace,
    accessibility_needs=SCENARIO_PROFILES[0].accessibility_needs,
    notes=(
        f"Base scenario: {DIAGNOSTIC_BASE_SCENARIO} (Local Experience). Every field "
        "held identical to that scenario's profile except `touristiness_pref`, "
        "flipped from -0.8 to +0.8 (opposite sign, same extreme magnitude)."
    ),
)

ALL_SCENARIO_PROFILES: tuple[ScenarioProfile, ...] = (*SCENARIO_PROFILES, DIAGNOSTIC_SCENARIO)

_PARTY_SIZE_BY_TYPE: dict[str, int] = {
    "solo": 1,
    "couple": 2,
    "family_young_kids": 4,
    "family_teens": 4,
    "friends": 3,
}

_EMPTY_INTERACTIONS_COLUMNS: tuple[str, ...] = ("traveler_id", "trip_id", "poi_id", "label")


def scenario_traveler_id(number: int) -> str:
    return f"SCENARIO_{number}_TRAVELER"


def scenario_trip_id(number: int) -> str:
    return f"SCENARIO_{number}_TRIP"


def stay_point_for_destination(pois_df: pd.DataFrame, destination: str) -> tuple[float, float]:
    """The destination's own real POI centroid (mean lat/lon of every
    `pois_prepared.parquet` row in that destination) -- see module docstring."""
    subset = pois_df.loc[pois_df["destination"] == destination]
    return float(subset["lat"].mean()), float(subset["lon"].mean())


def build_scenario_travelers_and_trips(
    pois_df: pd.DataFrame,
    real_trips_df: pd.DataFrame,
    destination: str,
    profiles: tuple[ScenarioProfile, ...] = ALL_SCENARIO_PROFILES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct `travelers.parquet`/`trips.parquet`-shaped rows for every
    `profiles` entry, matching `datagen/travelers.py`'s own schema exactly (module
    docstring) so every downstream function needs zero special-casing."""
    stay_lat, stay_lon = stay_point_for_destination(pois_df, destination)
    start_date = pd.Timestamp(real_trips_df["start_date"].max()) + pd.Timedelta(
        days=SCENARIO_START_DATE_LEAD_DAYS
    )
    end_date = start_date + pd.Timedelta(days=SCENARIO_TRIP_DURATION_DAYS)

    traveler_rows: list[dict[str, Any]] = []
    trip_rows: list[dict[str, Any]] = []
    for profile in profiles:
        traveler_id = scenario_traveler_id(profile.number)
        trip_id = scenario_trip_id(profile.number)
        traveler_rows.append(
            {
                "traveler_id": traveler_id,
                "home_market": "international",
                "home_destination": destination,
                "party_type": profile.party_type,
                "budget": profile.budget,
                "mobility": profile.mobility,
                "interests": list(profile.interests),
                "touristiness_pref": float(profile.touristiness_pref),
                "pace": profile.pace,
                "accessibility_needs": list(profile.accessibility_needs),
                "explicit_preferences": f"Scenario {profile.number}: {profile.name}.",
                "dietary": [],
            }
        )
        trip_rows.append(
            {
                "trip_id": trip_id,
                "traveler_id": traveler_id,
                "destination": destination,
                "start_date": start_date,
                "end_date": end_date,
                "trip_duration_days": SCENARIO_TRIP_DURATION_DAYS,
                "party_size": _PARTY_SIZE_BY_TYPE[profile.party_type],
                "stay_lat": stay_lat,
                "stay_lon": stay_lon,
                "trip_sequence": 1,
            }
        )
    return pd.DataFrame(traveler_rows), pd.DataFrame(trip_rows)


def compute_scenario_traveler_features(
    real_travelers_df: pd.DataFrame,
    real_trips_df: pd.DataFrame,
    synthetic_travelers_df: pd.DataFrame,
    synthetic_trips_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    feature_cfg: FeatureBuildConfig,
    artifacts_dir: Path,
) -> pd.DataFrame:
    """The 4 synthetic `(traveler_id, trip_id)` rows of `traveler_features.parquet`
    -- via `assemble_traveler_features` on `pd.concat([real, synthetic])`, never
    on the synthetic rows alone: `assemble_traveler_features` derives its
    `explicit_interest_*` one-hot vocabulary from whichever `travelers_df` it is
    given (`features.traveler_features.build_interest_vocabulary`), and every
    scenario interest is already a member of the real 31-label vocabulary (module
    docstring) -- concatenating keeps that vocabulary (and therefore the exact
    `explicit_interest_*` column set `artifacts/model.txt` was trained on)
    unchanged, rather than risking a silently narrower vocabulary computed from
    only 4 rows. `poi_emb.npy`'s committed cache makes this cheap (no TF-IDF/SVD
    refit)."""
    text_embeddings = build_poi_text_embeddings(
        pois_df, feature_cfg.text_embedding, feature_cfg.seed, artifacts_dir / "poi_emb.npy"
    )
    augmented_travelers = pd.concat([real_travelers_df, synthetic_travelers_df], ignore_index=True)
    augmented_trips = pd.concat([real_trips_df, synthetic_trips_df], ignore_index=True)
    all_features = assemble_traveler_features(
        augmented_travelers,
        augmented_trips,
        interactions_train,
        pois_df,
        text_embeddings,
        feature_cfg,
    )
    synthetic_trip_ids = set(synthetic_trips_df["trip_id"])
    return all_features.loc[all_features["trip_id"].isin(synthetic_trip_ids)].reset_index(drop=True)


def generate_scenario_candidates(
    pois_df: pd.DataFrame,
    real_travelers_df: pd.DataFrame,
    synthetic_travelers_df: pd.DataFrame,
    synthetic_trips_df: pd.DataFrame,
    poi_features_df: pd.DataFrame,
    synthetic_traveler_features_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    candidates_cfg: CandidatesConfig,
    learned_top_k: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """The 4 synthetic trips' candidate union table -- `candidates.union
    .generate_candidates`, unchanged, given an out-of-sample K-Means segment
    assignment (`segments_override`, module docstring) so `channel_archetype`
    reads the correct affinity columns for a traveler who was never part of the
    original K-Means fit."""
    segments = assign_traveler_segments_out_of_sample(
        real_travelers_df,
        synthetic_travelers_df,
        candidates_cfg.traveler_segment_clusters,
        candidates_cfg.seed,
    )
    return generate_candidates(
        pois_df,
        synthetic_travelers_df,
        synthetic_trips_df,
        poi_features_df,
        synthetic_traveler_features_df,
        interactions_train,
        candidates_cfg,
        segments_override=segments,
        learned_top_k=learned_top_k,
    )


def build_scenario_holdout_frame(
    candidates_df: pd.DataFrame,
    synthetic_trips_df: pd.DataFrame,
    synthetic_travelers_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    poi_features_df: pd.DataFrame,
    synthetic_traveler_features_df: pd.DataFrame,
    budget_target_price_level: BudgetTargetPriceLevel,
) -> pd.DataFrame:
    """The synthetic `(trip_id, poi_id)` ranking frame -- `models.ranking_data
    .build_ranking_frame`, unchanged, with an EMPTY interactions frame (these are
    brand-new trips with no logged interaction of any kind) so every candidate
    correctly gets `label=0` -- a structural placeholder (no ground truth exists
    for a never-run trip), never read by anything this module reports (utility/
    preference/compatibility/confidence all come from the model, not `label`)."""
    trip_ids = set(synthetic_trips_df["trip_id"])
    empty_interactions = pd.DataFrame(columns=list(_EMPTY_INTERACTIONS_COLUMNS))
    return build_ranking_frame(
        candidates_df,
        trip_ids,
        empty_interactions,
        synthetic_trips_df,
        synthetic_travelers_df,
        pois_df,
        poi_features_df,
        synthetic_traveler_features_df,
        budget_target_price_level,
    )


# -----------------------------------------------------------------------------------
# Overlap matrix (spec.md section 15: "the overlap matrix across scenarios")
# -----------------------------------------------------------------------------------


def compute_overlap_matrix(
    top10_by_scenario: dict[int, list[str]],
    diagnostic_scenario: int,
    base_scenario: int,
    diagnostic_candidate_pool_jaccard: float | None = None,
) -> dict[str, Any]:
    """Pairwise top-10 Jaccard across every scenario pair, as a full symmetric
    4x4 matrix (diagonal 1.0 by construction: `jaccard(s, s) == 1.0`) plus the
    single scenario-4-vs-base number spec.md section 15 explicitly asks be
    reported ("should be low, demonstrating the ranking is driven by the
    preference signal, not profile confounds").

    `diagnostic_candidate_pool_jaccard` (optional -- `run_scenarios` passes the
    ACTUAL candidate-set Jaccard between the diagnostic and base scenario's
    pre-ranking candidate pools, computed from `candidates.union
    .generate_candidates`'s own output): a genuine, measured root-cause input
    for `eval/report.py::render_scenarios`'s honest diagnosis when the top-10
    overlap does NOT come out low -- see that function's docstring."""
    order = sorted(top10_by_scenario)
    sets_by_scenario = {n: set(top10_by_scenario[n]) for n in order}

    matrix: list[list[float]] = []
    pairwise: dict[str, float] = {}
    for i in order:
        row: list[float] = []
        for j in order:
            value = jaccard(sets_by_scenario[i], sets_by_scenario[j])
            row.append(value)
            if i < j:
                pairwise[f"{i}_{j}"] = value
        matrix.append(row)

    diag_key = (
        f"{base_scenario}_{diagnostic_scenario}"
        if base_scenario < diagnostic_scenario
        else f"{diagnostic_scenario}_{base_scenario}"
    )
    return {
        "scenario_order": order,
        "matrix": matrix,
        "pairwise_jaccard_top10": pairwise,
        "diagnostic_scenario": diagnostic_scenario,
        "diagnostic_base_scenario": base_scenario,
        "diagnostic_vs_base_jaccard": pairwise[diag_key],
        "diagnostic_vs_base_candidate_pool_jaccard": diagnostic_candidate_pool_jaccard,
    }


# -----------------------------------------------------------------------------------
# Per-scenario output JSON assembly
# -----------------------------------------------------------------------------------


def _profile_dict(
    profile: ScenarioProfile,
    traveler_id: str,
    trip_id: str,
    traveler_row: pd.Series,
    trip_row: pd.Series,
) -> dict[str, Any]:
    return {
        "traveler_id": traveler_id,
        "trip_id": trip_id,
        "destination": str(trip_row["destination"]),
        "interests": list(traveler_row["interests"]),
        "touristiness_pref": float(traveler_row["touristiness_pref"]),
        "budget": str(traveler_row["budget"]),
        "mobility": str(traveler_row["mobility"]),
        "party_type": str(traveler_row["party_type"]),
        "pace": str(traveler_row["pace"]),
        "accessibility_needs": list(traveler_row["accessibility_needs"]),
        "trip_duration_days": int(trip_row["trip_duration_days"]),
        "stay_lat": float(trip_row["stay_lat"]),
        "stay_lon": float(trip_row["stay_lon"]),
        "start_date": pd.Timestamp(trip_row["start_date"]).isoformat(),
        "notes": profile.notes,
    }


def _attach_poi_display_fields(
    recommendations: list[dict[str, Any]], pois_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Add `name`/`category` (spec.md section 15's requested top-10 table columns
    that `scoring.output.assemble_output_payload`'s schema doesn't itself carry)
    to each recommendation entry, looked up from the real catalog."""
    lookup = pois_df.set_index("poi_id")[["name", "category"]]
    out: list[dict[str, Any]] = []
    for rec in recommendations:
        row = lookup.loc[rec["poi_id"]]
        enriched = dict(rec)
        enriched["name"] = str(row["name"])
        enriched["category"] = str(row["category"])
        out.append(enriched)
    return out


def run_scenarios(
    data_dir: Path,
    artifacts_dir: Path,
    results_dir: Path,
    feature_cfg: FeatureBuildConfig,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
) -> dict[str, Any]:
    """`poi_rank.cli scenarios`'s entry point (module docstring): build the 4
    synthetic profiles, run them through the real pipeline, write
    `results/scenarios/{1,2,3,4}.json` + `results/scenarios/overlap_matrix.json`,
    and return a summary dict for the CLI report and tests."""
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    poi_features_df = pd.read_parquet(data_dir / "poi_features.parquet")
    real_travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    real_trips_df = pd.read_parquet(data_dir / "trips.parquet")
    interactions_train = pd.read_parquet(data_dir / "interactions_train.parquet")

    synthetic_travelers_df, synthetic_trips_df = build_scenario_travelers_and_trips(
        pois_df, real_trips_df, SCENARIO_DESTINATION
    )

    synthetic_traveler_features_df = compute_scenario_traveler_features(
        real_travelers_df,
        real_trips_df,
        synthetic_travelers_df,
        synthetic_trips_df,
        interactions_train,
        pois_df,
        feature_cfg,
        artifacts_dir,
    )

    learned_top_k = None
    if candidates_cfg.learned is not None and candidates_cfg.learned.quota > 0:
        # The full-train retriever persisted by `poi_rank.cli candidates`: scenario travelers
        # are unseen, so they are scored out-of-sample exactly like holdout trips.
        scenario_scores = score_full_catalog(
            RetrieverInputs(
                pois_df,
                synthetic_travelers_df,
                synthetic_trips_df,
                poi_features_df,
                synthetic_traveler_features_df,
                interactions_train,
                feature_cfg.traveler_features.budget_target_price_level,
            ),
            Retriever.load(artifacts_dir),
            sorted(synthetic_trips_df["trip_id"]),
            candidates_cfg.learned.num_threads,
        )
        learned_top_k = top_k_by_trip(scenario_scores, candidates_cfg.learned.quota)

    candidates_df = generate_scenario_candidates(
        pois_df,
        real_travelers_df,
        synthetic_travelers_df,
        synthetic_trips_df,
        poi_features_df,
        synthetic_traveler_features_df,
        interactions_train,
        candidates_cfg,
        learned_top_k,
    )

    budget_target_price_level = feature_cfg.traveler_features.budget_target_price_level
    holdout_frame = build_scenario_holdout_frame(
        candidates_df,
        synthetic_trips_df,
        synthetic_travelers_df,
        pois_df,
        poi_features_df,
        synthetic_traveler_features_df,
        budget_target_price_level,
    )

    result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        candidates_cfg.geo,
        candidates_cfg.longtail.pop_pct_cutoff,
        holdout_frame_override=holdout_frame,
        trips_df_override=synthetic_trips_df,
        travelers_df_override=synthetic_travelers_df,
        candidates_df_override=candidates_df,
    )

    scenarios_dir = results_dir / SCENARIOS_DIRNAME
    figures_dir = scenarios_dir / "figures"
    enriched_payload = enrich_recommend_result(
        result,
        data_dir,
        artifacts_dir,
        figures_dir,
        scoring_cfg,
        feature_cfg,
        pois_df_override=pois_df,
        travelers_df_override=synthetic_travelers_df,
    )

    traveler_by_id = synthetic_travelers_df.set_index("traveler_id")
    trip_by_id = synthetic_trips_df.set_index("trip_id")
    generated_at = datetime.now(UTC).isoformat()

    scenarios_dir.mkdir(parents=True, exist_ok=True)
    written_paths: dict[int, Path] = {}
    top10_by_scenario: dict[int, list[str]] = {}
    for profile in ALL_SCENARIO_PROFILES:
        trip_id = scenario_trip_id(profile.number)
        traveler_id = scenario_traveler_id(profile.number)
        trip_payload = enriched_payload[trip_id]
        recommendations = _attach_poi_display_fields(trip_payload["recommendations"], pois_df)
        top10_by_scenario[profile.number] = [rec["poi_id"] for rec in recommendations]

        scenario_payload = {
            "scenario_number": profile.number,
            "scenario_name": profile.name,
            "profile": _profile_dict(
                profile,
                traveler_id,
                trip_id,
                traveler_by_id.loc[traveler_id],
                trip_by_id.loc[trip_id],
            ),
            "model_version": trip_payload["model_version"],
            "recommendations": recommendations,
            "generated_at": generated_at,
        }
        out_path = scenarios_dir / f"{profile.number}.json"
        out_path.write_text(json.dumps(scenario_payload, indent=2, sort_keys=True) + "\n", "utf-8")
        written_paths[profile.number] = out_path

    diagnostic_trip_id = scenario_trip_id(DIAGNOSTIC_SCENARIO.number)
    base_trip_id = scenario_trip_id(DIAGNOSTIC_BASE_SCENARIO)
    diagnostic_candidates = set(
        candidates_df.loc[candidates_df["trip_id"] == diagnostic_trip_id, "poi_id"]
    )
    base_candidates = set(candidates_df.loc[candidates_df["trip_id"] == base_trip_id, "poi_id"])
    diagnostic_candidate_pool_jaccard = jaccard(diagnostic_candidates, base_candidates)

    overlap_matrix = compute_overlap_matrix(
        top10_by_scenario,
        DIAGNOSTIC_SCENARIO.number,
        DIAGNOSTIC_BASE_SCENARIO,
        diagnostic_candidate_pool_jaccard,
    )
    overlap_matrix["generated_at"] = generated_at
    overlap_path = scenarios_dir / OVERLAP_MATRIX_FILENAME
    overlap_path.write_text(json.dumps(overlap_matrix, indent=2, sort_keys=True) + "\n", "utf-8")

    return {
        "scenario_paths": written_paths,
        "overlap_matrix_path": overlap_path,
        "overlap_matrix": overlap_matrix,
        "top10_by_scenario": top10_by_scenario,
        "n_scenarios": len(ALL_SCENARIO_PROFILES),
    }
