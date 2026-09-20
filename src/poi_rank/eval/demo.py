"""`poi_rank.cli demo` / `make demo`: live top-10 from the COMMITTED model artifacts.

Two forms (both print rank, POI, category, utility, preference, compatibility, confidence, the
top SHAP signals and the explanation lines):

    make demo TRAVELER=U0005
    make demo INTERESTS=local_food,neighborhoods BUDGET=medium MOBILITY=public_transport \
              TOURISTINESS=-0.8 PARTY=solo DEST=seoul

* `TRAVELER=<id>`: that traveler LAST real trip through the real pipeline (real candidate set,
  real features and history).
* the stated-profile form: a brand-new traveler/trip (no history, the cold-start path) built like
  the four assignment scenarios (`eval/scenarios.py`) and run through the same scoring pipeline.

It is the same code path as `recommend`/`scenarios` (never a second scoring implementation), with
one serving shortcut: the isotonic calibrator persisted by `recommend` is loaded instead of
re-fitted on the train frame. If `artifacts/calibrator.pkl` is missing the pipeline falls back to
fitting it (slow, ~1 minute).
"""

from __future__ import annotations

import pickle
import tempfile
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
from poi_rank.eval.scenarios import (
    ScenarioProfile,
    _attach_poi_display_fields,
    build_scenario_holdout_frame,
    build_scenario_travelers_and_trips,
    compute_scenario_traveler_features,
    generate_scenario_candidates,
    scenario_trip_id,
)
from poi_rank.explain.output_enrichment import enrich_recommend_result
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.config import ModelConfig
from poi_rank.models.cross_ranking import attach_cross_features
from poi_rank.models.ranking_data import _load_common, build_ranking_frame
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.output import CALIBRATOR_FILENAME, run_scoring_pipeline

# Friendly aliases -> the real stated-interest vocabulary (`explicit_interest_*`).
INTEREST_ALIASES: dict[str, tuple[str, ...]] = {
    "local_food": ("local", "foodie"),
    "neighborhoods": ("authentic",),
    "museums": ("museum",),
    "nightlife": ("nightlife",),
}
BUDGETS = ("low", "medium", "high")
MOBILITIES = ("walk", "public_transport", "car", "mixed")
PARTIES = ("solo", "couple", "family_young_kids", "family_teens", "friends")
PACES = ("relaxed", "moderate", "packed")


class DemoInputError(ValueError):
    """Raised with an actionable message for a bad CLI input."""


def _choice(name: str, value: str, options: tuple[str, ...]) -> str:
    if value not in options:
        raise DemoInputError(f"{name}={value!r} is not valid; choose one of {', '.join(options)}")
    return value


def parse_interests(raw: str, vocabulary: set[str]) -> tuple[str, ...]:
    out: list[str] = []
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        mapped = INTEREST_ALIASES.get(token, (token,))
        for label in mapped:
            if label not in vocabulary:
                raise DemoInputError(
                    f"unknown interest {token!r}; use one of the stated-interest labels "
                    f"({', '.join(sorted(vocabulary))}) or an alias "
                    f"({', '.join(sorted(INTEREST_ALIASES))})"
                )
            if label not in out:
                out.append(label)
    if not out:
        raise DemoInputError("INTERESTS is empty; give at least one, e.g. INTERESTS=local_food")
    return tuple(out)


def build_profile(
    interests: str,
    budget: str,
    mobility: str,
    touristiness: float,
    party: str,
    vocabulary: set[str],
    pace: str = "moderate",
) -> ScenarioProfile:
    if not -1.0 <= touristiness <= 1.0:
        raise DemoInputError(f"TOURISTINESS={touristiness} must be in [-1, 1]")
    return ScenarioProfile(
        number=1,
        name="Demo profile",
        interests=parse_interests(interests, vocabulary),
        touristiness_pref=touristiness,
        budget=_choice("BUDGET", budget, BUDGETS),
        mobility=_choice("MOBILITY", mobility, MOBILITIES),
        party_type=_choice("PARTY", party, PARTIES),
        pace=_choice("PACE", pace, PACES),
        accessibility_needs=(),
        notes="stated-profile demo",
    )


def _load_calibrator(artifacts_dir: Path) -> Any | None:
    path = artifacts_dir / CALIBRATOR_FILENAME
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301  # our own artifact, written by `recommend`


def recommend_for_traveler(
    traveler_id: str,
    data_dir: Path,
    artifacts_dir: Path,
    feature_cfg: FeatureBuildConfig,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
) -> list[dict[str, Any]]:
    d = _load_common(data_dir)
    trips = d["trips_df"].loc[d["trips_df"]["traveler_id"] == traveler_id].sort_values("start_date")
    if trips.empty:
        raise DemoInputError(f"unknown TRAVELER={traveler_id!r} (no trips in trips.parquet)")
    trip_id = str(trips.iloc[-1]["trip_id"])
    budget = feature_cfg.traveler_features.budget_target_price_level
    frame = build_ranking_frame(
        d["candidates_df"],
        {trip_id},
        pd.DataFrame(columns=["traveler_id", "trip_id", "poi_id", "label"]),
        d["trips_df"],
        d["travelers_df"],
        d["pois_df"],
        d["poi_features_df"],
        d["traveler_features_df"],
        budget,
    )
    frame = attach_cross_features(frame, data_dir, budget)
    result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        candidates_cfg.geo,
        candidates_cfg.longtail.pop_pct_cutoff,
        holdout_frame_override=frame,
        calibrator_override=_load_calibrator(artifacts_dir),
    )
    with tempfile.TemporaryDirectory() as tmp:
        payload = enrich_recommend_result(
            result, data_dir, artifacts_dir, Path(tmp), scoring_cfg, feature_cfg
        )
    return _attach_poi_display_fields(payload[trip_id]["recommendations"], d["pois_df"])


def recommend_for_profile(
    profile: ScenarioProfile,
    destination: str,
    data_dir: Path,
    artifacts_dir: Path,
    feature_cfg: FeatureBuildConfig,
    model_cfg: ModelConfig,
    scoring_cfg: ScoringConfig,
    candidates_cfg: CandidatesConfig,
) -> list[dict[str, Any]]:
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    if destination not in set(pois_df["destination"]):
        raise DemoInputError(
            f"DEST={destination!r} is not in the catalog; choose one of "
            f"{', '.join(sorted(set(pois_df['destination'])))}"
        )
    poi_features_df = pd.read_parquet(data_dir / "poi_features.parquet")
    real_travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    real_trips_df = pd.read_parquet(data_dir / "trips.parquet")
    interactions_train = pd.read_parquet(data_dir / "interactions_train.parquet")
    travelers_df, trips_df = build_scenario_travelers_and_trips(
        pois_df, real_trips_df, destination, (profile,)
    )
    features_df = compute_scenario_traveler_features(
        real_travelers_df,
        real_trips_df,
        travelers_df,
        trips_df,
        interactions_train,
        pois_df,
        feature_cfg,
        artifacts_dir,
    )
    budget = feature_cfg.traveler_features.budget_target_price_level
    learned_top_k = None
    if candidates_cfg.learned is not None and candidates_cfg.learned.quota > 0:
        scores = score_full_catalog(
            RetrieverInputs(
                pois_df,
                travelers_df,
                trips_df,
                poi_features_df,
                features_df,
                interactions_train,
                budget,
            ),
            Retriever.load(artifacts_dir),
            sorted(trips_df["trip_id"]),
            candidates_cfg.learned.num_threads,
        )
        learned_top_k = top_k_by_trip(scores, candidates_cfg.learned.quota)
    candidates_df = generate_scenario_candidates(
        pois_df,
        real_travelers_df,
        travelers_df,
        trips_df,
        poi_features_df,
        features_df,
        interactions_train,
        candidates_cfg,
        learned_top_k,
    )
    frame = build_scenario_holdout_frame(
        candidates_df,
        trips_df,
        travelers_df,
        pois_df,
        poi_features_df,
        features_df,
        budget,
        data_dir,
    )
    result = run_scoring_pipeline(
        data_dir,
        artifacts_dir,
        feature_cfg,
        model_cfg,
        scoring_cfg,
        candidates_cfg.geo,
        candidates_cfg.longtail.pop_pct_cutoff,
        holdout_frame_override=frame,
        trips_df_override=trips_df,
        travelers_df_override=travelers_df,
        candidates_df_override=candidates_df,
        calibrator_override=_load_calibrator(artifacts_dir),
    )
    with tempfile.TemporaryDirectory() as tmp:
        payload = enrich_recommend_result(
            result,
            data_dir,
            artifacts_dir,
            Path(tmp),
            scoring_cfg,
            feature_cfg,
            pois_df_override=pois_df,
            travelers_df_override=travelers_df,
        )
    return _attach_poi_display_fields(
        payload[scenario_trip_id(profile.number)]["recommendations"], pois_df
    )


def format_recommendations(recs: list[dict[str, Any]]) -> str:
    lines = []
    for r in recs:
        signals = ", ".join(
            f"{s['feature_group']} {s['contribution']:+.2f}" for s in r["top_signals"]
        )
        lines.append(
            f"{r['rank']:>2}. {r['name']}  [{r['category']}]  utility {r['utility']:.3f}  "
            f"preference {r['preference_score']:.3f}  compat {r['context_compatibility']:.3f}  "
            f"confidence {r['confidence']:.3f}"
        )
        lines.append(f"      top signals: {signals}")
        lines.extend(f"      - {e}" for e in r["explanation"])
    return "\n".join(lines)
