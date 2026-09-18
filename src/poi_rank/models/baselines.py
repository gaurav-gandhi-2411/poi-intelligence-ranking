"""Baselines 1-6 (spec.md section 8): every function shares one interface -- given a
ranking frame (`models.ranking_data`'s output, or a compatible test fixture), return a
`BaselineResult` carrying a per-row score (higher = better, aligned to the frame's
index) plus a free-form diagnostics dict (cold-start / geo-filter fallback rates,
etc.) -- so `eval/metrics.py` and the CLI report layer never special-case one
baseline's return type, and a later phase's LambdaMART/IPS system can be added to the
same table without reworking this interface.

**Firewall**: this module (like all of `models/`) must never import from
`poi_rank.scoring` and must never reference the oracle-only export directory --
enforced by `tests/test_firewall_models.py`. It legitimately imports from `poi_rank.data`,
`poi_rank.features`, and `poi_rank.candidates` (all upstream phases whose output/
machinery this phase consumes and reuses, never reimplements).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from poi_rank.candidates.channels import ItemItemCF, build_item_item_cf, trip_seed
from poi_rank.candidates.config import GeoChannelConfig
from poi_rank.candidates.union import cf_seed_poi_ids
from poi_rank.data.geo_prep import haversine_km
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.features.traveler_features import group_interactions_by_traveler
from poi_rank.models.config import ContentCosineConfig, LogisticRegressionConfig

FloatArray = npt.NDArray[np.float64]

_CATEGORICAL_PREFIX = "cat_"
_NUMERIC_FEATURE_PREFIXES: tuple[str, ...] = (
    "text_emb_",
    "num_",
    "geo_",
    "behav_",
    "explicit_",
    "implicit_",
    "interact_",
)
# `geo_h3_cell` is a string identifier, not a numeric feature -- the one column under
# the numeric-prefix set that must be excluded explicitly.
_NUMERIC_EXCLUDE: frozenset[str] = frozenset({"geo_h3_cell"})


@dataclass(frozen=True)
class BaselineResult:
    """Uniform baseline-function return type (see module docstring)."""

    score: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------------
# Baseline 1: Random
# -----------------------------------------------------------------------------------


def baseline_random(frame: pd.DataFrame, seed: int) -> BaselineResult:
    """Uniform-random ranking of each trip's candidate set, seeded per trip via
    `candidates.channels.trip_seed` (a SHA256 digest of `(seed, trip_id)`) -- same
    seeding discipline as the long-tail candidate channel, deterministic independent
    of `PYTHONHASHSEED` or row iteration order."""
    scores = np.empty(len(frame), dtype=np.float64)
    for trip_id, group in frame.groupby("trip_id", sort=False):
        rng = np.random.default_rng(trip_seed(seed, str(trip_id)))
        scores[group.index.to_numpy()] = rng.random(len(group))
    return BaselineResult(score=pd.Series(scores, index=frame.index, name="score_random"))


# -----------------------------------------------------------------------------------
# Baseline 2: Popularity
# -----------------------------------------------------------------------------------


def baseline_popularity(frame: pd.DataFrame) -> BaselineResult:
    """Rank by `num_pop_pct` (within-destination popularity percentile,
    `data/popularity.py`) descending -- zero traveler-specific signal, spec.md
    section 8 baseline 2."""
    return BaselineResult(score=frame["num_pop_pct"].astype(np.float64).rename("score_popularity"))


# -----------------------------------------------------------------------------------
# Baseline 3: Popularity + geo filter
# -----------------------------------------------------------------------------------


def baseline_popularity_geo_filter(
    frame: pd.DataFrame, geo_cfg: GeoChannelConfig
) -> BaselineResult:
    """Popularity ranking restricted to candidates within the same
    mobility-conditioned radius `candidates/channels.py`'s geo channel uses
    (`GeoChannelConfig.radius_km_for_mobility`, reused directly, spec.md section 8
    baseline 3). Candidates outside the radius are pushed to the bottom of the
    ranking (score `-inf`) rather than dropped, so every candidate still gets a
    score. **Fallback**: if the radius filter would exclude an entire trip's
    candidate set (possible since candidates are already geo/interest/semantic
    pre-filtered, not the raw catalog -- a tight radius can legitimately empty out),
    the filter is skipped for that trip and plain popularity is used instead;
    `diagnostics['fallback_rate']` reports how often this fires."""
    dist = haversine_km(
        frame["geo_lat"].to_numpy(dtype=np.float64),
        frame["geo_lon"].to_numpy(dtype=np.float64),
        frame["stay_lat"].to_numpy(dtype=np.float64),
        frame["stay_lon"].to_numpy(dtype=np.float64),
    )
    radius = frame["mobility"].map(geo_cfg.radius_km_for_mobility).to_numpy(dtype=np.float64)
    within = dist <= radius
    pop = frame["num_pop_pct"].to_numpy(dtype=np.float64)

    scores = np.empty(len(frame), dtype=np.float64)
    n_trips = 0
    n_fallback = 0
    for _trip_id, group in frame.groupby("trip_id", sort=False):
        n_trips += 1
        idx = group.index.to_numpy()
        w = within[idx]
        if w.any():
            s = pop[idx].copy()
            s[~w] = -np.inf
            scores[idx] = s
        else:
            n_fallback += 1
            scores[idx] = pop[idx]

    return BaselineResult(
        score=pd.Series(scores, index=frame.index, name="score_popularity_geo"),
        diagnostics={"fallback_rate": n_fallback / n_trips if n_trips else 0.0},
    )


# -----------------------------------------------------------------------------------
# Baseline 4: Content cosine (explicit interests only)
# -----------------------------------------------------------------------------------


def baseline_content_cosine(frame: pd.DataFrame, cfg: ContentCosineConfig) -> BaselineResult:
    """Rank by a weighted blend of `interact_interest_match` (stated-interest
    coverage) and price/budget fit derived from `interact_price_gap` -- both
    EXPLICIT-only signals (spec.md section 8 baseline 4), structurally distinct from
    the implicit taste vector / behavioral block a later LambdaMART system consumes.
    Price gap (0-3 scale) is normalized to a [0, 1] fit score via
    `1 - interact_price_gap / 3`."""
    price_fit = 1.0 - frame["interact_price_gap"].to_numpy(dtype=np.float64) / 3.0
    score = (
        cfg.interest_weight * frame["interact_interest_match"].to_numpy(dtype=np.float64)
        + cfg.price_fit_weight * price_fit
    )
    return BaselineResult(score=pd.Series(score, index=frame.index, name="score_content_cosine"))


# -----------------------------------------------------------------------------------
# Baseline 5: Item-kNN collaborative filtering
# -----------------------------------------------------------------------------------


def baseline_item_knn_cf(
    frame: pd.DataFrame,
    pois_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    trips_df: pd.DataFrame,
) -> BaselineResult:
    """Rank candidates by summed item-item co-interaction similarity to the
    traveler's own as-of-safe, positively-engaged history (spec.md section 8
    baseline 5) -- reuses `candidates.channels.build_item_item_cf` (the SAME global
    similarity matrix the CF candidate channel uses, never reimplemented) and
    `candidates.union.cf_seed_poi_ids` (the SAME as-of-safe seed-history lookup, same
    temporal-leakage discipline as Phase 3/4a). **Cold-start fallback**: for a
    traveler with no as-of-safe engaged history (their first trip), the CF score is
    undefined for every candidate -- falls back to the popularity ranking
    (`num_pop_pct`) rather than crashing or returning an uninformative all-zero
    ranking; `diagnostics['cold_start_fallback_rate']` reports how often this fires.
    """
    cf = build_item_item_cf(pois_df, interactions_train)
    canonical_map = build_poi_id_canonical_map(pois_df)
    remapped = remap_interaction_poi_ids(interactions_train, canonical_map)
    interactions_by_traveler = group_interactions_by_traveler(remapped)

    start_date_by_trip = dict(zip(trips_df["trip_id"], trips_df["start_date"], strict=True))
    trip_traveler = frame.drop_duplicates("trip_id").set_index("trip_id")["traveler_id"].to_dict()
    pop = frame["num_pop_pct"].to_numpy(dtype=np.float64)

    scores = np.zeros(len(frame), dtype=np.float64)
    n_trips = 0
    n_cold_start = 0
    for trip_id, group in frame.groupby("trip_id", sort=False):
        n_trips += 1
        idx = group.index.to_numpy()
        traveler_id = trip_traveler[trip_id]
        as_of = start_date_by_trip[trip_id]
        seed_ids = cf_seed_poi_ids(
            interactions_by_traveler, traveler_id, str(trip_id), as_of, remapped
        )
        if not seed_ids:
            n_cold_start += 1
            scores[idx] = pop[idx]
            continue
        scores[idx] = _cf_scores_for_candidates(cf, seed_ids, group["poi_id"].to_numpy())

    return BaselineResult(
        score=pd.Series(scores, index=frame.index, name="score_item_knn_cf"),
        diagnostics={"cold_start_fallback_rate": n_cold_start / n_trips if n_trips else 0.0},
    )


def _cf_scores_for_candidates(
    cf: ItemItemCF, seed_poi_ids: list[str], candidate_poi_ids: npt.NDArray[np.object_]
) -> FloatArray:
    """Summed co-interaction similarity from every seed POI to each candidate."""
    seed_idx = [cf.poi_index[pid] for pid in seed_poi_ids if pid in cf.poi_index]
    cand_idx = np.array([cf.poi_index.get(str(pid), -1) for pid in candidate_poi_ids])
    scores = np.zeros(len(candidate_poi_ids), dtype=np.float64)
    if not seed_idx:
        return scores
    valid = cand_idx >= 0
    for si in seed_idx:
        row = cf.similarity[si]
        scores[valid] += row[cand_idx[valid]]
    return scores


# -----------------------------------------------------------------------------------
# Baseline 6: Logistic regression, pointwise, full feature set
# -----------------------------------------------------------------------------------


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every numeric/boolean feature column (all blocks: text/POI-numeric/geo/
    behavioral/explicit/implicit/engineered-interaction), excluding the one
    string-valued column (`geo_h3_cell`) that happens to share the numeric-prefix
    set."""
    cols = [
        c
        for c in frame.columns
        if c.startswith(_NUMERIC_FEATURE_PREFIXES) and c not in _NUMERIC_EXCLUDE
    ]
    return sorted(cols)


def categorical_feature_columns(frame: pd.DataFrame) -> list[str]:
    """LightGBM-native `cat_*` categorical columns -- one-hot-encoded for sklearn
    here (spec.md section 8 baseline 6 explicitly diverges from the "keep categories
    native" rule LightGBM enjoys: `LogisticRegression` has no native categorical
    support, unlike a later LambdaMART system)."""
    return sorted(c for c in frame.columns if c.startswith(_CATEGORICAL_PREFIX))


def _design_matrix(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    dummy_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Numeric block: cast to float, NaN (e.g. a cold-start traveler's
    `implicit_mean_localness`) filled with 0.0 -- `LogisticRegression` has no native
    NaN handling, unlike LightGBM's later phase; the `_was_missing` indicator
    columns are themselves part of `numeric_columns` and carry the "this was
    imputed" signal forward. Categorical block: one-hot via `pd.get_dummies`. When
    `dummy_columns` is given (scoring an eval frame against a model fit on the train
    frame), the eval frame's dummy columns are reindexed to exactly the train
    frame's set (unseen categories dropped, missing ones filled 0) so the design
    matrix's column order/width always matches what the model was fit on."""
    numeric = frame[numeric_columns].astype(np.float64).fillna(0.0).reset_index(drop=True)
    cat = frame[categorical_columns].astype(str)
    dummies = pd.get_dummies(cat, prefix=categorical_columns).reset_index(drop=True)
    if dummy_columns is not None:
        dummies = dummies.reindex(columns=dummy_columns, fill_value=0)
    design = pd.concat([numeric, dummies], axis=1)
    return design, list(dummies.columns)


@dataclass
class LogisticRegressionModel:
    """A fitted baseline-6 pipeline plus the exact column layout it was fit on, so
    `score_logistic_regression` can reproduce an identical design matrix for any
    other ranking frame (e.g. the holdout evaluation frame)."""

    pipeline: Pipeline
    numeric_columns: list[str]
    categorical_columns: list[str]
    dummy_columns: list[str]


def fit_logistic_regression(
    train_frame: pd.DataFrame, cfg: LogisticRegressionConfig, seed: int
) -> LogisticRegressionModel:
    """Fit a plain `StandardScaler -> LogisticRegression` pipeline on the TRAIN
    ranking frame's full feature set, predicting `P(label >= 1)` (spec.md section 8
    baseline 6)."""
    numeric_columns = numeric_feature_columns(train_frame)
    categorical_columns = categorical_feature_columns(train_frame)
    design, dummy_columns = _design_matrix(train_frame, numeric_columns, categorical_columns)
    y = (train_frame["label"].to_numpy() >= 1).astype(int)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=cfg.max_iter, C=cfg.C, random_state=seed),
            ),
        ]
    )
    pipeline.fit(design, y)
    return LogisticRegressionModel(
        pipeline=pipeline,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dummy_columns=dummy_columns,
    )


def score_logistic_regression(
    model: LogisticRegressionModel, frame: pd.DataFrame
) -> BaselineResult:
    """Score any ranking frame with a fitted `LogisticRegressionModel`."""
    design, _ = _design_matrix(
        frame, model.numeric_columns, model.categorical_columns, model.dummy_columns
    )
    proba = model.pipeline.predict_proba(design)[:, 1]
    return BaselineResult(
        score=pd.Series(proba, index=frame.index, name="score_logistic_regression"),
        diagnostics={"n_features": design.shape[1]},
    )
