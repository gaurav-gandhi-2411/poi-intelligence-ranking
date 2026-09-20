"""Traveler representation (spec.md section 6): explicit + implicit blocks, plus the
traveler x POI interaction-feature functions and the observable traveler-segment
proxy used by `poi_features.py`'s "archetype affinity profile".

**Two different mechanisms for two different jobs** (mirrors `confidence.py`'s module
docstring, stated from the other side): this module's implicit-block taste vector
uses an EXPONENTIAL TIME-DECAY weight per interaction (`exp(-delta_t / tau)`,
discounting *individual events* by recency) -- unrelated to
`confidence.confidence_shrinkage_alpha` (`n_t / (n_t + k)`, an evidence-VOLUME
shrinkage applied only in the cold-start fallback path and the confidence score).
Never shared code.

**Temporal leakage discipline (the load-bearing design decision of this module,
rewritten for DGP remediation Block A RC3.1 -- docs/DATA_CARD.md "DGP remediation,
Block A"):** every implicit-block quantity computed here for a given `(traveler_id,
trip_id)` row now uses a genuine PER-IMPRESSION as-of cutoff: every one of the
traveler's own interaction rows (any trip, INCLUDING earlier impressions in the SAME
trip's own browsing session, plus pre-trip history -- see below) with `timestamp <
as_of` (strict). No `trip_id` exclusion is needed: `datagen/travelers.py::
generate_trips` guarantees a traveler's trips never overlap (>=60-day minimum gap
between trips vs. a <=45-day max in-trip session lead time), so a trip's own EARLIER
impressions (this trip's other slates, dated `trip.start_date - lead_days`,
`lead_days>=0`, i.e. always `< trip.start_date` except an exact `lead_days=0` tie,
correctly excluded by strict `<`) pass the pure timestamp filter, while any
CURRENT-OR-LATER impression (including this same trip's own `as_of`-time slate, or a
later trip) is correctly excluded by `timestamp < as_of` alone.

**Correction (experiment H, Amendment 1) -- the PER-TRIP row is as-of the trip FIRST own impression,
not `start_date`.** The rule above is right for `traveler_history_before(as_of=...)`, but this
module emits ONE row per trip, and calling it with `as_of = start_date` folded the trip whole
browsing session (dated 0-45 days BEFORE `start_date`, and the source of the graded labels) into the
features of every row of that trip: train/validation features carried label information that
holdout features (whose session is not in the history pool) cannot.
`assemble_traveler_features` now uses `as_of = min(start_date, first own logged impression)`; the
"earlier impressions of the same trip are visible" clause applies only to a per-impression consumer,
not to this per-trip table
(`tests/test_traveler_features.py::test_assemble_traveler_features_excludes_own_session_of_train_trips`,
docs/TECHNICAL.md section 5.1).

**Why this replaces the pre-remediation per-TRIP cutoff:** the old implementation
additionally excluded `trip_id == <this trip>` rows even when their timestamp was
already `< as_of` -- treating a trip's own earlier browsing sessions as forbidden
"future" information, which was over-conservative (Phase 3's own "future work,
deferred" note) and the direct cause of the pre-remediation 79%/68% cold-start rate
(`poi_rank.cli diagnose-dgp` D6): a traveler's FIRST trip could never see ANY of its
own in-trip history, regardless of how many earlier-in-session impressions it had.
Dropping the trip-scoped exclusion, keeping only the (already-correct) timestamp
comparison, is what makes "the taste vector at impression i uses impressions < t_i,
INCLUDING earlier impressions in the same trip" true by construction -- verified
directly against a hand-built multi-impression-per-trip fixture in
`tests/test_traveler_features.py`.

**Pre-trip history** (`datagen/pipeline.py`'s Block A RC3.2 pre-trip seeding,
`interactions_pretrip.parquet`) is concatenated into the SAME history pool this
module reads before grouping -- it needs no special-casing at all: every pre-trip row
is dated 46-180 days before the traveler's FIRST trip's start date, strictly earlier
than any of that traveler's own in-trip session timestamps (0-45 days before their
respective trip's start), so it always satisfies the same `timestamp < as_of` filter.
~15% of travelers deliberately get ZERO pre-trip rows (the intentional cold-start
cohort, `pretrip_history.cold_start_fraction`), so `n_interactions=0` remains a real,
expected outcome for that cohort's first trip -- not a bug.

`explicit_days_remaining` and `implicit_days_since_last_interaction` reduce to the
same underlying quantity in this implementation (see docs/DATA_CARD.md) since the
dataset provides no "trip booking" event distinct from browsing-interaction
timestamps -- both columns are still emitted under spec.md's two names.

**Implicit preference-summary stats use ENGAGED interactions only (label >= 1), never
raw impressions:** `implicit_category_distribution`, `implicit_mean_price_level`,
`implicit_mean_localness`, and `implicit_mean_pop_pct` would otherwise just mirror the
popularity-biased *exposure* policy's shape (what the traveler was shown), not what
they actually liked -- see docs/DATA_CARD.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from poi_rank.data.categories import CANONICAL_CATEGORIES
from poi_rank.features.config import BudgetTargetPriceLevel, FeatureBuildConfig, TasteWeights
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids

FloatArray = npt.NDArray[np.float64]

EXPLICIT_PREFIX = "explicit_"
IMPLICIT_PREFIX = "implicit_"

PARTY_TYPES: tuple[str, ...] = ("solo", "couple", "family_young_kids", "family_teens", "friends")
MOBILITY_TYPES: tuple[str, ...] = ("walk", "public_transport", "car", "mixed")
PACE_ORDER: dict[str, int] = {"relaxed": 0, "moderate": 1, "packed": 2}
BUDGET_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
SEASONS: tuple[str, ...] = ("winter", "spring", "summer", "autumn")
ACCESSIBILITY_VOCAB: tuple[str, ...] = ("wheelchair", "stroller")

_INTERACTIONS_COLUMNS: tuple[str, ...] = (
    "traveler_id",
    "trip_id",
    "poi_id",
    "interaction_type",
    "label",
    "timestamp",
)

# -----------------------------------------------------------------------------------
# Observed interest vocabulary + traveler-segment proxy (shared with poi_features.py)
# -----------------------------------------------------------------------------------


def build_interest_vocabulary(travelers_df: pd.DataFrame) -> list[str]:
    """Observed (not DGP-taxonomy-imported) interest vocabulary -- every distinct
    label appearing anywhere in `travelers_df['interests']`, sorted for determinism.

    Mirrors `data/categories.py`'s precedent of hand-deriving vocabulary from what
    the data actually contains rather than importing `datagen`'s internal taxonomy
    (`datagen/taxonomy.py::INTEREST_LABELS`) -- a real production system building
    traveler features would not have access to its data generator's internals either.
    Measured on the committed dataset: 31 distinct labels (not 32 = len(CATEGORIES) +
    len(TAGS) as a literal sum, because `datagen/taxonomy.py`'s `CATEGORIES` and
    `TAGS` both separately contain the string `"shopping"`; not spec.md's literal
    "14d" either -- see docs/DATA_CARD.md).
    """
    vocab: set[str] = set()
    for interests in travelers_df["interests"]:
        vocab.update(interests)
    return sorted(vocab)


def _interests_multi_hot(interests_col: pd.Series, vocab: list[str]) -> pd.DataFrame:
    cols = {
        f"{EXPLICIT_PREFIX}interest_{label}": interests_col.apply(
            lambda xs, target=label: target in set(xs)
        )
        for label in vocab
    }
    return pd.DataFrame(cols, index=interests_col.index)


def _traveler_segment_feature_matrix(travelers_df: pd.DataFrame, vocab: list[str]) -> FloatArray:
    """Observable feature matrix used only as K-Means clustering input for the
    traveler-segment proxy -- NOT the features shipped in the final explicit block
    (those are the full one-hot/ordinal columns built by `build_explicit_block`)."""
    interests_mh = _interests_multi_hot(travelers_df["interests"], vocab).to_numpy(dtype=float)
    budget_ord = travelers_df["budget"].map(BUDGET_ORDER).to_numpy(dtype=float).reshape(-1, 1)
    party_oh = (
        pd.get_dummies(travelers_df["party_type"])
        .reindex(columns=list(PARTY_TYPES), fill_value=0)
        .to_numpy(dtype=float)
    )
    touristiness = travelers_df["touristiness_pref"].to_numpy(dtype=float).reshape(-1, 1)
    return np.hstack([interests_mh, budget_ord, party_oh, touristiness])


def assign_traveler_segments(travelers_df: pd.DataFrame, n_clusters: int, seed: int) -> pd.Series:
    """Observable N-cluster proxy segmentation over stated traveler attributes only
    (interests multi-hot, budget ordinal, party_type one-hot, touristiness_pref) --
    K-Means, seeded, zero randomness beyond `random_state`.

    This is NOT the DGP's latent archetype mixture: `travelers.parquet` does not
    export archetype weights, and this module never reads the oracle-only export
    directory. Returns a `pd.Series` of integer cluster labels indexed by
    `traveler_id` -- see docs/DATA_CARD.md "archetype affinity profile: zero oracle
    information".
    """
    vocab = build_interest_vocabulary(travelers_df)
    features = _traveler_segment_feature_matrix(travelers_df, vocab)
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(scaled)
    return pd.Series(labels, index=travelers_df["traveler_id"].to_numpy(), name="segment")


def assign_traveler_segments_out_of_sample(
    fitted_travelers_df: pd.DataFrame,
    new_travelers_df: pd.DataFrame,
    n_clusters: int,
    seed: int,
) -> pd.Series:
    """Segment labels for `new_travelers_df` (e.g. `eval/scenarios.py`'s hand-built
    synthetic travelers) that are guaranteed to mean the SAME thing as
    `poi_features.py`'s already-persisted `behav_archetype_affinity_NN` columns.

    `assign_traveler_segments` above does a fresh `.fit_predict()` -- calling it on
    `pd.concat([real_travelers_df, new_travelers_df])` would silently REFIT the
    K-Means (a different input array, even with the same `random_state`), which can
    relabel/reorder cluster IDs relative to the ORIGINAL fit `poi_features.py` used
    to build `behav_archetype_affinity_00..NN` -- `channel_archetype` would then
    read a synthetic traveler's "segment 3" against an affinity column that was
    actually fit to mean a different cluster. This function instead fits the SAME
    `StandardScaler` + `KMeans` pipeline ONCE on `fitted_travelers_df` (the real
    population `poi_features.py`/`candidates/union.py` already used) and calls
    `.predict()` (out-of-sample assignment to the nearest already-fit centroid) on
    `new_travelers_df` -- cluster ID semantics are preserved by construction. The
    interest vocabulary is likewise built from `fitted_travelers_df` ALONE (never
    `new_travelers_df`), for the same "must match what was already fit" reason.
    """
    vocab = build_interest_vocabulary(fitted_travelers_df)
    fitted_features = _traveler_segment_feature_matrix(fitted_travelers_df, vocab)
    scaler = StandardScaler().fit(fitted_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(
        scaler.transform(fitted_features)
    )

    new_features = _traveler_segment_feature_matrix(new_travelers_df, vocab)
    labels = kmeans.predict(scaler.transform(new_features))
    return pd.Series(labels, index=new_travelers_df["traveler_id"].to_numpy(), name="segment")


# -----------------------------------------------------------------------------------
# Explicit block
# -----------------------------------------------------------------------------------


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def build_explicit_block(merged: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    """Explicit traveler/trip block (spec.md section 6), computed on `merged` =
    `trips_df` left-joined to `travelers_df` on `traveler_id` (one row per trip)."""
    out: dict[str, Any] = {}

    interests_mh = _interests_multi_hot(merged["interests"], vocab)
    out.update(interests_mh.to_dict("series"))

    out[f"{EXPLICIT_PREFIX}budget_ordinal"] = merged["budget"].map(BUDGET_ORDER).astype(float)

    party_oh = pd.get_dummies(merged["party_type"]).reindex(columns=list(PARTY_TYPES), fill_value=0)
    for party in PARTY_TYPES:
        out[f"{EXPLICIT_PREFIX}party_{party}"] = party_oh[party].astype(bool)

    mobility_oh = pd.get_dummies(merged["mobility"]).reindex(
        columns=list(MOBILITY_TYPES), fill_value=0
    )
    for mobility in MOBILITY_TYPES:
        out[f"{EXPLICIT_PREFIX}mobility_{mobility}"] = mobility_oh[mobility].astype(bool)

    out[f"{EXPLICIT_PREFIX}touristiness_pref"] = merged["touristiness_pref"].astype(float)
    out[f"{EXPLICIT_PREFIX}pace_ordinal"] = merged["pace"].map(PACE_ORDER).astype(float)

    for need in ACCESSIBILITY_VOCAB:
        out[f"{EXPLICIT_PREFIX}accessibility_{need}"] = merged["accessibility_needs"].apply(
            lambda xs, target=need: target in set(xs)
        )

    out[f"{EXPLICIT_PREFIX}trip_duration_days"] = merged["trip_duration_days"].astype(float)

    seasons = merged["start_date"].dt.month.apply(_season_from_month)
    season_oh = pd.get_dummies(seasons).reindex(columns=list(SEASONS), fill_value=0)
    for season in SEASONS:
        out[f"{EXPLICIT_PREFIX}season_{season}"] = season_oh[season].astype(bool)

    return pd.DataFrame(out, index=merged.index)


# -----------------------------------------------------------------------------------
# Implicit block (as-of-time-safe)
# -----------------------------------------------------------------------------------


def half_life_to_decay_constant(halflife_days: float) -> float:
    """`tau` such that `exp(-delta_t / tau) == 0.5` at `delta_t == halflife_days`
    (i.e. `tau = halflife_days / ln(2)`). spec.md section 6 writes `tau = 180 days
    half-life` then `exp(-delta_t/tau)`; taken literally as `tau = 180` the formula
    would decay to `exp(-1) ~= 0.368` (not 0.5) at `delta_t = 180`. This function
    honors the mathematical meaning of "half-life" -- see docs/DATA_CARD.md.
    """
    return float(halflife_days / np.log(2.0))


def _empty_interactions_like(interactions: pd.DataFrame) -> pd.DataFrame:
    return interactions.iloc[0:0]


def group_interactions_by_traveler(interactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-group (already poi-id-reconciled) train interactions by `traveler_id`,
    once, for fast per-trip as-of filtering."""
    return {str(tid): g for tid, g in interactions.groupby("traveler_id", sort=False)}


def traveler_history_before(
    interactions_by_traveler: dict[str, pd.DataFrame],
    traveler_id: str,
    trip_id: str,
    as_of: pd.Timestamp,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Per-impression as-of-safe interaction history (Block A RC3.1): every row for
    `traveler_id` with `timestamp < as_of`, from ANY trip -- including this SAME
    `trip_id`'s own earlier impressions and pre-trip history (module docstring
    explains why a pure timestamp filter is sufficient and correct; `trip_id` is
    kept as a parameter for call-site clarity/API stability but is no longer used
    to filter)."""
    del trip_id  # kept for call-site clarity; filtering is timestamp-only (see docstring)
    sub = interactions_by_traveler.get(traveler_id)
    if sub is None:
        return _empty_interactions_like(template)
    mask = sub["timestamp"] < as_of
    return sub.loc[mask]


def compute_taste_vector(
    history: pd.DataFrame,
    poi_emb_by_id: dict[str, FloatArray],
    as_of: pd.Timestamp,
    weights: TasteWeights,
    halflife_days: float,
    emb_dim: int,
) -> FloatArray:
    """`taste_t = normalize(sum_i w(type_i) * exp(-delta_t_i / tau) * emb(poi_i))`
    (spec.md section 6). Uses the FULL as-of-safe history (all interaction types,
    including `view` and `dismiss`, per the spec weight table) -- unlike the
    preference-summary implicit stats below, which deliberately restrict to engaged
    rows.
    """
    accum = np.zeros(emb_dim, dtype=np.float64)
    if history.empty:
        return accum
    tau = half_life_to_decay_constant(halflife_days)
    delta_days = (as_of - history["timestamp"]).dt.days.to_numpy(dtype=float)
    decay = np.exp(-delta_days / tau)
    for (poi_id, itype), d in zip(
        zip(history["poi_id"], history["interaction_type"], strict=True), decay, strict=True
    ):
        emb = poi_emb_by_id.get(poi_id)
        if emb is None:
            continue
        accum += weights.get(itype) * d * emb
    norm = np.linalg.norm(accum)
    return accum / norm if norm > 0 else accum


def _compute_history_derived_row(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    poi_emb_by_id: dict[str, FloatArray],
    poi_category_by_id: dict[str, str],
    poi_price_by_id: dict[str, float],
    poi_localness_by_id: dict[str, float],
    poi_pop_pct_by_id: dict[str, float],
    weights: TasteWeights,
    halflife_days: float,
    emb_dim: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}

    taste = compute_taste_vector(history, poi_emb_by_id, as_of, weights, halflife_days, emb_dim)
    for i in range(emb_dim):
        row[f"{IMPLICIT_PREFIX}taste_{i:02d}"] = taste[i]

    # "Interactions" = active responses only (view is an impression, not an
    # interaction, per spec.md's own taxonomy table framing) -- see module docstring.
    active = history.loc[history["interaction_type"] != "view"]
    row[f"{IMPLICIT_PREFIX}interaction_count"] = float(len(active))
    if len(active) > 0:
        last_ts = active["timestamp"].max()
        row[f"{IMPLICIT_PREFIX}days_since_last_interaction"] = float((as_of - last_ts).days)
        row[f"{IMPLICIT_PREFIX}days_since_last_interaction_was_missing"] = False
    else:
        row[f"{IMPLICIT_PREFIX}days_since_last_interaction"] = np.nan
        row[f"{IMPLICIT_PREFIX}days_since_last_interaction_was_missing"] = True

    # Preference-summary stats: engaged rows only (label >= 1), never raw impressions.
    engaged = history.loc[history["label"] >= 1]
    row[f"{IMPLICIT_PREFIX}breadth_categories"] = float(
        engaged["poi_id"].map(poi_category_by_id).nunique()
    )

    for cat in CANONICAL_CATEGORIES:
        row[f"{IMPLICIT_PREFIX}category_dist_{cat}"] = 0.0
    if len(engaged) > 0:
        cats = engaged["poi_id"].map(poi_category_by_id).dropna()
        if len(cats) > 0:
            dist = cats.value_counts(normalize=True)
            for cat, share in dist.items():
                if cat in CANONICAL_CATEGORIES:
                    row[f"{IMPLICIT_PREFIX}category_dist_{cat}"] = float(share)

        prices = engaged["poi_id"].map(poi_price_by_id).dropna()
        localness_vals = engaged["poi_id"].map(poi_localness_by_id).dropna()
        pop_pct_vals = engaged["poi_id"].map(poi_pop_pct_by_id).dropna()
    else:
        prices = pd.Series(dtype=float)
        localness_vals = pd.Series(dtype=float)
        pop_pct_vals = pd.Series(dtype=float)

    for name, series in (
        ("mean_price_level", prices),
        ("mean_localness", localness_vals),
        ("mean_pop_pct", pop_pct_vals),
    ):
        if len(series) > 0:
            row[f"{IMPLICIT_PREFIX}{name}"] = float(series.mean())
            row[f"{IMPLICIT_PREFIX}{name}_was_missing"] = False
        else:
            row[f"{IMPLICIT_PREFIX}{name}"] = np.nan
            row[f"{IMPLICIT_PREFIX}{name}_was_missing"] = True

    # explicit_days_remaining intentionally mirrors implicit_days_since_last_interaction
    # -- see module docstring.
    row[f"{EXPLICIT_PREFIX}days_remaining"] = row[f"{IMPLICIT_PREFIX}days_since_last_interaction"]

    return row


# -----------------------------------------------------------------------------------
# Traveler x POI interaction features (functions only -- applied at candidate-scoring
# time by a later phase; spec.md section 6)
# -----------------------------------------------------------------------------------


def cosine_similarity_taste_poi(taste_vecs: FloatArray, poi_embs: FloatArray) -> FloatArray:
    """`cos(taste_t, emb_p)`, row-aligned, vectorized over `(n, d)` arrays. Both are
    expected already L2-normalized (as this module and `text_embed.py` both
    guarantee), but the full cosine formula is used regardless so this stays correct
    even if a caller passes un-normalized vectors."""
    dot = np.sum(taste_vecs * poi_embs, axis=1)
    denom = np.linalg.norm(taste_vecs, axis=1) * np.linalg.norm(poi_embs, axis=1)
    denom = np.where(denom > 0, denom, 1.0)
    result: FloatArray = dot / denom
    return result


def localness_preference_gap(
    implicit_localness: FloatArray, touristiness_pref: FloatArray
) -> FloatArray:
    """`|implicit_localness - touristiness_pref|` (spec.md section 6, literal
    formula). `data/localness.py`'s `localness` column is an unbounded weighted
    z-score sum, not a normalized `[-1, 1]` scale like `touristiness_pref` -- this is
    a magnitude-of-mismatch proxy, not a normalized distance; no rescaling invented,
    documented as-is per the literal spec.md formula.
    """
    result: FloatArray = np.abs(implicit_localness - touristiness_pref)
    return result


def interest_match_score(
    stated_interests: list[set[str]], poi_category: list[str], poi_tags: list[list[str]]
) -> FloatArray:
    """Fraction of a traveler's stated interests that appear in the candidate POI's
    own `{category} u {tags}` set -- coverage, not Jaccard (a traveler with 3 stated
    interests where 1 matches scores 0.33, regardless of how many tags the POI has).
    """
    n = len(stated_interests)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        interests = stated_interests[i]
        if not interests:
            continue
        poi_terms = set(poi_tags[i]) | {poi_category[i]}
        scores[i] = len(interests & poi_terms) / len(interests)
    return scores


def price_gap(
    budget: list[str], poi_price_level: FloatArray, target_price_level: BudgetTargetPriceLevel
) -> FloatArray:
    """`|target_price_level(budget) - poi.price_level|`, both on the 1-4 scale.
    `target_price_level` is independently authored in `configs/features.yaml`
    (`traveler_features.budget_target_price_level`), never imported from
    `datagen/utility.py::BUDGET_TARGET_PRICE_LEVEL` -- spec.md / docs/DATA_CARD.md are
    explicit that the DGP's latent price_fit machinery and any later observable
    scoring share names but never code (docs/DATA_CARD.md resolved ambiguity #2).
    """
    targets = np.array([target_price_level.get(b) for b in budget], dtype=np.float64)
    result: FloatArray = np.abs(targets - poi_price_level)
    return result


# -----------------------------------------------------------------------------------
# Top-level assembly
# -----------------------------------------------------------------------------------


def assemble_traveler_features(
    travelers_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    poi_embeddings: npt.NDArray[np.float32],
    cfg: FeatureBuildConfig,
    interactions_pretrip: pd.DataFrame | None = None,
    only_trip_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Assemble the full traveler feature table (spec.md section 6), keyed by
    `(traveler_id, trip_id)` -- one row per trip, since trip-conditional fields
    (season, trip_duration, the implicit block's as-of cutoff) vary per trip even for
    the same traveler.

    `interactions_pretrip` (Block A RC3.2, `datagen/pipeline.py`'s
    `interactions_pretrip.parquet`) is concatenated into the SAME history pool as
    `interactions_train` before grouping -- optional (defaults to `None`/empty) so
    existing callers built against pre-remediation datasets without that file (e.g.
    `eval/dgp_diagnostics.py`'s MiniLM-path rebuild) keep working unchanged.
    """
    merged = trips_df.merge(travelers_df, on="traveler_id", how="left")
    vocab = build_interest_vocabulary(travelers_df)
    explicit_static = build_explicit_block(merged, vocab)

    canonical_map = build_poi_id_canonical_map(pois_df)
    history_source = interactions_train[list(_INTERACTIONS_COLUMNS)]
    if interactions_pretrip is not None and len(interactions_pretrip) > 0:
        history_source = pd.concat(
            [history_source, interactions_pretrip[list(_INTERACTIONS_COLUMNS)]],
            ignore_index=True,
        )
    remapped = remap_interaction_poi_ids(history_source, canonical_map)
    interactions_by_traveler = group_interactions_by_traveler(remapped)

    poi_emb_by_id = dict(zip(pois_df["poi_id"], poi_embeddings, strict=True))
    poi_category_by_id = dict(zip(pois_df["poi_id"], pois_df["category"], strict=True))
    poi_price_by_id = dict(zip(pois_df["poi_id"], pois_df["price_level_imputed"], strict=True))
    poi_localness_by_id = dict(zip(pois_df["poi_id"], pois_df["localness"], strict=True))
    poi_pop_pct_by_id = dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True))

    weights = cfg.traveler_features.taste_weights
    halflife = cfg.traveler_features.taste_halflife_days
    emb_dim = poi_embeddings.shape[1]

    # The implicit block is ONE row per trip, so its as-of cutoff must be no later than the trip's
    # FIRST logged impression: a trip's browsing session runs 0-45 days BEFORE `start_date`, and
    # `timestamp < start_date` would fold the session's own engagements -- the very interactions the
    # graded labels are made of -- into the features of the rows being labelled (train/validation
    # trips only; holdout trips have no session rows in the history pool, which is what serving
    # looks like). Holdout trips and trips without logged impressions keep `start_date`.
    session_start = interactions_train.groupby("trip_id")["timestamp"].min()
    as_of_by_trip = {
        str(t): min(start, session_start.get(t, start))
        for t, start in zip(merged["trip_id"], merged["start_date"], strict=True)
    }

    # `only_trip_ids` (serving path, `poi_rank.cli demo`): the vocabulary and explicit block above
    # still come from ALL travelers (same column set as training); only the expensive per-trip
    # history loop is restricted to the requested trips.
    selected = (
        merged["trip_id"].isin(only_trip_ids).to_numpy()
        if only_trip_ids is not None
        else np.ones(len(merged), dtype=bool)
    )
    merged = merged.loc[selected]
    explicit_static = explicit_static.loc[selected]

    history_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        trip_as_of = as_of_by_trip[str(row.trip_id)]
        history = traveler_history_before(
            interactions_by_traveler, row.traveler_id, row.trip_id, trip_as_of, remapped
        )
        history_rows.append(
            _compute_history_derived_row(
                history,
                trip_as_of,
                poi_emb_by_id,
                poi_category_by_id,
                poi_price_by_id,
                poi_localness_by_id,
                poi_pop_pct_by_id,
                weights,
                halflife,
                emb_dim,
            )
        )
    history_df = pd.DataFrame(history_rows, index=merged.index)

    explicit_days_remaining = history_df.pop(f"{EXPLICIT_PREFIX}days_remaining")
    explicit = pd.concat([explicit_static, explicit_days_remaining], axis=1)

    keys = merged[["traveler_id", "trip_id"]].reset_index(drop=True)
    return pd.concat(
        [keys, explicit.reset_index(drop=True), history_df.reset_index(drop=True)], axis=1
    )
