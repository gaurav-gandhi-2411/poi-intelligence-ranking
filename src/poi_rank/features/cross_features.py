"""Explicit traveler x POI cross features for the ranker (experiment H).

Why these exist: a GBDT recovers multiplicative traveler x POI interactions poorly from raw
factors, and traveler columns are constant within a trip group, so a split on them spends depth
without separating candidates. Each feature below is a stated traveler attribute crossed with a
derived POI column -- exactly the compatibility dimensions the assignment's section 11 asks for.

**Not oracle leakage**: every input is a stated traveler attribute (explicit touristiness
preference, interests, budget, party, mobility), a derived POI column the production system has
(localness index, price level, tags, category, embedding) or the traveler's own logged history
before the trip starts. Nothing here reads `_oracle/` or `datagen`; `tests/test_cross_features.py`
extends the firewall check to this module.

All features carry the `xf_` prefix. `candidates/retriever.py` drops that prefix, so adding them
never changes the retriever or the candidate sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.features.config import BudgetTargetPriceLevel

FloatArray = npt.NDArray[np.float64]

XF_PREFIX = "xf_"
# Engagement worth remembering as history: the graded label is >= 1 for click/navigate/save/
# share/visit/booking; `view` (impression) and `dismiss` are label 0.
POSITIVE_LABEL_MIN = 1
NEGATIVE_INTERACTION = "dismiss"
HISTORY_HALF_LIFE_DAYS = 90.0  # matches the scale of the taste-vector decay (features config)


@dataclass(frozen=True)
class CrossFeatureContext:
    """Everything the cross features need beyond the ranking frame itself."""

    trips_df: pd.DataFrame  # trip_id, start_date
    # trip_id -> timestamp of the trip FIRST logged impression (its own browsing session).
    # History must be strictly before this: a trip session runs 0-45 days BEFORE start_date, so
    # `timestamp < start_date` would include the very interactions the label is made of.
    session_start: pd.Series
    poi_localness_reference: FloatArray  # catalog localness values (frame-independent percentiles)
    history: pd.DataFrame  # traveler_id, poi_id, interaction_type, label, timestamp
    budget_target_price_level: BudgetTargetPriceLevel


def _localness_percentile(values: FloatArray, reference: FloatArray) -> FloatArray:
    """Percentile of each POI localness value within the CATALOG (not within the frame, so the
    train and holdout frames map identically)."""
    ref = np.sort(reference)
    out: FloatArray = np.searchsorted(ref, values, side="right") / float(len(ref))
    return out


def _hits_matrices(frame: pd.DataFrame) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per row: |interests ∩ tags|, category-in-interests (0/1), |interests ∩ (cat ∪ tags)| /
    |interests| -- via unique trips x unique POIs, gathered back (no row-wise set operations)."""
    trips = frame[["trip_id", "interests"]].drop_duplicates("trip_id")
    pois = frame[["poi_id", "poi_category_raw", "poi_tags_raw"]].drop_duplicates("poi_id")
    trip_sets = [set(x) for x in trips["interests"]]
    tag_sets = [set(t) for t in pois["poi_tags_raw"]]
    cats = [str(c) for c in pois["poi_category_raw"]]
    vocab = {t: i for i, t in enumerate(sorted(set().union(*trip_sets, *tag_sets, set(cats))))}

    def mh(sets: list[set[str]]) -> npt.NDArray[np.float32]:
        m = np.zeros((len(sets), len(vocab)), dtype=np.float32)
        for r, terms in enumerate(sets):
            for t in terms:
                m[r, vocab[t]] = 1.0
        return m

    trip_mh = mh(trip_sets)
    tag_hits = trip_mh @ mh(tag_sets).T
    cat_hits = trip_mh @ mh([{c} for c in cats]).T
    sizes = np.array([max(len(s), 1) for s in trip_sets], dtype=np.float64)
    t_idx = pd.Index(trips["trip_id"]).get_indexer(frame["trip_id"])
    p_idx = pd.Index(pois["poi_id"]).get_indexer(frame["poi_id"])
    th = tag_hits[t_idx, p_idx].astype(np.float64)
    ch = cat_hits[t_idx, p_idx].astype(np.float64)
    cover = (th + ch) / sizes[t_idx]
    return th, ch, cover


def _with_cutoff(keys: pd.DataFrame, ctx: CrossFeatureContext) -> pd.DataFrame:
    """Attach `cutoff` = the trip session start when it has logged impressions (train trips),
    else its start_date (holdout trips, whose own impressions are not in the history pool)."""
    out = keys.merge(ctx.trips_df[["trip_id", "start_date"]], on="trip_id", how="left")
    session = out["trip_id"].map(ctx.session_start)
    out["cutoff"] = session.fillna(out["start_date"])
    return out


def _prior_engagement(frame: pd.DataFrame, ctx: CrossFeatureContext) -> pd.DataFrame:
    """Time-decayed count of the traveler's OWN earlier positive engagements with this POI and
    with this POI's category, and cosine of the POI to the centroid of POIs they earlier
    dismissed. Strictly before the trip start (`timestamp < start_date`), so a trip never sees
    its own labels."""
    keys = _with_cutoff(frame[["trip_id", "traveler_id", "poi_id"]], ctx)
    hist = ctx.history
    pos = hist.loc[hist["label"] >= POSITIVE_LABEL_MIN, ["traveler_id", "poi_id", "timestamp"]]
    m = keys.reset_index().merge(pos, on=["traveler_id", "poi_id"], how="inner")
    m = m.loc[m["timestamp"] < m["cutoff"]]
    days = (m["cutoff"] - m["timestamp"]).dt.total_seconds() / 86400.0
    m = m.assign(w=np.power(0.5, days / HISTORY_HALF_LIFE_DAYS))
    poi_decay = m.groupby("index")["w"].sum()
    out = pd.DataFrame(index=frame.index)
    out["xf_prior_poi_engaged"] = poi_decay.reindex(frame.index).fillna(0.0).to_numpy()
    return out


def _dismissed_cosine(frame: pd.DataFrame, ctx: CrossFeatureContext) -> FloatArray:
    """Cosine between each candidate text embedding and the centroid of the embeddings of the
    POIs this traveler DISMISSED before the trip (negative taste); 0 without such history."""
    emb_cols = sorted(c for c in frame.columns if c.startswith("text_emb_"))
    pois = frame[["poi_id", *emb_cols]].drop_duplicates("poi_id")
    emb = pois[emb_cols].to_numpy(dtype=np.float64)
    poi_idx = pd.Index(pois["poi_id"])
    trips = _with_cutoff(frame[["trip_id", "traveler_id"]].drop_duplicates("trip_id"), ctx)
    hist = ctx.history
    neg = hist.loc[
        hist["interaction_type"] == NEGATIVE_INTERACTION, ["traveler_id", "poi_id", "timestamp"]
    ]
    m = trips.merge(neg, on="traveler_id", how="inner")
    m = m.loc[m["timestamp"] < m["cutoff"]]
    rows = poi_idx.get_indexer(m["poi_id"])
    keep = rows >= 0
    m, rows = m.loc[keep], rows[keep]
    trip_index = pd.Index(trips["trip_id"])
    trip_pos = trip_index.get_indexer(m["trip_id"])
    centroid = np.zeros((len(trips), emb.shape[1]))
    counts = np.zeros(len(trips))
    np.add.at(centroid, trip_pos, emb[rows])
    np.add.at(counts, trip_pos, 1.0)
    centroid /= np.maximum(counts, 1.0)[:, None]
    norm_c = np.linalg.norm(centroid, axis=1)
    r_emb = frame[emb_cols].to_numpy(dtype=np.float64)
    r_trip = trip_index.get_indexer(frame["trip_id"])
    dot = np.einsum("ij,ij->i", r_emb, centroid[r_trip])
    denom = np.linalg.norm(r_emb, axis=1) * norm_c[r_trip]
    out: FloatArray = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return out


def add_cross_features(frame: pd.DataFrame, ctx: CrossFeatureContext) -> pd.DataFrame:
    """Return `frame` plus the `xf_*` cross-feature columns (input is not mutated)."""
    out = frame.copy()
    pref = out["explicit_touristiness_pref"].to_numpy(dtype=np.float64)
    loc_z = out["num_localness"].to_numpy(dtype=np.float64)
    loc_c = 2.0 * _localness_percentile(loc_z, ctx.poi_localness_reference) - 1.0  # [-1, 1]
    want_local = -pref  # a traveler who wants "less touristy" wants MORE local
    out["xf_loc_align"] = want_local * loc_c
    out["xf_loc_gap"] = np.abs(loc_c - want_local)
    out["xf_loc_x_pref"] = loc_z * pref
    out["xf_pop_x_pref"] = out["num_pop_pct"].to_numpy(dtype=np.float64) * pref

    target = out["budget"].map(ctx.budget_target_price_level.get).to_numpy(dtype=np.float64)
    signed = out["num_price_level"].to_numpy(dtype=np.float64) - target
    out["xf_price_signed"] = signed
    out["xf_price_over"] = np.clip(signed, 0.0, None)
    out["xf_price_under"] = np.clip(-signed, 0.0, None)

    tag_hits, cat_hit, cover = _hits_matrices(out)
    out["xf_interest_tag_hits"] = tag_hits
    out["xf_interest_cat_hit"] = cat_hit
    out["xf_interest_cover"] = cover
    n_tags = out["poi_tags_raw"].map(len).to_numpy(dtype=np.float64)
    out["xf_interest_tag_ratio"] = tag_hits / np.maximum(n_tags, 1.0)

    prior = _prior_engagement(out, ctx)
    for col in prior.columns:
        out[col] = prior[col].to_numpy()
    out["xf_cos_dismissed"] = _dismissed_cosine(out, ctx)
    return out


def cross_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c.startswith(XF_PREFIX))
