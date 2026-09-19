"""Learned retriever (candidates/retriever.py): cross-fit leakage discipline + determinism.

Runs on a small slice of the committed dataset (real feature tables, tiny boosters) so the
structural properties are checked without the ~1 min full-scale fit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poi_rank.candidates import retriever as rt
from poi_rank.candidates.config import LearnedChannelConfig
from poi_rank.features.config import FeatureBuildConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"

pytestmark = pytest.mark.skipif(
    not (DATA / "traveler_features.parquet").exists(), reason="needs committed synthetic data"
)

CFG = LearnedChannelConfig(
    quota=20,
    n_estimators=5,
    learning_rate=0.1,
    num_leaves=7,
    n_folds=2,
    ips_clip_min=0.05,
    num_threads=1,
)


def _inputs(n_train: int = 60, n_holdout: int = 15) -> rt.RetrieverInputs:
    trips = pd.read_parquet(DATA / "trips.parquet")
    keep = pd.concat(
        [
            trips.loc[~trips["is_holdout"]].sort_values("trip_id").head(n_train),
            trips.loc[trips["is_holdout"]].sort_values("trip_id").head(n_holdout),
        ]
    )
    ids = set(keep["trip_id"])
    tf = pd.read_parquet(DATA / "traveler_features.parquet")
    ix = pd.read_parquet(DATA / "interactions_train.parquet")
    cfg = FeatureBuildConfig.from_yaml(ROOT / "configs" / "features.yaml")
    return rt.RetrieverInputs(
        pois_df=pd.read_parquet(DATA / "pois_prepared.parquet"),
        travelers_df=pd.read_parquet(DATA / "travelers.parquet"),
        trips_df=keep.reset_index(drop=True),
        poi_features_df=pd.read_parquet(DATA / "poi_features.parquet"),
        traveler_features_df=tf.loc[tf["trip_id"].isin(ids)].reset_index(drop=True),
        interactions_train=ix.loc[ix["trip_id"].isin(ids)].reset_index(drop=True),
        budget_target_price_level=cfg.traveler_features.budget_target_price_level,
    )


def test_scores_cover_every_trip_by_every_destination_poi() -> None:
    inp = _inputs()
    scores, _ = rt.crossfit_retriever_scores(inp, CFG, seed=42, num_threads=1)
    n_dest = inp.pois_df.groupby("destination").size()
    expected = inp.trips_df["destination"].map(n_dest).sum()
    assert len(scores) == expected
    assert scores["score"].notna().all()


def test_identical_scores_at_1_and_8_threads() -> None:
    inp = _inputs(n_train=40, n_holdout=10)
    a, model_a = rt.crossfit_retriever_scores(inp, CFG, seed=42, num_threads=1)
    b, model_b = rt.crossfit_retriever_scores(inp, CFG, seed=42, num_threads=8)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # The serialised model ends with a `parameters:` footer that records num_threads itself;
    # the trees (everything before it) are what must be identical.
    trees_a = model_a.booster.model_to_string().split("parameters:")[0]
    trees_b = model_b.booster.model_to_string().split("parameters:")[0]
    assert trees_a == trees_b


def test_train_trips_are_never_scored_by_a_model_fit_on_them(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    inp = _inputs(n_train=40, n_holdout=10)
    fit_ids_by_model: dict[int, set[str]] = {}
    real_fit = rt.fit_retriever
    real_score = rt.score_full_catalog
    scored: list[tuple[set[str], set[str]]] = []  # (trips scored, trips the scorer was fit on)

    def spy_fit(table, fit_ids, cfg, seed, num_threads):  # type: ignore[no-untyped-def]
        model = real_fit(table, fit_ids, cfg, seed, num_threads)
        fit_ids_by_model[id(model)] = set(fit_ids)
        return model

    def spy_score(inp_, model, trip_ids, num_threads):  # type: ignore[no-untyped-def]
        scored.append((set(trip_ids), fit_ids_by_model[id(model)]))
        return real_score(inp_, model, trip_ids, num_threads)

    monkeypatch.setattr(rt, "fit_retriever", spy_fit)
    monkeypatch.setattr(rt, "score_full_catalog", spy_score)
    rt.crossfit_retriever_scores(inp, CFG, seed=42, num_threads=1)

    holdout = set(inp.trips_df.loc[inp.trips_df["is_holdout"], "trip_id"])
    assert len(scored) == CFG.n_folds + 1
    for trips, model_fit_ids in scored:
        if trips & holdout:
            assert trips <= holdout
            assert not (model_fit_ids & holdout)  # holdout trips never enter any fit
        else:
            assert not (model_fit_ids & trips), "train trip scored by a model that saw it"


def test_top_k_is_deterministic_and_tie_broken_by_poi_id() -> None:
    s = pd.DataFrame(
        {"trip_id": ["T1"] * 4, "poi_id": ["c", "a", "b", "d"], "score": [0.5, 0.5, 0.9, 0.1]}
    )
    assert rt.top_k_by_trip(s, 3) == {"T1": ["b", "a", "c"]}
