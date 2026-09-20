"""E1: retrieval-vs-ranking gain decomposition.

Every system is scored over BOTH candidate sets on the same unbiased holdout: the original six
heuristic channels ("legacy6", `configs/features_legacy6.yaml`) and the shipped learned retriever
(+ long-tail + interest). Popularity, content-cosine and the random baseline read only frame
columns, so they are scored on exactly the candidates of the set they are placed in (content-cosine
is therefore like-for-like with the primary system in each column). The primary LambdaMART+IPS is
shown twice on the legacy set: the SHIPPED booster (trained on the new candidates) and one
RETRAINED on the legacy training candidates, so the ranker is never penalised for a train/serve
candidate mismatch.

Writes `results/parts/retrieval_ranking_decomposition.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.candidates.config import CandidatesConfig
from poi_rank.candidates.union import generate_candidates
from poi_rank.eval import oracle as oracle_reader
from poi_rank.eval.config import EvalConfig
from poi_rank.eval.decision_register import (
    Lab,
    fit_lgb,
    per_trip_ndcg10,
    score_lgb,
    summarize_ndcg,
)
from poi_rank.eval.metrics import _rank_order, dcg_at_k, paired_wilcoxon
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.pair_frame import build_ranking_frame, label_by_trip_poi
from poi_rank.features.reconcile import build_poi_id_canonical_map
from poi_rank.models import baselines as bl
from poi_rank.models import lambdamart as lm

LEGACY_CANDIDATES_FILENAME = "candidates_legacy6.parquet"


def _legacy_candidates(data_dir: Path, lab: Lab, legacy_cfg: CandidatesConfig) -> pd.DataFrame:
    path = data_dir / LEGACY_CANDIDATES_FILENAME
    if path.exists():
        return pd.read_parquet(path)
    trips = pd.read_parquet(data_dir / "trips.parquet")
    cand = generate_candidates(
        lab.pois_df,
        pd.read_parquet(data_dir / "travelers.parquet"),
        trips,
        pd.read_parquet(data_dir / "poi_features.parquet"),
        pd.read_parquet(data_dir / "traveler_features.parquet"),
        lab.interactions_train,
        legacy_cfg,
    )
    cand.to_parquet(path, index=False)
    return cand


def end_to_end_ndcg10(
    frame: pd.DataFrame, score: pd.Series, all_labels: dict[str, npt.NDArray[np.float64]]
) -> pd.Series:
    """NDCG@10 with a candidate-set-INDEPENDENT denominator.

    The usual NDCG@10 normalises by the ideal ranking of the candidate set itself, so a
    retriever that surfaces MORE positives raises the ideal and lowers NDCG for the same ranker:
    retrieval quality is normalised away. Here the ideal is computed over EVERY logged (exposed)
    label of the trip holdout slate; a relevant POI the retriever missed can never be ranked, so
    it costs DCG. The denominator is identical for every candidate set and system."""
    work = frame[["trip_id", "poi_id", "label"]].assign(score=score.to_numpy())
    out: dict[str, float] = {}
    for trip_id, g in work.groupby("trip_id", sort=True):
        ideal_labels = all_labels.get(str(trip_id))
        if ideal_labels is None or ideal_labels.max() < 1:
            continue
        idcg = dcg_at_k(np.sort(ideal_labels)[::-1].astype(np.int64), 10)
        order = _rank_order(g["poi_id"].to_numpy(dtype=object), g["score"].to_numpy(dtype=float))
        out[str(trip_id)] = dcg_at_k(g["label"].to_numpy(dtype=np.int64)[order], 10) / idcg
    return pd.Series(out, name="ndcg@10_end_to_end").rename_axis("trip_id")


def run_decomposition(
    data_dir: Path,
    artifacts_dir: Path,
    results_dir: Path,
    feature_cfg: FeatureBuildConfig,
    legacy_cfg: CandidatesConfig,
    lab: Lab,
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    from poi_rank.eval.compose import write_part

    trips = pd.read_parquet(data_dir / "trips.parquet")
    travelers = pd.read_parquet(data_dir / "travelers.parquet")
    pf = pd.read_parquet(data_dir / "poi_features.parquet")
    tf_all = pd.read_parquet(data_dir / "traveler_features.parquet")
    holdout_random = pd.read_parquet(data_dir / "interactions_holdout_random.parquet")
    budget = feature_cfg.traveler_features.budget_target_price_level
    holdout_ids = set(trips.loc[trips["is_holdout"], "trip_id"])
    train_ids = set(trips.loc[~trips["is_holdout"], "trip_id"])
    legacy = _legacy_candidates(data_dir, lab, legacy_cfg)

    def frame(cand: pd.DataFrame, ids: set[str], interactions: pd.DataFrame) -> pd.DataFrame:
        return build_ranking_frame(
            cand, ids, interactions, trips, travelers, lab.pois_df, pf, tf_all, budget
        )

    frames = {
        "learned_retriever": lab.holdout_frame,
        "legacy_6_channel": frame(legacy, holdout_ids, holdout_random),
    }
    legacy_train = frame(legacy, train_ids, lab.interactions_train)
    booster_l, num_l, cat_l, _, _ = fit_lgb(lab, train_frame=legacy_train)
    shipped = lm.load_boosters(artifacts_dir)["lambdamart_ips"]
    num = bl.numeric_feature_columns(lab.holdout_frame)
    cat = bl.categorical_feature_columns(lab.holdout_frame)
    oracle_dir = oracle_reader.oracle_dir_for(data_dir)
    cc_cfg = lab.model_cfg.baselines.content_cosine

    all_labels = {
        str(t): g.to_numpy()
        for t, g in label_by_trip_poi(
            holdout_random, build_poi_id_canonical_map(lab.pois_df)
        ).groupby(level=0)
    }
    e2e: dict[str, dict[str, pd.Series]] = {}
    per_trip: dict[str, dict[str, pd.Series]] = {}
    sizes: dict[str, float] = {}
    for set_name, fr in frames.items():
        scores = {
            "random": bl.baseline_random(fr, lab.model_cfg.seed).score,
            "popularity": bl.baseline_popularity(fr).score,
            "content_cosine": bl.baseline_content_cosine(fr, cc_cfg).score,
            "lambdamart_ips_shipped": lm.score_booster(shipped, fr, num, cat),
            "oracle": oracle_reader.oracle_ceiling_scores(fr[["trip_id", "poi_id"]], oracle_dir),
        }
        if set_name == "legacy_6_channel":
            scores["lambdamart_ips_retrained_on_this_set"] = score_lgb(booster_l, num_l, cat_l, fr)
        per_trip[set_name] = {k: per_trip_ndcg10(fr, v) for k, v in scores.items()}
        e2e[set_name] = {k: end_to_end_ndcg10(fr, v, all_labels) for k, v in scores.items()}
        sizes[set_name] = float(fr.groupby("trip_id").size().mean())

    table: dict[str, dict[str, Any]] = {}
    table_e2e: dict[str, dict[str, Any]] = {}
    for set_name, systems in per_trip.items():
        table[set_name] = {k: summarize_ndcg(v, eval_cfg) for k, v in systems.items()}
        table_e2e[set_name] = {k: summarize_ndcg(v, eval_cfg) for k, v in e2e[set_name].items()}

    def wil(a: pd.Series, b: pd.Series) -> dict[str, float]:
        r = paired_wilcoxon(a, b)
        return {"p_value": r.p_value, "n_pairs": r.n_pairs}

    new, old = per_trip["learned_retriever"], per_trip["legacy_6_channel"]
    tests = {
        "learned: primary vs content_cosine": wil(
            new["lambdamart_ips_shipped"], new["content_cosine"]
        ),
        "legacy: primary(retrained) vs content_cosine": wil(
            old["lambdamart_ips_retrained_on_this_set"], old["content_cosine"]
        ),
        "retrieval effect, primary (learned vs legacy-retrained)": wil(
            new["lambdamart_ips_shipped"], old["lambdamart_ips_retrained_on_this_set"]
        ),
        "retrieval effect, content_cosine": wil(new["content_cosine"], old["content_cosine"]),
        "retrieval effect, popularity": wil(new["popularity"], old["popularity"]),
    }
    ne, oe = e2e["learned_retriever"], e2e["legacy_6_channel"]
    tests_e2e = {
        "learned: primary vs content_cosine": wil(
            ne["lambdamart_ips_shipped"], ne["content_cosine"]
        ),
        "legacy: primary(retrained) vs content_cosine": wil(
            oe["lambdamart_ips_retrained_on_this_set"], oe["content_cosine"]
        ),
        "retrieval effect, primary (learned vs legacy-retrained)": wil(
            ne["lambdamart_ips_shipped"], oe["lambdamart_ips_retrained_on_this_set"]
        ),
        "retrieval effect, content_cosine": wil(ne["content_cosine"], oe["content_cosine"]),
        "retrieval effect, popularity": wil(ne["popularity"], oe["popularity"]),
    }
    m = {s: {k: v["mean"] for k, v in t.items()} for s, t in table_e2e.items()}
    lg, lc = m["learned_retriever"], m["legacy_6_channel"]
    total = lg["lambdamart_ips_shipped"] - lc["popularity"]
    retrieval_part = lg["popularity"] - lc["popularity"]
    payload = {
        "ndcg10_candidate_relative": table,
        "ndcg10_end_to_end": table_e2e,
        "mean_candidates_per_trip": sizes,
        "paired_tests_candidate_relative": tests,
        "paired_tests_end_to_end": tests_e2e,
        "decomposition_of_popularity_to_primary_gain_end_to_end": {
            "popularity_on_legacy": lc["popularity"],
            "primary_on_learned": lg["lambdamart_ips_shipped"],
            "total_gain": total,
            "from_retrieval_alone_popularity_learned_minus_legacy": retrieval_part,
            "from_ranker_on_legacy_set": lc["lambdamart_ips_retrained_on_this_set"]
            - lc["popularity"],
            "from_ranker_on_learned_set": lg["lambdamart_ips_shipped"] - lg["popularity"],
            "primary_minus_content_cosine_on_learned": lg["lambdamart_ips_shipped"]
            - lg["content_cosine"],
            "primary_minus_content_cosine_on_legacy_retrained": lc[
                "lambdamart_ips_retrained_on_this_set"
            ]
            - lc["content_cosine"],
        },
        "note": "Baselines read only frame columns, so each is scored over exactly the candidates "
        "of its column; content-cosine is like-for-like with the primary system.",
    }
    write_part(results_dir, "retrieval_ranking_decomposition", payload)
    return payload
