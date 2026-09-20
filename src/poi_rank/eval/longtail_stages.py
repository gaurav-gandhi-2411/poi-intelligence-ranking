"""E3: long-tail precision at every stage of the serving pipeline.

Stages (same holdout trips, same labels, same share/precision definition as the scorecard --
`eval/longtail.py`): the candidate pool itself (positive rate among long-tail candidates); the raw
ranker top-10; top-10 after the hard-constraint gate (raw-score order); top-10 by the utility;
and the final MMR-diversified list. The first difference between consecutive rows that
drops precision names the stage that destroys it. An MMR-lambda sweep is added as a DIAGNOSTIC
(post-hoc on the holdout; not a selection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from poi_rank.eval import longtail as lt
from poi_rank.eval.compose import write_part
from poi_rank.eval.personalization import top10_lists_from_payload
from poi_rank.scoring import diversity as div


def _lists(frame: pd.DataFrame, score: pd.Series, k: int = 10) -> dict[str, list[str]]:
    ranked = frame[["trip_id", "poi_id"]].assign(score=score.to_numpy())
    ranked = ranked.sort_values(["trip_id", "score", "poi_id"], ascending=[True, False, True])
    top = ranked.groupby("trip_id", sort=True).head(k)
    return {str(t): g["poi_id"].tolist() for t, g in top.groupby("trip_id", sort=True)}


def run_longtail_stages(
    scoring_result: dict[str, Any],
    pois_df: pd.DataFrame,
    cutoff: float,
    diversity_cfg: Any,
    lambda_default: float,
    top_k: int,
    results_dir: Path,
) -> dict[str, Any]:
    full: pd.DataFrame = scoring_result["full_frame"]
    pop = dict(zip(pois_df["poi_id"], pois_df["pop_pct"], strict=True))
    labels = {
        (str(t), str(p)): int(v)
        for t, p, v in zip(full["trip_id"], full["poi_id"], full["label"], strict=True)
    }
    raw = scoring_result["raw_score"]
    survivors = full.loc[full["hard_gate"] == 1.0]

    def measure(lists: dict[str, list[str]]) -> dict[str, float | int | None]:
        d = lt.longtail_share_and_precision(lists, pop, cutoff, labels).to_dict()
        return {
            "long_tail_share": d["share"],
            "long_tail_precision": d["precision"],
            "n_long_tail_recommended": d["n_longtail_recommended"],
            "n_total_recommended": d["n_total_recommended"],
        }

    is_lt = full["poi_id"].map(pop) < cutoff
    pool_lt = full.loc[is_lt]
    stages: dict[str, Any] = {
        "0_candidate_pool (positive rate among long-tail candidates)": {
            "long_tail_share": float(is_lt.mean()),
            "long_tail_precision": float((pool_lt["label"] >= 1).mean()),
            "head_positive_rate": float((full.loc[~is_lt, "label"] >= 1).mean()),
        },
        "1_raw_ranker_top10": measure(_lists(full, raw)),
        "2_after_hard_gate_raw_order": measure(_lists(survivors, raw.loc[survivors.index])),
        "3_after_utility": measure(_lists(survivors, survivors["utility"])),
        "4_final_after_mmr": measure(top10_lists_from_payload(scoring_result["payload"])),
    }
    sweep = {}
    for lam in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        reranked = div.mmr_rerank_all_trips(survivors, diversity_cfg, lam, top_k)
        lists = {
            str(t): g.sort_values("mmr_rank")["poi_id"].tolist()
            for t, g in reranked.groupby("trip_id")
        }
        sweep[f"{lam:g}"] = measure(lists)
    payload = {
        "stages": stages,
        "mmr_lambda_diagnostic_post_hoc_on_holdout": sweep,
        "lambda_default": lambda_default,
    }
    write_part(results_dir, "longtail_stages", payload)
    return payload
