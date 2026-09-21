"""Experiment L3b helpers: compare candidate-set designs on the train-carved VALIDATION trips
(`docs/experiments/L-final.md`, pre-registered before any run).

Everything is measured on validation trips with the logged (exposed) training interactions as the
ground truth, IPS-weighted, and with a candidate-set-INDEPENDENT NDCG denominator, so a design that
retrieves more positives is rewarded and one that misses a positive is charged for it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

EVAL_CLIP = 20.0  # same IPS clip as `eval/ranker_sweep.py::ips_weighted_ndcg10`
LONG_TAIL_RECALL_FLOOR = 0.75  # brief section 9 (Gate-B, blocking)
EFFECTIVE_K_CAP = 300.0  # serving budget (Gate-B amendment L3a)
ADOPTION_BAR = 0.010  # the bar used throughout the project (experiment H, E2)
_DISC = 1.0 / np.log2(np.arange(2, 12))


def ips_gain(label: FloatArray, p_expose: FloatArray) -> FloatArray:
    """`(2^label - 1) / clip(p_expose, 1/EVAL_CLIP, 1)`; NaN propensity (unexposed) has label 0."""
    w = 1.0 / np.clip(np.nan_to_num(p_expose, nan=1.0), 1.0 / EVAL_CLIP, 1.0)
    out: FloatArray = (np.power(2.0, label) - 1.0) * w
    return out


def ideal_dcg_by_trip(exposed: pd.DataFrame) -> dict[str, float]:
    """`exposed`: one row per logged `(trip_id, poi_id)` with `label` and `p_expose`. The ideal
    DCG@10 of each trip over EVERY exposed label, independent of any candidate set."""
    gains = ips_gain(
        exposed["label"].to_numpy(dtype=np.float64), exposed["p_expose"].to_numpy(dtype=np.float64)
    )
    work = pd.DataFrame({"trip_id": exposed["trip_id"].to_numpy(), "gain": gains})
    out: dict[str, float] = {}
    for trip_id, g in work.groupby("trip_id", sort=True):
        top = np.sort(g["gain"].to_numpy())[::-1][:10]
        out[str(trip_id)] = float((top * _DISC[: len(top)]).sum())
    return out


def fixed_denominator_ndcg10(
    frame: pd.DataFrame,
    score: FloatArray,
    p_expose: pd.Series,
    ideal: dict[str, float],
) -> float:
    """Mean over trips (with a positive ideal) of DCG@10 of the design's own candidates ranked by
    `score`, divided by the trip's candidate-set-independent ideal DCG@10."""
    gains = ips_gain(frame["label"].to_numpy(dtype=np.float64), p_expose.to_numpy(dtype=np.float64))
    work = frame[["trip_id", "poi_id"]].assign(score=score, gain=gains)
    vals: list[float] = []
    for trip_id, g in work.groupby("trip_id", sort=True):
        idcg = ideal.get(str(trip_id), 0.0)
        if idcg <= 0.0:
            continue
        order = np.lexsort((g["poi_id"].to_numpy(dtype=object), -g["score"].to_numpy()))
        top = g["gain"].to_numpy()[order][:10]
        vals.append(float((top * _DISC[: len(top)]).sum() / idcg))
    return float(np.mean(vals))


def candidate_recall(
    cand_sets: dict[str, set[str]],
    exposed: pd.DataFrame,
    trips: set[str],
    pop_pct_by_poi: dict[str, float],
    long_tail_cutoff: float,
    dest_pois: dict[str, set[str]],
    dest_of_trip: dict[str, str],
) -> dict[str, float]:
    """Per-trip IPS-weighted recall of exposed positives (label >= 1) by the candidate set, and
    the matching chance baseline (share of the stratum's destination POIs the set contains),
    averaged over trips with at least one positive in the stratum."""
    pos = exposed.loc[(exposed["label"] >= 1) & exposed["trip_id"].isin(trips)]
    w = 1.0 / np.clip(
        np.nan_to_num(pos["p_expose"].to_numpy(dtype=np.float64), nan=1.0), 1.0 / EVAL_CLIP, 1.0
    )
    pos = pos.assign(w=w)
    out: dict[str, float] = {}
    strata: dict[str, Callable[[str], bool]] = {
        "overall": lambda p: True,
        "long_tail": lambda p: pop_pct_by_poi.get(p, 1.0) < long_tail_cutoff,
    }
    for name, in_stratum in strata.items():
        recalls: list[float] = []
        chances: list[float] = []
        for trip_id, g in pos.groupby("trip_id", sort=True):
            g = g.loc[[in_stratum(p) for p in g["poi_id"]]]
            if g.empty:
                continue
            cset = cand_sets.get(str(trip_id), set())
            hit = g.loc[g["poi_id"].isin(cset), "w"].sum()
            recalls.append(float(hit / g["w"].sum()))
            catalog = {p for p in dest_pois[dest_of_trip[str(trip_id)]] if in_stratum(p)}
            chances.append(len(cset & catalog) / len(catalog))
        out[f"{name}_recall"] = float(np.mean(recalls))
        out[f"{name}_chance"] = float(np.mean(chances))
        out[f"{name}_lift"] = out[f"{name}_recall"] - out[f"{name}_chance"]
    return out


def select_design(designs: dict[str, dict[str, Any]], incumbent: str = "A") -> dict[str, Any]:
    """The pre-registered rule. `designs[name]` needs `v_ndcg10` (seed mean), `long_tail_recall`
    and `effective_k`."""
    feasible = {
        n: d
        for n, d in designs.items()
        if d["long_tail_recall"] >= LONG_TAIL_RECALL_FLOOR and d["effective_k"] <= EFFECTIVE_K_CAP
    }
    if not feasible:
        return {"adopted": incumbent, "reason": "no design is feasible; the incumbent stays"}
    best = max(feasible, key=lambda n: designs[n]["v_ndcg10"])
    if incumbent not in feasible:
        return {"adopted": best, "reason": "the incumbent fails a constraint; best feasible design"}
    gain = designs[best]["v_ndcg10"] - designs[incumbent]["v_ndcg10"]
    if best != incumbent and gain >= ADOPTION_BAR:
        return {"adopted": best, "reason": "beats the incumbent by the adoption bar", "gain": gain}
    return {
        "adopted": incumbent,
        "reason": "no feasible design beats the incumbent by the adoption bar",
        "best_feasible": best,
        "gain_of_best_feasible": gain,
    }
