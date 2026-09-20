"""`compose`: the ONLY writer of `results/metrics.json`.

Every pipeline stage writes its own `results/parts/<stage>.json`; this module merges them.
The old design had `evaluate` write `metrics.json` and later stages (`lodo`, ...) read-modify-
write the same file -- a lost-update hazard that clobbered results twice. One writer over
independent inputs cannot lose an update, and a part that is missing is reported as missing
rather than silently absent from the scorecard.

Layout of the composed file:
  * `evaluate.json` is spread at the top level (the evaluation harness payload -- systems,
    bias-gap table, personalization, ablations, ... -- keeps the keys downstream code/docs read).
  * every other part nests under its stem name (`lodo`, `gate_dgp`, `gate_representation`,
    `representation`, `decision_register`, ...).
  * `parts_present` lists exactly which parts were composed.

Deterministic: sorted keys, no timestamps, byte-identical for identical parts
(`tests/test_compose.py`, `make audit`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PARTS_DIRNAME = "parts"
METRICS_FILENAME = "metrics.json"
EVALUATE_PART = "evaluate"

# stem -> key in metrics.json. `evaluate` is spread flat (None). Parts absent from disk are
# skipped and named in `parts_present`; a REQUIRED part missing is an error.
COMPOSED_PARTS: dict[str, str | None] = {
    EVALUATE_PART: None,
    "train": "train",
    "lodo": "lodo",
    "dgp_gate": "gate_dgp",
    "dgp_diagnostics": "dgp_diagnostics",
    "gate_representation": "gate_representation",
    "representation": "representation",
    "decision_register": "decision_register",
    "seed_replication": "seed_replication",
    "a3_step0": "a3_step0",
    "a3_pairfeat": "a3_pairfeat",
    "a3_retriever_grid": "a3_retriever_grid",
    "retrieval_ranking_decomposition": "retrieval_ranking_decomposition",
    "longtail_stages": "longtail_stages",
    "e4_k_sweep": "e4_k_sweep",
    "ranker_sweep": "ranker_sweep",
    "fresh_clone_verification": "fresh_clone_verification",
    "h_experiments": "h_experiments",
    "timings": "timings",
    "timings_full": "timings_full",
}
REQUIRED_PARTS = (EVALUATE_PART,)


def derived_numbers(m: dict[str, Any]) -> dict[str, Any]:
    """Quantities the docs quote that are pure functions of composed parts (differences and
    ratios). They live in metrics.json so every number in the docs traces to a stored value."""
    out: dict[str, Any] = {}
    systems = m.get("systems", {})
    if {"popularity", "content_cosine", "lambdamart_ips"} <= systems.keys():
        pop, cos, prim = (
            systems[k]["metrics"]["ndcg@10"]["mean"]
            for k in ("popularity", "content_cosine", "lambdamart_ips")
        )
        out["ndcg10_gain_popularity_to_content_cosine"] = cos - pop
        out["ndcg10_gain_content_cosine_to_primary"] = prim - cos
    stages = m.get("longtail_stages", {}).get("stages")
    if stages:
        base_key = next(k for k in stages if k.startswith("0_candidate_pool"))
        base = stages[base_key]["long_tail_precision"]
        lift = {"base_rate": base}
        for name, key in (
            ("raw_ranker", "1_raw_ranker_top10"),
            ("after_gate", "2_after_hard_gate_raw_order"),
            ("after_utility", "3_after_utility"),
            ("final_after_mmr", "4_final_after_mmr"),
        ):
            lift[name] = stages[key]["long_tail_precision"] / base
        out["longtail_lift_over_base_rate"] = lift
        sweep = m["longtail_stages"].get("mmr_lambda_diagnostic_post_hoc_on_holdout", {})
        out["mmr_lambda_lift_over_base_rate"] = {
            lam: v["long_tail_precision"] / base for lam, v in sweep.items()
        }
    seeds = m.get("seed_replication", {}).get("metrics", {})
    if "ndcg10_lambdamart_ips" in seeds and "ndcg10_content_cosine" in seeds:
        from scipy import stats

        prim = seeds["ndcg10_lambdamart_ips"]["per_seed"]
        cos = seeds["ndcg10_content_cosine"]["per_seed"]
        keys = sorted(prim)
        a = [prim[k] for k in keys]
        b = [cos[k] for k in keys]
        gaps = [x - y for x, y in zip(a, b, strict=True)]
        mean_gap = sum(gaps) / len(gaps)
        sd_gap = (sum((g - mean_gap) ** 2 for g in gaps) / (len(gaps) - 1)) ** 0.5
        out["seed_paired_primary_vs_content_cosine"] = {
            "n_seeds": len(gaps),
            "n_primary_above": sum(g > 0 for g in gaps),
            "mean_gap": mean_gap,
            "sd_gap": sd_gap,
            "paired_t_p": float(stats.ttest_rel(a, b).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
        }
    abl = m.get("h_experiments", {}).get("holdout_ablation_no_h_seed42")
    if abl and "systems" in m:
        final = m["systems"]["lambdamart_ips"]["metrics"]["ndcg@10"]["mean"]
        no_h = abl["headline"]["ndcg10_lambdamart_ips"]
        out["h_holdout_effect"] = {"final": final, "no_h": no_h, "gain": final - no_h}
    sweep = m.get("ranker_sweep")
    if sweep:
        out["ranker_sweep_margin"] = {
            "winner_minus_shipped": sweep["winner"]["stage2_mean_val_ips_weighted_ndcg10"]
            - sweep["previous_shipped_config_4_seed"]["mean"],
            "adoption_bar": 0.010,
        }
    dr1 = next(
        (r for r in m.get("decision_register", {}).get("rows", []) if r["id"] == "DR1"), None
    )
    if dr1 and dr1.get("results"):
        res = dr1["results"]
        out["dr1_hard_constraints"] = {
            "brief_additive_violations": res["additive_no_gate (brief)"][
                "hard_constraint_violations_in_top10"
            ],
            "brief_additive_trips_with_violation": res["additive_no_gate (brief)"][
                "trips_with_violation"
            ],
            "shipped_multiplicative_gated_violations": res["multiplicative_gated (production)"][
                "hard_constraint_violations_in_top10"
            ],
            "n_trips": res["multiplicative_gated (production)"]["n_trips"],
        }
    return out


def parts_dir(results_dir: Path) -> Path:
    return results_dir / PARTS_DIRNAME


def write_part(results_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    """Write one stage's part file (deterministic JSON). Stages call this; only `compose_metrics`
    ever writes `metrics.json`."""
    directory = parts_dir(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compose_metrics(results_dir: Path) -> Path:
    directory = parts_dir(results_dir)
    for stem in REQUIRED_PARTS:
        if not (directory / f"{stem}.json").exists():
            raise FileNotFoundError(
                f"required part results/{PARTS_DIRNAME}/{stem}.json is missing -- run the "
                f"`{stem}` stage before `compose`."
            )
    composed: dict[str, Any] = {}
    present: list[str] = []
    for stem, key in COMPOSED_PARTS.items():
        path = directory / f"{stem}.json"
        if not path.exists():
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        present.append(stem)
        if key is None:
            composed.update(content)
        else:
            composed[key] = content
    composed["parts_present"] = present
    composed["derived"] = derived_numbers(composed)
    out = results_dir / METRICS_FILENAME
    out.write_text(json.dumps(composed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
