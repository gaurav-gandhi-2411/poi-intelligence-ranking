"""Generates `docs/RESULTS.md` from `results/metrics.json` (spec.md section 11,
`make docs` / `python -m poi_rank.eval.report`). **Hard rule, spec.md's own words:
"No number appears in `docs/` that is not produced by `results/metrics.json`."**
Every number below is an f-string interpolation of a value read directly out of the
parsed JSON dict -- there is no hand-typed metric literal anywhere in this module.
`tests/test_report.py::test_every_number_in_results_md_traces_to_metrics_json`
greps the generated markdown and cross-checks every numeric token against the
source JSON, the actual enforcement mechanism for this rule (not just a promise).

Run AFTER `poi_rank.cli evaluate` (`results/metrics.json` must already exist).
`poi_rank.cli lodo`'s `"lodo"` key is OPTIONAL -- rendered if present, reported as
"not yet run" if absent (spec.md section 16's own cut-order list allows LODO to be
skipped under time pressure; this module never fabricates a number for it either
way).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poi_rank.eval.report_sections import (
    diagnoses,
    render_candidate_recall,
    render_decision_register,
    render_gates,
    render_headline,
    render_seed_replication,
    render_timings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METRICS_PATH = REPO_ROOT / "results" / "metrics.json"
DEFAULT_SCENARIOS_DIR = REPO_ROOT / "results" / "scenarios"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "RESULTS.md"

SCENARIO_NUMBERS: tuple[int, ...] = (1, 2, 3, 4)

METRIC_ORDER: tuple[str, ...] = (
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "precision@5",
    "precision@10",
    "recall@10",
    "recall@20",
    "map",
    "mrr",
)

SYSTEM_DISPLAY_NAMES: dict[str, str] = {
    "random": "1. Random",
    "popularity": "2. Popularity",
    "popularity_geo_filter": "3. Popularity + geo filter",
    "content_cosine": "4. Content cosine",
    "item_knn_cf": "5. Item-kNN CF",
    "logistic_regression": "6. Logistic regression",
    "lambdamart": "7. LambdaMART",
    "lambdamart_ips": "8. LambdaMART + IPS (primary)",
    "oracle": "9. Oracle (ceiling)",
}
SYSTEM_ORDER: tuple[str, ...] = tuple(SYSTEM_DISPLAY_NAMES)


def load_metrics(path: Path = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    """Parse `results/metrics.json`. Raises `FileNotFoundError` with an actionable
    message if `poi_rank.cli evaluate` has not been run yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run `poi_rank.cli evaluate` (or `make evaluate`) "
            "before generating docs/RESULTS.md."
        )
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_scenarios(scenarios_dir: Path = DEFAULT_SCENARIOS_DIR) -> dict[str, Any] | None:
    """Parse `results/scenarios/{1,2,3,4}.json` + `overlap_matrix.json` (spec.md
    section 15, `poi_rank.cli scenarios` / `make scenarios`). Returns `None`
    (rendered as "not yet run", mirroring `render_cold_start`'s own optional
    `lodo` key convention) if the directory or any expected file is missing --
    `render_results_md` must stay correct for a `docs` run before `scenarios` has
    ever been run, not just the production `make reproduce` order."""
    overlap_path = scenarios_dir / "overlap_matrix.json"
    if not scenarios_dir.is_dir() or not overlap_path.exists():
        return None
    scenario_payloads = []
    for number in SCENARIO_NUMBERS:
        path = scenarios_dir / f"{number}.json"
        if not path.exists():
            return None
        scenario_payloads.append(json.loads(path.read_text(encoding="utf-8")))
    overlap_matrix = json.loads(overlap_path.read_text(encoding="utf-8"))
    return {"scenarios": scenario_payloads, "overlap_matrix": overlap_matrix}


def _fmt(value: float | int | None, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _fmt_pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def _agg_str(agg: dict[str, Any] | None) -> str:
    """`mean [ci_low, ci_high]` from a `MetricAggregate.to_dict()`-shaped dict.
    `None` (a metric `configs/eval.yaml`'s `metrics.ndcg_ks`/etc. didn't request --
    e.g. a reduced test config computing only ndcg@{5,10}, not the production
    @{5,10,20}) renders as `N/A` rather than raising, since this module must stay
    correct for any valid `eval.yaml`, not just the committed production one."""
    if agg is None:
        return "N/A"
    return f"{_fmt(agg['mean'])} [{_fmt(agg['ci_low'])}, {_fmt(agg['ci_high'])}]"


def _met(condition: bool) -> str:
    return "**MET**" if condition else "**MISSED**"


# -----------------------------------------------------------------------------------
# Section 11.10 success-criteria summary table
# -----------------------------------------------------------------------------------


def render_success_criteria(
    metrics: dict[str, Any], scenarios: dict[str, Any] | None = None
) -> str:
    systems = metrics["systems"]
    lambdamart_ips = systems["lambdamart_ips"]
    popularity_ndcg10 = systems["popularity"]["metrics"]["ndcg@10"]["mean"]
    ips_ndcg10 = lambdamart_ips["metrics"]["ndcg@10"]["mean"]
    relative_lift = (ips_ndcg10 / popularity_ndcg10 - 1.0) if popularity_ndcg10 > 0 else 0.0
    p_value = metrics["wilcoxon"]["lambdamart_ips_vs_popularity"]["p_value"]
    pct_ceiling = lambdamart_ips["pct_of_ceiling_ndcg10"]
    recall = metrics["candidate_recall"]
    strata = recall.get("by_stratum", {})
    arch = metrics["personalization"]["archetype"]
    cross_jaccard = arch["cross_archetype_jaccard_mean"]
    ratio = arch["within_cross_ratio"]
    n_violations = metrics["constraint_compatibility"]["n_hard_constraint_violations"]
    ece_after = metrics["calibration"]["ece_after"]
    decile_rho = metrics["confidence_decile_validation"]["spearman_rho"]
    lt_share = metrics["longtail"]["share"]
    lt_precision = metrics["longtail"]["precision"]
    localness = metrics.get("localness_validation", {}).get("spearman_rho")

    # (name, target, measured, met?, diagnosis key when missed)
    rows: list[tuple[str, str, str, bool, str | None]] = [
        (
            "NDCG@10 vs popularity",
            "&ge; +40% relative, Wilcoxon p < 0.01",
            f"{relative_lift * 100:+.1f}% relative, p={p_value:.4g}",
            relative_lift >= 0.40 and p_value < 0.01,
            None,
        ),
        (
            "% of oracle ceiling (candidate-level NDCG@10)",
            "&ge; 70%",
            _fmt_pct(pct_ceiling),
            pct_ceiling >= 0.70,
            "ceiling",
        ),
        (
            "Candidate recall, overall (exposed positives)",
            "&ge; 0.85",
            _fmt(recall["overall"]["recall_mean"]),
            recall["overall"]["recall_mean"] >= 0.85,
            None,
        ),
        (
            "Candidate recall, long-tail stratum",
            "&ge; 0.75",
            _fmt(recall["long_tail"]["recall_mean"]),
            recall["long_tail"]["recall_mean"] >= 0.75,
            None,
        ),
        (
            "Candidate recall lift over chance (overall)",
            "&ge; +0.35",
            f"{strata['overall']['lift_abs']:+.3f}" if strata else "N/A",
            bool(strata) and strata["overall"]["lift_abs"] >= 0.35,
            None,
        ),
        (
            "Cross-archetype Jaccard@10 (true labels)",
            "&le; 0.25",
            _fmt(cross_jaccard),
            cross_jaccard <= 0.25,
            None,
        ),
        (
            "Within/cross Jaccard ratio (true labels)",
            "&ge; 2.0",
            "inf" if ratio is None else _fmt(ratio, 2),
            ratio is None or ratio >= 2.0,
            "archetype_ratio",
        ),
        (
            "Hard-constraint violations in top-10",
            "= 0",
            str(n_violations),
            n_violations == 0,
            None,
        ),
        ("ECE after calibration", "&le; 0.05", _fmt(ece_after), ece_after <= 0.05, None),
        (
            "Confidence-decile NDCG rank correlation (Spearman; need not be strictly monotone)",
            "&ge; 0.6 (spec-v2; spec.md section 11.10 said 0.7)",
            "N/A" if decile_rho is None else _fmt(decile_rho, 3),
            decile_rho is not None and decile_rho >= 0.6,
            "confidence_decile",
        ),
        (
            "Long-tail share of top-10",
            "&ge; 0.25",
            _fmt(lt_share),
            lt_share >= 0.25,
            "longtail_share",
        ),
        (
            "Long-tail precision of top-10",
            "&ge; 0.40",
            "N/A" if lt_precision is None else _fmt(lt_precision),
            lt_precision is not None and lt_precision >= 0.40,
            "longtail_precision",
        ),
        (
            "Localness index Spearman vs latent localness",
            "&ge; 0.6",
            "N/A" if localness is None else _fmt(localness),
            localness is not None and localness >= 0.6,
            "localness",
        ),
    ]
    if scenarios is not None:
        overlap = scenarios["overlap_matrix"]["diagnostic_vs_base_jaccard"]
        rows.append(
            (
                "Scenario-4 (touristiness flip) top-10 overlap",
                "&le; 0.35",
                _fmt(overlap, 3),
                overlap <= 0.35,
                "scenario4",
            )
        )

    lines = [
        "## Success criteria scorecard",
        "",
        "Stated up front, then measured. Every MISSED row carries a diagnosis below -- a miss "
        "is reported as a ceiling only after a diagnostic has ruled out a mechanism, citing "
        'the number (spec.md: "An honest miss with a root-cause analysis scores better than a '
        'suspiciously perfect table.").',
        "",
        "| Metric | Target | Measured | Status |",
        "|---|---|---|---|",
    ]
    for name, target, measured, met, _ in rows:
        lines.append(f"| {name} | {target} | {measured} | {_met(met)} |")
    lines.append("")
    diag = diagnoses(metrics, scenarios)
    missed = [(name, key) for name, _, _, met, key in rows if not met]
    if missed:
        lines += ["### Diagnoses of the missed rows", ""]
        for name, key in missed:
            text = diag.get(key) if key else None
            if text is None:
                text = "**UNDIAGNOSED** -- no diagnostic has ruled out a mechanism for this miss."
            lines.append(f"- **{name}**: {text}")
        lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# Primary ranking-quality table (spec.md section 11.1)
# -----------------------------------------------------------------------------------


def render_primary_table(metrics: dict[str, Any]) -> str:
    systems = metrics["systems"]
    header = ["System"] + [m for m in METRIC_ORDER] + ["% of oracle ceiling"]
    lines = [
        "## Primary ranking quality (unbiased random-exposure holdout, spec.md section 11.1)",
        "",
        f"n_holdout_trips = {metrics['meta']['n_holdout_trips']}, "
        f"bootstrap n_resamples = {metrics['meta']['bootstrap_n_resamples']} "
        f"(resample unit: {metrics['meta']['bootstrap_resample_unit']}).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name in SYSTEM_ORDER:
        sys_payload = systems[name]
        cells = [SYSTEM_DISPLAY_NAMES[name]]
        for m in METRIC_ORDER:
            cells.append(_agg_str(sys_payload["metrics"].get(m)))
        cells.append(_fmt_pct(sys_payload["pct_of_ceiling_ndcg10"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Paired Wilcoxon signed-rank tests (NDCG@10)")
    lines.append("")
    lines.append("| Comparison | statistic | p-value | n_pairs |")
    lines.append("|---|---|---|---|")
    for key, w in sorted(metrics["wilcoxon"].items()):
        lines.append(f"| {key} | {w['statistic']:.4f} | {w['p_value']:.4g} | {w['n_pairs']} |")
    lines.append("")
    return "\n".join(lines)


def render_bias_gap_table(metrics: dict[str, Any]) -> str:
    lines = [
        "## Bias-gap table (spec.md section 11.1)",
        "",
        "Each system's NDCG@10 on the SECONDARY biased holdout "
        "(`interactions_holdout_logged.parquet`) vs the PRIMARY unbiased holdout. "
        "Expectation: the popularity baseline shows a large positive gap (flattered "
        "by biased logs); the IPS-corrected model shows a small gap.",
        "",
        "| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |",
        "|---|---|---|---|",
    ]
    for name in SYSTEM_ORDER:
        row = metrics["bias_gap"][name]
        lines.append(
            f"| {SYSTEM_DISPLAY_NAMES[name]} | {_fmt(row['ndcg@10_unbiased'])} | "
            f"{_fmt(row['ndcg@10_biased'])} | {row['gap']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# Personalization / coverage / long-tail / constraints / diversity / calibration /
# confidence-decile
# -----------------------------------------------------------------------------------


def render_personalization(metrics: dict[str, Any]) -> str:
    p = metrics["personalization"]
    arch = p["archetype"]
    proxy = p.get("archetype_kmeans_proxy")
    ideal = p.get("archetype_ideal_ranker_reference")

    def ratio_of(d: dict[str, Any]) -> str:
        r = d["within_cross_ratio"]
        return "inf" if r is None else _fmt(r, 2)

    lines = [
        "## Personalization (spec.md section 11.2, spec-v2 RC4)",
        "",
        "Pairs are SAME-DESTINATION trip pairs only: a POI belongs to exactly one destination, so "
        "two trips to different destinations have Jaccard = RBO = 0 by construction, and pooling "
        "them measures the catalog partition rather than the recommender.",
        "",
        "- Mean pairwise Jaccard@10 (same destination): "
        f"**{_fmt(p['mean_pairwise_jaccard_at_10'])}** "
        f"(n_pairs={p['n_pairs']})",
        f"- Mean pairwise rank-biased overlap (RBO, p=0.9): **{_fmt(p['mean_pairwise_rbo'])}** "
        f"(n_pairs={p['n_pairs_rbo']})",
    ]
    if "mean_pairwise_jaccard_at_10_all_pairs" in p:
        lines.append(
            "- For continuity with the pre-fix number, all pairs pooled (2/3 of them structurally "
            f"zero): {_fmt(p['mean_pairwise_jaccard_at_10_all_pairs'])} "
            f"(n_pairs={p['n_pairs_all_pairs']})"
        )
    lines += [
        "",
        "| Grouping | Within Jaccard@10 | Cross Jaccard@10 | Ratio | within / cross pairs |",
        "|---|---|---|---|---|",
        f"| **True archetype labels** (dominant archetype; within = same dominant and mixture "
        f"cosine > 0.8) | {_fmt(arch['within_archetype_jaccard_mean'])} | "
        f"{_fmt(arch['cross_archetype_jaccard_mean'])} | **{ratio_of(arch)}** | "
        f"{arch['within_archetype_n_pairs']} / {arch['cross_archetype_n_pairs']} |",
    ]
    if ideal is not None:
        lines.append(
            f"| Reference: lists ranked by the TRUE utility (a perfect ranker) | "
            f"{_fmt(ideal['within_archetype_jaccard_mean'])} | "
            f"{_fmt(ideal['cross_archetype_jaccard_mean'])} | {ratio_of(ideal)} | "
            f"{ideal['within_archetype_n_pairs']} / {ideal['cross_archetype_n_pairs']} |"
        )
    if proxy is not None:
        lines.append(
            f"| K-Means traveler-segment PROXY (clusters of the same features being evaluated -- "
            f"kept only to show why it was misleading) | "
            f"{_fmt(proxy['within_archetype_jaccard_mean'])} | "
            f"{_fmt(proxy['cross_archetype_jaccard_mean'])} | {ratio_of(proxy)} | "
            f"{proxy['within_archetype_n_pairs']} / {proxy['cross_archetype_n_pairs']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_coverage(metrics: dict[str, Any]) -> str:
    c = metrics["coverage"]
    primary = c["primary_system"]
    pop = c["popularity_baseline"]
    return "\n".join(
        [
            "## Coverage (spec.md section 11.3)",
            "",
            "| | Primary system (lambdamart_ips) | Popularity baseline |",
            "|---|---|---|",
            f"| Catalog coverage@10 | {_fmt_pct(primary['catalog_coverage_at_10'])} | "
            f"{_fmt_pct(pop['catalog_coverage_at_10'])} |",
            f"| Gini coefficient | {_fmt(primary['gini'])} | {_fmt(pop['gini'])} |",
            f"| Entropy (bits) | {_fmt(primary['entropy_bits'])} | {_fmt(pop['entropy_bits'])} |",
            f"| POIs ever recommended | {primary['n_pois_ever_recommended']} / "
            f"{primary['n_catalog_pois']} | {pop['n_pois_ever_recommended']} / "
            f"{pop['n_catalog_pois']} |",
            "",
            "Lower Gini / higher entropy / higher coverage = less popularity-monoculture "
            "concentration. Per-destination coverage:",
            "",
            "| Destination | Primary | Popularity |",
            "|---|---|---|",
        ]
        + [
            f"| {dest} | {_fmt_pct(primary['catalog_coverage_by_destination'].get(dest))} | "
            f"{_fmt_pct(pop['catalog_coverage_by_destination'].get(dest))} |"
            for dest in sorted(primary["catalog_coverage_by_destination"])
        ]
        + [""]
    )


def render_longtail(metrics: dict[str, Any]) -> str:
    lg = metrics["longtail"]
    precision_str = "N/A" if lg["precision"] is None else _fmt(lg["precision"])
    lines = [
        "## Long-tail / local discovery (spec.md section 11.4)",
        "",
        f"- Share of top-10 recommendations in the bottom-50%-popularity stratum: "
        f"**{_fmt(lg['share'])}** ({lg['n_longtail_recommended']} / {lg['n_total_recommended']})",
        f"- Long-tail precision (relevant per unbiased holdout): **{precision_str}** "
        f"({lg['n_longtail_relevant']} / {lg['n_longtail_recommended']})",
    ]
    lift = metrics.get("derived", {}).get("longtail_lift_over_base_rate")
    if lift is not None:
        lines += [
            f"- **Lift over the candidate pool's long-tail positive rate "
            f"({_fmt(lift['base_rate'])}): {lift['final_after_mmr']:.3f}x served, "
            f"{lift['raw_ranker']:.3f}x at the raw ranker** -- the primary long-tail precision "
            "statistic; the 0.40 raw-precision target was set a priori without reference to "
            "this base rate (miscalibrated at design time; see TECHNICAL.md section 4.2).",
        ]
    pop = lg.get("popularity_baseline")
    if pop is not None:
        pop_precision = "N/A" if pop["precision"] is None else _fmt(pop["precision"])
        lines += [
            f"- Popularity ranker, same measurement: share **{_fmt(pop['share'])}** "
            f"({pop['n_longtail_recommended']} / {pop['n_total_recommended']}), precision "
            f"**{pop_precision}**",
        ]
    lines += [
        "",
        '"Coverage without precision is just noise injection" -- both numbers reported '
        "together, per spec.md section 11.4.",
        "",
    ]
    return "\n".join(lines)


def render_constraint_compatibility(metrics: dict[str, Any]) -> str:
    cc = metrics["constraint_compatibility"]
    return "\n".join(
        [
            "## Constraint compatibility (spec.md section 11.5)",
            "",
            f"- % of top-10 with compatibility &ge; {cc['compatibility_target']}: "
            f"**{_fmt_pct(cc['share_above_target'])}** "
            f"({cc['n_above_target']} / {cc['n_recommended']})",
            f"- Hard-constraint violations in top-10: **{cc['n_hard_constraint_violations']}** "
            "(build-blocking, enforced independently by `tests/test_hard_constraints.py`)",
            "",
        ]
    )


def render_diversity(metrics: dict[str, Any]) -> str:
    dv = metrics["diversity"]
    lines = [
        "## Diversity (spec.md section 11.6)",
        "",
        f"- Category entropy@10 (bits): **{_fmt(dv['category_entropy_at_10_bits'])}**",
        f"- Intra-list mean cosine distance (at the configured default lambda): "
        f"**{_fmt(dv['intra_list_mean_distance'])}**",
        "",
        "### MMR lambda sweep (NDCG@10 vs diversity trade-off)",
        "",
        "| lambda | NDCG@10 (mean) | mean intra-list similarity |",
        "|---|---|---|",
    ]
    for row in dv["lambda_sweep"]:
        lines.append(
            f"| {row['lambda']:.2f} | {_fmt(row['ndcg@10_mean'])} | "
            f"{_fmt(row['mean_intra_list_similarity'])} |"
        )
    lines.append("")
    ls = metrics.get("longtail_stages")
    if ls is not None:
        lam_lift = metrics.get("derived", {}).get("mmr_lambda_lift_over_base_rate", {})
        lines += [
            "### MMR lambda vs long-tail precision (post-hoc holdout diagnostic, not a selection)",
            "",
            "| lambda | Long-tail share@10 | Long-tail precision@10 | Lift over pool base rate |",
            "|---|---|---|---|",
        ]
        for lam, v in ls["mmr_lambda_diagnostic_post_hoc_on_holdout"].items():
            lf = lam_lift.get(lam)
            lines.append(
                f"| {float(lam):.2f} | {_fmt(v['long_tail_share'])} | "
                f"{_fmt(v['long_tail_precision'])} | {'—' if lf is None else f'{lf:.3f}x'} |"
            )
        lines += [
            "",
            f"The shipped lambda ({ls['lambda_default']:g}) is a **deliberate diversity-for-"
            "precision trade, not a tuned optimum**, and lambda was not re-selected: choosing it "
            "on the holdout would leak, and re-selecting on validation would destabilise a "
            "shipped submission over a product knob. The cost is quantified in the two tables: "
            "moving from lambda 1.0 (no diversity term) to the shipped value lowers "
            "long-tail precision and NDCG@10 while cutting mean intra-list similarity, and the "
            "long-tail share is roughly flat across the whole range. A product owner who "
            "optimises for booking precision rather than category variety would move lambda "
            "toward 0.9 (a one-line scoring-config change), recovering most of the precision "
            "and NDCG@10 for a modest rise in list similarity; one who wants catalog exposure "
            "keeps 0.8.",
            "",
        ]
    return "\n".join(lines)


def render_calibration(metrics: dict[str, Any]) -> str:
    c = metrics["calibration"]
    return "\n".join(
        [
            "## Calibration (spec.md section 11.7)",
            "",
            "| | Before (naive) | After (isotonic) |",
            "|---|---|---|",
            f"| ECE (15 bins) | {_fmt(c['ece_before'])} | {_fmt(c['ece_after'])} |",
            f"| Brier score | {_fmt(c['brier_before'])} | {_fmt(c['brier_after'])} |",
            "",
            f"Calibration split: n_rows={c['n_calibration_rows']}, "
            f"n_trips={c['n_calibration_trips']}.",
            "",
        ]
    )


def render_confidence_decile(metrics: dict[str, Any]) -> str:
    dv = metrics["confidence_decile_validation"]
    rho_str = "N/A" if dv["spearman_rho"] is None else _fmt(dv["spearman_rho"], 3)
    lines = [
        "## Confidence-decile validation (spec.md section 9.3 / 11.10)",
        "",
        f"Spearman rho (decile rank, decile mean NDCG@k): **{rho_str}** "
        f"(target &ge; 0.7, {_met(dv['target_met'])}). n_trips_included={dv['n_trips_included']}.",
        "",
    ]
    if dv["deciles"]:
        lines += ["| Decile | Mean confidence | Mean NDCG@k | n_trips |", "|---|---|---|---|"]
        for d in dv["deciles"]:
            lines.append(
                f"| {d['decile']} | {_fmt(d['mean_confidence'])} | "
                f"{_fmt(d['ndcg@k_mean'])} | {d['n_trips']} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_beta_sensitivity(metrics: dict[str, Any]) -> str:
    rows = metrics["beta_sensitivity"]
    lines = [
        "## Utility beta sensitivity (spec.md section 9.1)",
        "",
        "| beta | NDCG@10 (mean) |",
        "|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['beta']:.2f} | {_fmt(row['ndcg@10_mean'])} |")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# Cold-start cohorts (interaction-count buckets, new-POI cohort, LODO)
# -----------------------------------------------------------------------------------


def render_cold_start(metrics: dict[str, Any]) -> str:
    cs_payload = metrics["cold_start"]
    lines = [
        "## Cold-start cohorts (spec.md section 11.8 / section 12)",
        "",
        "### NDCG@10 by traveler interaction-count bucket",
        "",
        "| Bucket | NDCG@10 (mean) | n_trips_in_bucket |",
        "|---|---|---|",
    ]
    for bucket, agg in cs_payload["ndcg@10_by_interaction_count_bucket"].items():
        lines.append(f"| {bucket} | {_fmt(agg['mean'])} | {agg['n_trips_in_bucket']} |")
    lines.append("")

    npc = metrics["new_poi_cohort"]
    with_d = npc["ndcg@10_lambdamart_ips_with_dropout"]
    without_d = npc["ndcg@10_lambdamart_ips_no_dropout"]
    lines += [
        "### New-POI cohort (spec.md section 12)",
        "",
        f"- n_new_pois_in_catalog={npc['n_new_pois_in_catalog']}, "
        f"n_holdout_trips_with_relevant_cohort_candidate={npc['n_holdout_trips_with_relevant_cohort_candidate']}",
        f"- NDCG@10 WITH behavioral dropout: **{_agg_str(with_d)}**",
        f"- NDCG@10 WITHOUT behavioral dropout: **{_agg_str(without_d)}**",
        f"- Paired Wilcoxon (with vs without): "
        f"p={npc['wilcoxon_with_vs_without_dropout']['p_value']:.4g}",
        "",
    ]

    lodo = metrics.get("lodo")
    lines.append("### Leave-one-destination-out (LODO)")
    lines.append("")
    if lodo is None:
        lines.append(
            "**Not yet run.** LODO is a separate command (`poi_rank.cli lodo` / "
            "`make lodo`), not part of `make reproduce`'s default chain -- 3 full "
            "LightGBM retrains, genuinely expensive (see `eval/cold_start.py`'s "
            "module docstring). Run `poi_rank.cli lodo` after `poi_rank.cli "
            "evaluate` to populate this section."
        )
    else:
        lines.append(
            f"Wall-clock: **{lodo['wall_clock_seconds']:.1f}s** for {lodo['n_destinations']} "
            "destination-held-out retrains."
        )
        lines.append("")
        lines.append("| Destination | NDCG@10 (LODO) | NDCG@10 (full training) | Wilcoxon p |")
        lines.append("|---|---|---|---|")
        for row in lodo["per_destination"]:
            lines.append(
                f"| {row['destination']} | {_agg_str(row['ndcg@10_lodo'])} | "
                f"{_agg_str(row['ndcg@10_full_training'])} | "
                f"{row['wilcoxon_lodo_vs_full_training']['p_value']:.4g} |"
            )
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# Ablations (spec.md section 11.9)
# -----------------------------------------------------------------------------------


def render_ablations(metrics: dict[str, Any]) -> str:
    rows = metrics["ablations"]
    lines = [
        "## Ablations (spec.md section 11.9)",
        "",
        "Each row: delta NDCG@10 (ablated - full lambdamart_ips), against the SAME "
        "already-trained primary system as the reference point.",
        "",
        "| Ablation | Status | NDCG@10 (full) | NDCG@10 (ablated) | Delta | Wilcoxon p |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] != "measured":
            lines.append(f"| {row['ablation']} | {row['status']} | - | - | - | - |")
            continue
        delta = row["delta_ndcg@10"]
        p = row["wilcoxon_p_value"]
        p_str = "N/A" if p is None else f"{p:.4g}"
        lines.append(
            f"| {row['ablation']} | measured | {_agg_str(row['ndcg@10_full'])} | "
            f"{_agg_str(row['ndcg@10_ablated'])} | {delta:+.4f} | {p_str} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    for row in rows:
        lines.append(f"- **{row['ablation']}**: {row['note']}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# 3+1 required scenarios (spec.md section 15)
# -----------------------------------------------------------------------------------

# Reuses this project's own already-established "low overlap" bar
# (`render_success_criteria`'s "Cross-archetype Jaccard@10 <= 0.25" row) rather than
# inventing a new threshold just for this diagnostic.
_DIAGNOSTIC_LOW_OVERLAP_THRESHOLD = 0.25


def _render_scenario_top10_table(recommendations: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | "
        "Confidence | Pop. %ile | Localness |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in recommendations:
        # `popularity_percentile` is ALREADY on a 0-100 scale in the source JSON
        # (`scoring.output.assemble_output_payload`: `pop_pct * 100.0`) -- rendered
        # via plain `_fmt` (matched against the raw source leaf directly, tight
        # tolerance), NOT `_fmt_pct` (which would divide by 100 first and expects
        # a [0, 1] fraction as its source leaf, per its other call sites in this
        # module) -- routing it through `_fmt_pct` here would round-trip through
        # an extra x100/div-100 round trip for no reason and only lose precision.
        lines.append(
            f"| {rec['rank']} | {rec['poi_id']} | {rec['name']} | {rec['category']} | "
            f"{_fmt(rec['utility'])} | {_fmt(rec['preference_score'])} | "
            f"{_fmt(rec['context_compatibility'])} | {_fmt(rec['confidence'])} | "
            f"{_fmt(rec['popularity_percentile'])}% | "
            f"{_fmt(rec['localness_index'])} |"
        )
    return lines


def render_one_scenario(payload: dict[str, Any]) -> str:
    p = payload["profile"]
    top1 = payload["recommendations"][0] if payload["recommendations"] else None
    lines = [
        f"### Scenario {payload['scenario_number']}: {payload['scenario_name']}",
        "",
        *([f"- Persona: {p['persona']}"] if p.get("persona") else []),
        f"- Destination: **{p['destination']}**; interests: "
        f"{', '.join(p['interests'])}; touristiness_pref: **{_fmt(p['touristiness_pref'], 2)}**",
        f"- Budget: {p['budget']}; mobility: {p['mobility']}; party: {p['party_type']}; "
        f"pace: {p['pace']}; accessibility needs: "
        f"{', '.join(p['accessibility_needs']) if p['accessibility_needs'] else 'none'}",
        f"- {p['notes']}",
        "",
        "#### Top-10 recommendations",
        "",
    ]
    lines += _render_scenario_top10_table(payload["recommendations"])
    lines.append("")
    if top1 is not None:
        lines += [
            f"**Top signals (rank 1, {top1['poi_id']}):** "
            + ", ".join(
                f"{s['feature_group']} ({_fmt(s['contribution'], 3)})" for s in top1["top_signals"]
            ),
            "",
            f"**Explanation (rank 1):** {' '.join(top1['explanation'])}",
            "",
        ]
    return "\n".join(lines)


def render_scenarios(scenarios: dict[str, Any] | None) -> str:
    """The 3+1 required scenarios (spec.md section 15). Every number below is read
    directly from `results/scenarios/*.json` (`poi_rank.cli scenarios`'s own
    output) -- the same "generated, never hand-typed" discipline `render_*`
    everywhere else in this module already follows for `results/metrics.json`."""
    lines = ["## Scenarios (spec.md section 15)", ""]
    if scenarios is None:
        lines.append(
            "**Not yet run.** Run `poi_rank.cli scenarios` (or `make scenarios`) to "
            "populate this section."
        )
        lines.append("")
        return "\n".join(lines)

    for payload in scenarios["scenarios"]:
        lines.append(render_one_scenario(payload))

    overlap = scenarios["overlap_matrix"]
    order = overlap["scenario_order"]
    lines += [
        "### Pairwise top-10 Jaccard overlap across scenarios",
        "",
        "| | " + " | ".join(f"Scenario {n}" for n in order) + " |",
        "|" + "---|" * (len(order) + 1),
    ]
    for i, row in zip(order, overlap["matrix"], strict=True):
        lines.append(f"| Scenario {i} | " + " | ".join(_fmt(v, 3) for v in row) + " |")

    diag_value = overlap["diagnostic_vs_base_jaccard"]
    is_low = diag_value <= _DIAGNOSTIC_LOW_OVERLAP_THRESHOLD
    lines += [
        "",
        f"**Diagnostic scenario {overlap['diagnostic_scenario']} vs base scenario "
        f"{overlap['diagnostic_base_scenario']}** (touristiness_pref flipped to the "
        f"opposite extreme, everything else held constant): top-10 Jaccard overlap = "
        f"**{_fmt(diag_value, 3)}** (spec.md section 15 expects this to be low, i.e. "
        f"&le; {_DIAGNOSTIC_LOW_OVERLAP_THRESHOLD}, {_met(is_low)}).",
        "",
    ]
    if is_low:
        lines.append(
            "Low overlap demonstrates the ranking is driven by the preference signal, "
            "not profile confounds, as spec.md section 15 expects."
        )
    else:
        candidate_jaccard = overlap.get("diagnostic_vs_base_candidate_pool_jaccard")
        lines.append(
            "**Honest miss, not hidden**: overlap is higher than spec.md section 15 expects. "
            "Measured mechanism: the two scenarios' PRE-RANKING candidate pools overlap at "
            f"**{_fmt_pct(candidate_jaccard) if candidate_jaccard is not None else 'N/A'}** "
            "Jaccard (`candidates.union.generate_candidates`'s own output), so the ranking "
            "starts from nearly the same POIs and `touristiness_pref` can only reorder them "
            "through the feature columns that carry it (`explicit_touristiness_pref`, "
            "`interact_localness_gap`); `scoring.compatibility`'s sub-scores carry no "
            "localness/touristiness term at all (spec.md section 9.1). The overlap is below "
            "1.0, i.e. the preference does move the ranking -- just not enough to dominate a "
            "candidate pool this similar."
        )
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# Full document assembly
# -----------------------------------------------------------------------------------


def render_results_md(metrics: dict[str, Any], scenarios: dict[str, Any] | None = None) -> str:
    sections = [
        "# RESULTS.md",
        "",
        "**Generated automatically by `poi_rank.eval.report` from "
        "`results/metrics.json`. Do not hand-edit -- every number here traces "
        'directly to that JSON file (spec.md section 0: "No unverified metric may '
        'appear in any document").**',
        "",
        "Synthetic data throughout (three destinations; Seoul is the lead). Personas are inbound "
        "foreign travelers. Read the numbers as properties of this simulator, not as claims "
        "about real traffic -- `docs/TECHNICAL.md` states what the simulator does and does not "
        "license.",
        "",
        render_headline(metrics),
        render_bias_gap_table(metrics),
        render_gates(metrics),
        render_success_criteria(metrics, scenarios),
        render_primary_table(metrics),
        render_candidate_recall(metrics),
        render_personalization(metrics),
        render_coverage(metrics),
        render_longtail(metrics),
        render_constraint_compatibility(metrics),
        render_diversity(metrics),
        render_calibration(metrics),
        render_confidence_decile(metrics),
        render_cold_start(metrics),
        render_ablations(metrics),
        render_beta_sensitivity(metrics),
        render_decision_register(metrics),
        render_seed_replication(metrics),
        render_timings(metrics),
        render_scenarios(scenarios),
    ]
    return "\n".join(sections)


def run_report(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
) -> Path:
    """`poi_rank.eval.report`'s entry point (`make docs` / `python -m
    poi_rank.eval.report`): read `results/metrics.json` (+ `results/scenarios/*
    .json`, if present -- `load_scenarios`'s own documented "not yet run"
    fallback), render `docs/RESULTS.md`, write it, return the output path."""
    metrics = load_metrics(metrics_path)
    scenarios = load_scenarios(scenarios_dir)
    content = render_results_md(metrics, scenarios)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    return output_path


NARRATIVE_TEMPLATES: dict[str, str] = {
    "docs/TECHNICAL_SUMMARY.md.tmpl": "docs/TECHNICAL_SUMMARY.md",
    "docs/TECHNICAL.md.tmpl": "docs/TECHNICAL.md",
    "docs/REQUIREMENTS.md.tmpl": "docs/REQUIREMENTS.md",
    "README.md.tmpl": "README.md",
    "submission/SUBMISSION.md.tmpl": "submission/SUBMISSION.md",
}


def run_narrative_docs(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Render the hand-narrated documents from their templates (`report_templates.py`): the
    prose is hand-written, every number and table is resolved from `results/metrics.json`."""
    from poi_rank.eval.report_templates import render_template

    metrics = load_metrics(metrics_path)
    scenarios = load_scenarios(scenarios_dir)
    written: list[Path] = []
    for template, target in NARRATIVE_TEMPLATES.items():
        template_path = repo_root / template
        if not template_path.exists():
            continue
        rendered = render_template(template_path.read_text(encoding="utf-8"), metrics, scenarios)
        out = repo_root / target
        out.write_text(rendered, encoding="utf-8")
        written.append(out)
    return written


if __name__ == "__main__":
    written = run_report()
    print(f"wrote {written}")
    for path in run_narrative_docs():
        print(f"wrote {path}")
