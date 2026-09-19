"""RESULTS.md sections added in the post-remediation rewrite: headline (local / long-tail
discovery), gates, candidate-recall-by-stratum, the scorecard's per-miss diagnoses, the
Decision Register and timings. Same rule as `eval/report.py`: every number is an f-string
interpolation of a value read out of `results/metrics.json` -- nothing is typed by hand.
"""

from __future__ import annotations

from typing import Any


def _f(value: float | int | None, decimals: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def _pct(value: float | int | None, decimals: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:.{decimals}f}%"


def render_headline(metrics: dict[str, Any]) -> str:
    """Lead result: local / long-tail discovery vs a popularity ranker. The incumbents already
    rank by popularity, so the differentiating claim is surfacing the genuinely relevant
    non-obvious POI -- backed by long-tail PRECISION, not just share."""
    lt = metrics["longtail"]
    pop = lt.get("popularity_baseline")
    cov = metrics["coverage"]
    ndcg = metrics["systems"]["lambdamart_ips"]["metrics"]["ndcg@10"]
    pop_ndcg = metrics["systems"]["popularity"]["metrics"]["ndcg@10"]
    gap = metrics["bias_gap"]
    lines = [
        "## Headline: local and long-tail discovery (Seoul-first, inbound travelers)",
        "",
        "konnect.kr serves foreign travelers in Korea, and the incumbents already rank by "
        "popularity -- so a popularity-shaped list is table stakes. The claim that matters is "
        "surfacing the genuinely relevant, non-obvious POI: **long-tail share and long-tail "
        "precision together** (coverage without precision is just noise injection), measured on "
        "the unbiased random-exposure holdout.",
        "",
        "| Top-10 lists, primary holdout | LambdaMART + IPS (primary) | Popularity ranker |",
        "|---|---|---|",
    ]
    if pop is not None:
        lines += [
            f"| Long-tail share (bottom-50% popularity stratum) | **{_f(lt['share'])}** | "
            f"{_f(pop['share'])} |",
            f"| Long-tail precision (relevant / recommended) | **{_f(lt['precision'])}** | "
            f"{_f(pop['precision'])} |",
        ]
    else:
        lines += [
            f"| Long-tail share | **{_f(lt['share'])}** | not measured |",
            f"| Long-tail precision | **{_f(lt['precision'])}** | not measured |",
        ]
    lines += [
        f"| Catalog coverage@10 | **{_pct(cov['primary_system']['catalog_coverage_at_10'])}** | "
        f"{_pct(cov['popularity_baseline']['catalog_coverage_at_10'])} |",
        f"| Gini of recommendation exposure (lower = less monoculture) | "
        f"**{_f(cov['primary_system']['gini'])}** | {_f(cov['popularity_baseline']['gini'])} |",
        f"| NDCG@10 (unbiased holdout, 95% CI) | **{_f(ndcg['mean'])}** "
        f"[{_f(ndcg['ci_low'])}, {_f(ndcg['ci_high'])}] | {_f(pop_ndcg['mean'])} "
        f"[{_f(pop_ndcg['ci_low'])}, {_f(pop_ndcg['ci_high'])}] |",
        "",
        "Popularity is flattered by exposure-biased logs; the bias-gap table below quantifies "
        f"exactly how much (popularity {gap['popularity']['gap']:+.4f}, primary "
        f"{gap['lambdamart_ips']['gap']:+.4f} NDCG@10 between the biased and unbiased holdouts).",
        "",
    ]
    return "\n".join(lines)


def render_gates(metrics: dict[str, Any]) -> str:
    lines = ["## Acceptance gates", ""]
    a = metrics.get("gate_dgp")
    b = metrics.get("gate_representation")
    lines += [
        "**Gate-A (simulator quality, frozen before any model work)** -- measured on the "
        "generator alone, so the model cannot have been tuned against it:",
        "",
    ]
    if a is None:
        lines += ["Not composed (run `gate-dgp`).", ""]
    else:
        lines += ["| Check | Threshold | Measured | Status |", "|---|---|---|---|"]
        for name, c in sorted(a["checks"].items()):
            status = "PASS" if c["passed"] else "FAIL"
            lines.append(
                f"| {name} | {c['op']} {c['threshold']} | {_f(c['measured'])} | {status} |"
            )
        lines.append("")
    lines += [
        "**Gate-B (representation / candidate generation)** -- blocking rows are oracle-free "
        "(recall against EXPOSED holdout positives); every representation and retrieval "
        "hyperparameter was selected on a train-carved validation split, and the oracle-based "
        "rows below are reporting-only, computed after the selection was frozen. The oracle "
        "never touched a decision; it only ever scored the result.",
        "",
    ]
    if b is None:
        lines += ["Not composed (run `gate-representation`).", ""]
    else:
        lines += ["| Blocking check | Threshold | Measured | Status |", "|---|---|---|---|"]
        for name, c in b["blocking_checks"].items():
            status = "PASS" if c["passed"] else "FAIL"
            lines.append(
                f"| {name} | {c['op']} {c['threshold']} | {_f(c['measured'])} | {status} |"
            )
        lines += ["", "| Reporting-only (after freeze) | Value |", "|---|---|"]
        for name, value in b["reporting_only_after_freeze"].items():
            if isinstance(value, float):
                lines.append(f"| {name} | {_f(value)} |")
        lines.append("")
    return "\n".join(lines)


def render_candidate_recall(metrics: dict[str, Any]) -> str:
    strata = metrics["candidate_recall"].get("by_stratum")
    lines = ["## Candidate recall by popularity stratum (with chance baselines)", ""]
    if strata is None:
        return "\n".join([*lines, "Not measured.", ""])
    lines += [
        "Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its "
        "own chance baseline: the share of that stratum's destination POIs a same-size random "
        "candidate set would contain. Raw recall without this lift hid a near-chance failure "
        "for four phases.",
        "",
        "| Stratum | Recall | Chance | Lift (abs) | Trips |",
        "|---|---|---|---|---|",
    ]
    for name, row in strata.items():
        lines.append(
            f"| {name} | {_f(row['recall'], 3)} | {_f(row['chance_recall'], 3)} | "
            f"{row['lift_abs']:+.3f} | {int(row['n_trips'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def diagnoses(metrics: dict[str, Any]) -> dict[str, str]:
    """A diagnosis (mechanism + the diagnostic number that ruled it in/out) for every scorecard
    row that can be missed. A missed row WITHOUT an entry here renders as UNDIAGNOSED -- loudly."""
    out: dict[str, str] = {}
    rep = metrics.get("representation", {})
    d9 = rep.get("d9_within_trip", {})
    p = metrics["personalization"]
    if d9:
        out["ceiling"] = (
            "The oracle ranks the SAME candidates by the DGP's true utility; the model only sees "
            "observable features. Text is not the limiting link: text-alone within-trip taste "
            f"fidelity is {_f(d9['text_alone']['mean_within_trip_spearman'], 3)} and D11 ridge R2 "
            f"is {_f(rep['d11_text_only']['ridge_r2_oof'], 3)}. The taste ESTIMATOR is: run over "
            "the TRUE semantic vectors it still reaches only "
            f"{_f(d9['taste_estimator_alone']['mean_within_trip_spearman'], 3)} "
            f"(shipped chain {_f(d9['shipped_chain_d9']['mean_within_trip_spearman'], 3)}), "
            "from sparse, exposure-biased histories."
        )
    ideal = p.get("archetype_ideal_ranker_reference")
    if ideal is not None:
        ratio = ideal["within_cross_ratio"]
        target_ok = ratio is None or ratio >= 2.0
        verdict = (
            "so the target IS attainable by a perfect ranker and the shortfall is ranking "
            "quality (see the ceiling diagnosis: taste-estimator fidelity)."
            if target_ok
            else "so the target is NOT attainable even by a perfect ranker in this simulator: "
            "the miss is a property of the DGP (archetype explains only part of a traveler's "
            "utility), not of the model."
        )
        out["archetype_ratio"] = (
            "Within/cross-archetype list similarity ratio for lists ranked by the TRUE utility "
            "(a perfect ranker, same-destination pairs): "
            f"{'inf' if ratio is None else _f(ratio, 2)} "
            f"(within {_f(ideal['within_archetype_jaccard_mean'])}, cross "
            f"{_f(ideal['cross_archetype_jaccard_mean'])}); {verdict}"
        )
    lt = metrics["longtail"]
    out["longtail_precision"] = (
        f"Long-tail precision is {_f(lt['precision'])} over "
        f"{lt['n_longtail_recommended']} long-tail recommendations, with candidate recall "
        f"{_f(metrics['candidate_recall']['long_tail']['recall_mean'], 3)} in that stratum, so "
        "retrieval is not the bottleneck. UNVERIFIED hypothesis: long-tail POIs have a lower "
        "positive base rate under exposure-uniform labels, capping precision mechanically; "
        "test = long-tail vs head positive rate among candidates (not yet run)."
    )
    dv = metrics["confidence_decile_validation"]
    out["confidence_decile"] = (
        f"Confidence-decile Spearman is {_f(dv['spearman_rho'], 3)}. UNVERIFIED hypothesis: the "
        "evidence-volume terms dominate the ensemble-sd term, so deciles separate by evidence "
        "volume rather than correctness; test = per-component Spearman vs decile NDCG (not yet "
        "run)."
    )
    return out


def render_decision_register(metrics: dict[str, Any]) -> str:
    reg = metrics.get("decision_register")
    lines = ["## Decision Register", ""]
    if reg is None:
        return "\n".join([*lines, "Not composed (run `dr`).", ""])
    lines += [
        "Every design choice not dictated by the assignment, the alternative, the experiment, "
        "and the measured result. Rows are generated from `results/parts/dr/*.json`; an "
        "experiment that was not run says NOT RUN.",
        "",
        "| # | Decision | Alternative | Experiment | Result / verdict | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in reg["rows"]:
        exp = row["experiment"].replace("|", "/")
        verdict = row["verdict"].replace("|", "/")
        lines.append(
            f"| {row['id']} | {row['decision']} | {row['alternative']} | {exp} | {verdict} | "
            f"{row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_timings(metrics: dict[str, Any]) -> str:
    lines = ["## Wall-clock", ""]
    found = False
    for key in ("timings", "timings_full"):
        t = metrics.get(key)
        if t is None:
            continue
        found = True
        lines += [
            f"**{t['target']}**: {t['total_seconds']:.1f} s total "
            "(16-logical-core laptop CPU, no GPU, no network).",
            "",
            "| Stage | Seconds |",
            "|---|---|",
        ]
        for st in t["stages"]:
            lines.append(f"| {st['stage']} | {st['seconds']:.1f} |")
        lines.append("")
    if not found:
        lines += ["Not recorded (run `scripts/time_reproduce.py`).", ""]
    return "\n".join(lines)
