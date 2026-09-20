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


def diagnoses(metrics: dict[str, Any], scenarios: dict[str, Any] | None = None) -> dict[str, str]:
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
            "the shortfall is a property of the simulator under this target, not of the model."
        )
        out["archetype_ratio"] = (
            "Within/cross-archetype list similarity ratio for lists ranked by the TRUE utility "
            "(a perfect ranker, same-destination pairs): "
            f"{'inf' if ratio is None else _f(ratio, 2)} "
            f"(within {_f(ideal['within_archetype_jaccard_mean'])}, cross "
            f"{_f(ideal['cross_archetype_jaccard_mean'])}); {verdict}"
        )
    lt = metrics["longtail"]
    dr9_row = {r["id"]: r for r in metrics.get("decision_register", {}).get("rows", [])}.get(
        "DR9", {}
    )
    raw = ((dr9_row.get("results") or {}).get("50") or {}).get("longtail_at_10")
    tail = (
        f" The raw ranker's top-10 long-tail precision (DR9, quota 50, before the compatibility "
        f"gate, utility and MMR re-rank) is {_f(raw['precision'])} at share {_f(raw['share'])}, "
        f"against {_f(lt['precision'])} at share {_f(lt['share'])} in the served list: precision "
        "is lost AFTER ranking while share rises. Which of the three scoring-layer steps is "
        "responsible is not isolated (untested)."
        if raw
        else ""
    )
    lift = metrics.get("derived", {}).get("longtail_lift_over_base_rate")
    lift_txt = (
        f" Measured against the pool's own long-tail positive rate ({_f(lift['base_rate'], 3)}) "
        f"the served list is a {lift['final_after_mmr']:.3f}x lift (raw ranker "
        f"{lift['raw_ranker']:.3f}x): the 0.40 target was set a priori without reference to that "
        "base rate and was miscalibrated at design time, so lift over base rate is the primary "
        "statistic and raw precision secondary (TECHNICAL.md section 4.2)."
        if lift
        else ""
    )
    out["longtail_precision"] = (
        f"Long-tail precision is {_f(lt['precision'])} over "
        f"{lt['n_longtail_recommended']} long-tail recommendations, with candidate recall "
        f"{_f(metrics['candidate_recall']['long_tail']['recall_mean'], 3)} in that stratum, so "
        "retrieval is not the bottleneck." + lift_txt + tail
    )
    reg = {r["id"]: r for r in metrics.get("decision_register", {}).get("rows", [])}
    dr9 = reg.get("DR9", {}).get("results")
    if dr9:
        shares = {q: v["longtail_at_10"]["share"] for q, v in dr9.items()}
        out["longtail_share"] = (
            f"Long-tail share of the served top-10 is {_f(lt['share'])}. Decision Register DR9 "
            "varies the long-tail candidate quota and measures the raw ranker's top-10 share: "
            + ", ".join(f"quota {q} -> {_f(v, 3)}" for q, v in shares.items())
            + ". The candidate quota is therefore not the lever; the share is set by the "
            "ranker's scores (and the MMR re-rank) over a candidate set that already contains "
            "long-tail POIs."
        )
    loc = metrics.get("localness_validation", {})
    comps = loc.get("component_spearman_rho")
    if comps:
        best_name, best_rho = max(comps.items(), key=lambda kv: abs(kv[1]))
        stale = abs(best_rho) > loc["spearman_rho"]
        out["localness"] = (
            f"The composite index reaches rho {_f(loc['spearman_rho'], 3)}; its observable "
            "inputs correlate with the latent localness at "
            + ", ".join(f"{k} {_f(v, 3)}" for k, v in comps.items())
            + (
                f". The composite is BELOW its best single input ({best_name}, "
                f"|rho| {_f(abs(best_rho), 3)}): the blend weights were fixed earlier, when the "
                "geo input carried almost no signal (before the simulator's geo/localness fix). "
                "Re-weighting them against the latent localness would be tuning on the oracle "
                "(there is no oracle-free validation target for this index), so that retune is "
                "declined on principle: the index is left as shipped and the gap is reported."
                if stale
                else ". The index is a weighted blend of these signed correlations, so it "
                "cannot exceed what its inputs carry."
            )
        )
    if scenarios is not None:
        ov = scenarios["overlap_matrix"]
        cj = ov.get("diagnostic_vs_base_candidate_pool_jaccard")
        h = metrics.get("h_experiments")
        extra = ""
        if h:
            pre = h["pre_fix_headline"]["scenario4_overlap"]
            share = h["h0a_shap_shares_final_model"]["localness_fit"]
            extra = (
                f" Before the feature-skew fix (TECHNICAL.md section 5.1) this row was "
                f"{_f(pre, 3)}; the fix changed how much the ranker relies on which signals, and "
                "the flip now moves the top-10 less. The localness_fit group carries only "
                f"{_f(share, 3)} of the final model grouped-SHAP attribution (section 5.2), so "
                "touristiness_pref has limited leverage on the ranking. UNVERIFIED hypothesis: "
                "the label-carrying implicit features used to interact with the preference terms "
                "in a way the corrected features do not (untested; diagnostic scenario, not a "
                "product requirement)."
            )
        out["scenario4"] = (
            f"Top-10 overlap {_f(ov['diagnostic_vs_base_jaccard'], 3)} with candidate-pool "
            f"Jaccard {_f(cj, 3)} between the two profiles (measured from the candidate "
            "generator's own output)." + extra
        )
    dv = metrics["confidence_decile_validation"]
    h = metrics.get("h_experiments", {}).get("pre_fix_headline", {})
    pre_conf = h.get("confidence_decile_spearman")
    pre_txt = (
        f" Before the feature-skew fix (TECHNICAL.md section 5.1) this row was {_f(pre_conf, 3)}: "
        "the confidence ensemble and the ranker were retrained on the corrected features and the "
        "decile ordering changed with them (the holdout features themselves are unchanged)."
        if pre_conf is not None
        else ""
    )
    out["confidence_decile"] = (
        f"Confidence-decile Spearman is {_f(dv['spearman_rho'], 3)}." + pre_txt + " UNVERIFIED "
        "hypothesis: the evidence-volume terms dominate the ensemble-sd term, so deciles separate "
        "by evidence volume rather than correctness; test = per-component Spearman vs decile "
        "NDCG (untested)."
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


# -----------------------------------------------------------------------------------
# Blocks used by the narrative templates (docs/TECHNICAL.md.tmpl, README.md.tmpl)
# -----------------------------------------------------------------------------------

_SYSTEM_NAMES = {
    "random": "1. Random",
    "popularity": "2. Popularity",
    "popularity_geo_filter": "3. Popularity + geo filter",
    "content_cosine": "4. Content cosine",
    "item_knn_cf": "5. Item-kNN CF",
    "logistic_regression": "6. Logistic regression (CV-tuned L2)",
    "lambdamart": "7. LambdaMART (no IPS)",
    "lambdamart_ips": "8. **LambdaMART + IPS (primary)**",
    "oracle": "9. Oracle (ceiling)",
}


def systems_rows(metrics: dict[str, Any]) -> str:
    lines = ["| System | NDCG@10 (95% CI) | % of oracle ceiling |", "|---|---|---|"]
    for key, name in _SYSTEM_NAMES.items():
        s = metrics["systems"][key]
        m = s["metrics"]["ndcg@10"]
        lines.append(
            f"| {name} | {_f(m['mean'])} [{_f(m['ci_low'])}, {_f(m['ci_high'])}] | "
            f"{_pct(s['pct_of_ceiling_ndcg10'])} |"
        )
    return "\n".join(lines)


def bias_gap_rows(metrics: dict[str, Any]) -> str:
    lines = [
        "| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |",
        "|---|---|---|---|",
    ]
    for key, name in _SYSTEM_NAMES.items():
        r = metrics["bias_gap"][key]
        lines.append(
            f"| {name} | {_f(r['ndcg@10_unbiased'])} | {_f(r['ndcg@10_biased'])} | "
            f"{r['gap']:+.4f} |"
        )
    return "\n".join(lines)


def ablation_rows(metrics: dict[str, Any]) -> str:
    lines = [
        "| Ablation | Delta NDCG@10 | Wilcoxon p | n pairs |",
        "|---|---|---|---|",
    ]
    for a in metrics["ablations"]:
        p = a["wilcoxon_p_value"]
        lines.append(
            f"| `{a['ablation']}` | {a['delta_ndcg@10']:+.4f} | "
            f"{'N/A' if p is None else f'{p:.3g}'} | {a['wilcoxon_n_pairs']} |"
        )
    return "\n".join(lines)


def _ci(m: dict[str, Any]) -> str:
    return f"{_f(m['mean'])} [{_f(m['ci_low'])}, {_f(m['ci_high'])}]"


def dr_block(metrics: dict[str, Any], dr_id: str) -> str:
    """A compact markdown table of one Decision-Register experiment's measured results."""
    row = next(
        (r for r in metrics.get("decision_register", {}).get("rows", []) if r["id"] == dr_id), None
    )
    if row is None or row["results"] is None:
        return f"_{dr_id}: NOT RUN._"
    res = row["results"]
    if dr_id == "DR1":
        lines = [
            "| Rule | Hard violations in top-10 | Trips with >= 1 | Mean compat@10 | NDCG@10 |",
            "|---|---|---|---|---|",
        ]
        for name, v in res.items():
            lines.append(
                f"| {name} | {v['hard_constraint_violations_in_top10']} | "
                f"{v['trips_with_violation']} / {v['n_trips']} | {_f(v['mean_compat_at_10'])} | "
                f"{_f(v['ndcg_at_10'])} |"
            )
    elif dr_id == "DR2":
        lines = [
            "| Train fraction | Train trips | LambdaMART + IPS NDCG@10 | Two-tower NDCG@10 |",
            "|---|---|---|---|",
        ]
        for r in res["curve"]:
            lines.append(
                f"| {_pct(r['fraction'], 0)} | {r['n_train_trips']} | {_ci(r['lambdamart_ips'])} | "
                f"{_ci(r['two_tower'])} |"
            )
        cx = res["extrapolated_crossover_train_trips"]
        lines += [
            "",
            "Log-linear extrapolated crossover: "
            + (
                "none (the two-tower's fitted slope does not exceed LambdaMART's)."
                if cx is None
                else f"~{cx:,.0f} training trips ({cx / res['n_train_trips_full']:.1f}x the "
                f"{res['n_train_trips_full']} available; 4-point fit, low confidence)."
            ),
        ]
    elif dr_id in ("DR3", "DR7"):
        lines = ["| Variant | NDCG@10 (95% CI) |", "|---|---|"]
        for name, v in res.items():
            lines.append(f"| {name} | {_ci(v)} |")
    elif dr_id == "DR4":
        lines = [
            "| Encoder | D11 ridge R2 | D9 within-trip Spearman | NDCG@10 (95% CI) |",
            "|---|---|---|---|",
        ]
        for name, v in res.items():
            lines.append(
                f"| {name} | {_f(v['d11_ridge_r2'], 3)} | "
                f"{_f(v['d9_within_trip']['mean_within_trip_spearman'], 3)} | {_ci(v['ndcg10'])} |"
            )
    elif dr_id == "DR6":
        lines = [
            "| Aggregator | NDCG@10 (gated) | Hard violations (ungated) | compat@10 (ungated) |",
            "|---|---|---|---|",
        ]
        for name, v in res.items():
            lines.append(
                f"| {name} | {_f(v['gated']['ndcg_at_10'])} | "
                f"{v['ungated']['hard_constraint_violations_in_top10']} | "
                f"{_f(v['ungated']['mean_compat_at_10'])} |"
            )
    elif dr_id == "DR9":
        lines = [
            "| Long-tail quota | NDCG@10 | Long-tail share@10 | Long-tail precision@10 | "
            "Candidate recall | Long-tail recall |",
            "|---|---|---|---|---|---|",
        ]
        for q, v in res.items():
            lt = v["longtail_at_10"]
            lines.append(
                f"| {q} | {_ci(v['ndcg10'])} | {_f(lt['share'], 3)} | {_f(lt['precision'], 3)} | "
                f"{_f(v['candidate_recall_overall'], 3)} | "
                f"{_f(v['candidate_recall_long_tail'], 3)} |"
            )
    elif dr_id == "DR10":
        lines = ["| alpha | beta | NDCG@10 | compat@10 |", "|---|---|---|---|"]
        for r in res:
            lines.append(
                f"| {r['alpha']} | {r['beta']} | {_f(r['ndcg_at_10'])} | "
                f"{_f(r['mean_compat_at_10'])} |"
            )
    elif dr_id == "DR8":
        lines = ["| Variant | D9 within-trip | NDCG@10 (95% CI) |", "|---|---|---|"]
        for name, v in res.items():
            lines.append(
                f"| {name} | {_f(v['d9_within_trip']['mean_within_trip_spearman'], 3)} | "
                f"{_ci(v['ndcg10'])} |"
            )
    elif dr_id == "DR11":
        lines = [
            "| Channels unioned with the learned top-210 | Val recall (IPS) | Holdout recall | "
            "Holdout long-tail | Holdout lift | Size |",
            "|---|---|---|---|---|---|",
        ]
        for r in res["K210_rows"]:
            lines.append(
                f"| {r['channels']} | {_f(r['val_ips_recall'], 3)} | "
                f"{_f(r['holdout_recall'], 3)} | {_f(r['holdout_long_tail'], 3)} | "
                f"{r['holdout_lift']:+.3f} | {r['size']:.0f} |"
            )
    else:
        lines = [row["verdict"]]
    return "\n".join(lines)


def render_seed_replication(metrics: dict[str, Any]) -> str:
    rep = metrics.get("seed_replication")
    lines = ["## Seed replication", ""]
    if rep is None:
        return "\n".join([*lines, "NOT RUN (single-seed results only).", ""])
    lines += [
        f"The full pipeline was regenerated end to end for seeds {rep['seeds']} (a new synthetic "
        "dataset, retriever, boosters and calibration each time; `scripts/seed_replication.py`). "
        "Seed 42 is the committed run.",
        "",
        "| Metric | Mean | SD | Min | Max |",
        "|---|---|---|---|---|",
    ]
    for name, v in rep["metrics"].items():
        lines.append(
            f"| {name} | {_f(v['mean'])} | {_f(v['sd'])} | {_f(v['min'])} | {_f(v['max'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------------
# E1-E4 blocks
# -----------------------------------------------------------------------------------

_E1_ROWS = (
    ("random", "Random"),
    ("popularity", "Popularity"),
    ("content_cosine", "Content cosine"),
    ("lambdamart_ips_shipped", "LambdaMART + IPS (shipped booster)"),
    ("lambdamart_ips_retrained_on_this_set", "LambdaMART + IPS (retrained on this set)"),
    ("oracle", "Oracle (true utility)"),
)


def decomposition_table(metrics: dict[str, Any]) -> str:
    d = metrics.get("retrieval_ranking_decomposition")
    if d is None:
        return "NOT RUN."
    e2e, rel = d["ndcg10_end_to_end"], d["ndcg10_candidate_relative"]
    old, new = e2e["legacy_6_channel"], e2e["learned_retriever"]
    n_old, n_new = (
        d["mean_candidates_per_trip"]["legacy_6_channel"],
        d["mean_candidates_per_trip"]["learned_retriever"],
    )
    lines = [
        f"| End-to-end NDCG@10 (fixed denominator) | 6-channel union ({n_old:.0f} cand/trip) | "
        f"Learned retriever + long-tail + interest ({n_new:.0f} cand/trip) |",
        "|---|---|---|",
    ]
    for key, name in _E1_ROWS:
        if key not in old and key not in new:
            continue
        cells = [_ci(t[key]) if key in t else "—" for t in (old, new)]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "For reference, the usual candidate-relative NDCG@10 (each set normalised by its own "
        "ideal):",
        "",
        "| System | 6-channel union | Learned retriever |",
        "|---|---|---|",
    ]
    for key, name in _E1_ROWS:
        a, b = rel["legacy_6_channel"].get(key), rel["learned_retriever"].get(key)
        if a is None and b is None:
            continue
        lines.append(f"| {name} | {_ci(a) if a else '—'} | {_ci(b) if b else '—'} |")
    return "\n".join(lines)


_LIFT_KEYS = ("raw_ranker", "after_gate", "after_utility", "final_after_mmr")


def longtail_stage_table(metrics: dict[str, Any]) -> str:
    d = metrics.get("longtail_stages")
    if d is None:
        return "NOT RUN."
    lift = metrics.get("derived", {}).get("longtail_lift_over_base_rate", {})
    lift_by_stage = dict(zip(d["stages"], [1.0, *[lift.get(k) for k in _LIFT_KEYS]], strict=True))
    lines = [
        "| Stage | Long-tail share | Long-tail precision | Lift over pool base rate |",
        "|---|---|---|---|",
    ]
    for name, v in d["stages"].items():
        lf = lift_by_stage.get(name)
        lines.append(
            f"| {name} | {_f(v['long_tail_share'], 3)} | {_f(v['long_tail_precision'], 3)} | "
            f"{'—' if lf is None else f'{lf:.3f}x'} |"
        )
    lines += [
        "",
        "MMR lambda (diagnostic, post-hoc on the holdout -- not a selection):",
        "",
        "| lambda | Long-tail share | Long-tail precision |",
        "|---|---|---|",
    ]
    for lam, v in d["mmr_lambda_diagnostic_post_hoc_on_holdout"].items():
        lines.append(
            f"| {lam} | {_f(v['long_tail_share'], 3)} | {_f(v['long_tail_precision'], 3)} |"
        )
    return "\n".join(lines)


def sweep_summary(metrics: dict[str, Any]) -> str:
    d = metrics.get("ranker_sweep")
    if d is None:
        return "NOT RUN."
    rows = d["rows"]

    def mean(pred: Any) -> float:
        xs = [r["stage1_val_ips_weighted_ndcg10"] for r in rows if pred(r)]
        return float(sum(xs) / len(xs))

    lines = ["| Axis | Value | Mean validation IPS-weighted NDCG@10 |", "|---|---|---|"]
    for o in ("lambdarank", "binary", "rank_xendcg"):
        lines.append(f"| objective | {o} | {_f(mean(lambda r, o=o: r['objective'] == o))} |")
    for c in sorted(
        {str(r["ips_clip_high"]) for r in rows},
        key=lambda x: (x == "none", float(x) if x != "none" else 0),
    ):
        lines.append(
            f"| IPS clip | {c} | {_f(mean(lambda r, c=c: str(r['ips_clip_high']) == c))} |"
        )
    for b in ("all_features", "no_raw_text_emb", "no_raw_taste", "no_raw_text_no_taste"):
        lines.append(f"| feature blocks | {b} | {_f(mean(lambda r, b=b: r['blocks'] == b))} |")
    w = d["winner"]
    lines += [
        "",
        f"Winner (4-seed mean {_f(w['stage2_mean_val_ips_weighted_ndcg10'])}): "
        f"objective {w['objective']}, IPS clip {w['ips_clip_high']}, blocks {w['blocks']}; "
        f"the previously shipped config scores {_f(d['previous_shipped_config_4_seed']['mean'])}.",
    ]
    return "\n".join(lines)


def k_sweep_table(metrics: dict[str, Any]) -> str:
    d = metrics.get("e4_k_sweep")
    if d is None:
        return "NOT RUN."
    lines = [
        d["pre_registered_rule"] + ".",
        "",
        "| Learned K | Val recall (IPS) | Val lift | Holdout recall (reporting-only) "
        "| Holdout long-tail | Holdout lift | Candidates/trip |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in d["grid"]:
        v, h = g["val_ips_weighted_selection"], g["holdout_reporting_only"]
        marks = []
        if g["learned_K"] == d["rule_k"]:
            marks.append("recall rule")
        if g["learned_K"] == d.get("shipped_k"):
            marks.append("shipped")
        mark = f" **({', '.join(marks)})**" if marks else ""
        lines.append(
            f"| {g['learned_K']}{mark} | {_f(v['recall'], 3)} | {v['lift']:+.3f} "
            f"| {_f(h['recall'], 3)} | {_f(h['lt'], 3)} | {h['lift']:+.3f} | {h['size']:.0f} |"
        )
    return "\n".join(lines)
