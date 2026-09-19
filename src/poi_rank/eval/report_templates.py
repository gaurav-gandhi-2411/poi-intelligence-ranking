"""Template renderer for the hand-narrated documents (`docs/TECHNICAL.md`, `README.md`).

The prose lives in `docs/*.tmpl` / `README.md.tmpl`; every number is a `{{path|fmt}}` token
resolved against `results/metrics.json` (+ `results/scenarios/*`), and every table is a
`{{@block}}` expanded from the same data. A token that cannot be resolved raises -- a missing
metric can never render as a stale or blank value. That is how the "no hand-typed numbers in
docs/" rule is enforced for narrative documents, not just for the fully generated RESULTS.md.

Token grammar:  `{{a.b.0.c}}` (dotted path, integer segments index lists), optional format spec
after `|` (`{{systems.popularity.metrics.ndcg@10.mean|.4f}}`, `|.1%`, `|+.1f`, `|d`);
`{{@name}}` block; `{{~ expr }}` derived value (see `_DERIVED`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from poi_rank.eval import report_sections as rs

_TOKEN = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\|\s*([^{}]+?)\s*)?\}\}")


def resolve(ctx: Any, path: str) -> Any:
    cur = ctx
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"template path '{path}': no key '{part}' (have {sorted(cur)[:12]})")
            cur = cur[part]
        else:
            raise KeyError(f"template path '{path}': cannot descend into {type(cur).__name__}")
    return cur


def _relative_lift_pct(ctx: dict[str, Any]) -> float:
    ips = ctx["systems"]["lambdamart_ips"]["metrics"]["ndcg@10"]["mean"]
    pop = ctx["systems"]["popularity"]["metrics"]["ndcg@10"]["mean"]
    return float((ips / pop - 1.0) * 100.0)


def _sig_vs_content(ctx: dict[str, Any]) -> str:
    """Words for the primary-vs-best-baseline Wilcoxon result, so the prose can never claim a
    win the p-value does not support."""
    p = ctx["wilcoxon"]["lambdamart_ips_vs_content_cosine"]["p_value"]
    return (
        "statistically significant at 0.05"
        if p < 0.05
        else "NOT statistically significant at 0.05, so not a supported win over that baseline"
    )


_DERIVED: dict[str, Callable[[dict[str, Any]], float | str]] = {
    "lift_vs_popularity_pct": _relative_lift_pct,
    "sig_vs_content": _sig_vs_content,
}

Block = Callable[[dict[str, Any], dict[str, Any] | None], str]

BLOCKS: dict[str, Block] = {
    "scorecard": lambda m, s: _strip_heading(rs_scorecard(m, s)),
    "bias_gap": lambda m, s: rs.bias_gap_rows(m),
    "systems_table": lambda m, s: rs.systems_rows(m),
    "ablation_table": lambda m, s: rs.ablation_rows(m),
    "recall_strata": lambda m, s: _strip_heading(rs.render_candidate_recall(m)),
    "gates": lambda m, s: _strip_heading(rs.render_gates(m)),
    "decision_register": lambda m, s: _strip_heading(rs.render_decision_register(m)),
    "timings": lambda m, s: _strip_heading(rs.render_timings(m)),
    "seed_table": lambda m, s: _strip_heading(rs.render_seed_replication(m)),
    "dr1": lambda m, s: rs.dr_block(m, "DR1"),
    "dr2": lambda m, s: rs.dr_block(m, "DR2"),
    "dr3": lambda m, s: rs.dr_block(m, "DR3"),
    "dr4": lambda m, s: rs.dr_block(m, "DR4"),
    "dr6": lambda m, s: rs.dr_block(m, "DR6"),
    "dr7": lambda m, s: rs.dr_block(m, "DR7"),
    "dr9": lambda m, s: rs.dr_block(m, "DR9"),
    "dr10": lambda m, s: rs.dr_block(m, "DR10"),
    "dr11": lambda m, s: rs.dr_block(m, "DR11"),
}


def _strip_heading(md: str) -> str:
    lines = md.splitlines()
    while lines and (lines[0].startswith("## ") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines)


def rs_scorecard(metrics: dict[str, Any], scenarios: dict[str, Any] | None) -> str:
    from poi_rank.eval.report import render_success_criteria

    return render_success_criteria(metrics, scenarios)


def render_template(text: str, metrics: dict[str, Any], scenarios: dict[str, Any] | None) -> str:
    ctx: dict[str, Any] = {**metrics, "scenarios": scenarios or {}}

    def repl(match: re.Match[str]) -> str:
        path, fmt = match.group(1), match.group(2)
        if path.startswith("@"):
            name = path[1:]
            if name not in BLOCKS:
                raise KeyError(f"unknown template block '{name}'")
            return BLOCKS[name](metrics, scenarios)
        if path.startswith("~"):
            name = path[1:].strip()
            value = _DERIVED[name](ctx)
        else:
            value = resolve(ctx, path)
        return format(value, fmt) if fmt else str(value)

    return _TOKEN.sub(repl, text)
