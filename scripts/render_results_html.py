"""Render `docs/results.html`: a single self-contained results explorer.

Reads ONLY `results/metrics.json` and `results/scenarios/*.json` (through the same loaders and
markdown renderers `docs/RESULTS.md` uses), so every number traces to JSON. No build step, no CDN,
no external assets, no server: inline CSS + a few lines of JS for the tabs; opens by double-click.

    uv run python scripts/render_results_html.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from poi_rank.eval.report import (
    load_metrics,
    load_scenarios,
    render_bias_gap_table,
    render_primary_table,
    render_success_criteria,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results.html"
NOTE = (
    "Results explorer for reviewing pipeline output. Not a product UI — frontend is out of scope "
    "per the assignment brief."
)

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a63;--line:#dcd9cf;--acc:#0b5d5a;--bad:#a4352b;--ok:#2c6e3f;
--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a199;--line:#33322e;
--acc:#63c2bd;--bad:#ef8a80;--ok:#7fc58f;--card:#1e1e1c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,
sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.2rem 0}
h2{font-size:1.2rem;margin:1.6rem 0 .5rem;border-bottom:1px solid var(--line);padding-bottom:.2rem}
h3{font-size:1rem;margin:1rem 0 .3rem}
.note{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);
padding:.6rem .8rem;margin:.6rem 0 1rem;color:var(--mut)}
nav[role=tablist]{display:flex;flex-wrap:wrap;gap:.4rem;margin:.6rem 0}
button[role=tab]{background:var(--card);color:var(--fg);border:1px solid var(--line);
padding:.4rem .8rem;border-radius:6px;cursor:pointer;font:inherit}
button[role=tab][aria-selected=true]{border-color:var(--acc);color:var(--acc);font-weight:600}
button[role=tab]:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
section[role=tabpanel][hidden]{display:none}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:.88rem}
th,td{border:1px solid var(--line);padding:.3rem .5rem;text-align:left;vertical-align:top}
th{background:var(--bg)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
ul.expl{margin:0;padding-left:1.1rem}
.mut{color:var(--mut)}
.miss strong{color:var(--bad)}
.met strong{color:var(--ok)}
code{background:var(--line);padding:0 .25rem;border-radius:3px}
.empty{color:var(--mut);font-style:italic}
"""

JS = """
const tabs=[...document.querySelectorAll('button[role=tab]')];
function show(id){for(const t of tabs){const on=t.dataset.target===id;
t.setAttribute('aria-selected',on);document.getElementById(t.dataset.target).hidden=!on;}}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.target)));
tabs.forEach((t,i)=>t.addEventListener('keydown',e=>{
if(e.key==='ArrowRight'||e.key==='ArrowLeft'){const n=tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];
n.focus();show(n.dataset.target);}}));
if(tabs.length)show(tabs[0].dataset.target);
"""


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text.replace("&amp;ge;", "&ge;").replace("&amp;le;", "&le;")


def md_to_html(md: str) -> str:
    """Minimal markdown -> HTML (headings, pipe tables, bullets, paragraphs); enough for the
    generated RESULTS sections, not a general converter."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].replace("|", ""))
            <= {
                "-",
                " ",
                ":",
            }
        ):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append('<div class="tablewrap"><table><thead><tr>')
            out.append("".join(f"<th>{_inline(c)}</th>" for c in head))
            out.append("</tr></thead><tbody>")
            for r in rows:
                cls = ""
                if any("**MISSED**" in c for c in r):
                    cls = ' class="miss"'
                elif any("**MET**" in c for c in r):
                    cls = ' class="met"'
                out.append(f"<tr{cls}>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table></div>")
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")) + 1, 4)
            out.append(f"<h{level}>{_inline(line.lstrip('#').strip())}</h{level}>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)


def _f(v: Any, nd: int = 3) -> str:
    return "—" if v is None else f"{float(v):.{nd}f}"


def scenario_panel(idx: int, payload: dict[str, Any]) -> str:
    prof = payload["profile"]
    title = f"Scenario {payload['scenario_number']}: {payload['scenario_name']}"
    diag = (
        ""
        if "iagnostic" in payload["scenario_name"]
        else (" (diagnostic)" if payload["scenario_number"] == 4 else "")
    )
    prof_items = "".join(
        f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
        for k, v in prof.items()
        if k != "notes"
    )
    head = [
        "Rank",
        "POI",
        "Category",
        "Utility",
        "Preference",
        "Context compat.",
        "Confidence",
        "Popularity pct",
        "Localness index",
        "Why (explanation)",
    ]
    rows = []
    for r in payload["recommendations"]:
        expl = "".join(f"<li>{html.escape(e)}</li>" for e in r["explanation"])
        rows.append(
            "<tr>"
            f"<td class='num'>{r['rank']}</td><td>{html.escape(r['name'])}</td>"
            f"<td>{html.escape(r['category'])}</td>"
            f"<td class='num'>{_f(r['utility'])}</td><td class='num'>{_f(r['preference_score'])}</td>"
            f"<td class='num'>{_f(r['context_compatibility'])}</td>"
            f"<td class='num'>{_f(r['confidence'])}</td>"
            f"<td class='num'>{_f(r['popularity_percentile'], 1)}</td>"
            f"<td class='num'>{_f(r['localness_index'])}</td>"
            f"<td><ul class='expl'>{expl}</ul></td></tr>"
        )
    subs = list(payload["recommendations"][0]["compatibility_breakdown"])
    comp_rows = []
    for r in payload["recommendations"][:3]:
        cells = "".join(f"<td class='num'>{_f(r['compatibility_breakdown'][k])}</td>" for k in subs)
        comp_rows.append(
            f"<tr><td class='num'>{r['rank']}</td><td>{html.escape(r['name'])}</td>{cells}</tr>"
        )
    return (
        f'<section role="tabpanel" id="scn{idx}" hidden><h2>{html.escape(title)}{diag}</h2>'
        f"<details open><summary>Traveler profile</summary><ul>{prof_items}</ul>"
        f"<p class='mut'>{html.escape(str(prof.get('notes', '')))}</p></details>"
        '<h3>Top-10</h3><div class="tablewrap"><table><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in head)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        "<h3>Compatibility sub-scores, top-3</h3>"
        '<div class="tablewrap"><table><thead><tr><th>Rank</th><th>POI</th>'
        + "".join(f"<th>{html.escape(k)}</th>" for k in subs)
        + "</tr></thead><tbody>"
        + "".join(comp_rows)
        + "</tbody></table></div></section>"
    )


def overlap_html(overlap: dict[str, Any]) -> str:
    order = overlap["scenario_order"]
    m = overlap["matrix"]
    head = "".join(f"<th>Scenario {n}</th>" for n in order)
    rows = "".join(
        f"<tr><th>Scenario {n}</th>"
        + "".join(f"<td class='num'>{_f(v)}</td>" for v in m[i])
        + "</tr>"
        for i, n in enumerate(order)
    )
    return (
        "<h2>Top-10 Jaccard overlap across scenarios</h2>"
        f'<div class="tablewrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody>'
        "</table></div>"
    )


def build() -> str:
    metrics = load_metrics()
    scenarios = load_scenarios()
    tabs, panels = [], []
    if scenarios is not None:
        for i, payload in enumerate(scenarios["scenarios"]):
            n = payload["scenario_number"]
            label = f"Scenario {n}" + (" (diagnostic)" if n == 4 else "")
            tabs.append(
                f'<button role="tab" data-target="scn{i}" aria-selected="false">{label}</button>'
            )
            panels.append(scenario_panel(i, payload))
    body = [
        f"<h1>poi-intelligence-ranking — results explorer</h1><p class='note'>{html.escape(NOTE)}</p>",
        '<h2>Scenarios</h2><nav role="tablist" aria-label="Scenarios">' + "".join(tabs) + "</nav>"
        if tabs
        else "<p class='empty'>Scenarios not generated yet (run <code>poi_rank.cli scenarios</code>).</p>",
        *panels,
    ]
    if scenarios is not None:
        body.append(overlap_html(scenarios["overlap_matrix"]))
    body.append("<h2>Scorecard (MET / MISSED, each miss with its diagnosis)</h2>")
    body.append(md_to_html(render_success_criteria(metrics, scenarios)))
    body.append("<h2>Baselines and the primary system, with 95% CIs</h2>")
    body.append(md_to_html(render_primary_table(metrics)))
    body.append("<h2>Bias gap (biased vs unbiased holdout)</h2>")
    body.append(md_to_html(render_bias_gap_table(metrics)))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Results explorer</title>"
        f"<style>{CSS}</style></head><body><main>"
        + "\n".join(body)
        + f"</main><script>{JS}</script></body></html>"
    )


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
