# poi-intelligence-ranking

> **Generated file.** Prose lives in `README.md.tmpl`; every number is resolved from
> `results/metrics.json` by `poi_rank.eval.report` (`make docs`). Do not hand-edit.

Personalized POI (point-of-interest) intelligence and ranking for **konnect.kr** — an
inbound-travel platform for foreigners in Korea — built on a fully synthetic, non-circular
dataset (three destinations, **Seoul first**). A two-factor (preference × compatibility)
recommender evaluated against an oracle ceiling, an exposure-bias-corrected primary metric, nine
measured ablations and a Decision Register in which every design choice carries a measured
number or an explicit `NOT RUN`.

## The result that matters commercially

The incumbents already rank by popularity. On the unbiased random-exposure holdout
(671 trips), top-10 lists:

| | This system | Popularity ranker |
|---|---|---|
| Long-tail share (bottom-50% popularity stratum) | **0.223** | 0.000 |
| Catalog coverage@10 | **64.9%** | 5.5% |
| NDCG@10 | **0.1480** [0.1380, 0.1582] | 0.0616 |

Long-tail **precision** is 0.145 (target 0.40 — a miss, diagnosed in
`docs/RESULTS.md`), so the honest claim is: it surfaces the long tail far more than a
popularity ranker at a real relevance gain, but not yet at the precision a product would want.
The bias-gap table (popularity +0.083 vs primary
+0.010 NDCG@10 between biased and unbiased holdouts) shows why the
comparison must be made on the unbiased holdout.

## Quick start

Requires Python 3.11/3.12 and [`uv`](https://docs.astral.sh/uv/). CPU only; no GPU, no API keys,
no network.

```bash
uv sync --frozen
export PYTHONHASHSEED=0      # byte-identical reruns (the Makefile sets it too)

uv run python -m poi_rank.cli generate
uv run python -m poi_rank.cli prepare
uv run python -m poi_rank.cli features
uv run python -m poi_rank.cli candidates
uv run python -m poi_rank.cli gate-dgp               # Gate-A: simulator quality (blocking)
uv run python -m poi_rank.cli train
uv run python -m poi_rank.cli evaluate
uv run python -m poi_rank.cli representation
uv run python -m poi_rank.cli gate-representation    # Gate-B: oracle-free recall rows (blocking)
uv run python -m poi_rank.cli compose                # the only writer of results/metrics.json
```

That is `make reproduce` (data → gates → train → evaluate → compose). `make reproduce-full` adds
`recommend` (a seeded 300-trip inspection sample), `scenarios`, `lodo` and the docs. Wall-clock
on a 16-logical-core laptop CPU (`scripts/time_reproduce.py`):
**367 s for `reproduce`**, **452 s for
`reproduce-full`**. Per-stage timings are in `docs/RESULTS.md`.

Integrity and tests:

```bash
uv run python -m poi_rank.cli audit          # deterministic invariant audit (JSON); --deep also
                                             # regenerates the dataset twice and compares SHA-256
uv run pytest -m "not slow" -n 12            # light tier, parallel
uv run pytest -m slow                        # heavy pipeline tests, serial lane
uv run ruff check src tests && uv run mypy
```

## Headline numbers (all generated from `results/metrics.json`)

| Metric | Value |
|---|---|
| Primary system NDCG@10 | **0.1480** |
| vs popularity | **+140.2% relative**, Wilcoxon p=5.69e-43 |
| vs best baseline (content cosine) | p=0.329 — NOT statistically significant at 0.05, so not a supported win over that baseline |
| % of candidate-level oracle ceiling | 39.6% (target ≥70% — missed, diagnosed) |
| Candidate recall (exposed positives), overall / long-tail | 0.873 / 0.848 |
| Hard-constraint violations in top-10 | **0** |
| ECE after isotonic calibration | **0.0418** |

Acceptance gates — Gate-A overall pass: **True**; Gate-B overall pass: **True**. Every scorecard row, every miss with its diagnosis, all nine
systems, the bias-gap table, per-stratum recall with chance baselines, cold-start cohorts, the
scenarios and the Decision Register are in **[`docs/RESULTS.md`](docs/RESULTS.md)**; the
reasoning is in **[`docs/TECHNICAL.md`](docs/TECHNICAL.md)**; every dataset decision in
**[`docs/DATA_CARD.md`](docs/DATA_CARD.md)**.

**What this does not show.** The data is synthetic and the labels come from a utility the
simulator author wrote; the text-derived fidelity numbers describe synthetic text; the labels do
not penalise hard-constraint violations. See TECHNICAL.md section 0 before reading anything as a
claim about real traffic.

## Architecture

```
datagen (firewalled) -> data prep -> features (POI + traveler + pair features)
                                          |
        candidates: learned full-catalog retriever (cross-fitted, IPS) + long-tail floor + interest
                                          |
                       LambdaMART ranker (IPS-weighted) -> isotonic calibration
                                          |
              compatibility (6 terms, hard-gated) -> utility = gate * rel^a * compat^b -> MMR
                                          |
                    explanations (grouped TreeSHAP on returned rows -> templates)
                                          |
                           ranked, weighted POI output (JSON)
```

## Repository layout

```
configs/         datagen / features / model / scoring / eval YAML (seed 42 everywhere)
src/poi_rank/    datagen (firewalled) data features candidates models scoring explain eval cli
data/synthetic/  committed dataset; _oracle/ is readable only by eval/oracle.py
artifacts/       models, retriever, confidence ensemble, poi_emb, calibrator
results/         parts/<stage>.json -> metrics.json (compose), scenarios/, figures/
docs/            TECHNICAL.md, RESULTS.md, DATA_CARD.md (+ TECHNICAL.md.tmpl)
scripts/         evidence scripts (A3/A4 experiments, timing)
tests/           firewall, oracle isolation, leakage, determinism, doc provenance, ...
```

## Reproducibility guarantees

- Seed 42 in every config; `PYTHONHASHSEED=0`; LightGBM trained with `deterministic=True`
  (trees identical at 1 vs 8 threads — asserted).
- The generator is byte-identical across runs; `poi_rank.cli audit --deep` checks this by SHA-256.
- `results/metrics.json` has a single writer (`compose`); a test and the audit enforce it.
- Dependencies are pinned in `uv.lock`; `sentence-transformers` is an opt-in extra used only by
  Decision Register DR4 and the D9 diagnostics.
