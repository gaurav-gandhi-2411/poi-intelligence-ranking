# poi-intelligence-ranking

> **Generated file.** Prose lives in `README.md.tmpl`; every number is resolved from
> `results/metrics.json` by `poi_rank.eval.report` (`make docs`). Do not hand-edit.

Personalized POI (point-of-interest) intelligence and ranking for **konnect.kr** — an
inbound-travel platform for foreigners in Korea — built on a fully synthetic, non-circular
dataset (three destinations, **Seoul first**). A two-factor (preference × compatibility)
recommender evaluated against an oracle ceiling, an exposure-bias-corrected primary metric, nine
measured ablations and a Decision Register in which every design choice carries a measured
number or an explicit `NOT RUN`.

## Headline — read this first

Primary system (IPS-weighted LambdaMART + hard-gated utility) NDCG@10 on the unbiased random-exposure
holdout: **0.1540 ± 0.0071 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1485, is number 2 of 5 counting from the lowest)**. It is a **significant** improvement over the popularity ranker the
incumbents already run (Wilcoxon p=4.34e-41, seed 42) and a
**not significant** improvement over a plain content-cosine baseline (p=0.445
at seed 42; the primary is above content cosine in 4 of 5 seeds (mean gap +0.0110 NDCG@10)). A measured decomposition
(TECHNICAL.md section 4.1) shows the gain over popularity comes from *ranking*, not from the learned
retriever: the retriever raised candidate recall without moving end-to-end NDCG@10 significantly.
The relative-lift number against popularity is real but is not the right headline on its own,
because it compares against the weakest reasonable baseline.

## The result that matters commercially

The incumbents already rank by popularity. On the unbiased random-exposure holdout
(671 trips), top-10 lists:

| | This system | Popularity ranker |
|---|---|---|
| Long-tail share (bottom-50% popularity stratum) | **0.225** | 0.000 |
| Catalog coverage@10 | **66.4%** | 5.7% |
| NDCG@10 | **0.1485** [0.1385, 0.1590] | 0.0629 |

Long-tail **precision** is 0.154 against a pool base rate of
0.099, i.e. a **1.556x lift over base rate**
(2.234x at the raw ranker); lift over base rate is the primary statistic. The
0.40 raw-precision target was set a priori without reference to that base rate and was miscalibrated
at design time (`docs/TECHNICAL.md` section 4.2), so the honest claim is: it surfaces the long tail
far more than a popularity ranker at a real relevance gain, and ranks long-tail POIs well above the
pool's base rate, but not to the absolute precision the a-priori target assumed.
The bias-gap table (popularity +0.083 vs primary
+0.009 NDCG@10 between biased and unbiased holdouts) shows why the
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
**396 s (6.6 min) for `reproduce`**, **493 s for
`reproduce-full`**. Per-stage timings are in `docs/RESULTS.md`. The original target was 5 minutes; the measured value is above it and was not chased further (the two largest stages are `train` and `evaluate`, i.e. LightGBM fits; the per-stage table is in `docs/RESULTS.md`).

**Cold fresh-clone reproduction verified.** Commit `60c399a` was cloned
into a clean directory, dependencies installed with `uv sync --frozen` on an empty cache
(108 s), and the ten stages above run in order:
**695 s in total** (measured on a busy machine, so an upper bound). The regenerated
`results/metrics.json` is **byte-identical** to the one committed at that commit (SHA-256 recorded in
`results/parts/fresh_clone_verification.json`); no key differs.

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
| Primary system NDCG@10 (seed 42, committed run) | **0.1485** — 5-seed: 0.1540 ± 0.0071 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1485, is number 2 of 5 counting from the lowest) |
| vs popularity | **+136.2% relative**, Wilcoxon p=4.34e-41 |
| vs best baseline (content cosine) | p=0.445 — NOT statistically significant at 0.05, so not a supported win over that baseline |
| % of candidate-level oracle ceiling | 39.6% (target ≥70% — missed, diagnosed) |
| Candidate recall (exposed positives), overall / long-tail | 0.857 / 0.826 |
| Hard-constraint violations in top-10 | **0** |
| ECE after isotonic calibration | **0.0466** |

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
