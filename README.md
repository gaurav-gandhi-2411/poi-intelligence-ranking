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
holdout: **0.1967 ± 0.0083 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1842, is the LOWEST of the five)**. It is a **significant** improvement over the popularity ranker the
incumbents already run (Wilcoxon p=3.54e-75, seed 42) **and over a
plain content-cosine baseline** (p=1.27e-14 at seed 42; above cosine in
5 of 5 seeds (mean gap +0.0623 NDCG@10); 5-seed paired t-test p=0.000239).

**A late-found bug, and what it changed.** As first tagged, this submission reported the ranker
*indistinguishable* from cosine (NDCG@10 0.1485 vs
0.1411, p=0.445) and explained that as
"a well-constructed baseline is hard to beat". That explanation was wrong: train-time traveler features
contained each trip's own labelled browsing session, which holdout features cannot. Fixing the as-of
cutoff (`docs/TECHNICAL.md` section 5.1; the conclusion in the first tag is retracted) is what moved the
ranker above cosine; the pre-registered follow-up experiment on explicit cross features (section 5.2)
added essentially nothing on the holdout. The relative lift over popularity remains real but compares
against the weakest reasonable baseline, so cosine is the right yardstick.

## The result that matters commercially

The incumbents already rank by popularity. On the unbiased random-exposure holdout
(671 trips), top-10 lists:

| | This system | Popularity ranker |
|---|---|---|
| Long-tail share (bottom-50% popularity stratum) | **0.144** | 0.000 |
| Catalog coverage@10 | **47.3%** | 3.5% |
| NDCG@10 | **0.1842** [0.1738, 0.1945] | 0.0523 |

Long-tail **precision** is 0.196 against a pool base rate of
0.097, i.e. a **2.026x lift over base rate**
(3.538x at the raw ranker); lift over base rate is the primary statistic. The
0.40 raw-precision target was set a priori without reference to that base rate and was miscalibrated
at design time (`docs/TECHNICAL.md` section 4.2), so the honest claim is: it surfaces the long tail
far more than a popularity ranker at a real relevance gain, and ranks long-tail POIs well above the
pool's base rate, but not to the absolute precision the a-priori target assumed.
The bias-gap table (popularity +0.080 vs primary
-0.005 NDCG@10 between biased and unbiased holdouts) shows why the
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
**335 s (5.6 min) for `reproduce`**, **432 s for
`reproduce-full`**. Per-stage timings are in `docs/RESULTS.md`. The measured `reproduce` value is above the original 5-minute target and was not chased further (the two largest stages are `train` and `evaluate`, i.e. LightGBM fits; the per-stage table is in `docs/RESULTS.md`).

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

## Results

**[`docs/results.html`](docs/results.html)** is a single self-contained results explorer (no build step,
no CDN, no server — open it by double-click): the four scenarios with their top-10 tables and
compatibility breakdowns, the scenario overlap matrix, the full MET/MISSED scorecard with each miss's
diagnosis, the baseline table with confidence intervals and the bias-gap table. It is generated from
`results/metrics.json` and `results/scenarios/*.json` by `scripts/render_results_html.py` (part of
`make docs`). It is a reviewer aid, not a product UI — frontend is out of scope per the assignment brief.

### Live demo (uses the committed model artifacts)

```bash
# a real traveler's last trip, through the real pipeline
make demo TRAVELER=U0005
# a stated profile (a new traveler, cold-start path)
make demo INTERESTS=local_food,neighborhoods BUDGET=medium MOBILITY=public_transport \
          TOURISTINESS=-0.8 PARTY=solo DEST=seoul
# without make:
uv run python -m poi_rank.cli demo --traveler U0005
uv run python -m poi_rank.cli demo --interests local_food,neighborhoods --budget medium \
    --mobility public_transport --touristiness -0.8 --party solo --dest seoul
```

Each prints the top-10 with utility, preference, compatibility and confidence scores, the top SHAP
signals and the explanation lines, in a few seconds (the calibrator persisted by `recommend` is
loaded rather than re-fitted). `TOURISTINESS` is in [-1, 1]; `INTERESTS` accepts the dataset's
interest labels or the aliases `local_food`, `neighborhoods`, `museums`, `nightlife`.

## Headline numbers (all generated from `results/metrics.json`)

| Metric | Value |
|---|---|
| Primary system NDCG@10 (seed 42, committed run) | **0.1842** — 5-seed: 0.1967 ± 0.0083 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1842, is the LOWEST of the five) |
| vs popularity | **+252.0% relative**, Wilcoxon p=3.54e-75 |
| vs best baseline (content cosine) | p=1.27e-14 — statistically significant at 0.05 |
| % of candidate-level oracle ceiling | 50.0% (target ≥70% — missed, diagnosed) |
| Candidate recall (exposed positives), overall / long-tail | 0.926 / 0.898 |
| Hard-constraint violations in top-10 | **0** |
| ECE after isotonic calibration | **0.0300** |

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
