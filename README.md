# poi-intelligence-ranking

Personalized POI (point-of-interest) intelligence and ranking system built on a
fully synthetic, non-circular dataset — a two-factor (preference × compatibility)
recommender evaluated with an oracle ceiling, an exposure-bias-corrected primary
metric, and 9 measured ablations, rather than a single leaderboard number.

No CI badge is wired up for this repository (a solo take-home project on a fixed
deadline, not a continuously-deployed service) — every number below is instead
independently reproducible by re-running the pipeline yourself, per the quick-start
below.

## Summary

Given a traveler's trip (destination, dates, budget, mobility, interests, party
composition, accessibility needs) and a POI catalog, this system ranks POIs by a
two-factor utility: `relevance` (a LightGBM `lambdarank` model, IPS-corrected for
exposure bias in the training logs) combined **multiplicatively** with
`compatibility` (a 6-term hard-gated context-fit score), then MMR-diversified and
explained with grouped TreeSHAP + deterministic natural-language templates (no
LLM anywhere). Because the dataset is synthetic and authored by the same project
that evaluates against it, the entire submission is built around defending against
that circularity: a firewalled data-generating process, an oracle-ceiling
comparison, and a primary evaluation set drawn from **uniform-random exposure**
rather than the (separately simulated) popularity-biased logs the model trains on.
See `docs/TECHNICAL.md` for the full ML-reasoning writeup and `docs/DATA_CARD.md`
for every dataset design decision and resolved ambiguity.

## Quick start

Requires Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/). No GPU, no
paid API keys, CPU only.

```bash
uv sync --frozen
```

**Primary, verified quick-start** — the exact 9-command sequence below was run
end-to-end against this repository's own committed dataset while writing this
README (`docs/DATA_CARD.md` "Phase 10" measured-results table has the full
wall-clock breakdown and confirms every substantive output number reproduced
byte-identically). `make` is documented as an equivalent convenience wrapper below,
but was not itself exercised in this project's own development environment — no
GNU Make binary is present there (`docs/DATA_CARD.md` #87) — so this is the
sequence to trust if `make reproduce` doesn't work on your machine either:

```bash
export PYTHONHASHSEED=0   # required for byte-identical reruns; Makefile sets this too

uv run python -m poi_rank.cli generate     # ~10s  -- synthetic DGP: catalog, travelers, trips, exposure logs, oracle export
uv run python -m poi_rank.cli prepare      # ~5s   -- dedup, category canonicalization, rating shrinkage, localness, geo
uv run python -m poi_rank.cli features     # ~5s   -- POI + traveler feature tables, text embeddings
uv run python -m poi_rank.cli candidates   # ~40s  -- 6-channel candidate generation + candidate_recall@250
uv run python -m poi_rank.cli train        # ~25s  -- LambdaMART + LambdaMART+IPS (LightGBM lambdarank)
uv run python -m poi_rank.cli evaluate     # ~2m40s -- baselines, oracle ceiling, bias-gap, full eval suite, 9 ablations
uv run python -m poi_rank.cli lodo         # ~21s  -- leave-one-destination-out (3 full retrains; must run AFTER evaluate)
uv run python -m poi_rank.cli scenarios    # ~45s  -- the 3+1 required scenarios (spec.md section 15)
uv run python -m poi_rank.eval.report      # <1s   -- regenerates docs/RESULTS.md from results/metrics.json
```

Total wall clock for the full sequence above: well under 5 minutes per individual
command (spec.md's own stated CPU budget), ~4-5 minutes summed end-to-end.

**Equivalent convenience wrapper**, if your system has GNU Make installed
(Linux/macOS, or Windows via WSL/MSYS2/choco):

```bash
make reproduce   # generate -> prepare -> features -> candidates -> train -> evaluate -> scenarios -> docs -> test
make lodo        # separate: 3 full destination-held-out retrains (deliberately not in `reproduce`'s default chain)
```

`make reproduce` does not include `lodo` or `recommend` — both are deliberately
separate commands (`docs/DATA_CARD.md` #75, #62): LODO is 3 extra full LightGBM
retrains kept out of the default chain per spec.md section 16's own cut-order, and
`recommend` (scores every holdout trip and writes `results/recommendations.json`
with full explanations) is a standalone, on-demand command, not part of the
evaluation pipeline itself.

**Run the test suite and lint separately**:

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
```

## Expected headline numbers

If your run matches this repository's committed dataset (the pipeline is fully
seeded and deterministic — `configs/*.yaml`'s `seed: 42`, `PYTHONHASHSEED=0`, two
independent runs produce byte-identical `results/metrics.json`), you should see, on
the primary unbiased random-exposure holdout (202 trips):

| Metric | Value |
|---|---|
| **lambdamart_ips NDCG@10 (primary system)** | **0.0875** [0.0715, 0.1042] |
| % of oracle ceiling | **66.2%** |
| vs. popularity baseline | **+40.6% relative, Wilcoxon p=0.004197** |
| Oracle ceiling NDCG@10 | 0.1322 [0.1139, 0.1500] |
| Hard-constraint violations in top-10 | **0** (build-blocking test) |
| ECE after isotonic calibration | **0.0457** |
| Catalog coverage@10 (vs. popularity's 4.7%) | **27.4%** |

Full results, every baseline, every ablation, the bias-gap table, cold-start
cohorts, and the 3+1 scenario comparison: `docs/RESULTS.md` (generated from
`results/metrics.json`, never hand-typed — enforced by
`tests/test_report.py::test_every_number_in_results_md_traces_to_metrics_json`).

**Honest misses** (reported, not hidden — see `docs/TECHNICAL.md`'s closing
section for the full success-criteria table and root-cause diagnosis for each):
% of oracle ceiling (66.2% vs a ≥70% target), long-tail candidate recall (0.3936 vs
≥0.80), within/cross-archetype Jaccard ratio (1.08 vs ≥2.0), confidence-decile NDCG
monotonicity (ρ=-0.382 vs ≥0.7), and long-tail share/precision (0.2338/0.0678 vs
≥0.25/≥0.5). Every one of these traces to one of two already-diagnosed root causes:
a ~0.44 candidate-recall ceiling that bounds every downstream ranking metric in this
project, or the coarseness of an 8-cluster K-Means archetype proxy relative to what
actually drives personalization. None was tuned away.

## Architecture

```
datagen (firewalled) -> data prep -> feature store (POI + traveler)
                                          |
                          candidate generation (6 channels + quotas)
                                          |
                       LambdaMART ranker (IPS-weighted) -> calibration
                                          |
                    compatibility scoring (multiplicative + hard gates)
                                          |
                     utility = relevance^a x compatibility^b -> MMR diversify
                                          |
                      explanations (grouped TreeSHAP -> templates)
                                          |
                           ranked, weighted POI output (JSON)
```

Full module-by-module reasoning, every justified deviation from the brief
(LambdaMART over a two-tower ranker, multiplicative over additive utility), and the
complete evaluation methodology: **[`docs/TECHNICAL.md`](docs/TECHNICAL.md)**.
Every dataset design decision, the dirtiness catalog, and 87 resolved ambiguities
between the spec and what was actually built: **[`docs/DATA_CARD.md`](docs/DATA_CARD.md)**.
Generated results (the single source of truth for every number in this project):
**[`docs/RESULTS.md`](docs/RESULTS.md)**.

## Repository layout

```
poi-intelligence-ranking/
├── configs/            # datagen.yaml, features.yaml, model.yaml, scoring.yaml, eval.yaml
├── src/poi_rank/
│   ├── datagen/         # FIREWALLED: never imports downstream code
│   ├── data/            # prepare, dedup, hours, geo, categories
│   ├── features/        # poi_features, traveler_features, text, behavioral
│   ├── candidates/      # channels, union, quotas
│   ├── models/          # lambdamart, baselines, ranking_data
│   ├── scoring/         # compatibility, utility, calibration, confidence, diversity
│   ├── explain/         # shap_groups, templates, counterfactual
│   ├── eval/            # metrics, oracle, personalization, coverage, ablations, report
│   └── cli.py           # typer: generate | prepare | features | candidates | train |
│                         #        evaluate | lodo | recommend | scenarios
├── data/synthetic/      # committed (+ _oracle/, read only by eval/oracle.py)
├── artifacts/           # poi_emb.npy, model.txt, calibrator.pkl (committed, small)
├── results/             # metrics.json, scenarios/*.json, figures/
├── docs/
│   ├── TECHNICAL.md     # ML reasoning + engineering decisions (this document's companion)
│   ├── RESULTS.md       # GENERATED from results/metrics.json -- never hand-edited
│   └── DATA_CARD.md     # DGP description, dirtiness catalog, resolved ambiguities
└── tests/               # firewall, no-leakage, hard-constraint, determinism, doc-provenance tests
```

## Reproducibility guarantees

- Global seed `42` in every `configs/*.yaml`; `PYTHONHASHSEED=0`.
- Two independent full pipeline runs produce byte-identical `results/metrics.json`,
  `results/scenarios/*.json`, and `artifacts/model*.txt` (verified directly, not
  merely configured — `docs/DATA_CARD.md` #46, #78).
- `tests/test_firewall.py` and its per-phase siblings assert the DGP never leaks
  into downstream code by parsing every module's imports with `ast`.
- Dependencies are pinned in `pyproject.toml`; `sentence-transformers` is an
  opt-in extra (`uv sync --extra text`) for the non-default text-embedding path —
  the canonical, always-tested path is TF-IDF → SVD-64, which is what the quick
  start above actually exercises.
