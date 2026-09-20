# Submission — poi-intelligence-ranking (konnect.kr take-home)

> Generated from `submission/SUBMISSION.md.tmpl`; every number resolves from `results/metrics.json`.

**Repo:** https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking · **Tag:** `v1.0-konnect-submission`
(cold fresh-clone reproduction verified at commit `60c399a`).

## Clone and run (CPU only, no API keys, no network; these stages, in this order, were run from a cold clone)

```bash
git clone https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking.git && cd poi-intelligence-ranking
git checkout v1.0-konnect-submission
uv sync --frozen && export PYTHONHASHSEED=0      # PowerShell: $env:PYTHONHASHSEED = "0"
for s in generate prepare features candidates gate-dgp train evaluate representation \
         gate-representation compose; do uv run python -m poi_rank.cli $s; done   # = make reproduce
```

Verified on an empty cache: **695 s** including a
108 s dependency install (busy machine, upper bound);
the regenerated `results/metrics.json` is byte-identical to the committed one.

## What a reviewer should see (`results/metrics.json`, seed 42)

- Primary NDCG@10 **0.1842** vs popularity
  0.0523 (p=3.54e-75) and content cosine
  0.1336 (p=1.27e-14);
  5-seed: 0.1967 ± 0.0083 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1842, is the LOWEST of the five); above cosine in 5 of 5 seeds (mean gap +0.0623 NDCG@10).
- Candidate recall 0.926 overall / 0.898 long-tail;
  Gate-A pass **True**, Gate-B pass **True**;
  hard-constraint violations **0**; ECE 0.0300.

## Three strongest results

1. **Hard constraints (DR1).** The brief's additive formula puts 974
   hard-constraint violations in the top-10s of 671 holdout trips
   (230 trips affected); the shipped multiplicative, hard-gated
   utility has 0.
2. **Exposure bias is measured, not assumed.** The NDCG@10 gap between the biased log and the
   random-exposure holdout is +0.080 for popularity but only
   -0.005 for the IPS-trained primary; the headline is computed on the unbiased holdout.
3. **Oracle-free selection.** Every choice (retriever K by a pre-registered rule, features, LR
   regularisation, ranker sweep) was made on train-carved validation; the true-utility oracle scored
   results but never chose anything (firewall tests + `poi_rank.cli audit --deep`).

## What was missed, and why (full scorecard: `docs/RESULTS.md`)

- **Late-found bug, fixed** (`docs/TECHNICAL.md` section 5.1): train-time traveler features contained each
  trip's own labelled browsing session. As first tagged the ranker looked no better than cosine
  (0.1485 vs 0.1411); after the fix it is well above
  it. The fix also moved scorecard rows the other way: confidence-decile Spearman
  0.879 → 0.358 and scenario-4 flip overlap
  0.176 → 0.538; both now MISSED.
- % of oracle ceiling 50.0% (target 70%); within/cross-archetype ratio
  1.20 (target 2.0); localness index rho 0.582 (target 0.6).
- Long-tail share 0.144 (target 0.25); precision 0.196 is a
  2.026x lift over the pool base rate — the 0.40 target was set without reference to that rate.
- `make reproduce` is above the original 5-minute target and was not chased further; DR5 (ANN benchmark) not run. All data is synthetic.
