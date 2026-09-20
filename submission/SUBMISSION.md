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

- Primary NDCG@10 **0.1485** vs popularity
  0.0629 (p=4.34e-41) and content cosine
  0.1411 (p=0.445, not significant);
  5-seed: 0.1540 ± 0.0071 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1485, is number 2 of 5 counting from the lowest).
- Candidate recall 0.857 overall / 0.826 long-tail;
  Gate-A pass **True**, Gate-B pass **True**;
  hard-constraint violations **0**; ECE 0.0466.

## Three strongest results

1. **Hard constraints (DR1).** The brief's additive formula puts 901
   hard-constraint violations in the top-10s of 671 holdout trips
   (229 trips affected); the shipped multiplicative, hard-gated
   utility has 0.
2. **Exposure bias is measured, not assumed.** The NDCG@10 gap between the biased log and the
   random-exposure holdout is +0.083 for popularity but only
   +0.009 for the IPS-trained primary; the headline is computed on the unbiased holdout.
3. **Oracle-free selection.** Every choice (retriever K by a pre-registered rule, features, LR
   regularisation, ranker sweep) was made on train-carved validation; the true-utility oracle scored
   results but never chose anything (firewall tests + `poi_rank.cli audit --deep`).

## What was missed, and why (full scorecard: `docs/RESULTS.md`)

- Ranker vs content cosine: +0.0074 NDCG@10, not significant on one seed
  (above cosine in 4 of 5 seeds (mean gap +0.0110 NDCG@10)); the popularity-to-cosine step is +0.0782.
- % of oracle ceiling 39.6% (target 70%); within/cross-archetype ratio
  1.32 (target 2.0); localness index rho 0.582 (target 0.6).
- Long-tail share 0.225 (target 0.25); precision 0.154 is a
  1.556x lift over the pool base rate — the 0.40 target was set without reference to that rate.
- `make reproduce` is above the 5-minute goal; DR5 (ANN benchmark) not run. All data is synthetic.
