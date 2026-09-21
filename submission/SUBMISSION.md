# Submission — poi-intelligence-ranking (konnect.kr take-home)

> Generated from `submission/SUBMISSION.md.tmpl`; every number resolves from `results/metrics.json`.

**Repo:** https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking · **Tag:** `v1.1-konnect-submission` · **Requirement-by-requirement coverage:** `docs/REQUIREMENTS.md`

## Headline: we found a train/serve skew in our own pipeline, and retracted our own conclusion

Train-time traveler features were computed as-of each trip's `start_date`, but a trip's browsing session
(the source of the labels) is dated 0-45 days *before* it, so train and validation features carried the
labels and holdout features could not: the median days since the last interaction is
**3 on train trips vs
102 on holdout trips**. The first tag reported the ranker as no better than content
cosine (0.1485 vs 0.1411, p=0.445); that
conclusion is retracted in the docs (`docs/TECHNICAL.md` sections 5.1, 10.1) and the whole pipeline was re-run.
The fix also *cost* two scorecard rows (confidence-decile Spearman
0.879 → 0.358, scenario-4 flip overlap 0.176 → 0.538);
both were diagnosed with measured mechanisms, not tuned away. Experiment L (v1.1, below) then fixed the second in the scoring layer
(0.538 → 0.000); the first fell further (0.236). **Holdout feature distributions were inspected
to diagnose a train/serve skew. No labels were used for selection and no hyperparameter was chosen on holdout.**

## Three supporting results

1. **Primary vs content cosine:** NDCG@10 **0.1842** vs
   0.1336 (p=1.27e-14); above cosine in 5 of 5 seeds (mean gap +0.0623 NDCG@10); 5-seed 0.1967 ± 0.0083 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1842, is the LOWEST of the five).
2. **Hard constraints (DR1):** the brief's additive formula puts 974 violations into the
   top-10s of 671 holdout trips; the gated multiplicative utility has
   0.
3. **Oracle-free selection:** every choice (K, features, regularisation, ranker sweep, experiment H) was made on
   train-carved validation; the true-utility oracle scored results but never chose anything.

## Clone and run (CPU only, no API keys, no network)

```bash
git clone https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking.git && cd poi-intelligence-ranking
git checkout v1.1-konnect-submission
uv sync --frozen && export PYTHONHASHSEED=0      # PowerShell: $env:PYTHONHASHSEED = "0"
for s in generate prepare features candidates gate-dgp train evaluate representation \
         gate-representation compose; do uv run python -m poi_rank.cli $s; done   # = make reproduce
make demo TRAVELER=U0005                          # live top-10 from the committed artifacts
```

`make reproduce` measured 377 s (6.3 min; reported as measured, no longer optimised).
The tagged commit was cold-cloned (empty uv cache, new venv), run in the order above, and its regenerated
`results/metrics.json` compared byte-for-byte with the committed file; the verified SHA and the result are in the
annotated tag (`git show v1.1-konnect-submission`), and `bash scripts/verify_fresh_clone.sh` repeats the check.

## What a reviewer should see (seed 42)

Primary NDCG@10 0.1842, popularity 0.0523; candidate recall
0.926 overall / 0.898 long-tail; Gate-A **True**, Gate-B **True**;
hard-constraint violations **0**; ECE 0.0300.

## What was missed, and why (6 MISSED rows; full scorecard: `docs/RESULTS.md`)

- % of oracle ceiling 50.0% (target 70%); localness index rho 0.582 (target 0.6).
- Long-tail share 0.220 (target 0.25) and precision 0.266. **Two rows trace to targets set without reference to the data's
  baselines**: long-tail precision 0.40 (pool base rate 0.097) and the within/cross-archetype ratio 2.0 (a perfect ranker reaches
  1.89); the recall 0.85 and chance-lift +0.35 targets were also set a priori and made the K rule infeasible (DR13).
- Decisions worth reading (DR3 objective, DR12 retrieval design, DR13 K selection): `docs/TECHNICAL.md` section 12.1. DR5 (ANN benchmark) not run; all data is synthetic.

## Experiment L (v1.1): what changed after the first submission

Three model-layer decisions, each by a rule written before it ran (`docs/experiments/L-final.md`), on the train-carved validation split, the holdout
read once: (1) **the stated touristiness preference steers the served ranking** through a scoring-layer factor `pref_align^gamma` (no retraining), with gamma
and the MMR lambda selected together; (2) **the MMR lambda was selected**, not defaulted (selected: 1.0, which turns the diversity
re-rank off and sits on the pre-registered entropy constraint); (3) **the retrieval design was re-decided** on validation with the ranker retrained per design, after
Gate-B's blocking rows were amended (long-tail recall and a 300-candidate serving budget block; the a-priori overall-recall and chance-lift targets report only): the
learned retriever stays. On the holdout the scorecard's MET rows are unchanged (seven MET rows before, eight now: scenario-4 overlap
0.538 -> 0.000 passes its target); one MISSED row got worse (confidence-decile Spearman
0.358 -> 0.236).

## Anticipated questions

**Why does the ranking barely change when the touristiness preference flips?** In the ranker alone it did, and we measured why (`docs/TECHNICAL.md` section 3.1; `results/parts/touristiness_axis.json`):
outcomes carry the preference (trip-level Spearman -0.51) but the ranker reproduced
83% of that gradient for cold-start trips and -1% for trips with history, because history-derived features are the cheaper split
(implicit-taste attribution 43.2%; removing the raw implicit block even improves holdout NDCG@10 by +0.0120). That is our
measured answer to the brief's explicit-versus-implicit question: the blend is learned and, with history, the implicit side won. A stated "prefer less touristy" is a per-trip instruction, so it now enters
the scoring layer (v1.1): on the real holdout trips the served top-10 flip-overlap falls 0.837 -> 0.353 (with history 0.854 -> 0.360), the
scenario-4 overlap is 0.000 against a target of at most 0.35, and served NDCG@10 does not fall (0.1345 -> 0.1636). Two honest caveats:
the factor **overshoots** the outcome gradient (about 11x), because the pre-registered rule minimised flip-overlap under two constraints rather than matching the slope; and the selection turned the diversity re-rank off.

**You shipped the retriever even though the legacy six-channel union scores higher end-to-end.** DR12 reports it: on the *holdout* the legacy union reaches NDCG@10
0.1919 against 0.1814 (paired p=0.0047),
with recall 0.926 against 0.596 and long-tail recall
0.898 against 0.540. A holdout comparison cannot drive a switch, so v1.1 re-decided the design on validation with a
rule fixed in advance (`docs/experiments/L-final.md`): three designs (learned, legacy, their union), the ranker retrained per design, four seeds, blocking constraints of long-tail recall at least 0.75 and at most
300 effective candidates, and a +0.010 adoption bar. Result: learned 0.1851, legacy 0.1745 (long-tail recall 0.592, below the brief's requirement),
union 0.1845 (311 candidates, over the serving budget). Nothing cleared the bar, so the learned retriever stays; on validation the legacy union is not ahead, so its holdout
advantage in DR12 is not a stable property of the design.

**Long-tail precision is 0.266 against a 0.40 target.** The 0.40 was set a priori, without reference to the data: the candidate pool's own long-tail positive rate is
0.097, so the served list is a 2.74x lift over the base rate (precision was 0.196 before v1.1) and the raw ranker's top-10 a 3.54x lift (precision 0.343). The E3 stage table shows where it goes: the hard gate takes it to
0.268, the utility (now with the preference factor) leaves it at 0.266 while the long-tail share rises to 0.220
(target 0.25), and MMR at the selected lambda 1.0 changes nothing. The post-hoc lambda curve (a diagnostic, not a selection) runs from 1.67x at lambda 0.5 to
2.74x at lambda 1, so the target is out of reach even without a diversity term. `docs/REQUIREMENTS.md` keeps the long-tail row PARTIAL.
