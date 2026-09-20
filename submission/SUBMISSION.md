# Submission — poi-intelligence-ranking (konnect.kr take-home)

> Generated from `submission/SUBMISSION.md.tmpl`; every number resolves from `results/metrics.json`.

**Repo:** https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking · **Tag:** `v1.0-konnect-submission` · **Requirement-by-requirement coverage:** `docs/REQUIREMENTS.md`

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
both are diagnosed with measured mechanisms, not tuned away. **Holdout feature distributions were inspected
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
git checkout v1.0-konnect-submission
uv sync --frozen && export PYTHONHASHSEED=0      # PowerShell: $env:PYTHONHASHSEED = "0"
for s in generate prepare features candidates gate-dgp train evaluate representation \
         gate-representation compose; do uv run python -m poi_rank.cli $s; done   # = make reproduce
make demo TRAVELER=U0005                          # live top-10 from the committed artifacts
```

`make reproduce` measured 335 s (5.6 min; reported as measured, no longer optimised).
The tagged commit was cold-cloned (empty uv cache, new venv), run in the order above, and its regenerated
`results/metrics.json` compared byte-for-byte with the committed file; the verified SHA and the result are in the
annotated tag (`git show v1.0-konnect-submission`), and `bash scripts/verify_fresh_clone.sh` repeats the check.

## What a reviewer should see (seed 42)

Primary NDCG@10 0.1842, popularity 0.0523; candidate recall
0.926 overall / 0.898 long-tail; Gate-A **True**, Gate-B **True**;
hard-constraint violations **0**; ECE 0.0300.

## What was missed, and why (7 MISSED rows; full scorecard: `docs/RESULTS.md`)

- % of oracle ceiling 50.0% (target 70%); localness index rho 0.582 (target 0.6).
- Long-tail share 0.144 (target 0.25) and precision 0.196. **Two rows trace to targets set without reference to the data's
  baselines**: long-tail precision 0.40 (pool base rate 0.097) and the within/cross-archetype ratio 2.0 (a perfect ranker reaches
  1.89); the recall 0.85 and chance-lift +0.35 targets were also set a priori and made the K rule infeasible (DR13).
- Decisions worth reading (DR3 objective, DR12 retrieval design, DR13 K selection): `docs/TECHNICAL.md` section 12.1. DR5 (ANN benchmark) not run; all data is synthetic.

## Anticipated questions

**Why does the ranking barely change when the touristiness preference flips?** It changes less than it should, and we
measured why before writing it up (`docs/TECHNICAL.md` section 3.1; `results/parts/touristiness_axis.json`). Negating the stated
preference on the 671 real holdout trips leaves the raw top-10 at Jaccard
0.904, but that average is diluted by travelers whose preference is near zero
(bottom |pref| tercile 0.973, top 0.834). It is not an
artifact of the simulator: outcomes carry the preference (trip-level Spearman -0.51) and the ranker
reproduces 83% of that gradient for cold-start trips but -1% for trips with history.
That is our measured answer to the brief's explicit-versus-implicit question: the blend is learned, and when the two conflict implicit
history wins (implicit-taste attribution 43.2% against
7.0% for the features that read the preference; removing the raw implicit block even improves
holdout NDCG@10 by +0.0120). Untested fixes: a preference-consistency term in the utility layer or a hard filter. For a platform whose
differentiator is non-touristy discovery, a stated preference should not be left for the ranker to learn.

**You shipped the retriever even though the legacy six-channel union scores higher end-to-end.** Correct, and it is reported
(DR12): the legacy union reaches NDCG@10 0.1919 against
0.1814 for the shipped retriever (paired p=0.0047), with the
ranker retrained on each set. The two sets differ on the criteria that were fixed before the comparison: candidate recall
0.926 against 0.596, long-tail recall
0.898 against 0.540, and
4 of 4 blocking Gate-B rows passed against
0. NDCG@10 over exposed labels is bounded by what the candidate set contains and does not reward
the long tail, which is the product's differentiator; the legacy set also holds fewer candidates per trip
(190 against 266). Switching to the legacy union
because it wins on the holdout would be selection on the holdout, so the row is reported and not acted on.

**Long-tail precision is 0.196 against a 0.40 target.** The 0.40 was set a priori, without reference to the data: the
candidate pool's own long-tail positive rate is 0.097, so the served list is a
2.03x lift over the base rate and the raw ranker's top-10 a 3.54x lift (precision
0.343). The MMR trade-off curve (a post-hoc holdout diagnostic, not a selection) runs from
1.90x at lambda 0.5 to 2.70x at lambda 1 with no diversity term (precision
0.262), so the target is out of reach even without diversity; the shipped lambda is
0.8. What is real: precision falls after ranking (raw 0.343 to served 0.196) and which of the gate,
utility and MMR steps causes it was not isolated. `docs/REQUIREMENTS.md` marks the long-tail row PARTIAL for that reason.
