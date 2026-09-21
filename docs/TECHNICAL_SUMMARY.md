# Technical summary — poi-intelligence-ranking

> Generated from `docs/TECHNICAL_SUMMARY.md.tmpl`; every number resolves from `results/metrics.json`. The brief's "concise technical document", in the brief's order. Depth: [TECHNICAL.md](TECHNICAL.md) (thorough), [RESULTS.md](RESULTS.md) (every measurement), [REQUIREMENTS.md](REQUIREMENTS.md) (requirement-by-requirement coverage). All data is synthetic.

## In five lines

1. **Ranking quality** (unbiased holdout, 671 trips): raw-ranker NDCG@10 0.1842; served top-10 (after gate, utility, MMR) 0.1636; content cosine 0.1336; popularity 0.0523.
2. **A train/serve skew was found and fixed:** train features saw each trip's own labelled session (median 3 days since last interaction on train trips, 102 on holdout); the first tag's "ranker no better than cosine" (0.1485 vs 0.1411) is retracted.
3. **A stated touristiness preference now steers the ranking:** served flip-overlap for trips with history 0.854 → 0.360; scenario-4 overlap 0.538 → 0.000.
4. **Constraints:** 0 hard-constraint violations in 6631 recommended slots, against 974 under the brief's additive formula.
5. **Oracle-free selection:** every choice was made on a train-carved validation split under a rule written beforehand; the true-utility oracle only scored results; the holdout was read once per decision.

## 1. Problem formulation

Given a traveler, a trip and a POI catalog, return a ranked, weighted list a downstream itinerary planner can consume. The problem is two-factor: a POI must be *preferred* and *compatible* with the trip (budget, hours, accessibility, party, mobility, reservation lead time). Two further concerns are first-class. **Exposure bias:** logged interactions record what a policy showed, so a popularity policy flatters a popularity ranker (popularity moves +0.0804 NDCG@10 between the biased and unbiased holdouts, the primary model -0.0046). **The commercial objective:** local and long-tail discovery for inbound travelers, not raw NDCG. The learning target is a graded relevance label per (trip, POI).
Depth: [TECHNICAL.md §1](TECHNICAL.md#1-problem-formulation).

## 2. Architecture

Datagen (firewalled) → data preparation → POI, traveler and pair features → candidate generation (a learned full-catalog retriever at K=240, a long-tail hard floor, an interest channel) → LambdaMART ranker (IPS-weighted) → isotonic calibration → hard gate and six compatibility terms → utility → MMR re-rank → grouped-TreeSHAP explanations → ranked, weighted JSON with scores, compatibility, confidence and planner weights. The diagram and module map are in the README.
Depth: [README architecture](../README.md#architecture), [TECHNICAL.md §1](TECHNICAL.md#1-problem-formulation).

## 3. Data assumptions

Everything is synthetic: 1875 travelers, 1446 catalog POIs in three destinations, inbound-traveler personas. The dataset is non-circular by construction: a latent-utility simulator whose truth lives only in `_oracle/`, readable solely by the evaluation oracle (tests scan for it). Two exposure logs: a popularity-biased training log with logged propensities, and a uniform-random holdout. Catalog dirtiness (duplicates, inconsistent categories, missing values) is injected. The simulator-quality gate passed (8 of 8 checks) and was frozen before any model work. Read every number as a property of this simulator, not of real traffic.
Depth: [TECHNICAL.md §0](TECHNICAL.md#0-what-the-simulator-does-and-does-not-license), [§2](TECHNICAL.md#2-dgp--circularity-defense).

## 4. Feature engineering

POI side: shrunk rating, review count, within-destination popularity percentile, a localness index, price, duration, opening hours, H3 geography, behavioural aggregates and a TF-IDF→SVD text embedding (out-of-fold R² to the latent semantics 0.865). Traveler side: an explicit block (interests, budget, party, mobility, stated touristiness preference) and an implicit block (an as-of-trip taste vector in the POI embedding space plus history summaries). Pair and cross features (taste cosine, price gap, localness × touristiness) join them; 258 columns reach the ranker. Every implicit feature is as-of the trip's own session start, which is the skew fix. Dropping the raw implicit-taste block changes holdout NDCG@10 by +0.0120; it was not applied, because choosing it on the holdout would be selection on the holdout.
Depth: [TECHNICAL.md §3](TECHNICAL.md#3-feature-architecture), [§3.1](TECHNICAL.md#31-how-explicit-and-implicit-signals-are-combined-brief-section-8--measured-for-trips-with-history-the-implicit-side-wins).

## 5. Model selection

LightGBM LambdaRank, one query group per trip. The brief prefers a simple model with thoughtful features to complexity, and a two-tower ranker was measured against it: 0.1624 against 0.1845 at full training data, with no crossover on the measured range. Pointwise objectives score higher on the holdout (binary 0.2046, regression 0.2033, against 0.1842); the objective was not switched because the validation-only sweep's best configuration beat the shipped one by 0.0052, under the pre-registered bar of 0.010.
Depth: [TECHNICAL.md §5](TECHNICAL.md#5-ranking-model-choice--lambdamart-justified-over-a-two-tower-ranker).

## 6. Training methodology

Train trips only. The label is the strongest interaction in a trip (view and dismiss lowest, click, then navigate, save and share, then visit and booking). Inverse-exposure-propensity weights, clipped and normalised per trip, correct the popularity-biased log. A share of training rows has its behavioural block masked so the ranker has seen the no-history regime. Early stopping uses train-carved validation trips; seed 42 throughout. Explicit cross features and a smaller tree were adopted under a pre-registered validation bar; on the holdout they added +0.0026 NDCG@10, so essentially all of the gain over cosine comes from fixing the skew.
Depth: [TECHNICAL.md §5.1](TECHNICAL.md#51-a-feature-skew-bug-found-late-and-its-fix), [§5.2](TECHNICAL.md#52-explicit-cross-features-and-ranker-tuning-experiment-h), [§6](TECHNICAL.md#6-ips-correction).

## 7. Ranking methodology

Candidates first, then the ranker over a few hundred per trip. Candidate recall of exposed holdout positives is 0.926 overall and 0.898 in the long-tail stratum, from a hard long-tail floor that the learned retriever cannot displace: the answer to "how do you avoid eliminating long-tail POIs". The retrieval design was decided on validation with the ranker retrained per design: learned retriever 0.1851, legacy six-channel union 0.1745 (long-tail recall 0.592, below the brief's requirement), their union 0.1845 (311 candidates, over the serving budget).
Depth: [TECHNICAL.md §4](TECHNICAL.md#4-candidate-generation), [§5](TECHNICAL.md#5-ranking-model-choice--lambdamart-justified-over-a-two-tower-ranker).

## 8. Scoring methodology

`utility = hard_gate × relevance^α × compatibility^β × pref_align^γ`. Relevance is the calibrated ranker score (ECE 0.471 → 0.030). Compatibility is the geometric mean of six terms; the hard gate removes POIs that are closed for the whole trip, inaccessible or unreachable. The multiplicative form deviates from the brief's additive example on purpose: a high relevance cannot outvote a failed constraint. `pref_align` is the stated touristiness preference as a per-trip factor in the scoring layer, where the brief puts trip context; γ and the MMR λ were selected together on validation. Each returned POI carries a calibrated confidence, a planner weight and a deterministic template explanation from grouped TreeSHAP.
Depth: [TECHNICAL.md §7](TECHNICAL.md#7-scoring-layer--multiplicative-utility-over-the-briefs-additive-formula), [§8](TECHNICAL.md#8-calibration).

## 9. Evaluation

Precision, recall and NDCG at K on the unbiased random-exposure holdout with 2000-resample trip bootstrap CIs and paired Wilcoxon tests: the ranker beats content cosine (p=1.3e-14; above cosine in 5 of 5 seeds (mean gap +0.0623 NDCG@10)). All six brief dimensions are measured: personalization, coverage, long-tail discovery, constraint compatibility, diversity, calibration. Success criteria were stated before measurement: eight met, six missed, each miss diagnosed with a measured mechanism. Some targets were set a priori without reference to the data's baselines (long-tail precision 0.266 against 0.40, where the pool's own rate is 0.097); they were reported as misses, not relaxed.
Depth: [TECHNICAL.md §10](TECHNICAL.md#10-evaluation-methodology), [RESULTS.md](RESULTS.md).

## 10. Cold-start strategy

One mechanism for all three cases: history-dependent features are as-of-trip and absent when there is no history, and the ranker is trained with the behavioural block masked. **New traveler:** NDCG@10 0.187 on 72 no-history trips against 0.182 with a long history. **New POI:** text, structured and geo features carry it. **New destination:** no feature is destination-specific; leave-one-destination-out NDCG@10 0.184 / 0.177 / 0.184 against 0.178 / 0.190 / 0.185 with the destination in training.
Depth: [TECHNICAL.md §10.3](TECHNICAL.md#103-cold-start-brief-section-15).

## 11. Production considerations

Prose only. At scale the retriever becomes an ANN stage over H3 shards while the ranker still scores a few hundred candidates (the ANN benchmark was not run, so this is reasoning). Offline nightly: embeddings, popularity percentiles, localness. Online: distance from the stay, open-now, availability, session context; one shared transformation library prevents train/serve skew. Weekly retrain with daily behavioural refresh, promoted on a fresh uniform-exposure slice plus interleaving. Log propensities and reserve uniform traffic as the feedback loop. Monitor feature drift, calibration, coverage, long-tail share, constraint violations (must stay zero) and cold-start share.
Depth: [TECHNICAL.md §11](TECHNICAL.md#11-production-considerations).

## Known limitations

- 50.0% of the oracle ceiling, against a 70% target: the taste estimator is the weak link.
- The preference factor overshoots: the served ordering follows the stated preference about 11x more steeply than observed behaviour.
- Confidence is weakly informative: decile Spearman 0.236.
- The new-POI dropout benefit is not demonstrated (p=0.33).
- Diversity sits on its constraint floor (entropy 2.94 bits) and MMR is effectively off (λ=1.0).
- The shipped LambdaRank objective is below binary and regression on the holdout (+0.0204); not switched, because validation did not clear the pre-registered bar.
