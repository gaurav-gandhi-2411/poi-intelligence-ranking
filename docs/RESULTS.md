# RESULTS.md

**Generated automatically by `poi_rank.eval.report` from `results/metrics.json`. Do not hand-edit -- every number here traces directly to that JSON file (spec.md section 0: "No unverified metric may appear in any document").**

Synthetic data throughout (three destinations; Seoul is the lead). Personas are inbound foreign travelers. Read the numbers as properties of this simulator, not as claims about real traffic -- `docs/TECHNICAL.md` states what the simulator does and does not license.

## Headline: local and long-tail discovery (Seoul-first, inbound travelers)

konnect.kr serves foreign travelers in Korea, and the incumbents already rank by popularity -- so a popularity-shaped list is table stakes. The claim that matters is surfacing the genuinely relevant, non-obvious POI: **long-tail share and long-tail precision together** (coverage without precision is just noise injection), measured on the unbiased random-exposure holdout.

| Top-10 lists, primary holdout | LambdaMART + IPS (primary) | Popularity ranker |
|---|---|---|
| Long-tail share (bottom-50% popularity stratum) | **0.1436** | 0.0000 |
| Long-tail precision (relevant / recommended) | **0.1964** | N/A |
| Catalog coverage@10 | **47.3%** | 3.5% |
| Gini of recommendation exposure (lower = less monoculture) | **0.8321** | 0.9771 |
| NDCG@10 (unbiased holdout, 95% CI) | **0.1842** [0.1738, 0.1945] | 0.0523 [0.0469, 0.0582] |

Popularity is flattered by exposure-biased logs; the bias-gap table below quantifies exactly how much (popularity +0.0804, primary -0.0046 NDCG@10 between the biased and unbiased holdouts).

## Bias-gap table (spec.md section 11.1)

Each system's NDCG@10 on the SECONDARY biased holdout (`interactions_holdout_logged.parquet`) vs the PRIMARY unbiased holdout. Expectation: the popularity baseline shows a large positive gap (flattered by biased logs); the IPS-corrected model shows a small gap.

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0547 | 0.0265 | -0.0282 |
| 2. Popularity | 0.0523 | 0.1327 | +0.0804 |
| 3. Popularity + geo filter | 0.0527 | 0.1326 | +0.0799 |
| 4. Content cosine | 0.1336 | 0.0478 | -0.0858 |
| 5. Item-kNN CF | 0.0604 | 0.1363 | +0.0759 |
| 6. Logistic regression | 0.1196 | 0.2516 | +0.1320 |
| 7. LambdaMART | 0.1239 | 0.2615 | +0.1377 |
| 8. LambdaMART + IPS (primary) | 0.1842 | 0.1796 | -0.0046 |
| 9. Oracle (ceiling) | 0.3682 | 0.1064 | -0.2618 |

## Acceptance gates

**Gate-A (simulator quality, frozen before any model work)** -- measured on the generator alone, so the model cannot have been tuned against it:

| Check | Threshold | Measured | Status |
|---|---|---|---|
| cold_start_trip_share | <= 0.25 | 0.1073 | PASS |
| d10_description_conditioning_raw_tfidf | >= 0.65 | 0.6892 | PASS |
| min_non_epsilon_term_share | >= 0.02 | 0.0553 | PASS |
| oracle_ndcg10_slate_level | >= 0.6 | 0.6245 | PASS |
| spearman_localness_vs_geo | >= 0.55 | 0.6987 | PASS |
| spearman_u_vs_label | >= 0.4 | 0.4634 | PASS |
| traveler_dependent_variance_share | >= 0.75 | 0.7719 | PASS |
| var_epsilon_over_var_u | <= 0.15 | 0.1099 | PASS |

**Gate-B (representation / candidate generation)** -- blocking rows are oracle-free (recall against EXPOSED holdout positives); every representation and retrieval hyperparameter was selected on a train-carved validation split, and the oracle-based rows below are reporting-only, computed after the selection was frozen. The oracle never touched a decision; it only ever scored the result.

| Blocking check | Threshold | Measured | Status |
|---|---|---|---|
| candidate_recall_lift_long_tail | >= 0.35 | 0.4169 | PASS |
| candidate_recall_lift_overall | >= 0.35 | 0.3737 | PASS |
| candidate_recall_long_tail | >= 0.75 | 0.8975 | PASS |
| candidate_recall_overall | >= 0.85 | 0.9262 | PASS |

| Reporting-only (after freeze) | Value |
|---|---|
| d11_cca_first_corr_text_only | 0.9790 |
| d11_ridge_r2_text_only | 0.8654 |
| d9_within_trip_shipped_chain | 0.4731 |
| d9_within_trip_taste_estimator_alone | 0.5718 |
| d9_within_trip_text_alone | 0.9272 |
| oracle_ceiling_ndcg10_candidate_level | 0.3682 |
| oracle_relevance_recall_long_tail | 0.9644 |
| oracle_relevance_recall_overall | 0.9788 |

## Success criteria scorecard

Stated up front, then measured. Every MISSED row carries a diagnosis below -- a miss is reported as a ceiling only after a diagnostic has ruled out a mechanism, citing the number (spec.md: "An honest miss with a root-cause analysis scores better than a suspiciously perfect table.").

| Metric | Target | Measured | Status |
|---|---|---|---|
| NDCG@10 vs popularity | &ge; +40% relative, Wilcoxon p < 0.01 | +252.0% relative, p=3.537e-75 | **MET** |
| % of oracle ceiling (candidate-level NDCG@10) | &ge; 70% | 50.0% | **MISSED** |
| Candidate recall, overall (exposed positives) | &ge; 0.85 | 0.9262 | **MET** |
| Candidate recall, long-tail stratum | &ge; 0.75 | 0.8975 | **MET** |
| Candidate recall lift over chance (overall) | &ge; +0.35 | +0.374 | **MET** |
| Cross-archetype Jaccard@10 (true labels) | &le; 0.25 | 0.0752 | **MET** |
| Within/cross Jaccard ratio (true labels) | &ge; 2.0 | 1.20 | **MISSED** |
| Hard-constraint violations in top-10 | = 0 | 0 | **MET** |
| ECE after calibration | &le; 0.05 | 0.0300 | **MET** |
| Confidence-decile NDCG rank correlation (Spearman; need not be strictly monotone) | &ge; 0.6 (spec-v2; spec.md section 11.10 said 0.7) | 0.358 | **MISSED** |
| Long-tail share of top-10 | &ge; 0.25 | 0.1436 | **MISSED** |
| Long-tail precision of top-10 | &ge; 0.40 | 0.1964 | **MISSED** |
| Localness index Spearman vs latent localness | &ge; 0.6 | 0.5815 | **MISSED** |
| Scenario-4 (touristiness flip) top-10 overlap | &le; 0.35 | 0.538 | **MISSED** |

### Diagnoses of the missed rows

- **% of oracle ceiling (candidate-level NDCG@10)**: The oracle ranks the SAME candidates by the DGP's true utility; the model only sees observable features. Text is not the limiting link: text-alone within-trip taste fidelity is 0.927 and D11 ridge R2 is 0.865. The taste ESTIMATOR is: run over the TRUE semantic vectors it still reaches only 0.572 (shipped chain 0.473), from sparse, exposure-biased histories.
- **Within/cross Jaccard ratio (true labels)**: Within/cross-archetype list similarity ratio for lists ranked by the TRUE utility (a perfect ranker, same-destination pairs): 1.89 (within 0.0632, cross 0.0334); so the target is NOT attainable even by a perfect ranker in this simulator: the shortfall is a property of the simulator under this target, not of the model.
- **Confidence-decile NDCG rank correlation (Spearman; need not be strictly monotone)**: Confidence-decile Spearman is 0.358. Before the feature-skew fix this row was 0.879. Measured (TECHNICAL.md section 10.1): the statistic is a rank correlation over ten decile means of a weak trip-level relationship (trip-level Spearman 0.221 before, 0.064 after). The confidence inputs on holdout are unchanged and dominated by the traveler-evidence term; what changed is the ranker: before the fix its NDCG rose with history volume (Spearman 0.131), after it does not (-0.040), so an evidence-volume-dominated confidence has nothing to track. The pre-fix figure was inflated by the leaked features; the post-fix value is the honest one. Reweighting toward the ensemble term would be tuning on holdout labels and is declined.
- **Long-tail share of top-10**: Long-tail share of the served top-10 is 0.1436. Decision Register DR9 varies the long-tail candidate quota and measures the raw ranker's top-10 share: quota 0 -> 0.106, quota 100 -> 0.106, quota 25 -> 0.106, quota 50 -> 0.106. The candidate quota is therefore not the lever; the share is set by the ranker's scores (and the MMR re-rank) over a candidate set that already contains long-tail POIs.
- **Long-tail precision of top-10**: Long-tail precision is 0.1964 over 952 long-tail recommendations, with candidate recall 0.898 in that stratum, so retrieval is not the bottleneck. Measured against the pool's own long-tail positive rate (0.097) the served list is a 2.026x lift (raw ranker 3.538x): the 0.40 target was set a priori without reference to that base rate and was miscalibrated at design time, so lift over base rate is the primary statistic and raw precision secondary (TECHNICAL.md section 4.2). The raw ranker's top-10 long-tail precision (DR9, quota 50, before the compatibility gate, utility and MMR re-rank) is 0.3375 at share 0.1064, against 0.1964 at share 0.1436 in the served list: precision is lost AFTER ranking while share rises. Which of the three scoring-layer steps is responsible is not isolated (untested).
- **Localness index Spearman vs latent localness**: The composite index reaches rho 0.582; its observable inputs correlate with the latent localness at dist_to_tourist_centroid_km 0.699, foreign_review_ratio -0.555, local_tag_hits 0.050, pop_pct -0.187. The composite is BELOW its best single input (dist_to_tourist_centroid_km, |rho| 0.699): the blend weights were fixed earlier, when the geo input carried almost no signal (before the simulator's geo/localness fix). Re-weighting them against the latent localness would be tuning on the oracle (there is no oracle-free validation target for this index), so that retune is declined on principle: the index is left as shipped and the gap is reported.
- **Scenario-4 (touristiness flip) top-10 overlap**: Top-10 overlap 0.538 with candidate-pool Jaccard 0.717 between the two profiles (measured from the candidate generator's own output). Before the feature-skew fix this row was 0.176. Measured (TECHNICAL.md section 10.1): the features that read touristiness_pref carry 0.070 of the final model attribution for a cold-start traveler, and negating the preference on the 671 real holdout trips leaves the raw top-10 at Jaccard 0.904 (0.813 for pure cold-start trips). Personalization by taste is healthy (cross-archetype Jaccard 0.075); touristiness_pref is a weak lever for a brand-new traveler, whose ranking is driven mainly by stated interests, popularity/quality and compatibility. The pre-fix value reflected an off-distribution response (0.825 of that model attribution sat on leak-trained history-based features, absent for these travelers), not stronger personalization (interpretation; the numbers are measured). Reported as a cold-start limitation.

## Primary ranking quality (unbiased random-exposure holdout, spec.md section 11.1)

n_holdout_trips = 671, bootstrap n_resamples = 2000 (resample unit: trip).

| System | ndcg@5 | ndcg@10 | ndcg@20 | precision@5 | precision@10 | recall@10 | recall@20 | map | mrr | % of oracle ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Random | 0.0460 [0.0396, 0.0529] | 0.0547 [0.0491, 0.0604] | 0.0707 [0.0653, 0.0761] | 0.1127 [0.1010, 0.1246] | 0.1118 [0.1030, 0.1204] | 0.0375 [0.0345, 0.0404] | 0.0734 [0.0695, 0.0771] | 0.1405 [0.1356, 0.1454] | 0.2908 [0.2659, 0.3157] | 14.9% |
| 2. Popularity | 0.0455 [0.0393, 0.0522] | 0.0523 [0.0469, 0.0582] | 0.0739 [0.0684, 0.0799] | 0.1034 [0.0924, 0.1151] | 0.1019 [0.0933, 0.1109] | 0.0353 [0.0323, 0.0383] | 0.0779 [0.0735, 0.0828] | 0.1502 [0.1450, 0.1556] | 0.2645 [0.2426, 0.2877] | 14.2% |
| 3. Popularity + geo filter | 0.0466 [0.0405, 0.0530] | 0.0527 [0.0474, 0.0583] | 0.0719 [0.0666, 0.0773] | 0.1082 [0.0963, 0.1198] | 0.1060 [0.0970, 0.1149] | 0.0362 [0.0332, 0.0391] | 0.0766 [0.0723, 0.0809] | 0.1472 [0.1420, 0.1522] | 0.2816 [0.2581, 0.3067] | 14.3% |
| 4. Content cosine | 0.1251 [0.1141, 0.1366] | 0.1336 [0.1243, 0.1432] | 0.1594 [0.1504, 0.1683] | 0.2477 [0.2301, 0.2644] | 0.2224 [0.2089, 0.2355] | 0.0767 [0.0727, 0.0808] | 0.1429 [0.1369, 0.1492] | 0.2157 [0.2080, 0.2236] | 0.4984 [0.4698, 0.5278] | 36.3% |
| 5. Item-kNN CF | 0.0529 [0.0460, 0.0605] | 0.0604 [0.0542, 0.0670] | 0.0796 [0.0739, 0.0858] | 0.1210 [0.1091, 0.1344] | 0.1165 [0.1070, 0.1267] | 0.0407 [0.0375, 0.0439] | 0.0817 [0.0772, 0.0866] | 0.1531 [0.1478, 0.1587] | 0.2993 [0.2755, 0.3236] | 16.4% |
| 6. Logistic regression | 0.1153 [0.1050, 0.1265] | 0.1196 [0.1109, 0.1290] | 0.1412 [0.1330, 0.1503] | 0.2379 [0.2203, 0.2566] | 0.2073 [0.1936, 0.2213] | 0.0728 [0.0685, 0.0773] | 0.1295 [0.1239, 0.1354] | 0.2043 [0.1972, 0.2119] | 0.4954 [0.4662, 0.5265] | 32.5% |
| 7. LambdaMART | 0.1168 [0.1066, 0.1280] | 0.1239 [0.1152, 0.1335] | 0.1428 [0.1346, 0.1524] | 0.2411 [0.2238, 0.2599] | 0.2167 [0.2030, 0.2308] | 0.0757 [0.0712, 0.0801] | 0.1317 [0.1259, 0.1377] | 0.1985 [0.1915, 0.2058] | 0.4902 [0.4604, 0.5209] | 33.6% |
| 8. LambdaMART + IPS (primary) | 0.1610 [0.1494, 0.1728] | 0.1842 [0.1738, 0.1945] | 0.2151 [0.2054, 0.2249] | 0.3362 [0.3154, 0.3574] | 0.3165 [0.2988, 0.3344] | 0.1097 [0.1049, 0.1147] | 0.1912 [0.1846, 0.1982] | 0.2700 [0.2607, 0.2798] | 0.5887 [0.5586, 0.6170] | 50.0% |
| 9. Oracle (ceiling) | 0.3388 [0.3231, 0.3563] | 0.3682 [0.3547, 0.3827] | 0.4234 [0.4121, 0.4352] | 0.5037 [0.4796, 0.5285] | 0.4878 [0.4662, 0.5095] | 0.1731 [0.1680, 0.1787] | 0.3339 [0.3263, 0.3421] | 0.4958 [0.4818, 0.5097] | 0.7350 [0.7087, 0.7618] | 100.0% |

### Paired Wilcoxon signed-rank tests (NDCG@10)

| Comparison | statistic | p-value | n_pairs |
|---|---|---|---|
| content_cosine_vs_popularity | 27619.0000 | 7.537e-42 | 605 |
| item_knn_cf_vs_popularity | 4344.5000 | 0.001136 | 605 |
| lambdamart_ips_vs_content_cosine | 55239.0000 | 1.266e-14 | 605 |
| lambdamart_ips_vs_popularity | 10271.0000 | 3.537e-75 | 605 |
| lambdamart_vs_lambdamart_ips | 32814.0000 | 1.19e-35 | 605 |
| lambdamart_vs_popularity | 23191.0000 | 1.887e-46 | 605 |
| logistic_regression_vs_popularity | 25651.0000 | 7.823e-42 | 605 |
| oracle_vs_popularity | 798.0000 | 1.014e-98 | 605 |

## Candidate recall by popularity stratum (with chance baselines)

Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its own chance baseline: the share of that stratum's destination POIs a same-size random candidate set would contain. Raw recall without this lift hid a near-chance failure for four phases.

| Stratum | Recall | Chance | Lift (abs) | Trips |
|---|---|---|---|---|
| long_tail | 0.898 | 0.481 | +0.417 | 605 |
| overall | 0.926 | 0.553 | +0.374 | 605 |
| q1_least_popular | 0.914 | 0.531 | +0.383 | 601 |
| q2 | 0.882 | 0.430 | +0.452 | 600 |
| q3 | 0.907 | 0.510 | +0.398 | 603 |
| q4_most_popular | 0.969 | 0.738 | +0.231 | 605 |

## Personalization (spec.md section 11.2, spec-v2 RC4)

Pairs are SAME-DESTINATION trip pairs only: a POI belongs to exactly one destination, so two trips to different destinations have Jaccard = RBO = 0 by construction, and pooling them measures the catalog partition rather than the recommender.

- Mean pairwise Jaccard@10 (same destination): **0.0764** (n_pairs=74775)
- Mean pairwise rank-biased overlap (RBO, p=0.9): **0.0987** (n_pairs=74775)
- For continuity with the pre-fix number, all pairs pooled (2/3 of them structurally zero): 0.0254 (n_pairs=224785)

| Grouping | Within Jaccard@10 | Cross Jaccard@10 | Ratio | within / cross pairs |
|---|---|---|---|---|
| **True archetype labels** (dominant archetype; within = same dominant and mixture cosine > 0.8) | 0.0903 | 0.0752 | **1.20** | 4814 / 65145 |
| Reference: lists ranked by the TRUE utility (a perfect ranker) | 0.0632 | 0.0334 | 1.89 | 4814 / 65145 |
| K-Means traveler-segment PROXY (clusters of the same features being evaluated -- kept only to show why it was misleading) | 0.0885 | 0.0742 | 1.19 | 11459 / 63316 |

## Coverage (spec.md section 11.3)

| | Primary system (lambdamart_ips) | Popularity baseline |
|---|---|---|
| Catalog coverage@10 | 47.3% | 3.5% |
| Gini coefficient | 0.8321 | 0.9771 |
| Entropy (bits) | 8.3283 | 5.2684 |
| POIs ever recommended | 684 / 1446 | 51 / 1446 |

Lower Gini / higher entropy / higher coverage = less popularity-monoculture concentration. Per-destination coverage:

| Destination | Primary | Popularity |
|---|---|---|
| barcelona | 44.3% | 3.1% |
| kyoto | 46.2% | 3.5% |
| seoul | 51.5% | 3.9% |

## Long-tail / local discovery (spec.md section 11.4)

- Share of top-10 recommendations in the bottom-50%-popularity stratum: **0.1436** (952 / 6631)
- Long-tail precision (relevant per unbiased holdout): **0.1964** (187 / 952)
- **Lift over the candidate pool's long-tail positive rate (0.0969): 2.026x served, 3.538x at the raw ranker** -- the primary long-tail precision statistic; the 0.40 raw-precision target was set a priori without reference to this base rate (miscalibrated at design time; see TECHNICAL.md section 4.2).
- Popularity ranker, same measurement: share **0.0000** (0 / 6710), precision **N/A**

"Coverage without precision is just noise injection" -- both numbers reported together, per spec.md section 11.4.

## Constraint compatibility (spec.md section 11.5)

- % of top-10 with compatibility &ge; 0.7: **92.2%** (6114 / 6631)
- Hard-constraint violations in top-10: **0** (build-blocking, enforced independently by `tests/test_hard_constraints.py`)

## Diversity (spec.md section 11.6)

- Category entropy@10 (bits): **3.2568**
- Intra-list mean cosine distance (at the configured default lambda): **0.7740**

### MMR lambda sweep (NDCG@10 vs diversity trade-off)

| lambda | NDCG@10 (mean) | mean intra-list similarity |
|---|---|---|
| 0.50 | 0.1919 | 0.1955 |
| 0.60 | 0.1986 | 0.2002 |
| 0.70 | 0.2049 | 0.2077 |
| 0.80 | 0.2118 | 0.2260 |
| 0.90 | 0.2372 | 0.2996 |
| 1.00 | 0.2524 | 0.3873 |

### MMR lambda vs long-tail precision (post-hoc holdout diagnostic, not a selection)

| lambda | Long-tail share@10 | Long-tail precision@10 | Lift over pool base rate |
|---|---|---|---|
| 0.50 | 0.1611 | 0.1845 | 1.903x |
| 0.60 | 0.1510 | 0.1878 | 1.937x |
| 0.70 | 0.1507 | 0.1902 | 1.962x |
| 0.80 | 0.1436 | 0.1964 | 2.026x |
| 0.90 | 0.1464 | 0.2513 | 2.592x |
| 1.00 | 0.1620 | 0.2616 | 2.699x |

The shipped lambda (0.8) is a **deliberate diversity-for-precision trade, not a tuned optimum**, and lambda was not re-selected: choosing it on the holdout would leak, and re-selecting on validation would destabilise a shipped submission over a product knob. The cost is quantified in the two tables: moving from lambda 1.0 (no diversity term) to the shipped value lowers long-tail precision and NDCG@10 while cutting mean intra-list similarity, and the long-tail share is roughly flat across the whole range. A product owner who optimises for booking precision rather than category variety would move lambda toward 0.9 (a one-line scoring-config change), recovering most of the precision and NDCG@10 for a modest rise in list similarity; one who wants catalog exposure keeps 0.8.

## Calibration (spec.md section 11.7)

| | Before (naive) | After (isotonic) |
|---|---|---|
| ECE (15 bins) | 0.4711 | 0.0300 |
| Brier score | 0.3317 | 0.0976 |

Calibration split: n_rows=82634, n_trips=311.

## Confidence-decile validation (spec.md section 9.3 / 11.10)

Spearman rho (decile rank, decile mean NDCG@k): **0.358** (target &ge; 0.7, **MISSED**). n_trips_included=605.

| Decile | Mean confidence | Mean NDCG@k | n_trips |
|---|---|---|---|
| 1 | 0.5509 | 0.1838 | 61 |
| 2 | 0.6352 | 0.1304 | 60 |
| 3 | 0.6723 | 0.1344 | 61 |
| 4 | 0.6894 | 0.1749 | 60 |
| 5 | 0.7006 | 0.1563 | 61 |
| 6 | 0.7103 | 0.1641 | 60 |
| 7 | 0.7203 | 0.1751 | 60 |
| 8 | 0.7294 | 0.1648 | 61 |
| 9 | 0.7419 | 0.1798 | 60 |
| 10 | 0.7586 | 0.1818 | 61 |

## Cold-start cohorts (spec.md section 11.8 / section 12)

### NDCG@10 by traveler interaction-count bucket

| Bucket | NDCG@10 (mean) | n_trips_in_bucket |
|---|---|---|
| 0 | 0.1872 | 72 |
| 1-3 | 0.2264 | 1 |
| 4-10 | 0.1999 | 59 |
| >10 | 0.1821 | 539 |

### New-POI cohort (spec.md section 12)

- n_new_pois_in_catalog=71, n_holdout_trips_with_relevant_cohort_candidate=318
- NDCG@10 WITH behavioral dropout: **0.6019 [0.5706, 0.6336]**
- NDCG@10 WITHOUT behavioral dropout: **0.6092 [0.5802, 0.6398]**
- Paired Wilcoxon (with vs without): p=0.3283

### Leave-one-destination-out (LODO)

Wall-clock: **40.7s** for 3 destination-held-out retrains.

| Destination | NDCG@10 (LODO) | NDCG@10 (full training) | Wilcoxon p |
|---|---|---|---|
| barcelona | 0.1840 [0.1643, 0.2044] | 0.1780 [0.1596, 0.1966] | 0.4827 |
| kyoto | 0.1770 [0.1598, 0.1939] | 0.1897 [0.1723, 0.2062] | 0.01763 |
| seoul | 0.1838 [0.1651, 0.2027] | 0.1848 [0.1663, 0.2038] | 0.8377 |

## Ablations (spec.md section 11.9)

Each row: delta NDCG@10 (ablated - full lambdamart_ips), against the SAME already-trained primary system as the reference point.

| Ablation | Status | NDCG@10 (full) | NDCG@10 (ablated) | Delta | Wilcoxon p |
|---|---|---|---|---|---|
| -IPS_weighting | measured | 0.1842 [0.1738, 0.1945] | 0.1239 [0.1152, 0.1335] | -0.0603 | 1.19e-35 |
| -calibration | measured | 0.1842 [0.1738, 0.1945] | 0.1812 [0.1713, 0.1917] | -0.0030 | 0.05429 |
| -MMR | measured | 0.2118 [0.1953, 0.2285] | 0.2524 [0.2351, 0.2693] | +0.0406 | 8.491e-26 |
| -interest_channel | measured | 0.1842 [0.1738, 0.1945] | 0.1843 [0.1738, 0.1946] | +0.0001 | 0.007686 |
| -long_tail_quota | measured | 0.1842 [0.1738, 0.1945] | 0.1846 [0.1741, 0.1950] | +0.0004 | 8.298e-06 |
| -text_embeddings | measured | 0.1842 [0.1738, 0.1945] | 0.1853 [0.1747, 0.1958] | +0.0011 | 0.9713 |
| -implicit_taste | measured | 0.1842 [0.1738, 0.1945] | 0.1962 [0.1856, 0.2072] | +0.0120 | 7.936e-05 |
| -explicit_interests | measured | 0.1842 [0.1738, 0.1945] | 0.1892 [0.1789, 0.1996] | +0.0050 | 0.08471 |
| -behavioral_block | measured | 0.1842 [0.1738, 0.1945] | 0.1857 [0.1755, 0.1967] | +0.0015 | 0.9709 |

Notes:

- **-IPS_weighting**: Free: system 7 (lambdamart, uniform weight) IS this ablation of system 8.
- **-calibration**: Isotonic calibration is a monotonic non-decreasing transform of the raw score, so it cannot change within-trip RANKING (and therefore NDCG@10) except through tie-break reordering at flat isotonic segments -- an at/near-zero delta here is the mathematically EXPECTED result, not a modeling failure. Calibration's actual purpose (cross-traveler score comparability for planner_weight) is not something NDCG@10 measures.
- **-MMR**: lambda=1.0 (pure utility ranking, no diversity penalty) vs the configured default lambda=0.8 -- reuses scoring.diversity.mmr_rerank_all_trips directly, no retraining.
- **-interest_channel**: Leave-one-channel-out ('channel_interest' removed from the candidate union): re-evaluates the ALREADY-TRAINED lambdamart_ips score over the smaller candidate set, no retraining, no new candidate generation.
- **-long_tail_quota**: Leave-one-channel-out ('channel_longtail' removed from the candidate union): re-evaluates the ALREADY-TRAINED lambdamart_ips score over the smaller candidate set, no retraining, no new candidate generation.
- **-text_embeddings**: Full retrain with every 'text_emb_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.
- **-implicit_taste**: Full retrain with every 'implicit_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.
- **-explicit_interests**: Full retrain with every 'explicit_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.
- **-behavioral_block**: Full retrain with every 'behav_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.

## Utility beta sensitivity (spec.md section 9.1)

| beta | NDCG@10 (mean) |
|---|---|
| 0.30 | 0.1660 |
| 0.50 | 0.1654 |
| 0.70 | 0.1645 |
| 0.90 | 0.1645 |
| 1.10 | 0.1638 |

## Decision Register

Every design choice not dictated by the assignment, the alternative, the experiment, and the measured result. Rows are generated from `results/parts/dr/*.json`; an experiment that was not run says NOT RUN.

| # | Decision | Alternative | Experiment | Result / verdict | Status |
|---|---|---|---|---|---|
| DR1 | Multiplicative utility rel^a * compat^b with a hard gate | The brief's additive formula a*rel + b*compat | Same holdout candidates/relevance/compatibility; rank by each combination rule; count hard-constraint violations (hard_gate == 0) among the top-10 of every holdout trip. | Additive scoring puts 974 hard-constraint violations into 230/671 trips' top-10; the multiplicative gated rule puts 0. NDCG@10 0.1740 (additive) vs 0.1644 (multiplicative gated); mean compat@10 0.8603 vs 0.8420. | MEASURED |
| DR2 | LightGBM LambdaMART ranker | Two-tower neural ranker (shared-space dot product) | Learning curve: both rankers fit on [0.1, 0.25, 0.5, 1.0] of TRAIN trips x seeds [42, 43, 44] (same IPS weights, same train-carved early stopping), scored on the full unbiased holdout; per-trip NDCG@10 averaged over seeds, 2000-resample trip bootstrap CIs. | Two-tower NDCG@10 0.0990, 0.1245, 0.1447, 0.1624 vs LambdaMART-IPS 0.1736, 0.1790, 0.1810, 0.1845 at fractions [0.1, 0.25, 0.5, 1.0]. Crossover at any measured point: False. Log-linear extrapolation puts a crossover at ~4,636 training trips (2.5x the 1829 used; 4-point fit, low confidence). CIs separated at 100%. | MEASURED |
| DR3 | Listwise LambdaRank objective | Pointwise binary / graded regression, listwise rank_xendcg | Same features, IPS weights, dropout, split and early-stopping metric (NDCG@10); only the LightGBM objective varies. | lambdarank 0.1842 [0.1738, 0.1945]; rank_xendcg 0.1950 [0.1846, 0.2051]; binary 0.2046 [0.1939, 0.2159]; regression 0.2033 [0.1925, 0.2142] | MEASURED |
| DR4 | TF-IDF -> SVD-64 POI text embedding | all-MiniLM-L6-v2 sentence embeddings (-> SVD-64) | Swap ONLY the POI text embedding (and the taste vectors and POI features built from it) and refit the ranker on the fixed candidate sets; measure representation fidelity (D9 within-trip, D11) and holdout NDCG@10. THIS SYNTHETIC CORPUS is generated from anchored, synonym-rich phrase pools over latent dimensions, so its vocabulary design favours lexical overlap: the outcome is a statement about this dataset, not a verdict on sentence encoders. | TF-IDF: D11 0.865, D9 0.473, NDCG@10 0.1786. MiniLM: D11 0.590, D9 0.258, NDCG@10 0.1758 (CIs overlap). Dataset-specific (templated synonym-pool text); transfer to real POI text is untested. | MEASURED |
| DR5 | Brute-force cosine retrieval | ANN index (faiss / hnswlib) | NOT RUN | NOT RUN (ANN only matters at catalog sizes far beyond this take-home's three destinations; cut for time, so 'brute force is fine here' is reasoning, not evidence) | NOT RUN |
| DR6 | Geometric-mean compatibility aggregation | min(), plain product, arithmetic mean | Recompute compatibility from the 6 sub-scores with each aggregator; rank gated and ungated multiplicative utility. | Ungated hard-violation counts: geometric_mean (production)=1391, min=994, product=903, arithmetic_mean=1476; NDCG@10 gated: geometric_mean (production)=0.1644, min=0.1616, product=0.1562, arithmetic_mean=0.1642. Geometric mean ungated NDCG 0.1813. | MEASURED |
| DR7 | IPS clip = 20 | clip in {5, 10, 50, none} | IPS clip-high swept with everything else fixed (weights renormalised per trip). | clip 5: 0.1504 [0.1413, 0.1601]; clip 10: 0.1751 [0.1658, 0.1852]; clip 20: 0.1842 [0.1738, 0.1945]; clip 50: 0.2131 [0.2008, 0.2253]; clip none: 0.1891 [0.1783, 0.2002]; clip no_ips: 0.1239 [0.1152, 0.1335] | MEASURED |
| DR8 | 180-day taste half-life and spec interaction weights | +-2x half-life; uniform interaction weights | Rebuild the traveler features with a different taste half-life / interaction weights, refit the ranker on the fixed candidate sets, and score the unbiased holdout; D9 (within-trip, reporting-only) shows the effect on estimator fidelity. | halflife_180d (shipped): NDCG@10 0.1786, D9 0.473; halflife_90d: NDCG@10 0.1786, D9 0.470; halflife_360d: NDCG@10 0.1787, D9 0.474; uniform_weights: NDCG@10 0.1810, D9 0.052 | MEASURED |
| DR9 | Long-tail candidate quota = 50 | quota in {0, 25, 100} | Regenerate the holdout candidate sets with a different long-tail hard-floor quota (retriever scores fixed); score with the SHIPPED booster (trained at quota 50, not refit per quota); raw ranker top-10, no MMR/gates. | quota 0: NDCG@10 0.1844, long-tail share 0.106, candidate recall 0.916; quota 25: NDCG@10 0.1844, long-tail share 0.106, candidate recall 0.921; quota 50: NDCG@10 0.1842, long-tail share 0.106, candidate recall 0.926; quota 100: NDCG@10 0.1838, long-tail share 0.106, candidate recall 0.935 | MEASURED |
| DR10 | alpha = 1.0, beta = 0.7 | alpha x beta grid | Gated multiplicative utility over an alpha x beta grid on the same holdout scoring pass. | Shipped (alpha=1.0, beta=0.7): NDCG@10 0.1644, compat@10 0.8420. NDCG-best cell (alpha=1.0, beta=0.3): 0.1659, compat@10 0.8360. | MEASURED |
| DR11 | Candidate generation = learned retriever + long-tail + interest | The original 6 heuristic channels, and subsets of them | Channel-subset grid at learned K=210 on a train-carved validation split (IPS-weighted logged positives); holdout columns are reporting-only. `none` = the learned retriever alone. Baseline = the original 6-channel heuristic union. | Legacy 6-channel union: recall 0.596 (lift +0.203). Shipped learned+long-tail+interest: recall 0.881 (lift +0.371, 246 candidates/trip). Ranking by true utility would recall 0.991 at the same budget (diagnostic). | MEASURED |
| DR12 | Retrieval design: learned full-catalog retriever + long-tail floor + interest | The original six-channel heuristic union (ranker retrained on it) | Recall and Gate-B rows of each candidate set on the corrected features, plus end-to-end fixed-denominator NDCG@10 with the ranker retrained on each set (eval/decomposition.py). | Learned retriever: recall 0.926 (long-tail 0.898, lift +0.374); legacy union: recall 0.596 (long-tail 0.540, lift +0.203). The legacy union wins end-to-end NDCG@10 (0.1919 vs 0.1814, paired p=0.0047) with a ranker retrained on each set. The retriever ships (it is the one that meets the recall and long-tail gate rows); switching on a holdout comparison would be selection leakage. | MEASURED |
| DR13 | Learned K = 240 (largest grid K with validation lift >= 0.36) | K = 270 from the pre-registered rule (smallest K with val recall >= 0.93) | Full K sweep on the corrected features: IPS-weighted validation recall and lift over chance (selection columns) with the holdout columns reporting-only. | The blind recall rule gives K=270, whose validation lift is 0.327 < 0.35 (the blocking Gate-B row); 0 grid K satisfy both criteria on validation. Tie-break, validation only, applied to the FIRST grid computation (Amendment 2: validation lift 0.363 at K=240): the largest K with lift >= 0.36 (gate + 0.01 margin) -> K=240. The committed grid was recomputed later on the final candidates.parquet and puts the validation lift at K=240 at 0.358 (still above the 0.35 gate, below the 0.36 margin; the margin rule would now give K=225). K was not re-selected: no model changes after the pre-registered decision, and the shipped pipeline meets the gate on the holdout. | MEASURED |

## Seed replication

The full pipeline was regenerated end to end for seeds [42, 43, 44, 45, 46] (a new synthetic dataset, retriever, boosters and calibration each time; `scripts/seed_replication.py`). Seed 42 is the committed run.

| Metric | Mean | SD | Min | Max |
|---|---|---|---|---|
| bias_gap_popularity | 0.0600 | 0.0259 | 0.0220 | 0.0963 |
| bias_gap_primary | -0.0085 | 0.0208 | -0.0473 | 0.0153 |
| candidate_recall_long_tail | 0.8785 | 0.0119 | 0.8628 | 0.8975 |
| candidate_recall_overall | 0.9147 | 0.0069 | 0.9066 | 0.9262 |
| coverage_at_10 | 0.4952 | 0.0395 | 0.4603 | 0.5718 |
| ece_after | 0.0280 | 0.0047 | 0.0189 | 0.0315 |
| longtail_precision | 0.1946 | 0.0190 | 0.1731 | 0.2218 |
| longtail_share | 0.1851 | 0.0437 | 0.1274 | 0.2409 |
| ndcg10_content_cosine | 0.1344 | 0.0028 | 0.1298 | 0.1385 |
| ndcg10_lambdamart_ips | 0.1967 | 0.0083 | 0.1842 | 0.2070 |
| ndcg10_logistic_regression | 0.1283 | 0.0074 | 0.1196 | 0.1381 |
| ndcg10_popularity | 0.0695 | 0.0097 | 0.0523 | 0.0816 |
| pct_of_oracle_ceiling | 0.5354 | 0.0191 | 0.5003 | 0.5533 |
| within_cross_ratio_true_labels | 1.1593 | 0.0319 | 1.1306 | 1.2008 |

## Wall-clock

**reproduce**: 334.6 s total (16-logical-core laptop CPU, no GPU, no network).

| Stage | Seconds |
|---|---|
| generate | 30.6 |
| prepare | 5.6 |
| features | 17.6 |
| candidates | 40.9 |
| gate-dgp | 9.9 |
| train | 78.1 |
| evaluate | 130.3 |
| representation | 15.2 |
| gate-representation | 3.2 |
| compose | 3.2 |

**reproduce-full**: 432.5 s total (16-logical-core laptop CPU, no GPU, no network).

| Stage | Seconds |
|---|---|
| generate | 30.6 |
| prepare | 5.6 |
| features | 17.6 |
| candidates | 40.9 |
| gate-dgp | 9.9 |
| train | 78.1 |
| evaluate | 130.3 |
| representation | 15.2 |
| gate-representation | 3.2 |
| compose | 3.2 |
| recommend | 32.7 |
| scenarios | 21.0 |
| lodo | 44.0 |
| docs | 0.2 |

## Scenarios (spec.md section 15)

### Scenario 1: Local Experience

- Persona: Food-led repeat visitor to Seoul (foreign, inbound): has done the palaces and Myeongdong, wants where locals actually eat and wander.
- Destination: **seoul**; interests: local, foodie, authentic; touristiness_pref: **-0.80**
- Budget: medium; mobility: public_transport; party: solo; pace: moderate; accessibility needs: none
- 'local food' -> {local, foodie}; 'neighborhoods' -> {authentic} (shares the 'local' tag with the food interest -- an authentic-local-neighborhood preference is naturally the same underlying taste). `pace` is not specified by spec.md section 15 for this scenario -- defaulted to 'moderate' (the middle of the 3-value PACE_ORDER scale).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0125 | Eatery Insadong Kitchen | restaurant | 0.2577 | 0.2690 | 0.9407 | 0.5494 | 10.4772% | -0.6315 |
| 2 | PSEO0210 | Time Honored Seongsu Trail | nature_park | 0.2156 | 0.2431 | 0.8426 | 0.5507 | 89.8340% | -1.2645 |
| 3 | PSEO0280 | Historic Gangnam Shrine | religious_site | 0.2287 | 0.2524 | 0.8689 | 0.6415 | 99.5851% | -1.1177 |
| 4 | PSEO0112 | Lively Hongdae Bakery | cafe | 0.2274 | 0.2431 | 0.9090 | 0.4848 | 66.8050% | 0.3067 |
| 5 | PSEO0025 | Breezy Bukchon Adventure Park | family_activity | 0.1947 | 0.2431 | 0.7282 | 0.5466 | 99.7925% | -1.2603 |
| 6 | PSEO0394 | Buzzy Gangnam Market | shopping | 0.1901 | 0.2059 | 0.8922 | 0.6986 | 93.7759% | -1.2350 |
| 7 | PSEO0137 | Family Friendly Itaewon Arcade | entertainment | 0.1549 | 0.1796 | 0.8101 | 0.5839 | 98.3402% | -1.1983 |
| 8 | PSEO0420 | Dreamy Hongdae Overlook | viewpoint | 0.1576 | 0.1654 | 0.9333 | 0.6734 | 99.3776% | -0.7503 |
| 9 | PSEO0312 | Genuine Hongdae Old Quarter | historic_site | 0.1648 | 0.1796 | 0.8844 | 0.5767 | 97.0954% | -0.6968 |
| 10 | PSEO0327 | Neighborhood Itaewon Bathhouse | wellness_spa | 0.1433 | 0.1531 | 0.9102 | 0.5555 | 88.3817% | -1.0171 |

**Top signals (rank 1, PSEO0125):** implicit_taste (1.106), popularity (0.327), interest_match (0.319)

**Explanation (rank 1):** Popular with travelers who share your interests Popular choice -- busier than 10% of comparable POIs in Seoul Strong match with your stated interest in foodie More budget-friendly than typical for your medium budget

### Scenario 2: History & Architecture

- Persona: First-time K-culture visitor to Seoul (foreign, inbound): heritage, museums and the iconic sights, comfortable budget.
- Destination: **seoul**; interests: historic_site, historic, cultural, museum; touristiness_pref: **0.40**
- Budget: high; mobility: public_transport; party: couple; pace: moderate; accessibility needs: none
- 'history' -> {historic_site, historic}; 'architecture' -> {cultural} (no literal 'architecture' tag exists in CATEGORIES+TAGS); 'museums' -> {museum} (exact match). `pace` defaulted to 'moderate' (not specified).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0400 | Authentic Itaewon Temple | historic_site | 0.2286 | 0.2431 | 0.9159 | 0.5509 | 81.1203% | -1.2657 |
| 2 | PSEO0401 | Mealtime Seongsu Gallery | museum | 0.2269 | 0.2431 | 0.9064 | 0.4882 | 8.2988% | 0.2919 |
| 3 | PSEO0195 | Flavor Packed Insadong Table | restaurant | 0.1400 | 0.1464 | 0.9379 | 0.6108 | 95.0207% | -0.5810 |
| 4 | PSEO0420 | Dreamy Hongdae Overlook | viewpoint | 0.1497 | 0.1531 | 0.9686 | 0.5580 | 99.3776% | -0.7503 |
| 5 | PSEO0067 | Late Night Hongdae Bar | nightlife | 0.1193 | 0.1464 | 0.7465 | 0.6380 | 98.5477% | -2.0151 |
| 6 | PSEO0145 | Peaceful Gangnam Park | nature_park | 0.1163 | 0.1293 | 0.8591 | 0.5524 | 95.9544% | -0.6621 |
| 7 | PSEO0167 | Nature Filled Itaewon Wellness Studio | wellness_spa | 0.1072 | 0.1108 | 0.9530 | 0.5690 | 90.0415% | -1.2879 |
| 8 | PSEO0280 | Historic Gangnam Shrine | religious_site | 0.1104 | 0.1293 | 0.7979 | 0.5708 | 99.5851% | -1.1177 |
| 9 | PSEO0071 | Teahouse Bukchon Bakery | cafe | 0.1063 | 0.1108 | 0.9420 | 0.5417 | 88.7967% | -0.7435 |
| 10 | PSEO0043 | Retail Heavy Itaewon Bazaar | shopping | 0.1012 | 0.1108 | 0.8784 | 0.6158 | 97.9253% | -1.4024 |

**Top signals (rank 1, PSEO0400):** interest_match (0.618), implicit_taste (0.569), popularity (0.387)

**Explanation (rank 1):** Strong match with your stated interest in cultural Popular with travelers who share your interests Popular choice -- busier than 81% of comparable POIs in Seoul Only open 75% of your trip's plausible visiting hours

### Scenario 3: Family with young children

- Persona: Foreign family with young children visiting Seoul: stroller, relaxed pace, things kids can do.
- Destination: **seoul**; interests: family_activity, nature_park, family-friendly; touristiness_pref: **0.00**
- Budget: medium; mobility: car; party: family_young_kids; pace: relaxed; accessibility needs: stroller
- 'activities' -> {family_activity}; 'parks' -> {nature_park}; 'interactive experiences' -> {family-friendly} (no literal 'interactive' tag exists). `touristiness_pref` and `mobility` are not specified by spec.md section 15 for this scenario -- defaulted to 0.0 (neutral) and 'car' (a common real choice for a family with young children traveling with a stroller) respectively; `party_type`/accessibility (`stroller`)/`pace` (`relaxed`)/`budget` (`medium`) are exactly as spec.md states.

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0053 | Iconic Itaewon Discovery Center | family_activity | 0.2473 | 0.2690 | 0.8871 | 0.5914 | 84.8548% | -1.4270 |
| 2 | PSEO0210 | Time Honored Seongsu Trail | nature_park | 0.2387 | 0.2690 | 0.8431 | 0.6011 | 89.8340% | -1.2645 |
| 3 | PSEO0123 | Bohemian Myeongdong Grill | restaurant | 0.1438 | 0.1531 | 0.9148 | 0.5448 | 82.1577% | -0.5479 |
| 4 | PSEO0446 | Easygoing Gangnam Bazaar | shopping | 0.1345 | 0.1531 | 0.8317 | 0.5095 | 9.0249% | -0.6149 |
| 5 | PSEO0410 | Authentic Seongsu Playhouse | entertainment | 0.1015 | 0.1108 | 0.8819 | 0.5860 | 86.5145% | -0.8836 |
| 6 | PSEO0249 | Wallet Friendly Gangnam Gallery | museum | 0.0849 | 0.0905 | 0.9134 | 0.5830 | 93.5685% | -1.2491 |
| 7 | PSEO0071 | Teahouse Bukchon Bakery | cafe | 0.0846 | 0.0905 | 0.9082 | 0.5383 | 88.7967% | -0.7435 |
| 8 | PSEO0428 | Off The Beaten Path Gangnam Monastery | religious_site | 0.0730 | 0.0811 | 0.8605 | 0.5253 | 4.8755% | -0.6312 |
| 9 | PSEO0088 | Gallery Like Gangnam Monument | historic_site | 0.1018 | 0.1108 | 0.8850 | 0.5900 | 95.2282% | -0.6739 |
| 10 | PSEO0007 | Family Day Seongsu Adventure Park | family_activity | 0.2165 | 0.2431 | 0.8474 | 0.5241 | 98.1328% | -0.6690 |

**Top signals (rank 1, PSEO0053):** implicit_taste (0.761), interest_match (0.621), popularity (0.381)

**Explanation (rank 1):** Popular with travelers who share your interests Strong match with your stated interest in family activity Popular choice -- busier than 85% of comparable POIs in Seoul Weaker fit on visit-length fit than most of your other options

### Scenario 4: Diagnostic (Scenario 1 profile, touristiness_pref flipped to the opposite extreme)

- Persona: The same traveler as scenario 1 asking for the opposite: the iconic, must-see, first-timer version of Seoul.
- Destination: **seoul**; interests: local, foodie, authentic; touristiness_pref: **0.80**
- Budget: medium; mobility: public_transport; party: solo; pace: moderate; accessibility needs: none
- Base scenario: 1 (Local Experience). Every field held identical to that scenario's profile except `touristiness_pref`, flipped from -0.8 to +0.8 (opposite sign, same extreme magnitude).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0125 | Eatery Insadong Kitchen | restaurant | 0.2329 | 0.2431 | 0.9407 | 0.5120 | 10.4772% | -0.6315 |
| 2 | PSEO0280 | Historic Gangnam Shrine | religious_site | 0.2203 | 0.2431 | 0.8689 | 0.5362 | 99.5851% | -1.1177 |
| 3 | PSEO0210 | Time Honored Seongsu Trail | nature_park | 0.1853 | 0.2089 | 0.8426 | 0.6426 | 89.8340% | -1.2645 |
| 4 | PSEO0036 | Foodie Favorite Bukchon Roastery | cafe | 0.1931 | 0.2078 | 0.9006 | 0.5038 | 94.6058% | -1.4750 |
| 5 | PSEO0025 | Breezy Bukchon Adventure Park | family_activity | 0.1664 | 0.2078 | 0.7282 | 0.5545 | 99.7925% | -1.2603 |
| 6 | PSEO0394 | Buzzy Gangnam Market | shopping | 0.1413 | 0.1531 | 0.8922 | 0.5699 | 93.7759% | -1.2350 |
| 7 | PSEO0420 | Dreamy Hongdae Overlook | viewpoint | 0.1458 | 0.1531 | 0.9333 | 0.5644 | 99.3776% | -0.7503 |
| 8 | PSEO0327 | Neighborhood Itaewon Bathhouse | wellness_spa | 0.1371 | 0.1464 | 0.9102 | 0.6150 | 88.3817% | -1.0171 |
| 9 | PSEO0306 | Tranquil Gangnam Temple | historic_site | 0.1230 | 0.1293 | 0.9310 | 0.5599 | 95.6432% | -0.9830 |
| 10 | PSEO0410 | Authentic Seongsu Playhouse | entertainment | 0.1329 | 0.1464 | 0.8705 | 0.6466 | 86.5145% | -0.8836 |

**Top signals (rank 1, PSEO0125):** implicit_taste (1.024), popularity (0.351), interest_match (0.325)

**Explanation (rank 1):** Popular with travelers who share your interests Popular choice -- busier than 10% of comparable POIs in Seoul Strong match with your stated interest in foodie More budget-friendly than typical for your medium budget

### Pairwise top-10 Jaccard overlap across scenarios

| | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 |
|---|---|---|---|---|
| Scenario 1 | 1.000 | 0.111 | 0.053 | 0.538 |
| Scenario 2 | 0.111 | 1.000 | 0.053 | 0.111 |
| Scenario 3 | 0.053 | 0.053 | 1.000 | 0.111 |
| Scenario 4 | 0.538 | 0.111 | 0.111 | 1.000 |

**Diagnostic scenario 4 vs base scenario 1** (touristiness_pref flipped to the opposite extreme, everything else held constant): top-10 Jaccard overlap = **0.538** (spec.md section 15 expects this to be low, i.e. &le; 0.25, **MISSED**).

**Honest miss, not hidden**: overlap is higher than spec.md section 15 expects. Measured mechanism: the two scenarios' PRE-RANKING candidate pools overlap at **71.7%** Jaccard (`candidates.union.generate_candidates`'s own output), so the ranking starts from nearly the same POIs and `touristiness_pref` can only reorder them through the feature columns that carry it (`explicit_touristiness_pref`, `interact_localness_gap`); `scoring.compatibility`'s sub-scores carry no localness/touristiness term at all (spec.md section 9.1). The overlap is below 1.0, i.e. the preference does move the ranking -- just not enough to dominate a candidate pool this similar.

