# RESULTS.md

**Generated automatically by `poi_rank.eval.report` from `results/metrics.json`. Do not hand-edit -- every number here traces directly to that JSON file (spec.md section 0: "No unverified metric may appear in any document").**

Synthetic data throughout (three destinations; Seoul is the lead). Personas are inbound foreign travelers. Read the numbers as properties of this simulator, not as claims about real traffic -- `docs/TECHNICAL.md` states what the simulator does and does not license.

## Headline: local and long-tail discovery (Seoul-first, inbound travelers)

konnect.kr serves foreign travelers in Korea, and the incumbents already rank by popularity -- so a popularity-shaped list is table stakes. The claim that matters is surfacing the genuinely relevant, non-obvious POI: **long-tail share and long-tail precision together** (coverage without precision is just noise injection), measured on the unbiased random-exposure holdout.

| Top-10 lists, primary holdout | LambdaMART + IPS (primary) | Popularity ranker |
|---|---|---|
| Long-tail share (bottom-50% popularity stratum) | **0.2252** | 0.0000 |
| Long-tail precision (relevant / recommended) | **0.1541** | N/A |
| Catalog coverage@10 | **66.4%** | 5.7% |
| Gini of recommendation exposure (lower = less monoculture) | **0.7016** | 0.9735 |
| NDCG@10 (unbiased holdout, 95% CI) | **0.1485** [0.1385, 0.1590] | 0.0629 [0.0567, 0.0698] |

Popularity is flattered by exposure-biased logs; the bias-gap table below quantifies exactly how much (popularity +0.0833, primary +0.0085 NDCG@10 between the biased and unbiased holdouts).

## Bias-gap table (spec.md section 11.1)

Each system's NDCG@10 on the SECONDARY biased holdout (`interactions_holdout_logged.parquet`) vs the PRIMARY unbiased holdout. Expectation: the popularity baseline shows a large positive gap (flattered by biased logs); the IPS-corrected model shows a small gap.

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0661 | 0.0306 | -0.0355 |
| 2. Popularity | 0.0629 | 0.1462 | +0.0833 |
| 3. Popularity + geo filter | 0.0623 | 0.1444 | +0.0821 |
| 4. Content cosine | 0.1411 | 0.0544 | -0.0867 |
| 5. Item-kNN CF | 0.0710 | 0.1494 | +0.0785 |
| 6. Logistic regression | 0.1130 | 0.2307 | +0.1177 |
| 7. LambdaMART | 0.1244 | 0.1917 | +0.0673 |
| 8. LambdaMART + IPS (primary) | 0.1485 | 0.1570 | +0.0085 |
| 9. Oracle (ceiling) | 0.3749 | 0.1125 | -0.2623 |

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
| candidate_recall_lift_long_tail | >= 0.35 | 0.3925 | PASS |
| candidate_recall_lift_overall | >= 0.35 | 0.3738 | PASS |
| candidate_recall_long_tail | >= 0.75 | 0.8262 | PASS |
| candidate_recall_overall | >= 0.85 | 0.8573 | PASS |

| Reporting-only (after freeze) | Value |
|---|---|
| d11_cca_first_corr_text_only | 0.9790 |
| d11_ridge_r2_text_only | 0.8654 |
| d9_within_trip_shipped_chain | 0.4731 |
| d9_within_trip_taste_estimator_alone | 0.5718 |
| d9_within_trip_text_alone | 0.9272 |
| oracle_ceiling_ndcg10_candidate_level | 0.3749 |
| oracle_relevance_recall_long_tail | 0.9107 |
| oracle_relevance_recall_overall | 0.9357 |

## Success criteria scorecard

Stated up front, then measured. Every MISSED row carries a diagnosis below -- a miss is reported as a ceiling only after a diagnostic has ruled out a mechanism, citing the number (spec.md: "An honest miss with a root-cause analysis scores better than a suspiciously perfect table.").

| Metric | Target | Measured | Status |
|---|---|---|---|
| NDCG@10 vs popularity | &ge; +40% relative, Wilcoxon p < 0.01 | +136.2% relative, p=4.336e-41 | **MET** |
| % of oracle ceiling (candidate-level NDCG@10) | &ge; 70% | 39.6% | **MISSED** |
| Candidate recall, overall (exposed positives) | &ge; 0.85 | 0.8573 | **MET** |
| Candidate recall, long-tail stratum | &ge; 0.75 | 0.8262 | **MET** |
| Candidate recall lift over chance (overall) | &ge; +0.35 | +0.374 | **MET** |
| Cross-archetype Jaccard@10 (true labels) | &le; 0.25 | 0.0385 | **MET** |
| Within/cross Jaccard ratio (true labels) | &ge; 2.0 | 1.32 | **MISSED** |
| Hard-constraint violations in top-10 | = 0 | 0 | **MET** |
| ECE after calibration | &le; 0.05 | 0.0466 | **MET** |
| Confidence-decile NDCG rank correlation (Spearman; need not be strictly monotone) | &ge; 0.6 (spec-v2; spec.md section 11.10 said 0.7) | 0.879 | **MET** |
| Long-tail share of top-10 | &ge; 0.25 | 0.2252 | **MISSED** |
| Long-tail precision of top-10 | &ge; 0.40 | 0.1541 | **MISSED** |
| Localness index Spearman vs latent localness | &ge; 0.6 | 0.5815 | **MISSED** |
| Scenario-4 (touristiness flip) top-10 overlap | &le; 0.35 | 0.176 | **MET** |

### Diagnoses of the missed rows

- **% of oracle ceiling (candidate-level NDCG@10)**: The oracle ranks the SAME candidates by the DGP's true utility; the model only sees observable features. Text is not the limiting link: text-alone within-trip taste fidelity is 0.927 and D11 ridge R2 is 0.865. The taste ESTIMATOR is: run over the TRUE semantic vectors it still reaches only 0.572 (shipped chain 0.473), from sparse, exposure-biased histories.
- **Within/cross Jaccard ratio (true labels)**: Within/cross-archetype list similarity ratio for lists ranked by the TRUE utility (a perfect ranker, same-destination pairs): 1.86 (within 0.0643, cross 0.0345); so the target is NOT attainable even by a perfect ranker in this simulator: the shortfall is a property of the simulator under this target, not of the model.
- **Long-tail share of top-10**: Long-tail share of the served top-10 is 0.2252. Decision Register DR9 varies the long-tail candidate quota and measures the raw ranker's top-10 share: quota 0 -> 0.156, quota 100 -> 0.168, quota 25 -> 0.161, quota 50 -> 0.161. The candidate quota is therefore not the lever; the share is set by the ranker's scores (and the MMR re-rank) over a candidate set that already contains long-tail POIs.
- **Long-tail precision of top-10**: Long-tail precision is 0.1541 over 1486 long-tail recommendations, with candidate recall 0.826 in that stratum, so retrieval is not the bottleneck. The raw ranker's top-10 long-tail precision (DR9, quota 50, before the compatibility gate, utility and MMR re-rank) is 0.2271 at share 0.1614, against 0.1541 at share 0.2252 in the served list: precision is lost AFTER ranking while share rises. Which of the three scoring-layer steps is responsible is not isolated (untested).
- **Localness index Spearman vs latent localness**: The composite index reaches rho 0.582; its observable inputs correlate with the latent localness at dist_to_tourist_centroid_km 0.699, foreign_review_ratio -0.555, local_tag_hits 0.050, pop_pct -0.187. The composite is BELOW its best single input (dist_to_tourist_centroid_km, |rho| 0.699): the blend weights were fixed earlier, when the geo input carried almost no signal (before the simulator's geo/localness fix). Re-weighting them against the latent localness would be tuning on the oracle (there is no oracle-free validation target for this index), so that retune is declined on principle: the index is left as shipped and the gap is reported.

## Primary ranking quality (unbiased random-exposure holdout, spec.md section 11.1)

n_holdout_trips = 671, bootstrap n_resamples = 2000 (resample unit: trip).

| System | ndcg@5 | ndcg@10 | ndcg@20 | precision@5 | precision@10 | recall@10 | recall@20 | map | mrr | % of oracle ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Random | 0.0566 [0.0498, 0.0637] | 0.0661 [0.0603, 0.0724] | 0.0824 [0.0770, 0.0882] | 0.1258 [0.1142, 0.1377] | 0.1237 [0.1145, 0.1329] | 0.0449 [0.0418, 0.0480] | 0.0862 [0.0819, 0.0907] | 0.1501 [0.1449, 0.1553] | 0.3123 [0.2884, 0.3369] | 17.6% |
| 2. Popularity | 0.0536 [0.0467, 0.0612] | 0.0629 [0.0567, 0.0698] | 0.0886 [0.0826, 0.0953] | 0.1165 [0.1046, 0.1288] | 0.1161 [0.1069, 0.1264] | 0.0439 [0.0404, 0.0476] | 0.0957 [0.0908, 0.1011] | 0.1641 [0.1584, 0.1701] | 0.2940 [0.2707, 0.3186] | 16.8% |
| 3. Popularity + geo filter | 0.0525 [0.0456, 0.0596] | 0.0623 [0.0562, 0.0688] | 0.0842 [0.0783, 0.0904] | 0.1180 [0.1061, 0.1300] | 0.1201 [0.1106, 0.1303] | 0.0446 [0.0413, 0.0482] | 0.0927 [0.0876, 0.0976] | 0.1595 [0.1540, 0.1651] | 0.3040 [0.2794, 0.3309] | 16.6% |
| 4. Content cosine | 0.1303 [0.1190, 0.1414] | 0.1411 [0.1313, 0.1511] | 0.1720 [0.1629, 0.1812] | 0.2587 [0.2411, 0.2754] | 0.2314 [0.2167, 0.2455] | 0.0867 [0.0820, 0.0912] | 0.1635 [0.1567, 0.1707] | 0.2337 [0.2255, 0.2422] | 0.5097 [0.4818, 0.5398] | 37.6% |
| 5. Item-kNN CF | 0.0598 [0.0522, 0.0679] | 0.0710 [0.0642, 0.0779] | 0.0937 [0.0873, 0.1005] | 0.1332 [0.1207, 0.1469] | 0.1320 [0.1221, 0.1428] | 0.0504 [0.0467, 0.0543] | 0.1004 [0.0953, 0.1063] | 0.1676 [0.1618, 0.1737] | 0.3235 [0.2991, 0.3488] | 18.9% |
| 6. Logistic regression | 0.1016 [0.0919, 0.1118] | 0.1130 [0.1045, 0.1226] | 0.1368 [0.1289, 0.1458] | 0.2221 [0.2042, 0.2405] | 0.1996 [0.1864, 0.2131] | 0.0749 [0.0704, 0.0795] | 0.1382 [0.1328, 0.1448] | 0.2106 [0.2030, 0.2183] | 0.4534 [0.4254, 0.4832] | 30.1% |
| 7. LambdaMART | 0.1087 [0.0981, 0.1196] | 0.1244 [0.1152, 0.1339] | 0.1510 [0.1426, 0.1600] | 0.2346 [0.2155, 0.2542] | 0.2168 [0.2030, 0.2317] | 0.0811 [0.0766, 0.0859] | 0.1506 [0.1441, 0.1571] | 0.2145 [0.2064, 0.2226] | 0.4650 [0.4362, 0.4942] | 33.2% |
| 8. LambdaMART + IPS (primary) | 0.1290 [0.1174, 0.1409] | 0.1485 [0.1385, 0.1590] | 0.1802 [0.1703, 0.1903] | 0.2641 [0.2438, 0.2841] | 0.2520 [0.2356, 0.2690] | 0.0921 [0.0872, 0.0972] | 0.1730 [0.1657, 0.1801] | 0.2402 [0.2314, 0.2497] | 0.5000 [0.4717, 0.5300] | 39.6% |
| 9. Oracle (ceiling) | 0.3426 [0.3263, 0.3606] | 0.3749 [0.3610, 0.3895] | 0.4333 [0.4217, 0.4452] | 0.5025 [0.4775, 0.5276] | 0.4878 [0.4662, 0.5091] | 0.1897 [0.1835, 0.1964] | 0.3623 [0.3542, 0.3712] | 0.5017 [0.4877, 0.5158] | 0.7317 [0.7049, 0.7589] | 100.0% |

### Paired Wilcoxon signed-rank tests (NDCG@10)

| Comparison | statistic | p-value | n_pairs |
|---|---|---|---|
| content_cosine_vs_popularity | 31942.0000 | 5.425e-36 | 605 |
| item_knn_cf_vs_popularity | 4550.0000 | 0.0004088 | 605 |
| lambdamart_ips_vs_content_cosine | 82860.5000 | 0.4445 | 605 |
| lambdamart_ips_vs_popularity | 28560.0000 | 4.336e-41 | 605 |
| lambdamart_vs_lambdamart_ips | 53402.5000 | 7.692e-11 | 605 |
| lambdamart_vs_popularity | 34244.5000 | 3.86e-31 | 605 |
| logistic_regression_vs_popularity | 37497.0000 | 4.302e-24 | 605 |
| oracle_vs_popularity | 1212.0000 | 1.686e-97 | 605 |

## Candidate recall by popularity stratum (with chance baselines)

Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its own chance baseline: the share of that stratum's destination POIs a same-size random candidate set would contain. Raw recall without this lift hid a near-chance failure for four phases.

| Stratum | Recall | Chance | Lift (abs) | Trips |
|---|---|---|---|---|
| long_tail | 0.826 | 0.434 | +0.392 | 605 |
| overall | 0.857 | 0.484 | +0.374 | 605 |
| q1_least_popular | 0.840 | 0.474 | +0.366 | 601 |
| q2 | 0.811 | 0.393 | +0.417 | 600 |
| q3 | 0.816 | 0.425 | +0.392 | 603 |
| q4_most_popular | 0.915 | 0.641 | +0.274 | 605 |

## Personalization (spec.md section 11.2, spec-v2 RC4)

Pairs are SAME-DESTINATION trip pairs only: a POI belongs to exactly one destination, so two trips to different destinations have Jaccard = RBO = 0 by construction, and pooling them measures the catalog partition rather than the recommender.

- Mean pairwise Jaccard@10 (same destination): **0.0394** (n_pairs=74775)
- Mean pairwise rank-biased overlap (RBO, p=0.9): **0.0531** (n_pairs=74775)
- For continuity with the pre-fix number, all pairs pooled (2/3 of them structurally zero): 0.0131 (n_pairs=224785)

| Grouping | Within Jaccard@10 | Cross Jaccard@10 | Ratio | within / cross pairs |
|---|---|---|---|---|
| **True archetype labels** (dominant archetype; within = same dominant and mixture cosine > 0.8) | 0.0508 | 0.0385 | **1.32** | 4814 / 65145 |
| Reference: lists ranked by the TRUE utility (a perfect ranker) | 0.0643 | 0.0345 | 1.86 | 4814 / 65145 |
| K-Means traveler-segment PROXY (clusters of the same features being evaluated -- kept only to show why it was misleading) | 0.0440 | 0.0386 | 1.14 | 11459 / 63316 |

## Coverage (spec.md section 11.3)

| | Primary system (lambdamart_ips) | Popularity baseline |
|---|---|---|
| Catalog coverage@10 | 66.4% | 5.7% |
| Gini coefficient | 0.7016 | 0.9735 |
| Entropy (bits) | 9.1430 | 5.5419 |
| POIs ever recommended | 960 / 1446 | 82 / 1446 |

Lower Gini / higher entropy / higher coverage = less popularity-monoculture concentration. Per-destination coverage:

| Destination | Primary | Popularity |
|---|---|---|
| barcelona | 64.4% | 6.0% |
| kyoto | 67.3% | 5.6% |
| seoul | 67.4% | 5.4% |

## Long-tail / local discovery (spec.md section 11.4)

- Share of top-10 recommendations in the bottom-50%-popularity stratum: **0.2252** (1486 / 6598)
- Long-tail precision (relevant per unbiased holdout): **0.1541** (229 / 1486)
- Popularity ranker, same measurement: share **0.0000** (0 / 6710), precision **N/A**

"Coverage without precision is just noise injection" -- both numbers reported together, per spec.md section 11.4.

## Constraint compatibility (spec.md section 11.5)

- % of top-10 with compatibility &ge; 0.7: **91.3%** (6027 / 6598)
- Hard-constraint violations in top-10: **0** (build-blocking, enforced independently by `tests/test_hard_constraints.py`)

## Diversity (spec.md section 11.6)

- Category entropy@10 (bits): **3.3674**
- Intra-list mean cosine distance (at the configured default lambda): **0.7566**

### MMR lambda sweep (NDCG@10 vs diversity trade-off)

| lambda | NDCG@10 (mean) | mean intra-list similarity |
|---|---|---|
| 0.50 | 0.1790 | 0.2069 |
| 0.60 | 0.1825 | 0.2135 |
| 0.70 | 0.1871 | 0.2237 |
| 0.80 | 0.1930 | 0.2434 |
| 0.90 | 0.2026 | 0.2870 |
| 1.00 | 0.2158 | 0.3791 |

## Calibration (spec.md section 11.7)

| | Before (naive) | After (isotonic) |
|---|---|---|
| ECE (15 bins) | 0.4444 | 0.0466 |
| Brier score | 0.3139 | 0.1047 |

Calibration split: n_rows=71391, n_trips=311.

## Confidence-decile validation (spec.md section 9.3 / 11.10)

Spearman rho (decile rank, decile mean NDCG@k): **0.879** (target &ge; 0.7, **MET**). n_trips_included=605.

| Decile | Mean confidence | Mean NDCG@k | n_trips |
|---|---|---|---|
| 1 | 0.4230 | 0.0594 | 61 |
| 2 | 0.5542 | 0.0902 | 60 |
| 3 | 0.5984 | 0.1406 | 61 |
| 4 | 0.6156 | 0.1432 | 60 |
| 5 | 0.6265 | 0.1431 | 61 |
| 6 | 0.6372 | 0.1458 | 60 |
| 7 | 0.6456 | 0.1640 | 60 |
| 8 | 0.6546 | 0.1585 | 61 |
| 9 | 0.6665 | 0.1614 | 60 |
| 10 | 0.6894 | 0.1509 | 61 |

## Cold-start cohorts (spec.md section 11.8 / section 12)

### NDCG@10 by traveler interaction-count bucket

| Bucket | NDCG@10 (mean) | n_trips_in_bucket |
|---|---|---|
| 0 | 0.0456 | 72 |
| 1-3 | 0.2242 | 1 |
| 4-10 | 0.1689 | 59 |
| >10 | 0.1599 | 539 |

### New-POI cohort (spec.md section 12)

- n_new_pois_in_catalog=71, n_holdout_trips_with_relevant_cohort_candidate=299
- NDCG@10 WITH behavioral dropout: **0.5553 [0.5250, 0.5864]**
- NDCG@10 WITHOUT behavioral dropout: **0.5565 [0.5295, 0.5871]**
- Paired Wilcoxon (with vs without): p=0.8101

### Leave-one-destination-out (LODO)

Wall-clock: **41.9s** for 3 destination-held-out retrains.

| Destination | NDCG@10 (LODO) | NDCG@10 (full training) | Wilcoxon p |
|---|---|---|---|
| barcelona | 0.1337 [0.1159, 0.1524] | 0.1374 [0.1194, 0.1569] | 0.7565 |
| kyoto | 0.1497 [0.1323, 0.1672] | 0.1553 [0.1358, 0.1746] | 0.4577 |
| seoul | 0.1537 [0.1352, 0.1731] | 0.1529 [0.1354, 0.1706] | 0.6976 |

## Ablations (spec.md section 11.9)

Each row: delta NDCG@10 (ablated - full lambdamart_ips), against the SAME already-trained primary system as the reference point.

| Ablation | Status | NDCG@10 (full) | NDCG@10 (ablated) | Delta | Wilcoxon p |
|---|---|---|---|---|---|
| -IPS_weighting | measured | 0.1485 [0.1385, 0.1590] | 0.1244 [0.1152, 0.1339] | -0.0241 | 7.692e-11 |
| -calibration | measured | 0.1485 [0.1385, 0.1590] | 0.1478 [0.1374, 0.1584] | -0.0007 | 0.7092 |
| -MMR | measured | 0.1930 [0.1764, 0.2102] | 0.2158 [0.1978, 0.2337] | +0.0228 | 3.575e-12 |
| -interest_channel | measured | 0.1485 [0.1385, 0.1590] | 0.1497 [0.1395, 0.1601] | +0.0012 | 6.823e-15 |
| -long_tail_quota | measured | 0.1485 [0.1385, 0.1590] | 0.1492 [0.1391, 0.1598] | +0.0007 | 6.259e-06 |
| -text_embeddings | measured | 0.1485 [0.1385, 0.1590] | 0.1579 [0.1465, 0.1695] | +0.0095 | 0.01209 |
| -implicit_taste | measured | 0.1485 [0.1385, 0.1590] | 0.1549 [0.1451, 0.1653] | +0.0064 | 0.1211 |
| -explicit_interests | measured | 0.1485 [0.1385, 0.1590] | 0.1467 [0.1364, 0.1570] | -0.0018 | 0.8217 |
| -behavioral_block | measured | 0.1485 [0.1385, 0.1590] | 0.1454 [0.1354, 0.1552] | -0.0030 | 0.3258 |

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
| 0.30 | 0.1349 |
| 0.50 | 0.1349 |
| 0.70 | 0.1357 |
| 0.90 | 0.1359 |
| 1.10 | 0.1361 |

## Decision Register

Every design choice not dictated by the assignment, the alternative, the experiment, and the measured result. Rows are generated from `results/parts/dr/*.json`; an experiment that was not run says NOT RUN.

| # | Decision | Alternative | Experiment | Result / verdict | Status |
|---|---|---|---|---|---|
| DR1 | Multiplicative utility rel^a * compat^b with a hard gate | The brief's additive formula a*rel + b*compat | Same holdout candidates/relevance/compatibility; rank by each combination rule; count hard-constraint violations (hard_gate == 0) among the top-10 of every holdout trip. | Additive scoring puts 901 hard-constraint violations into 229/671 trips' top-10; the multiplicative gated rule puts 0. NDCG@10 0.1408 (additive) vs 0.1357 (multiplicative gated); mean compat@10 0.8616 vs 0.8368. | MEASURED |
| DR2 | LightGBM LambdaMART ranker | Two-tower neural ranker (shared-space dot product) | Learning curve: both rankers fit on [0.1, 0.25, 0.5, 1.0] of TRAIN trips x seeds [42, 43, 44] (same IPS weights, same train-carved early stopping), scored on the full unbiased holdout; per-trip NDCG@10 averaged over seeds, 2000-resample trip bootstrap CIs. | Two-tower NDCG@10 0.1047, 0.1081, 0.1204, 0.1292 vs LambdaMART-IPS 0.1409, 0.1481, 0.1495, 0.1464 at fractions [0.1, 0.25, 0.5, 1.0]. Crossover at any measured point: False. Log-linear extrapolation puts a crossover at ~22,045 training trips (12.1x the 1829 used; 4-point fit, low confidence). CIs overlap at 100%. | MEASURED |
| DR3 | Listwise LambdaRank objective | Pointwise binary / graded regression, listwise rank_xendcg | Same features, IPS weights, dropout, split and early-stopping metric (NDCG@10); only the LightGBM objective varies. | lambdarank 0.1485 [0.1385, 0.1590]; rank_xendcg 0.1541 [0.1432, 0.1652]; binary 0.1590 [0.1483, 0.1701]; regression 0.1591 [0.1482, 0.1700] | MEASURED |
| DR4 | TF-IDF -> SVD-64 POI text embedding | all-MiniLM-L6-v2 sentence embeddings (-> SVD-64) | Swap ONLY the POI text embedding (and the taste vectors and POI features built from it) and refit the ranker on the fixed candidate sets; measure representation fidelity (D9 within-trip, D11) and holdout NDCG@10. THIS SYNTHETIC CORPUS is generated from anchored, synonym-rich phrase pools over latent dimensions, so its vocabulary design favours lexical overlap: the outcome is a statement about this dataset, not a verdict on sentence encoders. | TF-IDF: D11 0.865, D9 0.473, NDCG@10 0.1485. MiniLM: D11 0.590, D9 0.258, NDCG@10 0.1432 (CIs overlap). Dataset-specific (templated synonym-pool text); transfer to real POI text is untested. | MEASURED |
| DR5 | Brute-force cosine retrieval | ANN index (faiss / hnswlib) | NOT RUN | NOT RUN (cut for time; no evidence either way) | NOT RUN |
| DR6 | Geometric-mean compatibility aggregation | min(), plain product, arithmetic mean | Recompute compatibility from the 6 sub-scores with each aggregator; rank gated and ungated multiplicative utility. | Ungated hard-violation counts: geometric_mean (production)=1484, min=990, product=884, arithmetic_mean=1585; NDCG@10 gated: geometric_mean (production)=0.1357, min=0.1347, product=0.1322, arithmetic_mean=0.1341. Geometric mean ungated NDCG 0.1526. | MEASURED |
| DR7 | IPS clip = 20 | clip in {5, 10, 50, none} | IPS clip-high swept with everything else fixed (weights renormalised per trip). | clip 5: 0.1356 [0.1256, 0.1458]; clip 10: 0.1439 [0.1335, 0.1545]; clip 20: 0.1485 [0.1385, 0.1590]; clip 50: 0.1618 [0.1513, 0.1727]; clip none: 0.1571 [0.1466, 0.1678]; clip no_ips: 0.1244 [0.1152, 0.1339] | MEASURED |
| DR8 | 180-day taste half-life and spec interaction weights | +-2x half-life; uniform interaction weights | Rebuild the traveler features with a different taste half-life / interaction weights, refit the ranker on the fixed candidate sets, and score the unbiased holdout; D9 (within-trip, reporting-only) shows the effect on estimator fidelity. | halflife_180d (shipped): NDCG@10 0.1485, D9 0.473; halflife_90d: NDCG@10 0.1496, D9 0.470; halflife_360d: NDCG@10 0.1614, D9 0.474; uniform_weights: NDCG@10 0.1624, D9 0.052 | MEASURED |
| DR9 | Long-tail candidate quota = 50 | quota in {0, 25, 100} | Regenerate the holdout candidate sets with a different long-tail hard-floor quota (retriever scores fixed); score with the SHIPPED booster (trained at quota 50, not refit per quota); raw ranker top-10, no MMR/gates. | quota 0: NDCG@10 0.1492, long-tail share 0.156, candidate recall 0.845; quota 25: NDCG@10 0.1487, long-tail share 0.161, candidate recall 0.851; quota 50: NDCG@10 0.1485, long-tail share 0.161, candidate recall 0.857; quota 100: NDCG@10 0.1472, long-tail share 0.168, candidate recall 0.874 | MEASURED |
| DR10 | alpha = 1.0, beta = 0.7 | alpha x beta grid | Gated multiplicative utility over an alpha x beta grid on the same holdout scoring pass. | Shipped (alpha=1.0, beta=0.7): NDCG@10 0.1357, compat@10 0.8368. NDCG-best cell (alpha=1.0, beta=1.0): 0.1362, compat@10 0.8413. | MEASURED |
| DR11 | Candidate generation = learned retriever + long-tail + interest | The original 6 heuristic channels, and subsets of them | Channel-subset grid at learned K=210 on a train-carved validation split (IPS-weighted logged positives); holdout columns are reporting-only. `none` = the learned retriever alone. Baseline = the original 6-channel heuristic union. | Legacy 6-channel union: recall 0.596 (lift +0.203). Shipped learned+long-tail+interest: recall 0.881 (lift +0.371, 246 candidates/trip). Ranking by true utility would recall 0.991 at the same budget (diagnostic). | MEASURED |

## Seed replication

The full pipeline was regenerated end to end for seeds [42, 43, 44, 45, 46] (a new synthetic dataset, retriever, boosters and calibration each time; `scripts/seed_replication.py`). Seed 42 is the committed run.

| Metric | Mean | SD | Min | Max |
|---|---|---|---|---|
| bias_gap_popularity | 0.0643 | 0.0262 | 0.0253 | 0.1016 |
| bias_gap_primary | 0.0012 | 0.0184 | -0.0268 | 0.0300 |
| candidate_recall_long_tail | 0.8135 | 0.0078 | 0.8044 | 0.8262 |
| candidate_recall_overall | 0.8548 | 0.0032 | 0.8504 | 0.8574 |
| coverage_at_10 | 0.6382 | 0.0242 | 0.6004 | 0.6639 |
| ece_after | 0.0406 | 0.0076 | 0.0281 | 0.0484 |
| longtail_precision | 0.1680 | 0.0119 | 0.1541 | 0.1875 |
| longtail_share | 0.2556 | 0.0179 | 0.2252 | 0.2764 |
| ndcg10_content_cosine | 0.1430 | 0.0038 | 0.1386 | 0.1490 |
| ndcg10_lambdamart_ips | 0.1540 | 0.0071 | 0.1460 | 0.1664 |
| ndcg10_logistic_regression | 0.1231 | 0.0080 | 0.1130 | 0.1338 |
| ndcg10_popularity | 0.0783 | 0.0088 | 0.0629 | 0.0878 |
| pct_of_oracle_ceiling | 0.4128 | 0.0172 | 0.3961 | 0.4448 |
| within_cross_ratio_true_labels | 1.2593 | 0.0382 | 1.2126 | 1.3209 |

## Wall-clock

**reproduce**: 395.7 s total (16-logical-core laptop CPU, no GPU, no network).

| Stage | Seconds |
|---|---|
| generate | 35.2 |
| prepare | 5.9 |
| features | 19.9 |
| candidates | 48.1 |
| gate-dgp | 11.4 |
| train | 123.4 |
| evaluate | 128.1 |
| representation | 16.8 |
| gate-representation | 3.4 |
| compose | 3.5 |

**reproduce-full**: 492.9 s total (16-logical-core laptop CPU, no GPU, no network).

| Stage | Seconds |
|---|---|
| generate | 35.2 |
| prepare | 5.9 |
| features | 19.9 |
| candidates | 48.1 |
| gate-dgp | 11.4 |
| train | 123.4 |
| evaluate | 128.1 |
| representation | 16.8 |
| gate-representation | 3.4 |
| compose | 3.5 |
| recommend | 24.4 |
| scenarios | 26.8 |
| lodo | 46.0 |
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
| 1 | PSEO0426 | Mealtime Bukchon Wellness Studio | wellness_spa | 0.0560 | 0.0599 | 0.9101 | 0.4152 | 31.3278% | 0.5269 |
| 2 | PSEO0464 | Bohemian Insadong Exhibition Hall | museum | 0.0485 | 0.0533 | 0.8730 | 0.3622 | 71.7842% | -0.0535 |
| 3 | PSEO0114 | Evening Out Bukchon Speakeasy | nightlife | 0.0383 | 0.0496 | 0.6920 | 0.4220 | 43.5685% | 0.8511 |
| 4 | PSEO0071 | Teahouse Bukchon Bakery | cafe | 0.0453 | 0.0484 | 0.9076 | 0.5282 | 88.7967% | -0.7435 |
| 5 | PSEO0500 | Garden Bukchon Green (Branch) | nature_park | 0.0486 | 0.0533 | 0.8757 | 0.2773 | 21.3693% | 0.2222 |
| 6 | PSEO0188 | Dining Insadong Grill | restaurant | 0.0485 | 0.0533 | 0.8732 | 0.4349 | 82.3651% | 0.0727 |
| 7 | PSEO0457 | Show Insadong Theater | entertainment | 0.0417 | 0.0459 | 0.8707 | 0.4201 | 8.0913% | 0.3725 |
| 8 | PSEO0252 | On Trend Myeongdong Shrine | religious_site | 0.0457 | 0.0533 | 0.8019 | 0.4510 | 94.8133% | 0.1439 |
| 9 | PSEO0403 | Fun Filled Bukchon Fun Zone | family_activity | 0.0407 | 0.0496 | 0.7552 | 0.3626 | 38.1743% | 0.3558 |
| 10 | PSEO0326 | Curated Itaewon Arcade | shopping | 0.0438 | 0.0496 | 0.8385 | 0.4774 | 87.7593% | -0.5020 |

**Top signals (rank 1, PSEO0426):** interest_match (1.226), price_fit (0.128), localness_fit (0.052)

**Explanation (rank 1):** Matches your stated medium budget Less touristy than 69% of comparable POIs in Seoul 15 min from your stay by public transport -- a bit of a trek

### Scenario 2: History & Architecture

- Persona: First-time K-culture visitor to Seoul (foreign, inbound): heritage, museums and the iconic sights, comfortable budget.
- Destination: **seoul**; interests: historic_site, historic, cultural, museum; touristiness_pref: **0.40**
- Budget: high; mobility: public_transport; party: couple; pace: moderate; accessibility needs: none
- 'history' -> {historic_site, historic}; 'architecture' -> {cultural} (no literal 'architecture' tag exists in CATEGORIES+TAGS); 'museums' -> {museum} (exact match). `pace` defaulted to 'moderate' (not specified).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0182 | Laid Back Myeongdong Overlook | viewpoint | 0.0521 | 0.0533 | 0.9659 | 0.4893 | 90.8714% | -0.8629 |
| 2 | PSEO0343 | Luxurious Hongdae Retreat | wellness_spa | 0.0491 | 0.0533 | 0.8891 | 0.4629 | 85.5809% | -1.5229 |
| 3 | PSEO0381 | Eclectic Hongdae Lounge | nightlife | 0.0371 | 0.0533 | 0.5960 | 0.4564 | 86.9295% | -0.4450 |
| 4 | PSEO0071 | Teahouse Bukchon Bakery | cafe | 0.0475 | 0.0496 | 0.9420 | 0.4898 | 88.7967% | -0.7435 |
| 5 | PSEO0363 | Dreamy Seongsu Noodle House | restaurant | 0.0336 | 0.0359 | 0.9091 | 0.3780 | 7.0539% | -0.1695 |
| 6 | PSEO0325 | Historic Itaewon Shrine | religious_site | 0.0363 | 0.0459 | 0.7134 | 0.4652 | 66.1826% | 0.0740 |
| 7 | PSEO0391 | Storefront Seongsu Boutique | shopping | 0.0341 | 0.0359 | 0.9267 | 0.1658 | 13.4855% | 0.6000 |
| 8 | PSEO0457 | Show Insadong Theater | entertainment | 0.0335 | 0.0359 | 0.9037 | 0.3367 | 8.0913% | 0.3725 |
| 9 | PSEO0303 | Spiritual Gangnam Trail | nature_park | 0.0370 | 0.0459 | 0.7337 | 0.5491 | 72.1992% | -0.6232 |
| 10 | PSEO0378 | Landmark Site Insadong Fortress | historic_site | 0.0333 | 0.0359 | 0.8966 | 0.3741 | 72.6141% | -0.4826 |

**Top signals (rank 1, PSEO0182):** interest_match (0.658), popularity (0.137), price_fit (0.116)

**Explanation (rank 1):** Popular choice -- busier than 91% of comparable POIs in Seoul Matches your stated high budget More budget-friendly than typical for your high budget

### Scenario 3: Family with young children

- Persona: Foreign family with young children visiting Seoul: stroller, relaxed pace, things kids can do.
- Destination: **seoul**; interests: family_activity, nature_park, family-friendly; touristiness_pref: **0.00**
- Budget: medium; mobility: car; party: family_young_kids; pace: relaxed; accessibility needs: stroller
- 'activities' -> {family_activity}; 'parks' -> {nature_park}; 'interactive experiences' -> {family-friendly} (no literal 'interactive' tag exists). `touristiness_pref` and `mobility` are not specified by spec.md section 15 for this scenario -- defaulted to 0.0 (neutral) and 'car' (a common real choice for a family with young children traveling with a stroller) respectively; `party_type`/accessibility (`stroller`)/`pace` (`relaxed`)/`budget` (`medium`) are exactly as spec.md states.

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0074 | Neighborhood Bukchon Coffee House | cafe | 0.0558 | 0.0599 | 0.9035 | 0.3721 | 71.9917% | 0.3865 |
| 2 | PSEO0031 | Sun Drenched Bukchon Discovery Center | family_activity | 0.0505 | 0.0533 | 0.9239 | 0.4851 | 67.6349% | -0.4157 |
| 3 | PSEO0230 | Must See Insadong Noodle House | restaurant | 0.0422 | 0.0484 | 0.8219 | 0.4156 | 17.8423% | 1.0202 |
| 4 | PSEO0006 | Amusement Myeongdong Theater | entertainment | 0.0407 | 0.0459 | 0.8395 | 0.4273 | 52.3859% | 0.7186 |
| 5 | PSEO0090 | Tucked Away Bukchon Park | nature_park | 0.0445 | 0.0496 | 0.8565 | 0.4850 | 81.3278% | -0.2884 |
| 6 | PSEO0249 | Wallet Friendly Gangnam Gallery | museum | 0.0337 | 0.0359 | 0.9134 | 0.4738 | 93.5685% | -1.2491 |
| 7 | PSEO0251 | Retail Insadong Bazaar | shopping | 0.0336 | 0.0359 | 0.9104 | 0.2957 | 5.1867% | -0.9500 |
| 8 | PSEO0088 | Gallery Like Gangnam Monument | historic_site | 0.0422 | 0.0459 | 0.8850 | 0.5050 | 95.2282% | -0.6739 |
| 9 | PSEO0132 | Screening Itaewon Skyline Point | viewpoint | 0.0331 | 0.0359 | 0.8883 | 0.3567 | 9.7510% | -1.0274 |
| 10 | PSEO0123 | Bohemian Myeongdong Grill | restaurant | 0.0338 | 0.0359 | 0.9148 | 0.4199 | 82.1577% | -0.5479 |

**Top signals (rank 1, PSEO0074):** interest_match (0.958), price_fit (0.167), popularity (0.058)

**Explanation (rank 1):** Matches your stated medium budget Popular choice -- busier than 72% of comparable POIs in Seoul 11 min from your stay by car -- a bit of a trek

### Scenario 4: Diagnostic (Scenario 1 profile, touristiness_pref flipped to the opposite extreme)

- Persona: The same traveler as scenario 1 asking for the opposite: the iconic, must-see, first-timer version of Seoul.
- Destination: **seoul**; interests: local, foodie, authentic; touristiness_pref: **0.80**
- Budget: medium; mobility: public_transport; party: solo; pace: moderate; accessibility needs: none
- Base scenario: 1 (Local Experience). Every field held identical to that scenario's profile except `touristiness_pref`, flipped from -0.8 to +0.8 (opposite sign, same extreme magnitude).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0343 | Luxurious Hongdae Retreat | wellness_spa | 0.0585 | 0.0599 | 0.9681 | 0.4426 | 85.5809% | -1.5229 |
| 2 | PSEO0182 | Laid Back Myeongdong Overlook | viewpoint | 0.0507 | 0.0533 | 0.9306 | 0.4934 | 90.8714% | -0.8629 |
| 3 | PSEO0139 | Fun Filled Bukchon Adventure Park | family_activity | 0.0489 | 0.0533 | 0.8830 | 0.5051 | 87.3444% | -0.9920 |
| 4 | PSEO0071 | Teahouse Bukchon Bakery | cafe | 0.0463 | 0.0496 | 0.9076 | 0.4907 | 88.7967% | -0.7435 |
| 5 | PSEO0464 | Bohemian Insadong Exhibition Hall | museum | 0.0418 | 0.0459 | 0.8730 | 0.3386 | 71.7842% | -0.0535 |
| 6 | PSEO0454 | Crowd Pulling Myeongdong Bistro | restaurant | 0.0405 | 0.0459 | 0.8350 | 0.4671 | 69.9170% | -0.5462 |
| 7 | PSEO0359 | Parkland Bukchon Riverside Walk | nature_park | 0.0338 | 0.0359 | 0.9163 | 0.3399 | 45.0207% | 0.4073 |
| 8 | PSEO0252 | On Trend Myeongdong Shrine | religious_site | 0.0415 | 0.0484 | 0.8019 | 0.5686 | 94.8133% | 0.1439 |
| 9 | PSEO0347 | Fashionable Seongsu Cinema | entertainment | 0.0409 | 0.0459 | 0.8476 | 0.5457 | 91.4938% | -1.5244 |
| 10 | PSEO0094 | Gallery Like Itaewon Old Quarter | historic_site | 0.0337 | 0.0359 | 0.9116 | 0.3418 | 22.4066% | 0.9826 |

**Top signals (rank 1, PSEO0343):** interest_match (1.240), price_fit (0.159), popularity (0.108)

**Explanation (rank 1):** Matches your stated medium budget Popular choice -- busier than 86% of comparable POIs in Seoul More budget-friendly than typical for your medium budget

### Pairwise top-10 Jaccard overlap across scenarios

| | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 |
|---|---|---|---|---|
| Scenario 1 | 1.000 | 0.111 | 0.000 | 0.176 |
| Scenario 2 | 0.111 | 1.000 | 0.000 | 0.176 |
| Scenario 3 | 0.000 | 0.000 | 1.000 | 0.000 |
| Scenario 4 | 0.176 | 0.176 | 0.000 | 1.000 |

**Diagnostic scenario 4 vs base scenario 1** (touristiness_pref flipped to the opposite extreme, everything else held constant): top-10 Jaccard overlap = **0.176** (spec.md section 15 expects this to be low, i.e. &le; 0.25, **MET**).

Low overlap demonstrates the ranking is driven by the preference signal, not profile confounds, as spec.md section 15 expects.

