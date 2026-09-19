# RESULTS.md

**Generated automatically by `poi_rank.eval.report` from `results/metrics.json`. Do not hand-edit -- every number here traces directly to that JSON file (spec.md section 0: "No unverified metric may appear in any document").**

Synthetic data throughout (three destinations; Seoul is the lead). Personas are inbound foreign travelers. Read the numbers as properties of this simulator, not as claims about real traffic -- `docs/TECHNICAL.md` states what the simulator does and does not license.

## Headline: local and long-tail discovery (Seoul-first, inbound travelers)

konnect.kr serves foreign travelers in Korea, and the incumbents already rank by popularity -- so a popularity-shaped list is table stakes. The claim that matters is surfacing the genuinely relevant, non-obvious POI: **long-tail share and long-tail precision together** (coverage without precision is just noise injection), measured on the unbiased random-exposure holdout.

| Top-10 lists, primary holdout | LambdaMART + IPS (primary) | Popularity ranker |
|---|---|---|
| Long-tail share (bottom-50% popularity stratum) | **0.2234** | 0.0000 |
| Long-tail precision (relevant / recommended) | **0.1455** | N/A |
| Catalog coverage@10 | **64.9%** | 5.5% |
| Gini of recommendation exposure (lower = less monoculture) | **0.7149** | 0.9738 |
| NDCG@10 (unbiased holdout, 95% CI) | **0.1480** [0.1380, 0.1582] | 0.0616 [0.0557, 0.0683] |

Popularity is flattered by exposure-biased logs; the bias-gap table below quantifies exactly how much (popularity +0.0826, primary +0.0096 NDCG@10 between the biased and unbiased holdouts).

## Bias-gap table (spec.md section 11.1)

Each system's NDCG@10 on the SECONDARY biased holdout (`interactions_holdout_logged.parquet`) vs the PRIMARY unbiased holdout. Expectation: the popularity baseline shows a large positive gap (flattered by biased logs); the IPS-corrected model shows a small gap.

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0589 | 0.0281 | -0.0309 |
| 2. Popularity | 0.0616 | 0.1442 | +0.0826 |
| 3. Popularity + geo filter | 0.0605 | 0.1411 | +0.0806 |
| 4. Content cosine | 0.1400 | 0.0548 | -0.0852 |
| 5. Item-kNN CF | 0.0700 | 0.1478 | +0.0779 |
| 6. Logistic regression | 0.1129 | 0.2281 | +0.1151 |
| 7. LambdaMART | 0.1257 | 0.1963 | +0.0706 |
| 8. LambdaMART + IPS (primary) | 0.1480 | 0.1576 | +0.0096 |
| 9. Oracle (ceiling) | 0.3733 | 0.1125 | -0.2608 |

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
| candidate_recall_lift_long_tail | >= 0.35 | 0.3872 | PASS |
| candidate_recall_lift_overall | >= 0.35 | 0.3639 | PASS |
| candidate_recall_long_tail | >= 0.75 | 0.8481 | PASS |
| candidate_recall_overall | >= 0.85 | 0.8731 | PASS |

| Reporting-only (after freeze) | Value |
|---|---|
| d11_cca_first_corr_text_only | 0.9790 |
| d11_ridge_r2_text_only | 0.8654 |
| d9_within_trip_shipped_chain | 0.4731 |
| d9_within_trip_taste_estimator_alone | 0.5718 |
| d9_within_trip_text_alone | 0.9272 |
| oracle_ceiling_ndcg10_candidate_level | 0.3733 |
| oracle_relevance_recall_long_tail | 0.9254 |
| oracle_relevance_recall_overall | 0.9442 |

## Success criteria scorecard

Stated up front, then measured. Every MISSED row carries a diagnosis below -- a miss is reported as a ceiling only after a diagnostic has ruled out a mechanism, citing the number (spec.md: "An honest miss with a root-cause analysis scores better than a suspiciously perfect table.").

| Metric | Target | Measured | Status |
|---|---|---|---|
| NDCG@10 vs popularity | &ge; +40% relative, Wilcoxon p < 0.01 | +140.2% relative, p=5.695e-43 | **MET** |
| % of oracle ceiling (candidate-level NDCG@10) | &ge; 70% | 39.6% | **MISSED** |
| Candidate recall, overall (exposed positives) | &ge; 0.85 | 0.8731 | **MET** |
| Candidate recall, long-tail stratum | &ge; 0.75 | 0.8481 | **MET** |
| Candidate recall lift over chance (overall) | &ge; +0.35 | +0.364 | **MET** |
| Cross-archetype Jaccard@10 (true labels) | &le; 0.25 | 0.0398 | **MET** |
| Within/cross Jaccard ratio (true labels) | &ge; 2.0 | 1.27 | **MISSED** |
| Hard-constraint violations in top-10 | = 0 | 0 | **MET** |
| ECE after calibration | &le; 0.05 | 0.0418 | **MET** |
| Confidence-decile NDCG rank correlation (Spearman; need not be strictly monotone) | &ge; 0.6 (spec-v2; spec.md section 11.10 said 0.7) | 0.721 | **MET** |
| Long-tail share of top-10 | &ge; 0.25 | 0.2234 | **MISSED** |
| Long-tail precision of top-10 | &ge; 0.40 | 0.1455 | **MISSED** |
| Localness index Spearman vs latent localness | &ge; 0.6 | 0.5815 | **MISSED** |
| Scenario-4 (touristiness flip) top-10 overlap | &le; 0.35 | 0.111 | **MET** |

### Diagnoses of the missed rows

- **% of oracle ceiling (candidate-level NDCG@10)**: The oracle ranks the SAME candidates by the DGP's true utility; the model only sees observable features. Text is not the limiting link: text-alone within-trip taste fidelity is 0.927 and D11 ridge R2 is 0.865. The taste ESTIMATOR is: run over the TRUE semantic vectors it still reaches only 0.572 (shipped chain 0.473), from sparse, exposure-biased histories.
- **Within/cross Jaccard ratio (true labels)**: Within/cross-archetype list similarity ratio for lists ranked by the TRUE utility (a perfect ranker, same-destination pairs): 1.87 (within 0.0640, cross 0.0342); so the target is NOT attainable even by a perfect ranker in this simulator: the shortfall is a property of the simulator under this target, not of the model.
- **Long-tail share of top-10**: Long-tail share of the served top-10 is 0.2234. Decision Register DR9 varies the long-tail candidate quota and measures the raw ranker's top-10 share: quota 0 -> 0.156, quota 100 -> 0.166, quota 25 -> 0.157, quota 50 -> 0.160. The candidate quota is therefore not the lever; the share is set by the ranker's scores (and the MMR re-rank) over a candidate set that already contains long-tail POIs.
- **Long-tail precision of top-10**: Long-tail precision is 0.1455 over 1478 long-tail recommendations, with candidate recall 0.848 in that stratum, so retrieval is not the bottleneck. The raw ranker's top-10 long-tail precision (DR9, quota 50, before the compatibility gate, utility and MMR re-rank) is 0.2390 at share 0.1596, against 0.1455 at share 0.2234 in the served list: precision is lost AFTER ranking while share rises. Which of the three scoring-layer steps is responsible is not isolated (untested).
- **Localness index Spearman vs latent localness**: The composite index reaches rho 0.582; its observable inputs correlate with the latent localness at dist_to_tourist_centroid_km 0.699, foreign_review_ratio -0.555, local_tag_hits 0.050, pop_pct -0.187. The composite is BELOW its best single input (dist_to_tourist_centroid_km, |rho| 0.699): the blend weights were fixed earlier, when the geo input carried almost no signal (before the simulator's geo/localness fix). Re-weighting against the latent value would leak the oracle into a decision, so the index is left as shipped and the gap is reported.

## Primary ranking quality (unbiased random-exposure holdout, spec.md section 11.1)

n_holdout_trips = 671, bootstrap n_resamples = 2000 (resample unit: trip).

| System | ndcg@5 | ndcg@10 | ndcg@20 | precision@5 | precision@10 | recall@10 | recall@20 | map | mrr | % of oracle ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Random | 0.0510 [0.0440, 0.0581] | 0.0589 [0.0529, 0.0651] | 0.0759 [0.0705, 0.0812] | 0.1192 [0.1061, 0.1323] | 0.1173 [0.1079, 0.1271] | 0.0419 [0.0387, 0.0451] | 0.0822 [0.0780, 0.0866] | 0.1460 [0.1409, 0.1511] | 0.2894 [0.2657, 0.3139] | 15.8% |
| 2. Popularity | 0.0529 [0.0462, 0.0601] | 0.0616 [0.0557, 0.0683] | 0.0866 [0.0807, 0.0932] | 0.1168 [0.1046, 0.1291] | 0.1152 [0.1060, 0.1255] | 0.0427 [0.0393, 0.0462] | 0.0932 [0.0884, 0.0984] | 0.1602 [0.1546, 0.1660] | 0.2916 [0.2687, 0.3156] | 16.5% |
| 3. Popularity + geo filter | 0.0514 [0.0449, 0.0582] | 0.0605 [0.0546, 0.0667] | 0.0824 [0.0767, 0.0882] | 0.1168 [0.1046, 0.1288] | 0.1182 [0.1088, 0.1280] | 0.0431 [0.0399, 0.0466] | 0.0902 [0.0853, 0.0949] | 0.1553 [0.1498, 0.1607] | 0.3016 [0.2773, 0.3284] | 16.2% |
| 4. Content cosine | 0.1291 [0.1179, 0.1403] | 0.1400 [0.1302, 0.1498] | 0.1698 [0.1610, 0.1791] | 0.2554 [0.2379, 0.2724] | 0.2297 [0.2155, 0.2434] | 0.0845 [0.0800, 0.0890] | 0.1596 [0.1530, 0.1664] | 0.2297 [0.2217, 0.2381] | 0.5075 [0.4787, 0.5376] | 37.5% |
| 5. Item-kNN CF | 0.0596 [0.0520, 0.0675] | 0.0700 [0.0632, 0.0769] | 0.0921 [0.0858, 0.0988] | 0.1329 [0.1204, 0.1464] | 0.1306 [0.1207, 0.1410] | 0.0488 [0.0453, 0.0525] | 0.0976 [0.0926, 0.1033] | 0.1636 [0.1579, 0.1695] | 0.3226 [0.2981, 0.3478] | 18.7% |
| 6. Logistic regression | 0.1012 [0.0916, 0.1114] | 0.1129 [0.1045, 0.1223] | 0.1360 [0.1281, 0.1451] | 0.2212 [0.2030, 0.2399] | 0.2000 [0.1866, 0.2134] | 0.0736 [0.0691, 0.0782] | 0.1360 [0.1303, 0.1425] | 0.2082 [0.2007, 0.2159] | 0.4538 [0.4255, 0.4833] | 30.3% |
| 7. LambdaMART | 0.1099 [0.0993, 0.1205] | 0.1257 [0.1161, 0.1352] | 0.1555 [0.1465, 0.1645] | 0.2313 [0.2128, 0.2507] | 0.2186 [0.2048, 0.2337] | 0.0812 [0.0768, 0.0862] | 0.1529 [0.1465, 0.1597] | 0.2147 [0.2066, 0.2232] | 0.4855 [0.4549, 0.5155] | 33.7% |
| 8. LambdaMART + IPS (primary) | 0.1285 [0.1172, 0.1399] | 0.1480 [0.1380, 0.1582] | 0.1769 [0.1671, 0.1867] | 0.2659 [0.2462, 0.2858] | 0.2548 [0.2390, 0.2714] | 0.0932 [0.0881, 0.0982] | 0.1694 [0.1622, 0.1764] | 0.2370 [0.2282, 0.2465] | 0.4974 [0.4685, 0.5258] | 39.6% |
| 9. Oracle (ceiling) | 0.3419 [0.3257, 0.3593] | 0.3733 [0.3594, 0.3877] | 0.4309 [0.4194, 0.4430] | 0.5031 [0.4781, 0.5282] | 0.4870 [0.4659, 0.5083] | 0.1852 [0.1791, 0.1917] | 0.3552 [0.3471, 0.3642] | 0.5001 [0.4857, 0.5142] | 0.7299 [0.7027, 0.7564] | 100.0% |

### Paired Wilcoxon signed-rank tests (NDCG@10)

| Comparison | statistic | p-value | n_pairs |
|---|---|---|---|
| content_cosine_vs_popularity | 31691.0000 | 3.983e-36 | 605 |
| item_knn_cf_vs_popularity | 4541.0000 | 0.0003863 | 605 |
| lambdamart_ips_vs_content_cosine | 83128.0000 | 0.3289 | 605 |
| lambdamart_ips_vs_popularity | 27026.5000 | 5.695e-43 | 605 |
| lambdamart_vs_lambdamart_ips | 50894.0000 | 8.156e-12 | 605 |
| lambdamart_vs_popularity | 32735.5000 | 7.056e-32 | 605 |
| logistic_regression_vs_popularity | 36573.5000 | 5.235e-25 | 605 |
| oracle_vs_popularity | 992.0000 | 5.689e-98 | 605 |

## Candidate recall by popularity stratum (with chance baselines)

Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its own chance baseline: the share of that stratum's destination POIs a same-size random candidate set would contain. Raw recall without this lift hid a near-chance failure for four phases.

| Stratum | Recall | Chance | Lift (abs) | Trips |
|---|---|---|---|---|
| long_tail | 0.848 | 0.461 | +0.387 | 605 |
| overall | 0.873 | 0.509 | +0.364 | 605 |
| q1_least_popular | 0.859 | 0.500 | +0.359 | 601 |
| q2 | 0.836 | 0.422 | +0.414 | 600 |
| q3 | 0.834 | 0.454 | +0.380 | 603 |
| q4_most_popular | 0.922 | 0.660 | +0.262 | 605 |

## Personalization (spec.md section 11.2, spec-v2 RC4)

Pairs are SAME-DESTINATION trip pairs only: a POI belongs to exactly one destination, so two trips to different destinations have Jaccard = RBO = 0 by construction, and pooling them measures the catalog partition rather than the recommender.

- Mean pairwise Jaccard@10 (same destination): **0.0406** (n_pairs=74775)
- Mean pairwise rank-biased overlap (RBO, p=0.9): **0.0548** (n_pairs=74775)
- For continuity with the pre-fix number, all pairs pooled (2/3 of them structurally zero): 0.0135 (n_pairs=224785)

| Grouping | Within Jaccard@10 | Cross Jaccard@10 | Ratio | within / cross pairs |
|---|---|---|---|---|
| **True archetype labels** (dominant archetype; within = same dominant and mixture cosine > 0.8) | 0.0506 | 0.0398 | **1.27** | 4814 / 65145 |
| Reference: lists ranked by the TRUE utility (a perfect ranker) | 0.0640 | 0.0342 | 1.87 | 4814 / 65145 |
| K-Means traveler-segment PROXY (clusters of the same features being evaluated -- kept only to show why it was misleading) | 0.0454 | 0.0397 | 1.14 | 11459 / 63316 |

## Coverage (spec.md section 11.3)

| | Primary system (lambdamart_ips) | Popularity baseline |
|---|---|---|
| Catalog coverage@10 | 64.9% | 5.5% |
| Gini coefficient | 0.7149 | 0.9738 |
| Entropy (bits) | 9.0861 | 5.5212 |
| POIs ever recommended | 938 / 1446 | 80 / 1446 |

Lower Gini / higher entropy / higher coverage = less popularity-monoculture concentration. Per-destination coverage:

| Destination | Primary | Popularity |
|---|---|---|
| barcelona | 63.6% | 6.0% |
| kyoto | 64.8% | 5.6% |
| seoul | 66.2% | 5.0% |

## Long-tail / local discovery (spec.md section 11.4)

- Share of top-10 recommendations in the bottom-50%-popularity stratum: **0.2234** (1478 / 6615)
- Long-tail precision (relevant per unbiased holdout): **0.1455** (215 / 1478)
- Popularity ranker, same measurement: share **0.0000** (0 / 6710), precision **N/A**

"Coverage without precision is just noise injection" -- both numbers reported together, per spec.md section 11.4.

## Constraint compatibility (spec.md section 11.5)

- % of top-10 with compatibility &ge; 0.7: **91.3%** (6040 / 6615)
- Hard-constraint violations in top-10: **0** (build-blocking, enforced independently by `tests/test_hard_constraints.py`)

## Diversity (spec.md section 11.6)

- Category entropy@10 (bits): **3.3688**
- Intra-list mean cosine distance (at the configured default lambda): **0.7657**

### MMR lambda sweep (NDCG@10 vs diversity trade-off)

| lambda | NDCG@10 (mean) | mean intra-list similarity |
|---|---|---|
| 0.50 | 0.1689 | 0.2031 |
| 0.60 | 0.1719 | 0.2086 |
| 0.70 | 0.1769 | 0.2187 |
| 0.80 | 0.1831 | 0.2343 |
| 0.90 | 0.1927 | 0.2764 |
| 1.00 | 0.2108 | 0.3744 |

## Calibration (spec.md section 11.7)

| | Before (naive) | After (isotonic) |
|---|---|---|
| ECE (15 bins) | 0.4768 | 0.0418 |
| Brier score | 0.3410 | 0.1012 |

Calibration split: n_rows=75261, n_trips=311.

## Confidence-decile validation (spec.md section 9.3 / 11.10)

Spearman rho (decile rank, decile mean NDCG@k): **0.721** (target &ge; 0.7, **MET**). n_trips_included=605.

| Decile | Mean confidence | Mean NDCG@k | n_trips |
|---|---|---|---|
| 1 | 0.4642 | 0.0518 | 61 |
| 2 | 0.5484 | 0.1124 | 60 |
| 3 | 0.5881 | 0.1163 | 61 |
| 4 | 0.6041 | 0.1451 | 60 |
| 5 | 0.6155 | 0.1236 | 61 |
| 6 | 0.6266 | 0.1711 | 60 |
| 7 | 0.6377 | 0.1523 | 60 |
| 8 | 0.6497 | 0.1503 | 61 |
| 9 | 0.6625 | 0.1253 | 60 |
| 10 | 0.6844 | 0.1506 | 61 |

## Cold-start cohorts (spec.md section 11.8 / section 12)

### NDCG@10 by traveler interaction-count bucket

| Bucket | NDCG@10 (mean) | n_trips_in_bucket |
|---|---|---|
| 0 | 0.0451 | 72 |
| 1-3 | 0.2282 | 1 |
| 4-10 | 0.1784 | 59 |
| >10 | 0.1583 | 539 |

### New-POI cohort (spec.md section 12)

- n_new_pois_in_catalog=71, n_holdout_trips_with_relevant_cohort_candidate=307
- NDCG@10 WITH behavioral dropout: **0.5305 [0.5005, 0.5608]**
- NDCG@10 WITHOUT behavioral dropout: **0.5096 [0.4823, 0.5371]**
- Paired Wilcoxon (with vs without): p=0.09435

### Leave-one-destination-out (LODO)

Wall-clock: **32.3s** for 3 destination-held-out retrains.

| Destination | NDCG@10 (LODO) | NDCG@10 (full training) | Wilcoxon p |
|---|---|---|---|
| barcelona | 0.1362 [0.1177, 0.1566] | 0.1347 [0.1171, 0.1541] | 0.8623 |
| kyoto | 0.1455 [0.1270, 0.1647] | 0.1546 [0.1363, 0.1733] | 0.09778 |
| seoul | 0.1500 [0.1323, 0.1685] | 0.1548 [0.1377, 0.1722] | 0.2966 |

## Ablations (spec.md section 11.9)

Each row: delta NDCG@10 (ablated - full lambdamart_ips), against the SAME already-trained primary system as the reference point.

| Ablation | Status | NDCG@10 (full) | NDCG@10 (ablated) | Delta | Wilcoxon p |
|---|---|---|---|---|---|
| -IPS_weighting | measured | 0.1480 [0.1380, 0.1582] | 0.1257 [0.1161, 0.1352] | -0.0223 | 8.156e-12 |
| -calibration | measured | 0.1480 [0.1380, 0.1582] | 0.1458 [0.1355, 0.1560] | -0.0021 | 0.1352 |
| -MMR | measured | 0.1831 [0.1660, 0.2001] | 0.2108 [0.1928, 0.2286] | +0.0278 | 1.933e-13 |
| -CF_channel | measured | 0.1480 [0.1380, 0.1582] | 0.1480 [0.1380, 0.1582] | +0.0000 | 1 |
| -long_tail_quota | measured | 0.1480 [0.1380, 0.1582] | 0.1483 [0.1383, 0.1585] | +0.0004 | 2.958e-05 |
| -text_embeddings | measured | 0.1480 [0.1380, 0.1582] | 0.1557 [0.1451, 0.1662] | +0.0077 | 0.01339 |
| -implicit_taste | measured | 0.1480 [0.1380, 0.1582] | 0.1539 [0.1438, 0.1642] | +0.0059 | 0.1203 |
| -explicit_interests | measured | 0.1480 [0.1380, 0.1582] | 0.1481 [0.1378, 0.1586] | +0.0002 | 0.8168 |
| -behavioral_block | measured | 0.1480 [0.1380, 0.1582] | 0.1481 [0.1378, 0.1585] | +0.0001 | 0.5828 |

Notes:

- **-IPS_weighting**: Free: system 7 (lambdamart, uniform weight) IS this ablation of system 8.
- **-calibration**: Isotonic calibration is a monotonic non-decreasing transform of the raw score, so it cannot change within-trip RANKING (and therefore NDCG@10) except through tie-break reordering at flat isotonic segments -- an at/near-zero delta here is the mathematically EXPECTED result, not a modeling failure. Calibration's actual purpose (cross-traveler score comparability for planner_weight) is not something NDCG@10 measures.
- **-MMR**: lambda=1.0 (pure utility ranking, no diversity penalty) vs the configured default lambda=0.8 -- reuses scoring.diversity.mmr_rerank_all_trips directly, no retraining.
- **-CF_channel**: Leave-one-channel-out ('channel_cf' removed from the 6-channel candidate union): re-evaluates the ALREADY-TRAINED lambdamart_ips score over the smaller candidate set, no retraining, no new candidate generation.
- **-long_tail_quota**: Leave-one-channel-out ('channel_longtail' removed from the 6-channel candidate union): re-evaluates the ALREADY-TRAINED lambdamart_ips score over the smaller candidate set, no retraining, no new candidate generation.
- **-text_embeddings**: Full retrain with every 'text_emb_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.
- **-implicit_taste**: Full retrain with every 'implicit_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.
- **-explicit_interests**: Full retrain with every 'explicit_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.
- **-behavioral_block**: Full retrain with every 'behav_*' numeric feature column dropped from the design matrix (models.baselines.numeric_feature_columns filtered before training) -- same IPS + behavioral-dropout recipe as system 8 otherwise.

## Utility beta sensitivity (spec.md section 9.1)

| beta | NDCG@10 (mean) |
|---|---|
| 0.30 | 0.1297 |
| 0.50 | 0.1304 |
| 0.70 | 0.1298 |
| 0.90 | 0.1292 |
| 1.10 | 0.1282 |

## Decision Register

Every design choice not dictated by the assignment, the alternative, the experiment, and the measured result. Rows are generated from `results/parts/dr/*.json`; an experiment that was not run says NOT RUN.

| # | Decision | Alternative | Experiment | Result / verdict | Status |
|---|---|---|---|---|---|
| DR1 | Multiplicative utility rel^a * compat^b with a hard gate | The brief's additive formula a*rel + b*compat | Same holdout candidates/relevance/compatibility; rank by each combination rule; count hard-constraint violations (hard_gate == 0) among the top-10 of every holdout trip. | Additive scoring puts 844 hard-constraint violations into 216/671 trips' top-10; the multiplicative gated rule puts 0. NDCG@10 0.1341 (additive) vs 0.1296 (multiplicative gated); mean compat@10 0.8655 vs 0.8371. | MEASURED |
| DR2 | LightGBM LambdaMART ranker | Two-tower neural ranker (shared-space dot product) | Learning curve: both rankers fit on [0.1, 0.25, 0.5, 1.0] of TRAIN trips x seeds [42, 43, 44] (same IPS weights, same train-carved early stopping), scored on the full unbiased holdout; per-trip NDCG@10 averaged over seeds, 2000-resample trip bootstrap CIs. | Two-tower NDCG@10 0.1039, 0.1075, 0.1188, 0.1286 vs LambdaMART-IPS 0.1404, 0.1442, 0.1488, 0.1499 at fractions [0.1, 0.25, 0.5, 1.0]. Crossover at any measured point: False. Log-linear extrapolation puts a crossover at ~65,598 training trips (35.9x the 1829 used; 4-point fit, low confidence). CIs separated at 100%. | MEASURED |
| DR3 | Listwise LambdaRank objective | Pointwise binary / graded regression, listwise rank_xendcg | Same features, IPS weights, dropout, split and early-stopping metric (NDCG@10); only the LightGBM objective varies. | lambdarank 0.1480 [0.1380, 0.1582]; rank_xendcg 0.1556 [0.1453, 0.1666]; binary 0.1596 [0.1491, 0.1705]; regression 0.1596 [0.1492, 0.1700] | MEASURED |
| DR4 | TF-IDF -> SVD-64 POI text embedding | all-MiniLM-L6-v2 sentence embeddings (-> SVD-64) | Swap ONLY the POI text embedding (and the taste vectors and POI features built from it) and refit the ranker on the fixed candidate sets; measure representation fidelity (D9 within-trip, D11) and holdout NDCG@10. THIS SYNTHETIC CORPUS is generated from anchored, synonym-rich phrase pools over latent dimensions, so its vocabulary design favours lexical overlap: the outcome is a statement about this dataset, not a verdict on sentence encoders. | TF-IDF: D11 0.865, D9 0.473, NDCG@10 0.1480. MiniLM: D11 0.590, D9 0.258, NDCG@10 0.1549 (CIs overlap). Dataset-specific (templated synonym-pool text); transfer to real POI text is untested. | MEASURED |
| DR5 | Brute-force cosine retrieval | ANN index (faiss / hnswlib) | NOT RUN | NOT RUN (cut for time; no evidence either way) | NOT RUN |
| DR6 | Geometric-mean compatibility aggregation | min(), plain product, arithmetic mean | Recompute compatibility from the 6 sub-scores with each aggregator; rank gated and ungated multiplicative utility. | Ungated hard-violation counts: geometric_mean (production)=1422, min=940, product=830, arithmetic_mean=1550; NDCG@10 gated: geometric_mean (production)=0.1296, min=0.1269, product=0.1273, arithmetic_mean=0.1292. Geometric mean ungated NDCG 0.1465. | MEASURED |
| DR7 | IPS clip = 20 | clip in {5, 10, 50, none} | IPS clip-high swept with everything else fixed (weights renormalised per trip). | clip 5: 0.1351 [0.1255, 0.1451]; clip 10: 0.1390 [0.1289, 0.1495]; clip 20: 0.1480 [0.1380, 0.1582]; clip 50: 0.1574 [0.1465, 0.1684]; clip none: 0.1603 [0.1496, 0.1711]; clip no_ips: 0.1257 [0.1161, 0.1352] | MEASURED |
| DR8 | 180-day taste half-life and spec interaction weights | +-2x half-life; uniform interaction weights | Rebuild the traveler features with a different taste half-life / interaction weights, refit the ranker on the fixed candidate sets, and score the unbiased holdout; D9 (within-trip, reporting-only) shows the effect on estimator fidelity. | halflife_180d (shipped): NDCG@10 0.1480, D9 0.473; halflife_90d: NDCG@10 0.1487, D9 0.470; halflife_360d: NDCG@10 0.1544, D9 0.474; uniform_weights: NDCG@10 0.1621, D9 0.052 | MEASURED |
| DR9 | Long-tail candidate quota = 50 | quota in {0, 25, 100} | Regenerate the holdout candidate sets with a different long-tail hard-floor quota (retriever scores fixed); score with the SHIPPED booster (trained at quota 50, not refit per quota); raw ranker top-10, no MMR/gates. | quota 0: NDCG@10 0.1483, long-tail share 0.156, candidate recall 0.862; quota 25: NDCG@10 0.1483, long-tail share 0.157, candidate recall 0.868; quota 50: NDCG@10 0.1480, long-tail share 0.160, candidate recall 0.873; quota 100: NDCG@10 0.1469, long-tail share 0.166, candidate recall 0.887 | MEASURED |
| DR10 | alpha = 1.0, beta = 0.7 | alpha x beta grid | Gated multiplicative utility over an alpha x beta grid on the same holdout scoring pass. | Shipped (alpha=1.0, beta=0.7): NDCG@10 0.1296, compat@10 0.8371. NDCG-best cell (alpha=2.0, beta=1.0): 0.1303, compat@10 0.8336. | MEASURED |
| DR11 | Candidate generation = learned retriever + long-tail + interest | The original 6 heuristic channels, and subsets of them | Channel-subset grid at learned K=210 on a train-carved validation split (IPS-weighted logged positives); holdout columns are reporting-only. `none` = the learned retriever alone. Baseline = the original 6-channel heuristic union. | Legacy 6-channel union: recall 0.596 (lift +0.203). Shipped learned+long-tail+interest: recall 0.881 (lift +0.371, 246 candidates/trip). Ranking by true utility would recall 0.991 at the same budget (diagnostic). | MEASURED |

## Seed replication

The full pipeline was regenerated end to end for seeds [42, 43, 44, 45, 46] (a new synthetic dataset, retriever, boosters and calibration each time; `scripts/seed_replication.py`). Seed 42 is the committed run.

| Metric | Mean | SD | Min | Max |
|---|---|---|---|---|
| bias_gap_popularity | 0.0634 | 0.0263 | 0.0243 | 0.1008 |
| bias_gap_primary | -0.0029 | 0.0202 | -0.0311 | 0.0282 |
| candidate_recall_long_tail | 0.8339 | 0.0077 | 0.8272 | 0.8481 |
| candidate_recall_overall | 0.8702 | 0.0022 | 0.8671 | 0.8731 |
| coverage_at_10 | 0.6340 | 0.0249 | 0.6052 | 0.6740 |
| ece_after | 0.0432 | 0.0074 | 0.0296 | 0.0507 |
| longtail_precision | 0.1688 | 0.0163 | 0.1455 | 0.1915 |
| longtail_share | 0.2526 | 0.0319 | 0.2168 | 0.3061 |
| ndcg10_content_cosine | 0.1416 | 0.0037 | 0.1365 | 0.1471 |
| ndcg10_lambdamart_ips | 0.1560 | 0.0074 | 0.1480 | 0.1691 |
| ndcg10_logistic_regression | 0.1230 | 0.0078 | 0.1129 | 0.1337 |
| ndcg10_popularity | 0.0773 | 0.0089 | 0.0616 | 0.0872 |
| pct_of_oracle_ceiling | 0.4195 | 0.0187 | 0.3964 | 0.4527 |
| within_cross_ratio_true_labels | 1.2578 | 0.0296 | 1.2218 | 1.3071 |

## Wall-clock

**reproduce**: 366.8 s total (16-logical-core laptop CPU, no GPU, no network).

| Stage | Seconds |
|---|---|
| generate | 34.3 |
| prepare | 5.8 |
| features | 21.5 |
| candidates | 41.9 |
| gate-dgp | 10.6 |
| train | 102.2 |
| evaluate | 131.0 |
| representation | 13.3 |
| gate-representation | 3.0 |
| compose | 3.2 |

**reproduce-full**: 452.3 s total (16-logical-core laptop CPU, no GPU, no network).

| Stage | Seconds |
|---|---|
| generate | 34.3 |
| prepare | 5.8 |
| features | 21.5 |
| candidates | 41.9 |
| gate-dgp | 10.6 |
| train | 102.2 |
| evaluate | 131.0 |
| representation | 13.3 |
| gate-representation | 3.0 |
| compose | 3.2 |
| recommend | 24.6 |
| scenarios | 25.2 |
| lodo | 35.4 |
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
| 1 | PSEO0457 | Show Insadong Theater | entertainment | 0.0814 | 0.0896 | 0.8707 | 0.3793 | 8.0913% | 0.3725 |
| 2 | PSEO0426 | Mealtime Bukchon Wellness Studio | wellness_spa | 0.0719 | 0.0768 | 0.9101 | 0.3788 | 31.3278% | 0.5269 |
| 3 | PSEO0074 | Neighborhood Bukchon Coffee House | cafe | 0.0806 | 0.0873 | 0.8925 | 0.4420 | 71.9917% | 0.3865 |
| 4 | PSEO0280 | Historic Gangnam Shrine | religious_site | 0.0580 | 0.0640 | 0.8689 | 0.5098 | 99.5851% | -1.1177 |
| 5 | PSEO0277 | Heritage Bukchon Collection | museum | 0.0692 | 0.0768 | 0.8614 | 0.4226 | 3.1120% | 0.1187 |
| 6 | PSEO0331 | Romantic Hongdae Adventure Park | family_activity | 0.0548 | 0.0640 | 0.8020 | 0.3862 | 14.3154% | 0.4634 |
| 7 | PSEO0188 | Dining Insadong Grill | restaurant | 0.0582 | 0.0640 | 0.8732 | 0.4962 | 82.3651% | 0.0727 |
| 8 | PSEO0138 | Greenspace Hongdae Garden | nature_park | 0.0681 | 0.0768 | 0.8421 | 0.4411 | 15.7676% | -0.4128 |
| 9 | PSEO0349 | Kid Approved Itaewon Bar | nightlife | 0.0594 | 0.0768 | 0.6939 | 0.3770 | 47.9253% | 0.5295 |
| 10 | PSEO0232 | Heritage Site Gangnam Temple | historic_site | 0.0590 | 0.0640 | 0.8904 | 0.4106 | 75.5187% | -0.3887 |

**Top signals (rank 1, PSEO0457):** interest_match (1.030), price_fit (0.094), popularity (0.066)

**Explanation (rank 1):** Matches your stated medium budget Popular choice -- busier than 8% of comparable POIs in Seoul Priced above what's typical for your medium budget

### Scenario 2: History & Architecture

- Persona: First-time K-culture visitor to Seoul (foreign, inbound): heritage, museums and the iconic sights, comfortable budget.
- Destination: **seoul**; interests: historic_site, historic, cultural, museum; touristiness_pref: **0.40**
- Budget: high; mobility: public_transport; party: couple; pace: moderate; accessibility needs: none
- 'history' -> {historic_site, historic}; 'architecture' -> {cultural} (no literal 'architecture' tag exists in CATEGORIES+TAGS); 'museums' -> {museum} (exact match). `pace` defaulted to 'moderate' (not specified).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0167 | Nature Filled Itaewon Wellness Studio | wellness_spa | 0.0844 | 0.0873 | 0.9530 | 0.5217 | 90.0415% | -1.2879 |
| 2 | PSEO0071 | Teahouse Bukchon Bakery | cafe | 0.0736 | 0.0768 | 0.9420 | 0.4671 | 88.7967% | -0.7435 |
| 3 | PSEO0457 | Show Insadong Theater | entertainment | 0.0813 | 0.0873 | 0.9037 | 0.4109 | 8.0913% | 0.3725 |
| 4 | PSEO0325 | Historic Itaewon Shrine | religious_site | 0.0606 | 0.0768 | 0.7134 | 0.3835 | 66.1826% | 0.0740 |
| 5 | PSEO0182 | Laid Back Myeongdong Overlook | viewpoint | 0.0749 | 0.0768 | 0.9659 | 0.5196 | 90.8714% | -0.8629 |
| 6 | PSEO0067 | Late Night Hongdae Bar | nightlife | 0.0626 | 0.0768 | 0.7465 | 0.4690 | 98.5477% | -2.0151 |
| 7 | PSEO0331 | Romantic Hongdae Adventure Park | family_activity | 0.0505 | 0.0575 | 0.8324 | 0.4258 | 14.3154% | 0.4634 |
| 8 | PSEO0363 | Dreamy Seongsu Noodle House | restaurant | 0.0598 | 0.0640 | 0.9091 | 0.4378 | 7.0539% | -0.1695 |
| 9 | PSEO0138 | Greenspace Hongdae Garden | nature_park | 0.0599 | 0.0768 | 0.7012 | 0.4453 | 15.7676% | -0.4128 |
| 10 | PSEO0040 | Serene Seongsu Boutique | shopping | 0.0541 | 0.0575 | 0.9178 | 0.4102 | 62.6556% | 0.3393 |

**Top signals (rank 1, PSEO0167):** interest_match (1.271), popularity (0.156), price_fit (0.119)

**Explanation (rank 1):** Popular choice -- busier than 90% of comparable POIs in Seoul Matches your stated high budget More budget-friendly than typical for your high budget

### Scenario 3: Family with young children

- Persona: Foreign family with young children visiting Seoul: stroller, relaxed pace, things kids can do.
- Destination: **seoul**; interests: family_activity, nature_park, family-friendly; touristiness_pref: **0.00**
- Budget: medium; mobility: car; party: family_young_kids; pace: relaxed; accessibility needs: stroller
- 'activities' -> {family_activity}; 'parks' -> {nature_park}; 'interactive experiences' -> {family-friendly} (no literal 'interactive' tag exists). `touristiness_pref` and `mobility` are not specified by spec.md section 15 for this scenario -- defaulted to 0.0 (neutral) and 'car' (a common real choice for a family with young children traveling with a stroller) respectively; `party_type`/accessibility (`stroller`)/`pace` (`relaxed`)/`budget` (`medium`) are exactly as spec.md states.

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0031 | Sun Drenched Bukchon Discovery Center | family_activity | 0.0826 | 0.0873 | 0.9239 | 0.5226 | 67.6349% | -0.4157 |
| 2 | PSEO0074 | Neighborhood Bukchon Coffee House | cafe | 0.0813 | 0.0873 | 0.9035 | 0.4344 | 71.9917% | 0.3865 |
| 3 | PSEO0132 | Screening Itaewon Skyline Point | viewpoint | 0.0746 | 0.0811 | 0.8883 | 0.6002 | 9.7510% | -1.0274 |
| 4 | PSEO0277 | Heritage Bukchon Collection | museum | 0.0655 | 0.0768 | 0.7972 | 0.4101 | 3.1120% | 0.1187 |
| 5 | PSEO0156 | No Frills Seongsu Boutique | shopping | 0.0588 | 0.0640 | 0.8858 | 0.4146 | 42.9461% | 0.3222 |
| 6 | PSEO0303 | Spiritual Gangnam Trail | nature_park | 0.0704 | 0.0768 | 0.8831 | 0.4715 | 72.1992% | -0.6232 |
| 7 | PSEO0410 | Authentic Seongsu Playhouse | entertainment | 0.0586 | 0.0640 | 0.8819 | 0.4790 | 86.5145% | -0.8836 |
| 8 | PSEO0088 | Gallery Like Gangnam Monument | historic_site | 0.0587 | 0.0640 | 0.8850 | 0.4995 | 95.2282% | -0.6739 |
| 9 | PSEO0003 | Spiritual Gangnam Cathedral | religious_site | 0.0648 | 0.0768 | 0.7849 | 0.3883 | 4.1494% | -0.7932 |
| 10 | PSEO0123 | Bohemian Myeongdong Grill | restaurant | 0.0576 | 0.0613 | 0.9148 | 0.5132 | 82.1577% | -0.5479 |

**Top signals (rank 1, PSEO0031):** interest_match (0.983), price_fit (0.116), popularity (0.106)

**Explanation (rank 1):** Strong match with your stated interest in family-friendly Matches your stated medium budget Popular choice -- busier than 68% of comparable POIs in Seoul Only open 77% of your trip's plausible visiting hours

### Scenario 4: Diagnostic (Scenario 1 profile, touristiness_pref flipped to the opposite extreme)

- Persona: The same traveler as scenario 1 asking for the opposite: the iconic, must-see, first-timer version of Seoul.
- Destination: **seoul**; interests: local, foodie, authentic; touristiness_pref: **0.80**
- Budget: medium; mobility: public_transport; party: solo; pace: moderate; accessibility needs: none
- Base scenario: 1 (Local Experience). Every field held identical to that scenario's profile except `touristiness_pref`, flipped from -0.8 to +0.8 (opposite sign, same extreme magnitude).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0343 | Luxurious Hongdae Retreat | wellness_spa | 0.0876 | 0.0896 | 0.9681 | 0.5004 | 85.5809% | -1.5229 |
| 2 | PSEO0132 | Screening Itaewon Skyline Point | viewpoint | 0.0706 | 0.0768 | 0.8876 | 0.4442 | 9.7510% | -1.0274 |
| 3 | PSEO0139 | Fun Filled Bukchon Adventure Park | family_activity | 0.0704 | 0.0768 | 0.8830 | 0.5172 | 87.3444% | -0.9920 |
| 4 | PSEO0074 | Neighborhood Bukchon Coffee House | cafe | 0.0709 | 0.0768 | 0.8925 | 0.4015 | 71.9917% | 0.3865 |
| 5 | PSEO0325 | Historic Itaewon Shrine | religious_site | 0.0689 | 0.0768 | 0.8567 | 0.3908 | 66.1826% | 0.0740 |
| 6 | PSEO0138 | Greenspace Hongdae Garden | nature_park | 0.0584 | 0.0659 | 0.8421 | 0.5584 | 15.7676% | -0.4128 |
| 7 | PSEO0113 | Museum Insadong Exhibition Hall | museum | 0.0568 | 0.0640 | 0.8433 | 0.4795 | 92.9461% | -0.4470 |
| 8 | PSEO0363 | Dreamy Seongsu Noodle House | restaurant | 0.0559 | 0.0613 | 0.8759 | 0.4862 | 7.0539% | -0.1695 |
| 9 | PSEO0410 | Authentic Seongsu Playhouse | entertainment | 0.0521 | 0.0575 | 0.8705 | 0.5049 | 86.5145% | -0.8836 |
| 10 | PSEO0118 | Wallet Friendly Bukchon Arcade | shopping | 0.0535 | 0.0553 | 0.9543 | 0.5133 | 9.5436% | -0.9778 |

**Top signals (rank 1, PSEO0343):** interest_match (1.254), popularity (0.161), price_fit (0.125)

**Explanation (rank 1):** Popular choice -- busier than 86% of comparable POIs in Seoul Matches your stated medium budget More budget-friendly than typical for your medium budget

### Pairwise top-10 Jaccard overlap across scenarios

| | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 |
|---|---|---|---|---|
| Scenario 1 | 1.000 | 0.176 | 0.111 | 0.111 |
| Scenario 2 | 0.176 | 1.000 | 0.000 | 0.176 |
| Scenario 3 | 0.111 | 0.000 | 1.000 | 0.176 |
| Scenario 4 | 0.111 | 0.176 | 0.176 | 1.000 |

**Diagnostic scenario 4 vs base scenario 1** (touristiness_pref flipped to the opposite extreme, everything else held constant): top-10 Jaccard overlap = **0.111** (spec.md section 15 expects this to be low, i.e. &le; 0.25, **MET**).

Low overlap demonstrates the ranking is driven by the preference signal, not profile confounds, as spec.md section 15 expects.

