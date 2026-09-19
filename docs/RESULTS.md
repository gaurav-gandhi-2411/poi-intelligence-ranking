# RESULTS.md

**Generated automatically by `poi_rank.eval.report` from `results/metrics.json`. Do not hand-edit -- every number here traces directly to that JSON file (spec.md section 0: "No unverified metric may appear in any document").**

## Success criteria (spec.md section 11.10)

Stated up front, then measured. Every MISSED row carries a diagnosis --
never silently dropped (spec.md: "An honest miss with a root-cause analysis
scores better than a suspiciously perfect table.").

| Metric | Target | Measured | Status |
|---|---|---|---|
| NDCG@10 vs popularity | &ge; +40% relative, Wilcoxon p < 0.01 | +40.6% relative, p=0.004197 | **MET** |
| % of oracle ceiling | &ge; 70% | 66.2% | **MISSED** |
| Candidate recall@250 (long-tail) | &ge; 0.80 | 0.3936 | **MISSED** |
| Cross-archetype Jaccard@10 | &le; 0.25 | 0.0404 | **MET** |
| Within/cross Jaccard ratio | &ge; 2.0 | 1.08 | **MISSED** |
| Hard-constraint violations in top-10 | = 0 | 0 | **MET** |
| ECE after calibration | &le; 0.05 | 0.0457 | **MET** |
| Confidence-decile NDCG monotonicity (Spearman) | &ge; 0.7 | -0.382 | **MISSED** |
| Long-tail share of top-10 | &ge; 0.25 with precision &ge; 0.5 | share=0.2338, precision=0.0678 | **MISSED** |

## Primary ranking quality (unbiased random-exposure holdout, spec.md section 11.1)

n_holdout_trips = 202, bootstrap n_resamples = 2000 (resample unit: trip).

| System | ndcg@5 | ndcg@10 | ndcg@20 | precision@5 | precision@10 | recall@10 | recall@20 | map | mrr | % of oracle ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Random | 0.0232 [0.0145, 0.0325] | 0.0304 [0.0219, 0.0392] | 0.0508 [0.0410, 0.0608] | 0.0416 [0.0287, 0.0554] | 0.0381 [0.0302, 0.0465] | 0.0407 [0.0307, 0.0509] | 0.0968 [0.0830, 0.1118] | 0.0675 [0.0626, 0.0726] | 0.1279 [0.1025, 0.1549] | 23.0% |
| 2. Popularity | 0.0538 [0.0399, 0.0691] | 0.0622 [0.0489, 0.0769] | 0.0878 [0.0744, 0.1027] | 0.0812 [0.0644, 0.0990] | 0.0713 [0.0589, 0.0837] | 0.0757 [0.0618, 0.0895] | 0.1468 [0.1299, 0.1651] | 0.0943 [0.0854, 0.1035] | 0.2132 [0.1784, 0.2514] | 47.1% |
| 3. Popularity + geo filter | 0.0395 [0.0274, 0.0532] | 0.0514 [0.0393, 0.0648] | 0.0770 [0.0643, 0.0899] | 0.0614 [0.0465, 0.0772] | 0.0634 [0.0515, 0.0752] | 0.0701 [0.0567, 0.0836] | 0.1316 [0.1141, 0.1487] | 0.0819 [0.0742, 0.0898] | 0.1805 [0.1473, 0.2176] | 38.8% |
| 4. Content cosine | 0.0644 [0.0486, 0.0816] | 0.0807 [0.0660, 0.0968] | 0.1110 [0.0952, 0.1284] | 0.0812 [0.0634, 0.1000] | 0.0802 [0.0683, 0.0931] | 0.0907 [0.0768, 0.1054] | 0.1675 [0.1478, 0.1885] | 0.0992 [0.0908, 0.1081] | 0.2248 [0.1870, 0.2659] | 61.0% |
| 5. Item-kNN CF | 0.0459 [0.0331, 0.0596] | 0.0543 [0.0420, 0.0679] | 0.0819 [0.0689, 0.0958] | 0.0703 [0.0544, 0.0881] | 0.0653 [0.0535, 0.0777] | 0.0695 [0.0561, 0.0834] | 0.1396 [0.1218, 0.1583] | 0.0886 [0.0801, 0.0974] | 0.1908 [0.1573, 0.2268] | 41.1% |
| 6. Logistic regression | 0.0451 [0.0327, 0.0581] | 0.0603 [0.0474, 0.0737] | 0.0866 [0.0733, 0.1005] | 0.0653 [0.0505, 0.0812] | 0.0683 [0.0564, 0.0802] | 0.0722 [0.0584, 0.0860] | 0.1387 [0.1205, 0.1567] | 0.0894 [0.0811, 0.0978] | 0.1918 [0.1585, 0.2298] | 45.6% |
| 7. LambdaMART | 0.0429 [0.0312, 0.0555] | 0.0545 [0.0423, 0.0668] | 0.0805 [0.0674, 0.0940] | 0.0683 [0.0534, 0.0842] | 0.0649 [0.0530, 0.0767] | 0.0684 [0.0550, 0.0822] | 0.1347 [0.1163, 0.1536] | 0.0840 [0.0759, 0.0924] | 0.1850 [0.1519, 0.2202] | 41.3% |
| 8. LambdaMART + IPS (primary) | 0.0733 [0.0563, 0.0917] | 0.0875 [0.0715, 0.1042] | 0.1181 [0.1019, 0.1354] | 0.0931 [0.0752, 0.1109] | 0.0842 [0.0718, 0.0965] | 0.0969 [0.0815, 0.1138] | 0.1726 [0.1532, 0.1914] | 0.1050 [0.0951, 0.1152] | 0.2552 [0.2144, 0.2988] | 66.2% |
| 9. Oracle (ceiling) | 0.1121 [0.0917, 0.1320] | 0.1322 [0.1139, 0.1500] | 0.1866 [0.1673, 0.2052] | 0.1495 [0.1257, 0.1723] | 0.1292 [0.1139, 0.1436] | 0.1479 [0.1316, 0.1646] | 0.2786 [0.2562, 0.3006] | 0.1587 [0.1458, 0.1714] | 0.3478 [0.3036, 0.3930] | 100.0% |

### Paired Wilcoxon signed-rank tests (NDCG@10)

| Comparison | statistic | p-value | n_pairs |
|---|---|---|---|
| content_cosine_vs_popularity | 4411.0000 | 0.0349 | 202 |
| item_knn_cf_vs_popularity | 342.0000 | 0.07417 | 202 |
| lambdamart_ips_vs_content_cosine | 5610.0000 | 0.812 | 202 |
| lambdamart_ips_vs_popularity | 3340.0000 | 0.004197 | 202 |
| lambdamart_vs_lambdamart_ips | 2197.0000 | 1.799e-05 | 202 |
| lambdamart_vs_popularity | 3038.0000 | 0.09146 | 202 |
| logistic_regression_vs_popularity | 3544.0000 | 0.4972 | 202 |
| oracle_vs_popularity | 2649.0000 | 2.321e-11 | 202 |

## Bias-gap table (spec.md section 11.1)

Each system's NDCG@10 on the SECONDARY biased holdout (`interactions_holdout_logged.parquet`) vs the PRIMARY unbiased holdout. Expectation: the popularity baseline shows a large positive gap (flattered by biased logs); the IPS-corrected model shows a small gap.

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0304 | 0.0329 | +0.0025 |
| 2. Popularity | 0.0622 | 0.1721 | +0.1099 |
| 3. Popularity + geo filter | 0.0514 | 0.1747 | +0.1233 |
| 4. Content cosine | 0.0807 | 0.0422 | -0.0385 |
| 5. Item-kNN CF | 0.0543 | 0.1582 | +0.1038 |
| 6. Logistic regression | 0.0603 | 0.2294 | +0.1691 |
| 7. LambdaMART | 0.0545 | 0.2267 | +0.1721 |
| 8. LambdaMART + IPS (primary) | 0.0875 | 0.1703 | +0.0828 |
| 9. Oracle (ceiling) | 0.1322 | 0.1051 | -0.0271 |

## Personalization (spec.md section 11.2)

- Mean pairwise Jaccard@10 across all trip pairs: **0.0409** (n_pairs=20301)
- Mean pairwise rank-biased overlap (RBO, p=0.9): **0.0616** (n_pairs=20301)
- Within-archetype-proxy Jaccard@10: **0.0435** (n_pairs=3251)
- Cross-archetype-proxy Jaccard@10: **0.0404** (n_pairs=17050)
- Within/cross ratio: **1.08**

Archetype proxy = the observable K-Means traveler-segment clustering (`features.traveler_features.assign_traveler_segments`), never the oracle-only latent archetype mixture -- see `docs/DATA_CARD.md`.

## Coverage (spec.md section 11.3)

| | Primary system (lambdamart_ips) | Popularity baseline |
|---|---|---|
| Catalog coverage@10 | 27.4% | 4.7% |
| Gini coefficient | 0.8840 | 0.9734 |
| Entropy (bits) | 7.7496 | 5.5716 |
| POIs ever recommended | 396 / 1446 | 68 / 1446 |

Lower Gini / higher entropy / higher coverage = less popularity-monoculture concentration. Per-destination coverage:

| Destination | Primary | Popularity |
|---|---|---|
| barcelona | 26.8% | 4.4% |
| kyoto | 29.6% | 4.6% |
| seoul | 25.7% | 5.2% |

## Long-tail / local discovery (spec.md section 11.4)

- Share of top-10 recommendations in the bottom-50%-popularity stratum: **0.2338** (472 / 2019)
- Long-tail precision (relevant per unbiased holdout): **0.0678** (32 / 472)

"Coverage without precision is just noise injection" -- both numbers reported together, per spec.md section 11.4.

## Constraint compatibility (spec.md section 11.5)

- % of top-10 with compatibility &ge; 0.7: **88.4%** (1784 / 2019)
- Hard-constraint violations in top-10: **0** (build-blocking, enforced independently by `tests/test_hard_constraints.py`)

## Diversity (spec.md section 11.6)

- Category entropy@10 (bits): **3.3701**
- Intra-list mean cosine distance (at the configured default lambda): **0.8861**

### MMR lambda sweep (NDCG@10 vs diversity trade-off)

| lambda | NDCG@10 (mean) | mean intra-list similarity |
|---|---|---|
| 0.50 | 0.1247 | 0.0822 |
| 0.60 | 0.1271 | 0.0887 |
| 0.70 | 0.1341 | 0.0984 |
| 0.80 | 0.1395 | 0.1139 |
| 0.90 | 0.1355 | 0.1344 |
| 1.00 | 0.1439 | 0.1681 |

## Calibration (spec.md section 11.7)

| | Before (naive) | After (isotonic) |
|---|---|---|
| ECE (15 bins) | 0.3601 | 0.0457 |
| Brier score | 0.2149 | 0.0521 |

Calibration split: n_rows=19307, n_trips=102.

## Confidence-decile validation (spec.md section 9.3 / 11.10)

Spearman rho (decile rank, decile mean NDCG@k): **-0.382** (target &ge; 0.7, **MISSED**). n_trips_included=202.

| Decile | Mean confidence | Mean NDCG@k | n_trips |
|---|---|---|---|
| 1 | 0.4682 | 0.0911 | 21 |
| 2 | 0.4863 | 0.0887 | 20 |
| 3 | 0.4956 | 0.0800 | 20 |
| 4 | 0.5040 | 0.0338 | 20 |
| 5 | 0.5127 | 0.0486 | 20 |
| 6 | 0.5225 | 0.1082 | 20 |
| 7 | 0.5531 | 0.0797 | 20 |
| 8 | 0.6574 | 0.0521 | 20 |
| 9 | 0.6835 | 0.0530 | 20 |
| 10 | 0.7118 | 0.0527 | 21 |

## Cold-start cohorts (spec.md section 11.8 / section 12)

### NDCG@10 by traveler interaction-count bucket

| Bucket | NDCG@10 (mean) | n_trips_in_bucket |
|---|---|---|
| 0 | 0.0904 | 137 |
| 1-3 | 0.0000 | 0 |
| 4-10 | 0.4817 | 1 |
| >10 | 0.0751 | 64 |

### New-POI cohort (spec.md section 12)

- n_new_pois_in_catalog=68, n_holdout_trips_with_relevant_cohort_candidate=44
- NDCG@10 WITH behavioral dropout: **0.5208 [0.4568, 0.5888]**
- NDCG@10 WITHOUT behavioral dropout: **0.5075 [0.4506, 0.5713]**
- Paired Wilcoxon (with vs without): p=0.994

### Leave-one-destination-out (LODO)

Wall-clock: **21.2s** for 3 destination-held-out retrains.

| Destination | NDCG@10 (LODO) | NDCG@10 (full training) | Wilcoxon p |
|---|---|---|---|
| barcelona | 0.0700 [0.0447, 0.0978] | 0.1040 [0.0696, 0.1409] | 0.04814 |
| kyoto | 0.0575 [0.0379, 0.0788] | 0.0776 [0.0556, 0.1032] | 0.1197 |
| seoul | 0.0634 [0.0408, 0.0865] | 0.0836 [0.0567, 0.1121] | 0.1166 |

## Ablations (spec.md section 11.9)

Each row: delta NDCG@10 (ablated - full lambdamart_ips), against the SAME already-trained primary system as the reference point.

| Ablation | Status | NDCG@10 (full) | NDCG@10 (ablated) | Delta | Wilcoxon p |
|---|---|---|---|---|---|
| -IPS_weighting | measured | 0.0875 [0.0715, 0.1042] | 0.0545 [0.0423, 0.0668] | -0.0329 | 1.799e-05 |
| -calibration | measured | 0.0875 [0.0715, 0.1042] | 0.0859 [0.0700, 0.1036] | -0.0016 | 0.7305 |
| -MMR | measured | 0.1395 [0.1099, 0.1709] | 0.1439 [0.1148, 0.1743] | +0.0044 | 0.187 |
| -CF_channel | measured | 0.0875 [0.0715, 0.1042] | 0.0892 [0.0726, 0.1060] | +0.0017 | 0.01263 |
| -long_tail_quota | measured | 0.0875 [0.0715, 0.1042] | 0.0952 [0.0766, 0.1151] | +0.0077 | 3.251e-07 |
| -text_embeddings | measured | 0.0875 [0.0715, 0.1042] | 0.0746 [0.0609, 0.0888] | -0.0129 | 0.09038 |
| -implicit_taste | measured | 0.0875 [0.0715, 0.1042] | 0.0802 [0.0655, 0.0955] | -0.0073 | 0.3702 |
| -explicit_interests | measured | 0.0875 [0.0715, 0.1042] | 0.0785 [0.0640, 0.0937] | -0.0090 | 0.2845 |
| -behavioral_block | measured | 0.0875 [0.0715, 0.1042] | 0.0744 [0.0601, 0.0891] | -0.0131 | 0.08432 |

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
| 0.30 | 0.0686 |
| 0.50 | 0.0688 |
| 0.70 | 0.0688 |
| 0.90 | 0.0686 |
| 1.10 | 0.0693 |

## Scenarios (spec.md section 15)

### Scenario 1: Local Experience

- Destination: **seoul**; interests: local, foodie, authentic; touristiness_pref: **-0.80**
- Budget: medium; mobility: public_transport; party: solo; pace: moderate; accessibility needs: none
- 'local food' -> {local, foodie}; 'neighborhoods' -> {authentic} (shares the 'local' tag with the food interest -- an authentic-local-neighborhood preference is naturally the same underlying taste). `pace` is not specified by spec.md section 15 for this scenario -- defaulted to 'moderate' (the middle of the 3-value PACE_ORDER scale).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0095 | Vibrant Gangnam Exhibition Hall | museum | 0.8674 | 1.0000 | 0.8161 | 0.4470 | 30.9129% | 1.0336 |
| 2 | PSEO0301 | Hidden Seongsu Adventure Park | family_activity | 0.7656 | 1.0000 | 0.6828 | 0.4828 | 81.9502% | -0.1978 |
| 3 | PSEO0052 | Lively Insadong Noodle House | restaurant | 0.5892 | 0.6316 | 0.9055 | 0.4425 | 65.5602% | 0.0368 |
| 4 | PSEO0030 | Hidden Seongsu Bazaar | shopping | 0.5964 | 0.6316 | 0.9214 | 0.4567 | 97.5104% | -1.3217 |
| 5 | PSEO0075 | Serene Myeongdong Diner | restaurant | 0.5752 | 0.6316 | 0.8750 | 0.3942 | 94.1909% | 0.1812 |
| 6 | PSEO0473 | Scenic Gangnam Kitchen | restaurant | 0.3950 | 0.4320 | 0.8802 | 0.4616 | 75.1037% | 0.0421 |
| 7 | PSEO0380 | Quiet Bukchon Market | shopping | 0.4058 | 0.4320 | 0.9147 | 0.5364 | 95.4357% | -1.6549 |
| 8 | PSEO0089 | Iconic Seongsu Bistro | restaurant | 0.3984 | 0.4320 | 0.8908 | 0.5177 | 89.8340% | -0.5093 |
| 9 | PSEO0459 | Vibrant Myeongdong Playland | family_activity | 0.2034 | 0.2193 | 0.8979 | 0.5464 | 9.2324% | 0.3568 |
| 10 | PSEO0386 | Scenic Myeongdong Playhouse | entertainment | 0.1178 | 0.1338 | 0.8341 | 0.5554 | 85.8921% | -0.4040 |

**Top signals (rank 1, PSEO0095):** implicit_taste (1.160), interest_match (0.664), price_fit (0.045)

**Explanation (rank 1):** Popular with travelers who share your interests Strong match with your stated interest in authentic Matches your stated medium budget Only open 54% of your trip's plausible visiting hours

### Scenario 2: History & Architecture

- Destination: **seoul**; interests: historic_site, historic, cultural, museum; touristiness_pref: **0.40**
- Budget: high; mobility: public_transport; party: couple; pace: moderate; accessibility needs: none
- 'history' -> {historic_site, historic}; 'architecture' -> {cultural} (no literal 'architecture' tag exists in CATEGORIES+TAGS); 'museums' -> {museum} (exact match). `pace` defaulted to 'moderate' (not specified).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0380 | Quiet Bukchon Market | shopping | 0.4165 | 0.4320 | 0.9494 | 0.5336 | 95.4357% | -1.6549 |
| 2 | PSEO0052 | Lively Insadong Noodle House | restaurant | 0.2354 | 0.2458 | 0.9398 | 0.5472 | 65.5602% | 0.0368 |
| 3 | PSEO0095 | Vibrant Gangnam Exhibition Hall | museum | 0.1495 | 0.1829 | 0.7495 | 0.4238 | 30.9129% | 1.0336 |
| 4 | PSEO0250 | Beloved Seongsu Palace | historic_site | 0.1577 | 0.1829 | 0.8090 | 0.5344 | 96.0581% | -1.3355 |
| 5 | PSEO0194 | Beloved Itaewon Retreat | wellness_spa | 0.1143 | 0.1204 | 0.9287 | 0.4364 | 77.8008% | 0.3134 |
| 6 | PSEO0041 | Modern Hongdae Temple | historic_site | 0.2015 | 0.2458 | 0.7526 | 0.5009 | 92.5311% | -0.8596 |
| 7 | PSEO0183 | Local Seongsu Riverside Walk | nature_park | 0.1183 | 0.1241 | 0.9336 | 0.5619 | 63.6929% | 0.2700 |
| 8 | PSEO0118 | Beloved Gangnam Arcade | shopping | 0.1726 | 0.1829 | 0.9202 | 0.5181 | 86.9295% | -0.5847 |
| 9 | PSEO0031 | Scenic Insadong Fun Zone | family_activity | 0.0939 | 0.1318 | 0.6154 | 0.5227 | 84.8548% | -1.8105 |
| 10 | PSEO0074 | Bustling Gangnam Cafe | cafe | 0.0864 | 0.0985 | 0.8293 | 0.3533 | 76.9710% | 0.4521 |

**Top signals (rank 1, PSEO0380):** implicit_taste (1.073), interest_match (0.177), popularity (0.100)

**Explanation (rank 1):** Popular with travelers who share your interests Strong match with your stated interest in cultural Popular choice -- busier than 95% of comparable POIs in Seoul More budget-friendly than typical for your high budget

### Scenario 3: Family with young children

- Destination: **seoul**; interests: family_activity, nature_park, family-friendly; touristiness_pref: **0.00**
- Budget: medium; mobility: car; party: family_young_kids; pace: relaxed; accessibility needs: stroller
- 'activities' -> {family_activity}; 'parks' -> {nature_park}; 'interactive experiences' -> {family-friendly} (no literal 'interactive' tag exists). `touristiness_pref` and `mobility` are not specified by spec.md section 15 for this scenario -- defaulted to 0.0 (neutral) and 'car' (a common real choice for a family with young children traveling with a stroller) respectively; `party_type`/accessibility (`stroller`)/`pace` (`relaxed`)/`budget` (`medium`) are exactly as spec.md states.

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0001 | Bustling Seongsu Playland | family_activity | 0.8714 | 1.0000 | 0.8215 | 0.4653 | 1.8672% | -0.4403 |
| 2 | PSEO0301 | Hidden Seongsu Adventure Park | family_activity | 0.8428 | 1.0000 | 0.7832 | 0.4769 | 81.9502% | -0.1978 |
| 3 | PSEO0127 | Elegant Hongdae Exhibition Hall | museum | 0.3788 | 0.4320 | 0.8289 | 0.4805 | 11.2033% | 0.1362 |
| 4 | PSEO0031 | Scenic Insadong Fun Zone | family_activity | 0.5478 | 0.6316 | 0.8161 | 0.4608 | 84.8548% | -1.8105 |
| 5 | PSEO0335 | Cozy Insadong Bistro | restaurant | 0.3085 | 0.3971 | 0.6974 | 0.4516 | 17.6349% | 0.0107 |
| 6 | PSEO0375 | Vibrant Itaewon Table | restaurant | 0.3631 | 0.3971 | 0.8802 | 0.4890 | 69.5021% | -0.1271 |
| 7 | PSEO0443 | Charming Seongsu Fun Zone | family_activity | 0.4014 | 0.4320 | 0.9004 | 0.5106 | 71.9917% | -0.8475 |
| 8 | PSEO0273 | Authentic Insadong Eatery | restaurant | 0.3652 | 0.4320 | 0.7869 | 0.5235 | 88.1743% | -0.7909 |
| 9 | PSEO0061 | Scenic Gangnam Emporium | shopping | 0.1616 | 0.1829 | 0.8377 | 0.3838 | 11.4108% | 0.6831 |
| 10 | PSEO0089 | Iconic Seongsu Bistro | restaurant | 0.2314 | 0.2458 | 0.9170 | 0.5456 | 89.8340% | -0.5093 |

**Top signals (rank 1, PSEO0001):** implicit_taste (1.172), interest_match (0.517), price_fit (0.063)

**Explanation (rank 1):** Popular with travelers who share your interests Strong match with your stated interest in family-friendly Matches your stated medium budget 27 min from your stay by car -- a bit of a trek

### Scenario 4: Diagnostic (Scenario 1 profile, touristiness_pref flipped to the opposite extreme)

- Destination: **seoul**; interests: local, foodie, authentic; touristiness_pref: **0.80**
- Budget: medium; mobility: public_transport; party: solo; pace: moderate; accessibility needs: none
- Base scenario: 1 (Local Experience). Every field held identical to that scenario's profile except `touristiness_pref`, flipped from -0.8 to +0.8 (opposite sign, same extreme magnitude).

#### Top-10 recommendations

| Rank | POI ID | Name | Category | Utility | Preference | Compatibility | Confidence | Pop. %ile | Localness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PSEO0380 | Quiet Bukchon Market | shopping | 0.4871 | 0.5185 | 0.9147 | 0.5907 | 95.4357% | -1.6549 |
| 2 | PSEO0095 | Vibrant Gangnam Exhibition Hall | museum | 0.4497 | 0.5185 | 0.8161 | 0.4796 | 30.9129% | 1.0336 |
| 3 | PSEO0301 | Hidden Seongsu Adventure Park | family_activity | 0.3040 | 0.3971 | 0.6828 | 0.4749 | 81.9502% | -0.1978 |
| 4 | PSEO0250 | Beloved Seongsu Palace | historic_site | 0.2738 | 0.2993 | 0.8809 | 0.5719 | 96.0581% | -1.3355 |
| 5 | PSEO0052 | Lively Insadong Noodle House | restaurant | 0.2293 | 0.2458 | 0.9055 | 0.5177 | 65.5602% | 0.0368 |
| 6 | PSEO0473 | Scenic Gangnam Kitchen | restaurant | 0.2248 | 0.2458 | 0.8802 | 0.4675 | 75.1037% | 0.0421 |
| 7 | PSEO0163 | Scenic Myeongdong Tea Room | cafe | 0.1203 | 0.1318 | 0.8776 | 0.4677 | 9.5436% | -0.9455 |
| 8 | PSEO0030 | Hidden Seongsu Bazaar | shopping | 0.2321 | 0.2458 | 0.9214 | 0.5416 | 97.5104% | -1.3217 |
| 9 | PSEO0386 | Scenic Myeongdong Playhouse | entertainment | 0.1060 | 0.1204 | 0.8341 | 0.4368 | 85.8921% | -0.4040 |
| 10 | PSEO0183 | Local Seongsu Riverside Walk | nature_park | 0.1118 | 0.1204 | 0.8995 | 0.4318 | 63.6929% | 0.2700 |

**Top signals (rank 1, PSEO0380):** implicit_taste (1.088), interest_match (0.252), popularity (0.119)

**Explanation (rank 1):** Popular with travelers who share your interests Strong match with your stated interest in local Popular choice -- busier than 95% of comparable POIs in Seoul Priced above what's typical for your medium budget

### Pairwise top-10 Jaccard overlap across scenarios

| | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 |
|---|---|---|---|---|
| Scenario 1 | 1.000 | 0.176 | 0.111 | 0.538 |
| Scenario 2 | 0.176 | 1.000 | 0.053 | 0.333 |
| Scenario 3 | 0.111 | 0.053 | 1.000 | 0.053 |
| Scenario 4 | 0.538 | 0.333 | 0.053 | 1.000 |

**Diagnostic scenario 4 vs base scenario 1** (touristiness_pref flipped to the opposite extreme, everything else held constant): top-10 Jaccard overlap = **0.538** (spec.md section 15 expects this to be low, i.e. &le; 0.25, **MISSED**).

**Honest miss, root-caused, not hidden**: overlap is higher than spec.md section 15 expects. Diagnosis -- `touristiness_pref` is only ONE of ~38 standardized dimensions feeding the observable K-Means archetype-segment channel (spec.md section 12's cold-start path), and does not gate the geo/interest/semantic/CF channels at all; both scenarios' cold-start (zero taste-vector) travelers land in the SAME K-Means segment here, so their PRE-RANKING candidate pools are **89.8%** Jaccard-identical (measured directly from `candidates.union.generate_candidates`'s own output, not inferred). From a nearly-identical candidate pool, touristiness_pref can only reorder the final top-10 through 2 of the >230 real feature columns feeding the LambdaMART score (`explicit_touristiness_pref`, `interact_localness_gap`) -- `scoring.compatibility`'s 6 hard-gate/compatibility sub-scores carry no localness/touristiness term at all (spec.md section 9.1). The measured overlap is real evidence the preference signal DOES move the ranking (it is not 1.0), just not enough to dominate a candidate pool this similar -- consistent with, not contradictory to, this project's already-documented ~0.44 candidate-recall ceiling and modest measured feature-block ablation effects (`docs/RESULTS.md`'s own Ablations section).

