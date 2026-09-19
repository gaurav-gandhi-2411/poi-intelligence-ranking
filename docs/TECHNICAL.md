# TECHNICAL.md — `poi-intelligence-ranking`

> **Generated document.** Prose is hand-written in `docs/TECHNICAL.md.tmpl`; every number and
> table is resolved from `results/metrics.json` by `poi_rank.eval.report`
> (`report_templates.py`) — a token that cannot be resolved fails the build, so no figure here
> can be stale or hand-typed. `docs/RESULTS.md` is the fully generated results record;
> `docs/DATA_CARD.md` is the append-only log of every dataset decision.

This is the ML-reasoning and engineering-decision record. It covers what the audit checklist
requires — problem formulation, the circularity defense, every architectural choice, why
LambdaMART rather than a two-tower ranker, why the scoring layer is multiplicative rather than
the brief's additive example — and, since the remediation, a **Decision Register** (section 12)
in which every choice not dictated by the assignment carries a measured number or an explicit
`NOT RUN`.

## 0. What the simulator does and does not license

All data is synthetic (three destinations; **Seoul leads**, the personas are inbound foreign
travelers). Every result below is a property of *this* simulator. Three limits matter when
reading it:

1. The DGP's labels are generated from a utility that the simulator author wrote; a model
   scoring well means it recovered that utility from observable columns, not that it would do
   so on real behaviour. The defenses (section 2) make the number *non-circular*, not
   *transferable*.
2. POI text is generated from per-dimension phrase pools. Text-derived fidelity numbers (D10,
   D9, D11, the TF-IDF-vs-MiniLM comparison) describe that synthetic text; whether they transfer to real POI
   descriptions is untested, and they are not statements about encoders in general.
3. The labels do not penalise a hard-constraint violation (a closed or inaccessible POI is
   still "engaged with" if the simulator's utility says so), so NDCG cannot show the cost of
   violations; the constraint guarantee is enforced and measured separately (section 7).

## 1. Problem formulation

The brief frames POI recommendation as a **two-factor** problem: a POI is a good
recommendation only if it is both *preferred* (relevant to the traveler's taste) and
*compatible* (it fits the trip's hard and soft constraints — budget, hours, accessibility,
party, mobility, reservation lead time). This project is genuinely two-stage:
`src/poi_rank/models/` produces relevance, `src/poi_rank/scoring/` produces compatibility and
combines the two (section 7).

**Exposure bias** is the second first-class concern. Logged interactions conflate "the
traveler liked it" with "the serving policy showed it". The DGP simulates two exposure policies
(a popularity-biased training log and a uniform-random evaluation log) so the bias is
measurable and correctable rather than hand-waved. Measured on the bias-gap table below:
popularity's NDCG@10 moves by +0.0826 between the unbiased and biased
holdouts (it is flattered by the biased log), the primary IPS-corrected model by
+0.0096.

**The commercial objective is local / long-tail discovery, not raw NDCG.** The incumbents in
inbound-travel already rank by popularity, so a popularity-shaped list is table stakes. The
differentiating claim is surfacing the genuinely relevant, non-obvious POI, and it has to be
backed by long-tail **precision**, not just share. Measured on the unbiased holdout, top-10
lists: long-tail share 0.2234 (popularity ranker:
0.0000), catalog coverage@10
64.9% (popularity:
5.5%); long-tail precision
0.1455 on 1478 long-tail recommendations —
below the 0.40 target, diagnosed in the scorecard (section 10).

## 2. DGP / circularity defense

Synthetic data means the same team writes the generator and the model evaluated against it.
Without a structural firewall a model can simply re-learn the generator. Defenses, all
independently tested:

1. **The DGP firewall.** `src/poi_rank/datagen/**` never imports `features/`, `models/`,
   `candidates/` or `scoring/` (`tests/test_firewall.py`, AST-based). Each later phase has its
   own firewall test (`tests/test_firewall_*.py`); `candidates/` in particular may import
   `features/` but never `models/` or `scoring/`, which is why the shared pair-frame assembly
   lives in `features/pair_frame.py`.
2. **Oracle isolation.** The true taste vectors, POI latent quality/localness/semantic vectors,
   the true archetype mixtures and the noise-free utility `u(t,p)` live only in
   `data/synthetic/_oracle/`, readable solely by `eval/oracle.py`
   (`tests/test_oracle_isolation.py` is a raw text scan that allow-lists exactly one writer and
   one reader).
3. **Stated interests are a lossy projection of latent taste** (top-k labels omitted or
   replaced at random), so an explicit-only model cannot reach the ceiling.
4. **Gate-A is frozen before any model work; Gate-B's blocking rows are oracle-free; the
   oracle never touched a decision.** The simulator was calibrated against a blocking
   acceptance gate (below) *before* features/candidates/models were touched, and that
   calibration is documented in `docs/DATA_CARD.md` (it tunes the simulator, not the model).
   Every representation and retrieval hyperparameter was then selected on a **train-carved
   validation split** (`results/parts/a3_*.json`); the oracle-based numbers (D9, D11,
   oracle-relevance recall, the ceiling) are reporting-only, computed once after the selection
   was frozen.

**Gate-A (simulator quality)** and **Gate-B (representation / candidate generation)**:

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

**Two exposure logs, one primary evaluation set.** `interactions_train.parquet` uses a
popularity-biased policy; `interactions_holdout_random.parquet` uses uniform-random exposure
over a wide slate and is the **primary** evaluation population
(671 holdout trips); `interactions_holdout_logged.parquet` repeats the
biased policy over the holdout window only to compute the bias gap. Every logged interaction
carries its `p_expose`, which is what makes IPS possible.

**The bias-gap table** (every system, biased vs unbiased holdout):

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0589 | 0.0281 | -0.0309 |
| 2. Popularity | 0.0616 | 0.1442 | +0.0826 |
| 3. Popularity + geo filter | 0.0605 | 0.1411 | +0.0806 |
| 4. Content cosine | 0.1400 | 0.0548 | -0.0852 |
| 5. Item-kNN CF | 0.0700 | 0.1478 | +0.0779 |
| 6. Logistic regression (CV-tuned L2) | 0.1129 | 0.2281 | +0.1151 |
| 7. LambdaMART (no IPS) | 0.1257 | 0.1963 | +0.0706 |
| 8. **LambdaMART + IPS (primary)** | 0.1480 | 0.1576 | +0.0096 |
| 9. Oracle (ceiling) | 0.3733 | 0.1125 | -0.2608 |

**Oracle ceiling.** Ranking the same candidates by the true noise-free utility gives NDCG@10
0.3733; the primary system reaches
39.6% of it. The ceiling is candidate-level and
exposure-capped (unexposed candidates count as label 0), which is why it sits far below the
slate-level oracle measured by Gate-A.

## 3. Feature architecture

**POI representation** is hybrid by necessity: structured columns (shrunk rating, log review
count, `pop_pct`, localness index, price, duration, hours), categorical columns (native
LightGBM category dtype), geo (H3 cell, distance to the tourist centroid / transit, local
density), behavioural aggregates (CTR, save/visit/dismiss rates, archetype-affinity profile)
and a text embedding (name + description + tags → TF-IDF → SVD-64).

**Traveler representation** combines an explicit block (stated interests multi-hot, budget,
party, mobility, touristiness preference, pace, accessibility, logistics) with an implicit
block: a taste vector in the POI embedding space,
`taste_t = normalize(Σ w(type_i)·exp(-Δt_i/τ)·emb(poi_i))`, plus history-derived category
distribution and mean price/localness/popularity. Both blocks and their **pair features**
(`interact_cos_taste_poi`, `interact_localness_gap`, `interact_interest_match`,
`interact_price_gap`, `interact_category_affinity`) go to LambdaMART, which learns the blend
conditioned on evidence volume.

**What was measured about the representation** (oracle-free selection, oracle-based reporting):

- *Is text the bottleneck?* No. Out-of-fold ridge R² from the text embedding to the latent
  semantic space is 0.865 (first canonical
  correlation 0.979); within-trip taste
  fidelity with the TRUE taste and observable text is
  0.927.
- *Is the taste estimator?* It is the weak link: run over the TRUE semantic vectors it reaches
  0.572, and the
  shipped chain 0.473
  (D9, within-trip Spearman; pooled Spearman mixes between-trip scale differences into the
  statistic).
- *Behavioural item embedding (item2vec / ALS)*: **skipped.** The premise (a weak text channel)
  did not hold at this text fidelity, and the item side was already not the limiting link
  (`results/parts/a3_step0.json`).
- *Pair features, oracle-free validation NDCG@10, two seeds* (`results/parts/a3_pairfeat.json`,
  re-measured under the final training config): category-affinity mean delta
  +0.0099 (both seeds positive) —
  **adopted**; centred taste cosine mean delta
  +0.0079 — **not adopted**: an
  earlier pass under the previous training config had it negative on one seed, the choice was made
  then, and A3 was a single iteration; the re-measurement is recorded as an open follow-up rather
  than acted on after the fact. The gains are small relative to the seed spread.

**Feature-block ablation** (four full LightGBM retrains dropping one block each, plus the five
cheaper ablations; deltas are against the full model on the unbiased holdout):

| Ablation | Delta NDCG@10 | Wilcoxon p | n pairs |
|---|---|---|---|
| `-IPS_weighting` | -0.0223 | 8.16e-12 | 605 |
| `-calibration` | -0.0021 | 0.135 | 605 |
| `-MMR` | +0.0278 | 1.93e-13 | 579 |
| `-CF_channel` | +0.0000 | 1 | 605 |
| `-long_tail_quota` | +0.0004 | 2.96e-05 | 605 |
| `-text_embeddings` | +0.0077 | 0.0134 | 605 |
| `-implicit_taste` | +0.0059 | 0.12 | 605 |
| `-explicit_interests` | +0.0002 | 0.817 | 605 |
| `-behavioral_block` | +0.0001 | 0.583 | 605 |

Sign convention: delta = ablated minus full, so a POSITIVE delta means removing the block
*improved* NDCG@10. Several block rows are positive, and one is significant: dropping the raw
text-embedding block changes NDCG@10 by +0.0077
(p=0.0134), the raw implicit-taste block by
+0.0059 (p=0.12). Reading: the raw
embedding and taste-vector columns add noise the trees fit; their information still reaches the
model through the engineered `interact_cos_taste_poi` (which is not dropped with the raw block),
so the ablation says the *raw blocks* are not earning their columns, not that taste is useless.
Dropping them is a candidate improvement that was deliberately **not** applied: choosing it on
this holdout number would be selecting on the holdout. Explicit interests and the behavioural
block are within noise.

## 4. Candidate generation

**What failed first, measured.** The original six heuristic channels (geo, interest, semantic,
item-item CF, long-tail, archetype) recalled only a little more than chance of the exposed
holdout positives at roughly 40% of the catalog — while ranking by the true utility would have
recalled 0.991 at the
same budget. So the ceiling was the retrieval *design*, not the data
(`results/parts/a3_step0_recall.json`, Decision Register DR11).

**What ships.** A learned first-stage retriever (`candidates/retriever.py`): an IPS-weighted
LightGBM over the same observable pair features the ranker uses, scoring the *full* destination
catalog and taking the top-K per trip, unioned with the long-tail hard floor and the interest
channel. Train trips are scored **cross-fitted** (K-fold by trip — no trip is scored by a model
that saw it), holdout trips by the full-train model; a test asserts both the leakage discipline
and thread-count invariance. Geo, semantic, CF and archetype quotas are set to 0 (their
marginal recall was measured, not assumed; DR11). K was chosen on a train-carved validation
split against IPS-weighted logged positives (smallest K with validation recall ≥ 0.90; the
margin above the gate was set after seeing that validation runs above the holdout). **Selection
audit:** that rule gave K=210 on the grid computed
before the category-affinity feature entered the retriever; re-running the same rule on the final
feature set gives K=180, whose
holdout recall is 0.849
(vs 0.881 at the
shipped K, in the grid's simplified non-cross-fitted setup). K was not re-selected after the
fact, so the overall-recall gate pass is a **knife-edge result** that partly reflects a choice
made with knowledge of the holdout gap.

Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its own chance baseline: the share of that stratum's destination POIs a same-size random candidate set would contain. Raw recall without this lift hid a near-chance failure for four phases.

| Stratum | Recall | Chance | Lift (abs) | Trips |
|---|---|---|---|---|
| long_tail | 0.848 | 0.461 | +0.387 | 605 |
| overall | 0.873 | 0.509 | +0.364 | 605 |
| q1_least_popular | 0.859 | 0.500 | +0.359 | 601 |
| q2 | 0.836 | 0.422 | +0.414 | 600 |
| q3 | 0.834 | 0.454 | +0.380 | 603 |
| q4_most_popular | 0.922 | 0.660 | +0.262 | 605 |

Oracle-relevance recall (top 5% of the catalog by true utility; reporting-only):
0.944 overall,
0.925 long-tail.

**Long-tail quota sweep (DR9)** — the quota barely moves ranking quality or long-tail share, so
the share is set by the ranker, not the quota:

| Long-tail quota | NDCG@10 | Long-tail share@10 | Long-tail precision@10 | Candidate recall | Long-tail recall |
|---|---|---|---|---|---|
| 0 | 0.1483 [0.1383, 0.1585] | 0.156 | 0.243 | 0.862 | 0.820 |
| 100 | 0.1469 [0.1369, 0.1571] | 0.166 | 0.225 | 0.887 | 0.884 |
| 25 | 0.1483 [0.1382, 0.1585] | 0.157 | 0.244 | 0.868 | 0.834 |
| 50 | 0.1480 [0.1380, 0.1582] | 0.160 | 0.239 | 0.873 | 0.848 |

## 5. Ranking model choice — LambdaMART justified over a two-tower ranker

**Chosen:** LightGBM `objective="lambdarank"`, grouped by trip, IPS-weighted, behavioural-block
dropout (designed for new-POI robustness; its measured effect is in section 10).

**Measured against a two-tower neural ranker (DR2, learning curve).** A small two-tower model
(traveler tower and POI tower into a shared space, dot-product score, same IPS weights, same
train-carved early stopping) was trained on 10/25/50/100% of the training trips over three
seeds each, and scored against LambdaMART on the unbiased holdout with 2000-resample trip
bootstrap CIs. Note the two-tower can only use *separable* features; the `interact_*` pair
features cannot enter a dot product, which is part of what is being compared.

| Train fraction | Train trips | LambdaMART + IPS NDCG@10 | Two-tower NDCG@10 |
|---|---|---|---|
| 10% | 183 | 0.1404 [0.1312, 0.1493] | 0.1039 [0.0975, 0.1103] |
| 25% | 457 | 0.1442 [0.1349, 0.1534] | 0.1075 [0.1013, 0.1145] |
| 50% | 914 | 0.1488 [0.1391, 0.1586] | 0.1188 [0.1119, 0.1258] |
| 100% | 1829 | 0.1499 [0.1401, 0.1597] | 0.1286 [0.1208, 0.1365] |

Log-linear extrapolated crossover: ~65,598 training trips (35.9x the 1829 available; 4-point fit, low confidence).

There is no crossover anywhere on the measured range. The extrapolation is a four-point
log-linear fit and is an illustration of scale, not a forecast.

**Honest result on the objective (DR3).** The listwise choice is *not* supported by
measurement here: pointwise objectives tie or beat `lambdarank` on this data (CIs overlap
widely). `lambdarank` stays because the assignment asks for a learning-to-rank model and the
differences are within noise, not because it won.

| Variant | NDCG@10 (95% CI) |
|---|---|
| binary | 0.1596 [0.1491, 0.1705] |
| lambdarank | 0.1480 [0.1380, 0.1582] |
| rank_xendcg | 0.1556 [0.1453, 0.1666] |
| regression | 0.1596 [0.1492, 0.1700] |

**All nine systems, unbiased holdout** (671 trips, bootstrap 95% CI):

| System | NDCG@10 (95% CI) | % of oracle ceiling |
|---|---|---|
| 1. Random | 0.0589 [0.0529, 0.0651] | 15.8% |
| 2. Popularity | 0.0616 [0.0557, 0.0683] | 16.5% |
| 3. Popularity + geo filter | 0.0605 [0.0546, 0.0667] | 16.2% |
| 4. Content cosine | 0.1400 [0.1302, 0.1498] | 37.5% |
| 5. Item-kNN CF | 0.0700 [0.0632, 0.0769] | 18.7% |
| 6. Logistic regression (CV-tuned L2) | 0.1129 [0.1045, 0.1223] | 30.3% |
| 7. LambdaMART (no IPS) | 0.1257 [0.1161, 0.1352] | 33.7% |
| 8. **LambdaMART + IPS (primary)** | 0.1480 [0.1380, 0.1582] | 39.6% |
| 9. Oracle (ceiling) | 0.3733 [0.3594, 0.3877] | 100.0% |

Paired Wilcoxon: `lambdamart_ips` vs popularity p=5.69e-43
(relative lift +140.2%); vs the best baseline (content cosine)
p=0.329, NOT statistically significant at 0.05, so not a supported win over that baseline (the simple
interest-plus-price content baseline is strong on this data; the tested wins are over popularity and
the no-IPS ranker; the logistic-regression baseline's CI sits below the primary's, but no paired
test against it is stored); `lambdamart` vs `lambdamart_ips`
p=8.16e-12.
The logistic-regression baseline is now regularised (L2 strength chosen by trip-grouped CV
log-loss: C=0.01); the original unregularised fit on 300+
columns risked being a straw man.

## 6. IPS correction

Training weight `clip(1/p_expose, 1, clip_high)` renormalised per trip, unexposed candidates at
neutral weight 1.0 (dropping them would remove most negatives). Exposure rate of fit rows:
24.7%. Ablating IPS changes NDCG@10 by the `-IPS_weighting` row above.

**The clip is not NDCG-optimal on the holdout (DR7).** Less clipping is monotonically better
here, and no-clip is highest. Selecting the clip on the holdout would leak it into a decision,
so the shipped value stays and this is reported as an open improvement:

| Variant | NDCG@10 (95% CI) |
|---|---|
| 10 | 0.1390 [0.1289, 0.1495] |
| 20 | 0.1480 [0.1380, 0.1582] |
| 5 | 0.1351 [0.1255, 0.1451] |
| 50 | 0.1574 [0.1465, 0.1684] |
| no_ips | 0.1257 [0.1161, 0.1352] |
| none | 0.1603 [0.1496, 0.1711] |

## 7. Scoring layer — multiplicative utility over the brief's additive formula

```
hard_gate     = 0 if closed_entire_trip or accessibility_need_unmet or unreachable else 1
compatibility = (budget · mobility · hours · reservation · party · duration) ^ (1/6)
utility       = hard_gate · relevance^α · compatibility^β          (α=1.0, β=0.7)
```

**DR1 (uncuttable): the additive formula lets violations through.** Same candidates, same
relevance and compatibility; only the combination rule differs. The gate is applied *before*
ranking in production; violations are rows with `hard_gate == 0` in a returned top-10.

| Rule | Hard violations in top-10 | Trips with >= 1 | Mean compat@10 | NDCG@10 |
|---|---|---|---|---|
| additive_no_gate (brief) | 844 | 216 / 671 | 0.8655 | 0.1341 |
| additive_with_gate_filter | 0 | 0 / 671 | 0.8742 | 0.1215 |
| multiplicative_gated (production) | 0 | 0 / 671 | 0.8371 | 0.1296 |
| multiplicative_no_gate | 1422 | 317 / 671 | 0.8146 | 0.1465 |

The production rule returns **0**
violations across 6615 recommended slots
(build-blocking test + `make audit`). Additive scoring returns violations because a high
relevance score can outvote a failed factor. NDCG does *not* show a cost for the violations
(section 0, limit 3): the additive rule's NDCG@10
(0.1341) is slightly above
the gated production rule's (0.1296),
and the ungated multiplicative variant is higher still
(0.1465) — i.e. the gate itself
costs a little NDCG on these labels, by construction, and buys the guarantee. That the labels
ignore constraints is a limitation of the simulator, not evidence for the additive rule.

**DR6 — aggregator.** Geometric mean vs min / product / arithmetic mean:

| Aggregator | NDCG@10 (gated) | Hard violations (ungated) | compat@10 (ungated) |
|---|---|---|---|
| arithmetic_mean | 0.1292 | 1550 | 0.8451 |
| geometric_mean (production) | 0.1296 | 1422 | 0.8146 |
| min | 0.1269 | 940 | 0.6168 |
| product | 0.1273 | 830 | 0.4624 |

**DR10 — α×β grid** (gated multiplicative; shipped α=1.0, β=0.7 sits on a flat surface):

| alpha | beta | NDCG@10 | compat@10 |
|---|---|---|---|
| 0.5 | 0.0 | 0.1281 | 0.7924 |
| 0.5 | 0.3 | 0.1301 | 0.8353 |
| 0.5 | 0.7 | 0.1280 | 0.8476 |
| 0.5 | 1.0 | 0.1267 | 0.8550 |
| 0.5 | 1.5 | 0.1269 | 0.8645 |
| 1.0 | 0.0 | 0.1281 | 0.7924 |
| 1.0 | 0.3 | 0.1296 | 0.8299 |
| 1.0 | 0.7 | 0.1296 | 0.8371 |
| 1.0 | 1.0 | 0.1279 | 0.8418 |
| 1.0 | 1.5 | 0.1274 | 0.8489 |
| 2.0 | 0.0 | 0.1281 | 0.7924 |
| 2.0 | 0.3 | 0.1297 | 0.8274 |
| 2.0 | 0.7 | 0.1293 | 0.8308 |
| 2.0 | 1.0 | 0.1303 | 0.8336 |
| 2.0 | 1.5 | 0.1290 | 0.8382 |

## 8. Calibration

Raw LambdaMART scores are unbounded and not comparable across travelers; the output's
`planner_weight` needs calibrated scores. Isotonic regression is fit on a calibration split
carved by trip from the fit frame (disjoint from the early-stopping validation trips and from
the holdout). Measured on the holdout: ECE 0.4768 (naive min-max) →
**0.0418** (isotonic), Brier 0.3410 →
0.1012. The `-calibration` ablation is ~0 by construction (isotonic is
monotone, so it cannot change within-trip ranking); calibration's job is cross-traveler
comparability, which NDCG does not measure.

## 9. Explainability

Grouped TreeSHAP (`shap.TreeExplainer` on the primary booster; exact for a tree ensemble) maps
every feature column to one of ~10 semantic groups (`explain/shap_groups.py`, fails closed on an
unrecognised column); grouped values sum to the raw margin (asserted). TreeSHAP now runs **only
on the rows actually returned** — we never explain a POI we do not return — which took
`recommend` from minutes to seconds. `novelty` is a genuinely empty group and is excluded from
`top_signals` rather than fabricated. No LLM anywhere in the explanation layer: deterministic
templates with real numeric fill-ins; counterfactual lines re-score with the real
compatibility/utility functions.

## 10. Evaluation methodology

**Bootstrap + paired Wilcoxon.** Per-trip means, 2000-resample trip bootstrap 95% CIs, paired
Wilcoxon for every comparison; point estimates are never reported alone.

**Selection vs reporting.** Everything tuned (retriever K, category-affinity feature, LR
regularisation, learning rate/bin count) was selected on train-carved, oracle-free validation.
The oracle-based statistics are computed once afterwards and can never fail a gate.

**Personalization — three bugs in the old measurement, fixed.** (1) Pairs were pooled across
destinations, but a POI belongs to exactly one destination, so 2/3 of the pairs were zero by
construction; pairs are now same-destination only. (2) The within/cross-archetype groups were
K-Means clusters of the very features being evaluated; they now use the DGP's TRUE archetype
labels (dominant archetype, within-pairs also requiring mixture cosine > 0.8), with the K-Means
version kept and labelled as a proxy. (3) The ratio target is now bounded by a reference: the
same statistic for lists ranked by the true utility.

- same-destination mean pairwise Jaccard@10: 0.0406
  (74775 pairs); all-pairs (pooled, structurally diluted):
  0.0135
- true-label within / cross Jaccard: 0.0506
  / 0.0398, ratio
  **1.27**
- reference (perfect ranker): ratio
  1.87
- K-Means proxy ratio: 1.14

**Confidence.** The confidence-decile Spearman is now 0.721
(previously negative on the broken simulator), passing the ≥0.6 target.

**Cold start / new POIs / LODO.** New-POI cohort with vs without behavioural dropout: NDCG@10
0.5305 vs 0.5096
(paired Wilcoxon p=0.094: directionally
consistent with the intent, not statistically significant at this cohort size).
Leave-one-destination-out (three retrains): see the cold-start section of `docs/RESULTS.md`.

**Scenarios** (Seoul; spec.md section 15's three required profiles plus the touristiness-flip
diagnostic, narrated as inbound personas): the flip's top-10 overlap is
0.111 (target ≤ 0.35), from candidate pools
that overlap at 0.624.

### Scorecard — every miss carries a diagnosis

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

## 11. Production considerations

Prose only; no implementation in this repository backs the items below.

**What is different for an inbound-travel platform.** Most users are first-time visitors, so
cold start is the norm, not the edge (behavioural-block dropout in training is aimed at
this; an archetype-prior candidate channel was tried and disabled because it added no measured
recall, DR11); language and locale features matter for both retrieval and
explanation; seasonality (cherry blossom, festivals, extreme weather) shifts relevance and
hours; and event / pop-up POIs have short lifecycles, so behavioural aggregates must decay and
new-POI robustness is a first-class requirement.

**Scale.** The retriever scores the full catalog per request today (fine at this catalog size);
at production scale it becomes an ANN candidate stage plus H3 geo shards, with the ranker still
scoring a few hundred candidates. DR5 (ANN latency/recall at 1M vectors) was **not run**; the
claim "no ANN at this scale" is therefore reasoning, not evidence.

**Online vs offline features.** Offline nightly: text embeddings, popularity percentiles,
localness index, CF, archetype affinities. Online: distance from the stay location, open-now,
live availability, session context. A shared transformation library prevents train/serve skew.

**Retraining and promotion.** Weekly retrain, daily behavioural-aggregate refresh; promotion
gated on offline NDCG over a fresh uniform-exposure slice plus interleaving, never on offline
NDCG alone.

**Feedback loop.** Log propensities from the live policy (as `p_expose` does here) and reserve
a small share of traffic for uniform exposure — the production analogue of the random holdout.

**Monitoring.** Feature drift (PSI), interleaved NDCG, calibration drift (ECE), catalog
coverage / Gini / long-tail share, hard-constraint violation rate (must stay exactly 0), cold
start share, p99 latency and a candidate-recall proxy.

## 12. Decision Register

Every design choice not dictated by the assignment, the alternative, the experiment and the
measured result. Generated from `results/parts/dr/*.json`; an experiment that was not run says
`NOT RUN`. The detail tables are in the sections above (DR1/DR6/DR10 section 7, DR2/DR3
section 5, DR7 section 6, DR9/DR11 section 4) and DR4 below.

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

**DR4 (TF-IDF vs MiniLM) — read the caveat.** The synthetic corpus is generated from anchored,
synonym-rich phrase pools over latent dimensions, so its vocabulary design determines the
outcome: the conclusion is about *this dataset*, not a verdict on sentence encoders, and its
transfer to real POI text is untested. Note also that the fidelity ordering does not carry
through to the ranking metric: MiniLM's NDCG@10 is nominally *higher* than TF-IDF's with
overlapping CIs, despite much lower D11 / D9.

| Encoder | D11 ridge R2 | D9 within-trip Spearman | NDCG@10 (95% CI) |
|---|---|---|---|
| minilm_svd64 | 0.590 | 0.258 | 0.1549 [0.1446, 0.1655] |
| tfidf_svd64 | 0.865 | 0.473 | 0.1480 [0.1380, 0.1582] |

## 13. Reproduction, performance and integrity

`make reproduce` is CPU-only, single-seed, no cloud, no API keys, and does not run pytest (that
is CI). It runs both acceptance gates and ends in `compose`, the **only** writer of
`results/metrics.json` (each stage writes `results/parts/<stage>.json`; a test and `make audit`
enforce single-writer). `make reproduce-full` adds the inspection artifacts (`recommend`,
capped at a seeded 300-trip sample; `scenarios`), leave-one-destination-out, and the docs.

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

Where the time went (measured): the confidence ensemble was being retrained inside every
`evaluate` / `recommend` / `scenarios` call — it is now fit once in `train` and loaded; TreeSHAP
runs on returned rows only; MMR and its lambda sweep are vectorised with lambda-independent
pools; LightGBM uses all logical cores (`deterministic=True` makes the trees thread-count
invariant — asserted at 1 vs 8 threads). What did **not** help: running several LightGBM fits
concurrently (LightGBM already saturates the cores; concurrent fits were slower), so no
thread-pool machinery was kept.

`make audit` replaces the former verifier subagent with a deterministic script that emits
JSON: firewalls and oracle isolation, single writer of `metrics.json` (static scan + compose
twice), hard-constraint violations = 0, both gates, no hand-typed numbers in `docs/`,
LightGBM thread determinism, and (`--deep`) byte-identical regeneration of the dataset.

Cloud is *not* used for reproduction.

**Seed replication** (the whole pipeline regenerated per seed; the bootstrap CIs above are
within-seed, this is between-seed):

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
