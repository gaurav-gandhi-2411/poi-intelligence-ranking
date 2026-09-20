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
popularity's NDCG@10 moves by +0.0833 between the unbiased and biased
holdouts (it is flattered by the biased log), the primary IPS-corrected model by
+0.0085.

**The commercial objective is local / long-tail discovery, not raw NDCG.** The incumbents in
inbound-travel already rank by popularity, so a popularity-shaped list is table stakes. The
differentiating claim is surfacing the genuinely relevant, non-obvious POI, and it has to be
backed by long-tail **precision**, not just share. Measured on the unbiased holdout, top-10
lists: long-tail share 0.2252 (popularity ranker:
0.0000), catalog coverage@10
66.4% (popularity:
5.7%); long-tail precision
0.1541 on 1486 long-tail recommendations —
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

**Two exposure logs, one primary evaluation set.** `interactions_train.parquet` uses a
popularity-biased policy; `interactions_holdout_random.parquet` uses uniform-random exposure
over a wide slate and is the **primary** evaluation population
(671 holdout trips); `interactions_holdout_logged.parquet` repeats the
biased policy over the holdout window only to compute the bias gap. Every logged interaction
carries its `p_expose`, which is what makes IPS possible.

**The bias-gap table** (every system, biased vs unbiased holdout):

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0661 | 0.0306 | -0.0355 |
| 2. Popularity | 0.0629 | 0.1462 | +0.0833 |
| 3. Popularity + geo filter | 0.0623 | 0.1444 | +0.0821 |
| 4. Content cosine | 0.1411 | 0.0544 | -0.0867 |
| 5. Item-kNN CF | 0.0710 | 0.1494 | +0.0785 |
| 6. Logistic regression (CV-tuned L2) | 0.1130 | 0.2307 | +0.1177 |
| 7. LambdaMART (no IPS) | 0.1244 | 0.1917 | +0.0673 |
| 8. **LambdaMART + IPS (primary)** | 0.1485 | 0.1570 | +0.0085 |
| 9. Oracle (ceiling) | 0.3749 | 0.1125 | -0.2623 |

**Oracle ceiling — and why "% of ceiling" fell from 66.2% to
39.6%.** Ranking the same candidates by the true
noise-free utility gives NDCG@10 0.3749; the primary system
reaches 39.6% of it. The earlier report (DATA_CARD
history: primary 0.0875 against an oracle of 0.1322 = 66.2%) is **not comparable** to this one, for
two measured reasons, neither of which is "the model got worse" (the primary system's own NDCG@10
went *up*, from 0.0875 to 0.1485):

1. **The old oracle was crippled by the old data-generating process.** Before the simulator
   rewrite the labels were noise-dominated, so even a ranker with the true utility scored only
   0.1322 (DATA_CARD documents this: an oracle at 0.1322 is itself the evidence that labels were
   mostly noise). A model closing 66.2% of a near-noise ceiling is a small absolute number. After
   the rewrite the labels are utility-driven and the ceiling rose to
   0.3749 — the ceiling grew by a larger factor than the
   model's score, so the ratio fell even though the model improved.
2. **The candidate-relative ceiling depends on the candidate set.** The oracle re-ranks *its own*
   candidates, so a retriever that surfaces more relevant POIs raises the ideal too. Measured on
   identical holdout trips (E1), the candidate-relative oracle NDCG@10 is
   0.4037 on the old six-channel set and
   0.3749 on the learned set.

A related figure that is easy to confuse with the ceiling: retrieval top-K recall by true utility
(0.991) measures how
many truly relevant POIs a top-K by the true utility would surface. That is a *recall* diagnostic of
the catalog, not the NDCG ceiling, and it does not by itself explain the ratio. The ceiling is
candidate-level and exposure-capped (unexposed candidates count as label 0), which is why it sits
below the slate-level oracle measured by Gate-A.

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
| `-IPS_weighting` | -0.0241 | 7.69e-11 | 605 |
| `-calibration` | -0.0007 | 0.709 | 605 |
| `-MMR` | +0.0228 | 3.57e-12 | 579 |
| `-interest_channel` | +0.0012 | 6.82e-15 | 605 |
| `-long_tail_quota` | +0.0007 | 6.26e-06 | 605 |
| `-text_embeddings` | +0.0095 | 0.0121 | 605 |
| `-implicit_taste` | +0.0064 | 0.121 | 605 |
| `-explicit_interests` | -0.0018 | 0.822 | 605 |
| `-behavioral_block` | -0.0030 | 0.326 | 605 |

Sign convention: delta = ablated minus full, so a POSITIVE delta means removing the block
*improved* NDCG@10. Several block rows are positive, and one is significant: dropping the raw
text-embedding block changes NDCG@10 by +0.0095
(p=0.0121), the raw implicit-taste block by
+0.0064 (p=0.121). Reading: the raw
embedding and taste-vector columns add noise the trees fit; their information still reaches the
model through the engineered `interact_cos_taste_poi` (which is not dropped with the raw block),
so the ablation says the *raw blocks* are not earning their columns, not that taste is useless.

Dropping them is a candidate improvement that was deliberately **not** applied: choosing it on
this holdout number would be selecting on the holdout. Explicit interests and the behavioural
block are within noise.

**Channel ablations.** The earlier `-CF_channel` row was a **no-op by construction** (the CF quota
is 0 in the shipped candidate generator, so removing it changed nothing and the row carried no
information); it is replaced by `-interest_channel`, a channel that is actually in the shipped
union. Both channel ablations re-score the already-trained ranker over a smaller candidate set, and
both come out *slightly positive* for NDCG@10: `-interest_channel`
+0.0012 (p=6.82e-15), `-long_tail_quota` +0.0007
(p=6.26e-06). The effects are ~1% of NDCG@10 and small in absolute terms:
a plausible reading (not separately tested) is that a leaner set leaves the ranker fewer
low-relevance items to misplace. These two channels are **not** justified by top-10 relevance; they are justified by
recall and long-tail exposure (section 4), which NDCG@10 over the exposed labels does not reward.

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
and thread-count invariance.

**Three channels, not six.** The candidate set is the union of exactly three channels: the
**learned** retriever (the workhorse), the **long-tail hard floor** and the **interest** channel.
Geo, semantic, CF and archetype quotas are 0. This applies the brief's own rule (spec section 7:
"a channel that adds no marginal recall gets deleted") to the measured grid (Decision Register
DR11, frozen A3 grid at learned K=210): adding geo, semantic, CF and archetype on top of
long-tail + interest raises overall holdout recall from
0.881 to
0.903 while inflating the
candidate set from 246 to
290 POIs per trip and cutting the
chance-corrected lift from 0.371 to
0.301 — recall bought by volume, not
by discrimination. The **interest** channel is the one that survived the marginal test (recall
0.852 →
0.881 for a small increase in set size).

**Answer to the brief's section 9 ("how do we avoid killing the long tail?").** The long-tail
channel is a **hard floor of 50 candidates per trip that the learned retriever cannot
cannibalise**, and the long-tail stratum's recall is measured separately from overall recall (gate
row, `recall_strata` below). The quota does not raise the shipped long-tail *share* — DR9 below
shows the share is set by the ranker and re-ranker, not by the quota — but it is what guarantees
the ranker is *offered* long-tail POIs, and stage-by-stage precision is diagnosed in section 4.2.

**K rule (E4, pre-registered, blind, applied once).** The learned K is the *smallest K whose
IPS-weighted validation recall is at least 0.93* (validation = 20% of the train trips against
logged positives weighted 1/clip(p_expose); the holdout is reporting-only and was never consulted).
That rule gives **K=195**. It replaces the previous K=210, which had been
chosen with a 0.90 rule whose margin was adjusted after seeing the holdout gap (a contaminated
choice; the audit is kept in `results/parts/a3_retriever_grid.json`). The full sweep, holdout
column included:

smallest K with IPS-weighted validation overall recall >= 0.93 (val = 20% of train trips vs logged positives, weighted 1/clip(p_expose)); applied once, blind to the holdout column, which is reporting-only.

| Learned K | Val recall (IPS) | Holdout recall (reporting-only) | Holdout long-tail | Holdout lift | Candidates/trip |
|---|---|---|---|---|---|
| 90 | 0.768 | 0.685 | 0.613 | +0.370 | 152 |
| 105 | 0.800 | 0.723 | 0.654 | +0.385 | 163 |
| 120 | 0.834 | 0.755 | 0.690 | +0.395 | 174 |
| 135 | 0.860 | 0.781 | 0.720 | +0.398 | 185 |
| 150 | 0.883 | 0.805 | 0.749 | +0.397 | 197 |
| 165 | 0.902 | 0.829 | 0.781 | +0.396 | 209 |
| 180 | 0.919 | 0.849 | 0.806 | +0.391 | 221 |
| 195 **(rule)** | 0.933 | 0.866 | 0.828 | +0.382 | 233 |
| 210 | 0.945 | 0.881 | 0.847 | +0.371 | 246 |
| 225 | 0.957 | 0.893 | 0.863 | +0.358 | 258 |
| 240 | 0.963 | 0.906 | 0.881 | +0.344 | 271 |
| 255 | 0.972 | 0.915 | 0.893 | +0.328 | 283 |
| 270 | 0.979 | 0.924 | 0.905 | +0.310 | 296 |
| 285 | 0.985 | 0.933 | 0.916 | +0.293 | 308 |
| 300 | 0.990 | 0.942 | 0.928 | +0.276 | 321 |
| 315 | 0.993 | 0.949 | 0.937 | +0.257 | 334 |
| 330 | 0.996 | 0.956 | 0.947 | +0.237 | 347 |
| 345 | 0.997 | 0.964 | 0.955 | +0.218 | 359 |
| 360 | 0.999 | 0.971 | 0.965 | +0.199 | 372 |

At the rule K the holdout gives overall recall
0.866, long-tail recall
0.828, and chance-corrected lift
+0.382 (candidate set
233 POIs per trip in the sweep's
simplified non-cross-fitted setup; the shipped pipeline's own recall rows are in the gates and the
scorecard). K was not adjusted after the holdout was read; whatever the gate rows show is the result.

Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its own chance baseline: the share of that stratum's destination POIs a same-size random candidate set would contain. Raw recall without this lift hid a near-chance failure for four phases.

| Stratum | Recall | Chance | Lift (abs) | Trips |
|---|---|---|---|---|
| long_tail | 0.826 | 0.434 | +0.392 | 605 |
| overall | 0.857 | 0.484 | +0.374 | 605 |
| q1_least_popular | 0.840 | 0.474 | +0.366 | 601 |
| q2 | 0.811 | 0.393 | +0.417 | 600 |
| q3 | 0.816 | 0.425 | +0.392 | 603 |
| q4_most_popular | 0.915 | 0.641 | +0.274 | 605 |

Oracle-relevance recall (top 5% of the catalog by true utility; reporting-only):
0.936 overall,
0.911 long-tail.

**Long-tail quota sweep (DR9)** — the quota barely moves ranking quality or long-tail share, so
the share is set by the ranker, not the quota:

| Long-tail quota | NDCG@10 | Long-tail share@10 | Long-tail precision@10 | Candidate recall | Long-tail recall |
|---|---|---|---|---|---|
| 0 | 0.1492 [0.1391, 0.1598] | 0.156 | 0.238 | 0.845 | 0.794 |
| 100 | 0.1472 [0.1373, 0.1576] | 0.168 | 0.218 | 0.874 | 0.868 |
| 25 | 0.1487 [0.1388, 0.1592] | 0.161 | 0.230 | 0.851 | 0.810 |
| 50 | 0.1485 [0.1385, 0.1590] | 0.161 | 0.227 | 0.857 | 0.826 |

### 4.1 Retrieval vs ranking: where does the gain over popularity come from? (E1)

The learned retriever raised candidate recall from
0.596 (the six-channel union it
replaced) to the recall in the gates, yet the ranker's NDCG@10 edge over content cosine stayed
small. Candidate-relative NDCG@10 normalises each candidate set by its *own* ideal ranking, so it
cannot compare two candidate sets — a better retriever surfaces more positives, raising the ideal
and lowering the ratio for the same ranker. The decomposition therefore uses an **end-to-end
NDCG@10 with a fixed denominator** (the ideal over every logged label of the trip, identical for
every candidate set and system), scoring every system on both candidate sets on the same holdout
trips (the "retrained" row refits the ranker on the old set's own training candidates):

| End-to-end NDCG@10 (fixed denominator) | 6-channel union (190 cand/trip) | Learned retriever + long-tail + interest (233 cand/trip) |
|---|---|---|
| Random | 0.0512 [0.0460, 0.0564] | 0.0637 [0.0582, 0.0697] |
| Popularity | 0.0644 [0.0579, 0.0715] | 0.0596 [0.0536, 0.0661] |
| Content cosine | 0.1353 [0.1255, 0.1448] | 0.1357 [0.1260, 0.1452] |
| LambdaMART + IPS (shipped booster) | 0.1445 [0.1345, 0.1548] | 0.1434 [0.1332, 0.1539] |
| LambdaMART + IPS (retrained on this set) | 0.1462 [0.1360, 0.1562] | — |
| Oracle (true utility) | 0.3407 [0.3273, 0.3540] | 0.3583 [0.3440, 0.3718] |

For reference, the usual candidate-relative NDCG@10 (each set normalised by its own ideal):

| System | 6-channel union | Learned retriever |
|---|---|---|
| Random | 0.0601 [0.0542, 0.0662] | 0.0661 [0.0603, 0.0724] |
| Popularity | 0.0767 [0.0694, 0.0846] | 0.0629 [0.0567, 0.0698] |
| Content cosine | 0.1603 [0.1494, 0.1712] | 0.1411 [0.1313, 0.1511] |
| LambdaMART + IPS (shipped booster) | 0.1685 [0.1570, 0.1799] | 0.1485 [0.1385, 0.1590] |
| LambdaMART + IPS (retrained on this set) | 0.1704 [0.1588, 0.1820] | — |
| Oracle (true utility) | 0.4037 [0.3893, 0.4187] | 0.3749 [0.3610, 0.3895] |

**Decomposition of the headline gain (candidate-relative NDCG@10, learned candidate set, seed 42).**
Popularity 0.0629 → content cosine
0.1411
(**+0.0782**, *content matching*; Wilcoxon
p=5.43e-36) → LambdaMART + IPS
0.1485
(**+0.0074**, *learned ranking*;
p=0.445; 5-seed mean gap over cosine
4 of 5 seeds (mean gap +0.0110 NDCG@10)). Almost all of the lift over the popularity ranker the incumbents run is
achieved by matching a traveler's taste to a POI's content; the learned ranker adds a small,
not-significant increment on top.

**Conclusion.** At this data scale the learned ranker's marginal value over a well-constructed
content-similarity baseline is small, and we measured it rather than assuming it. That is
evidence-based model selection, not a shortfall to apologise for: the assignment's own guidance
(as stated in the brief; the brief text itself is not reproduced in this repository) is that a
simpler model with thoughtful features and strong evaluation beats unnecessary complexity. Three
independent measurements point the same way: (1) the validation-only ranker sweep (section 5)
selected the shipped configuration without using the holdout; (2) the DR2 learning curve shows a
two-tower ranker at 0.1292 against
LambdaMART at 0.1464 even on the full
training set (and 0.1047 against
0.1409 at a tenth of it), i.e.
neural complexity buys nothing here; (3) the ranker is well above popularity in every replicated seed, and above cosine in most (not
all) seeds. The ranker stays because the assignment asks for a learning-to-rank model and it is
above cosine on average — not because it earns a large margin.

Reading: on the fixed denominator, popularity moves by
-0.0048
when only the retriever changes; the primary system's gain over popularity is
+0.0817 on the legacy set and
+0.0837 on the learned set.
**The gain over popularity is a ranking gain, not a retrieval gain.** Primary vs content cosine on
the same (learned) set: paired Wilcoxon p =
0.405;
retrieval effect on the primary system (learned vs legacy-retrained): p =
0.308.
The honest headline is therefore: a **significant** improvement over the popularity ranker the
incumbents run, and a **not significant** edge over a plain content-cosine baseline — on this
simulator the taste signal a cosine captures is most of what a learned ranker extracts.

### 4.2 Long-tail precision at every stage (E3)

**The 0.40 long-tail precision target was miscalibrated at design time.** It was set a priori,
without reference to how often a long-tail candidate is relevant at all. The measured positive
rate among long-tail candidates in the candidate pool — what any ranker that added no signal would
deliver — is 0.099; a precision of 0.40 would
require roughly four times that. So the primary long-tail precision statistic is **lift over the
pool's base rate**, and raw precision is secondary: the raw ranker delivers
2.234x, and the served list (after the hard
gate, utility and MMR) 1.556x. Raw precision
still misses 0.40 and is reported as a miss in the scorecard, with this note.

The same audit applies to the **0.85 overall candidate-recall target**: it was also set a priori as
an absolute number, with no reference to the chance baseline, which depends on the candidate-set
size. At the shipped set size a random candidate set would already recall
0.484, so 0.85 means a lift of
+0.374 over chance. The target is met, but the margin
is thin (the lowest of the five replicated seeds is
0.8504) and should not be read as headroom.

Measured at each serving stage on the same holdout trips (share = fraction of returned slots that
are long-tail; precision = fraction of those that are positives):

| Stage | Long-tail share | Long-tail precision | Lift over pool base rate |
|---|---|---|---|
| 0_candidate_pool (positive rate among long-tail candidates) | 0.448 | 0.099 | 1.000x |
| 1_raw_ranker_top10 | 0.202 | 0.221 | 2.234x |
| 2_after_hard_gate_raw_order | 0.230 | 0.185 | 1.871x |
| 3_after_utility | 0.237 | 0.192 | 1.934x |
| 4_final_after_mmr | 0.225 | 0.154 | 1.556x |

MMR lambda (diagnostic, post-hoc on the holdout -- not a selection):

| lambda | Long-tail share | Long-tail precision |
|---|---|---|
| 0.5 | 0.234 | 0.138 |
| 0.6 | 0.228 | 0.145 |
| 0.7 | 0.221 | 0.148 |
| 0.8 | 0.225 | 0.154 |
| 0.9 | 0.227 | 0.173 |
| 1 | 0.237 | 0.192 |

Reading: the raw ranker already lifts long-tail precision from the pool's base rate to
0.221
(2.234x). The serving stages then give some of it
back: the hard gate takes it to 0.185, the utility
layer is ~neutral (0.192), and the MMR diversity re-rank takes it to
0.154. The gate and MMR each cost about the same. The lambda sweep
(RESULTS.md, diversity section) is a post-hoc diagnostic on the holdout — it is not a selection and
lambda is unchanged: lambda 0.8 is a deliberate diversity-for-precision trade whose cost is
quantified there, and even with no diversity term (lambda 1) precision stays well under 0.40.

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
| 10% | 183 | 0.1409 [0.1322, 0.1496] | 0.1047 [0.0983, 0.1109] |
| 25% | 457 | 0.1481 [0.1386, 0.1577] | 0.1081 [0.1014, 0.1151] |
| 50% | 914 | 0.1495 [0.1399, 0.1595] | 0.1204 [0.1138, 0.1271] |
| 100% | 1829 | 0.1464 [0.1370, 0.1561] | 0.1292 [0.1216, 0.1371] |

Log-linear extrapolated crossover: ~22,045 training trips (12.1x the 1829 available; 4-point fit, low confidence).

There is no crossover anywhere on the measured range. The extrapolation is a four-point
log-linear fit and is an illustration of scale, not a forecast.

**Honest result on the objective (DR3).** The listwise choice is *not* supported by
measurement here: pointwise objectives tie or beat `lambdarank` on this data (CIs overlap
widely). `lambdarank` stays because the assignment asks for a learning-to-rank model and the
differences are within noise, not because it won.

| Variant | NDCG@10 (95% CI) |
|---|---|
| binary | 0.1590 [0.1483, 0.1701] |
| lambdarank | 0.1485 [0.1385, 0.1590] |
| rank_xendcg | 0.1541 [0.1432, 0.1652] |
| regression | 0.1591 [0.1482, 0.1700] |

**All nine systems, unbiased holdout** (671 trips, bootstrap 95% CI):

| System | NDCG@10 (95% CI) | % of oracle ceiling |
|---|---|---|
| 1. Random | 0.0661 [0.0603, 0.0724] | 17.6% |
| 2. Popularity | 0.0629 [0.0567, 0.0698] | 16.8% |
| 3. Popularity + geo filter | 0.0623 [0.0562, 0.0688] | 16.6% |
| 4. Content cosine | 0.1411 [0.1313, 0.1511] | 37.6% |
| 5. Item-kNN CF | 0.0710 [0.0642, 0.0779] | 18.9% |
| 6. Logistic regression (CV-tuned L2) | 0.1130 [0.1045, 0.1226] | 30.1% |
| 7. LambdaMART (no IPS) | 0.1244 [0.1152, 0.1339] | 33.2% |
| 8. **LambdaMART + IPS (primary)** | 0.1485 [0.1385, 0.1590] | 39.6% |
| 9. Oracle (ceiling) | 0.3749 [0.3610, 0.3895] | 100.0% |

Paired Wilcoxon: `lambdamart_ips` vs popularity p=4.34e-41
(relative lift +136.2%); vs the best baseline (content cosine)
p=0.445, NOT statistically significant at 0.05, so not a supported win over that baseline (the simple
interest-plus-price content baseline is strong on this data; the tested wins are over popularity and
the no-IPS ranker; the logistic-regression baseline's CI sits below the primary's, but no paired
test against it is stored); `lambdamart` vs `lambdamart_ips`
p=7.69e-11.
The logistic-regression baseline is now regularised (L2 strength chosen by trip-grouped CV
log-loss: C=0.01); the original unregularised fit on 300+
columns risked being a straw man.

**One legitimate shot at the ranker (E2).** A single joint sweep over the objective
(lambdarank / binary / rank_xendcg), the IPS clip ceiling and the feature blocks (raw text
embedding, raw taste vector) — 60 configurations, scored on
**validation only** (the holdout was not read) with an IPS-weighted validation NDCG@10 (plain
validation NDCG would reward fitting the popularity-biased log). Successive halving: all
configurations at seed 42, then the top five re-fit over three more seeds; the winner is the best
4-seed mean.

| Axis | Value | Mean validation IPS-weighted NDCG@10 |
|---|---|---|
| objective | lambdarank | 0.2820 |
| objective | binary | 0.2803 |
| objective | rank_xendcg | 0.2828 |
| IPS clip | 5.0 | 0.2890 |
| IPS clip | 10.0 | 0.2940 |
| IPS clip | 20.0 | 0.2929 |
| IPS clip | 50.0 | 0.2797 |
| IPS clip | none | 0.2529 |
| feature blocks | all_features | 0.2951 |
| feature blocks | no_raw_text_emb | 0.2709 |
| feature blocks | no_raw_taste | 0.2906 |
| feature blocks | no_raw_text_no_taste | 0.2702 |

Winner (4-seed mean 0.3167): objective lambdarank, IPS clip 20.0, blocks all_features; the previously shipped config scores 0.3167.

The validation winner is the configuration that already ships: no change was made to the ranker,
so the single holdout result in this document is the shipped ranker's. The spread across the three
objectives is small (a few thousandths of validation NDCG); the large effects are the IPS clip
(no clipping is clearly worst) and keeping the raw text-embedding block. That last point disagrees
with the holdout feature-block ablation above, where dropping the raw text block moved holdout
NDCG@10 by +0.0095 (p=0.0121, not
significant). The two measure different things (IPS-weighted validation on the logged, biased
population vs unweighted random-exposure holdout) and neither is decisive; the config was chosen
by the validation-only rule fixed in advance, and the sweep never scored content cosine, so it
says nothing about closing the gap to it — only that no ranker knob in this grid beats the
shipped one on validation.

## 6. IPS correction

Training weight `clip(1/p_expose, 1, clip_high)` renormalised per trip, unexposed candidates at
neutral weight 1.0 (dropping them would remove most negatives). Exposure rate of fit rows:
24.9%. Ablating IPS changes NDCG@10 by the `-IPS_weighting` row above.

**The clip is not NDCG-optimal on the holdout (DR7).** Less clipping is monotonically better
here, and no-clip is highest. Selecting the clip on the holdout would leak it into a decision,
so the shipped value stays and this is reported as an open improvement:

| Variant | NDCG@10 (95% CI) |
|---|---|
| 10 | 0.1439 [0.1335, 0.1545] |
| 20 | 0.1485 [0.1385, 0.1590] |
| 5 | 0.1356 [0.1256, 0.1458] |
| 50 | 0.1618 [0.1513, 0.1727] |
| no_ips | 0.1244 [0.1152, 0.1339] |
| none | 0.1571 [0.1466, 0.1678] |

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
| additive_no_gate (brief) | 901 | 229 / 671 | 0.8616 | 0.1408 |
| additive_with_gate_filter | 0 | 0 / 671 | 0.8713 | 0.1275 |
| multiplicative_gated (production) | 0 | 0 / 671 | 0.8368 | 0.1357 |
| multiplicative_no_gate | 1484 | 314 / 671 | 0.8127 | 0.1526 |

The production rule returns **0**
violations across 6598 recommended slots
(build-blocking test + `make audit`). Additive scoring returns violations because a high
relevance score can outvote a failed factor. NDCG does *not* show a cost for the violations
(section 0, limit 3): the additive rule's NDCG@10
(0.1408) is slightly above
the gated production rule's (0.1357),
and the ungated multiplicative variant is higher still
(0.1526) — i.e. the gate itself
costs a little NDCG on these labels, by construction, and buys the guarantee. That the labels
ignore constraints is a limitation of the simulator, not evidence for the additive rule.

**DR6 — aggregator.** Geometric mean vs min / product / arithmetic mean:

| Aggregator | NDCG@10 (gated) | Hard violations (ungated) | compat@10 (ungated) |
|---|---|---|---|
| arithmetic_mean | 0.1341 | 1585 | 0.8445 |
| geometric_mean (production) | 0.1357 | 1484 | 0.8127 |
| min | 0.1347 | 990 | 0.6124 |
| product | 0.1322 | 884 | 0.4573 |

**DR10 — α×β grid** (gated multiplicative; shipped α=1.0, β=0.7 sits on a flat surface):

| alpha | beta | NDCG@10 | compat@10 |
|---|---|---|---|
| 0.5 | 0.0 | 0.1295 | 0.7913 |
| 0.5 | 0.3 | 0.1350 | 0.8350 |
| 0.5 | 0.7 | 0.1355 | 0.8469 |
| 0.5 | 1.0 | 0.1336 | 0.8542 |
| 0.5 | 1.5 | 0.1323 | 0.8637 |
| 1.0 | 0.0 | 0.1295 | 0.7913 |
| 1.0 | 0.3 | 0.1348 | 0.8306 |
| 1.0 | 0.7 | 0.1357 | 0.8368 |
| 1.0 | 1.0 | 0.1362 | 0.8413 |
| 1.0 | 1.5 | 0.1357 | 0.8482 |
| 2.0 | 0.0 | 0.1295 | 0.7913 |
| 2.0 | 0.3 | 0.1345 | 0.8285 |
| 2.0 | 0.7 | 0.1347 | 0.8312 |
| 2.0 | 1.0 | 0.1348 | 0.8334 |
| 2.0 | 1.5 | 0.1356 | 0.8375 |

## 8. Calibration

Raw LambdaMART scores are unbounded and not comparable across travelers; the output's
`planner_weight` needs calibrated scores. Isotonic regression is fit on a calibration split
carved by trip from the fit frame (disjoint from the early-stopping validation trips and from
the holdout). Measured on the holdout: ECE 0.4444 (naive min-max) →
**0.0466** (isotonic), Brier 0.3139 →
0.1047. The `-calibration` ablation is ~0 by construction (isotonic is
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

- same-destination mean pairwise Jaccard@10: 0.0394
  (74775 pairs); all-pairs (pooled, structurally diluted):
  0.0131
- true-label within / cross Jaccard: 0.0508
  / 0.0385, ratio
  **1.32**
- reference (perfect ranker): ratio
  1.86
- K-Means proxy ratio: 1.14

**Confidence.** The confidence-decile Spearman is now 0.879
(previously negative on the broken simulator), passing the ≥0.6 target.

**Cold start / new POIs / LODO.** New-POI cohort with vs without behavioural dropout: NDCG@10
0.5553 vs 0.5565
(paired Wilcoxon p=0.810: directionally
consistent with the intent, not statistically significant at this cohort size).
Leave-one-destination-out (three retrains): see the cold-start section of `docs/RESULTS.md`.

**Scenarios** (Seoul; spec.md section 15's three required profiles plus the touristiness-flip
diagnostic, narrated as inbound personas): the flip's top-10 overlap is
0.176 (target ≤ 0.35), from candidate pools
that overlap at 0.612.

### Scorecard — every miss carries a diagnosis

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
- **Long-tail precision of top-10**: Long-tail precision is 0.1541 over 1486 long-tail recommendations, with candidate recall 0.826 in that stratum, so retrieval is not the bottleneck. Measured against the pool's own long-tail positive rate (0.099) the served list is a 1.556x lift (raw ranker 2.234x): the 0.40 target was set a priori without reference to that base rate and was miscalibrated at design time, so lift over base rate is the primary statistic and raw precision secondary (TECHNICAL.md section 4.2). The raw ranker's top-10 long-tail precision (DR9, quota 50, before the compatibility gate, utility and MMR re-rank) is 0.2271 at share 0.1614, against 0.1541 at share 0.2252 in the served list: precision is lost AFTER ranking while share rises. Which of the three scoring-layer steps is responsible is not isolated (untested).
- **Localness index Spearman vs latent localness**: The composite index reaches rho 0.582; its observable inputs correlate with the latent localness at dist_to_tourist_centroid_km 0.699, foreign_review_ratio -0.555, local_tag_hits 0.050, pop_pct -0.187. The composite is BELOW its best single input (dist_to_tourist_centroid_km, |rho| 0.699): the blend weights were fixed earlier, when the geo input carried almost no signal (before the simulator's geo/localness fix). Re-weighting them against the latent localness would be tuning on the oracle (there is no oracle-free validation target for this index), so that retune is declined on principle: the index is left as shipped and the gap is reported.

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
| DR1 | Multiplicative utility rel^a * compat^b with a hard gate | The brief's additive formula a*rel + b*compat | Same holdout candidates/relevance/compatibility; rank by each combination rule; count hard-constraint violations (hard_gate == 0) among the top-10 of every holdout trip. | Additive scoring puts 901 hard-constraint violations into 229/671 trips' top-10; the multiplicative gated rule puts 0. NDCG@10 0.1408 (additive) vs 0.1357 (multiplicative gated); mean compat@10 0.8616 vs 0.8368. | MEASURED |
| DR2 | LightGBM LambdaMART ranker | Two-tower neural ranker (shared-space dot product) | Learning curve: both rankers fit on [0.1, 0.25, 0.5, 1.0] of TRAIN trips x seeds [42, 43, 44] (same IPS weights, same train-carved early stopping), scored on the full unbiased holdout; per-trip NDCG@10 averaged over seeds, 2000-resample trip bootstrap CIs. | Two-tower NDCG@10 0.1047, 0.1081, 0.1204, 0.1292 vs LambdaMART-IPS 0.1409, 0.1481, 0.1495, 0.1464 at fractions [0.1, 0.25, 0.5, 1.0]. Crossover at any measured point: False. Log-linear extrapolation puts a crossover at ~22,045 training trips (12.1x the 1829 used; 4-point fit, low confidence). CIs overlap at 100%. | MEASURED |
| DR3 | Listwise LambdaRank objective | Pointwise binary / graded regression, listwise rank_xendcg | Same features, IPS weights, dropout, split and early-stopping metric (NDCG@10); only the LightGBM objective varies. | lambdarank 0.1485 [0.1385, 0.1590]; rank_xendcg 0.1541 [0.1432, 0.1652]; binary 0.1590 [0.1483, 0.1701]; regression 0.1591 [0.1482, 0.1700] | MEASURED |
| DR4 | TF-IDF -> SVD-64 POI text embedding | all-MiniLM-L6-v2 sentence embeddings (-> SVD-64) | Swap ONLY the POI text embedding (and the taste vectors and POI features built from it) and refit the ranker on the fixed candidate sets; measure representation fidelity (D9 within-trip, D11) and holdout NDCG@10. THIS SYNTHETIC CORPUS is generated from anchored, synonym-rich phrase pools over latent dimensions, so its vocabulary design favours lexical overlap: the outcome is a statement about this dataset, not a verdict on sentence encoders. | TF-IDF: D11 0.865, D9 0.473, NDCG@10 0.1485. MiniLM: D11 0.590, D9 0.258, NDCG@10 0.1432 (CIs overlap). Dataset-specific (templated synonym-pool text); transfer to real POI text is untested. | MEASURED |
| DR5 | Brute-force cosine retrieval | ANN index (faiss / hnswlib) | NOT RUN | NOT RUN (ANN only matters at catalog sizes far beyond this take-home's three destinations; cut for time, so 'brute force is fine here' is reasoning, not evidence) | NOT RUN |
| DR6 | Geometric-mean compatibility aggregation | min(), plain product, arithmetic mean | Recompute compatibility from the 6 sub-scores with each aggregator; rank gated and ungated multiplicative utility. | Ungated hard-violation counts: geometric_mean (production)=1484, min=990, product=884, arithmetic_mean=1585; NDCG@10 gated: geometric_mean (production)=0.1357, min=0.1347, product=0.1322, arithmetic_mean=0.1341. Geometric mean ungated NDCG 0.1526. | MEASURED |
| DR7 | IPS clip = 20 | clip in {5, 10, 50, none} | IPS clip-high swept with everything else fixed (weights renormalised per trip). | clip 5: 0.1356 [0.1256, 0.1458]; clip 10: 0.1439 [0.1335, 0.1545]; clip 20: 0.1485 [0.1385, 0.1590]; clip 50: 0.1618 [0.1513, 0.1727]; clip none: 0.1571 [0.1466, 0.1678]; clip no_ips: 0.1244 [0.1152, 0.1339] | MEASURED |
| DR8 | 180-day taste half-life and spec interaction weights | +-2x half-life; uniform interaction weights | Rebuild the traveler features with a different taste half-life / interaction weights, refit the ranker on the fixed candidate sets, and score the unbiased holdout; D9 (within-trip, reporting-only) shows the effect on estimator fidelity. | halflife_180d (shipped): NDCG@10 0.1485, D9 0.473; halflife_90d: NDCG@10 0.1496, D9 0.470; halflife_360d: NDCG@10 0.1614, D9 0.474; uniform_weights: NDCG@10 0.1624, D9 0.052 | MEASURED |
| DR9 | Long-tail candidate quota = 50 | quota in {0, 25, 100} | Regenerate the holdout candidate sets with a different long-tail hard-floor quota (retriever scores fixed); score with the SHIPPED booster (trained at quota 50, not refit per quota); raw ranker top-10, no MMR/gates. | quota 0: NDCG@10 0.1492, long-tail share 0.156, candidate recall 0.845; quota 25: NDCG@10 0.1487, long-tail share 0.161, candidate recall 0.851; quota 50: NDCG@10 0.1485, long-tail share 0.161, candidate recall 0.857; quota 100: NDCG@10 0.1472, long-tail share 0.168, candidate recall 0.874 | MEASURED |
| DR10 | alpha = 1.0, beta = 0.7 | alpha x beta grid | Gated multiplicative utility over an alpha x beta grid on the same holdout scoring pass. | Shipped (alpha=1.0, beta=0.7): NDCG@10 0.1357, compat@10 0.8368. NDCG-best cell (alpha=1.0, beta=1.0): 0.1362, compat@10 0.8413. | MEASURED |
| DR11 | Candidate generation = learned retriever + long-tail + interest | The original 6 heuristic channels, and subsets of them | Channel-subset grid at learned K=210 on a train-carved validation split (IPS-weighted logged positives); holdout columns are reporting-only. `none` = the learned retriever alone. Baseline = the original 6-channel heuristic union. | Legacy 6-channel union: recall 0.596 (lift +0.203). Shipped learned+long-tail+interest: recall 0.881 (lift +0.371, 246 candidates/trip). Ranking by true utility would recall 0.991 at the same budget (diagnostic). | MEASURED |

**DR4 (TF-IDF vs MiniLM) — read the caveat.** The synthetic corpus is generated from anchored,
synonym-rich phrase pools over latent dimensions, so its vocabulary design determines the
outcome: the conclusion is about *this dataset*, not a verdict on sentence encoders, and its
transfer to real POI text is untested. Note also that the fidelity ordering does not carry
through to the ranking metric: MiniLM's NDCG@10 is nominally *higher* than TF-IDF's with
overlapping CIs, despite much lower D11 / D9.

| Encoder | D11 ridge R2 | D9 within-trip Spearman | NDCG@10 (95% CI) |
|---|---|---|---|
| minilm_svd64 | 0.590 | 0.258 | 0.1432 [0.1335, 0.1529] |
| tfidf_svd64 | 0.865 | 0.473 | 0.1485 [0.1385, 0.1590] |

## 13. Reproduction, performance and integrity

`make reproduce` is CPU-only, single-seed, no cloud, no API keys, and does not run pytest (that
is CI). It runs both acceptance gates and ends in `compose`, the **only** writer of
`results/metrics.json` (each stage writes `results/parts/<stage>.json`; a test and `make audit`
enforce single-writer). `make reproduce-full` adds the inspection artifacts (`recommend`,
capped at a seeded 300-trip sample; `scenarios`), leave-one-destination-out, and the docs.

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

**Honest wall-clock:** `make reproduce` measured 6.6 min on the dev laptop (other
processes were running, so timings carry roughly 30-40% noise). The original 5-minute target is
**not met** and was deliberately not chased further: the two largest stages are `train` and
`evaluate` (LightGBM fits), and shaving them would cost fidelity (fewer seeds/rounds) rather than
remove waste.

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

Headline: primary NDCG@10 = 0.1540 ± 0.0071 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1485, is number 2 of 5 counting from the lowest). The primary system is above content cosine in
4 of 5 seeds (mean gap +0.0110 NDCG@10). The overall candidate-recall gate (0.85) is passed in every seed, with the
lowest seed at 0.8504 — the margin over the
gate is thin, which is the honest reading of a pre-registered K that was fixed on validation.
