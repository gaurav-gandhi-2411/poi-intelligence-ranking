# TECHNICAL.md — `poi-intelligence-ranking`

> **Generated document.** Prose is hand-written in `docs/TECHNICAL.md.tmpl`; every number and
> table is resolved from `results/metrics.json` by `poi_rank.eval.report`
> (`report_templates.py`) — a token that cannot be resolved fails the build, so no figure here
> can be stale or hand-typed. `docs/RESULTS.md` is the fully generated results record;
> `docs/DATA_CARD.md` is the append-only log of every dataset decision.

## Summary — lead with the skew

**The strongest result in this repository is a negative one about our own pipeline.** A train/serve
feature skew was found in it, quantified, its consequence for a *published* conclusion retracted, and
the whole pipeline re-run. The implicit traveler block was computed as-of each trip's `start_date`,
but a trip's browsing session — the source of the graded labels — is dated 0-45 days *before*
`start_date`, so train and validation features carried the labels and holdout features could not. The
skew is visible in the committed data: the median days since the last interaction is
**3 on train trips against
102 on holdout trips**. The first tagged
submission reported the learned ranker as indistinguishable from content cosine
(0.1485 vs
0.1411, p=0.445)
and explained that as evidence for a simple model; that conclusion is **retracted** here (section 5.1).

The three results that stand on top of that:

1. **The primary system beats a well-built content baseline decisively.** NDCG@10
   0.1842 against
   0.1336 for content cosine
   (Wilcoxon p=1.27e-14), above cosine in
   5 of
   5 independently regenerated seeds
   (section 4.1).
2. **The brief's own additive scoring formula lets hard constraints through.** It puts
   974 hard-constraint violations into the
   top-10s of the holdout, against 0
   under the gated multiplicative rule (DR1, section 7).
3. **Oracle-free selection.** The true-utility oracle scored results but never chose anything: every
   selection (retriever K, features, regularisation, ranker sweep, experiment H) was made on train-carved
   validation.

**The holdout, stated precisely.** Holdout feature distributions were inspected to diagnose a
train/serve skew. No labels were used for selection and no hyperparameter was chosen on holdout.
(For completeness, holdout NDCG was *evaluated* — never selected on — three times outside the routine
pipeline run: once on the shipped hyperparameters to measure the effect of the fix, once for the
one-time read of experiment H, and once in the sandbox ablation that isolates H's effect; the K sweep and
the Decision Register also print holdout columns that are explicitly reporting-only.)

**The brief on model complexity** (assignment section 10, quoted exactly): "You are not required to
use a sophisticated model. A simpler model with thoughtful features, a clear learning objective, and
strong evaluation is preferable to unnecessary model complexity." The evidence here is consistent
with that in both directions: the neural alternative is not competitive at this data scale (DR2),
the objective choice is documented against a pointwise alternative (DR3), and — the reason the
retraction above mattered — the claim that a simple model was "enough" was itself an artifact of a
feature bug, not a finding.

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
combines the two (section 7). The end-to-end architecture diagram and the module map are in
`README.md` (sections "Architecture" and "Repository layout").

**Exposure bias** is the second first-class concern. Logged interactions conflate "the
traveler liked it" with "the serving policy showed it". The DGP simulates two exposure policies
(a popularity-biased training log and a uniform-random evaluation log) so the bias is
measurable and correctable rather than hand-waved. Measured on the bias-gap table below:
popularity's NDCG@10 moves by +0.0804 between the unbiased and biased
holdouts (it is flattered by the biased log), the primary IPS-corrected model by
-0.0046.

**The commercial objective is local / long-tail discovery, not raw NDCG.** The incumbents in
inbound-travel already rank by popularity, so a popularity-shaped list is table stakes. The
differentiating claim is surfacing the genuinely relevant, non-obvious POI, and it has to be
backed by long-tail **precision**, not just share. Measured on the unbiased holdout, top-10
lists: long-tail share 0.1436 (popularity ranker:
0.0000), catalog coverage@10
47.3% (popularity:
3.5%); long-tail precision
0.1964 on 952 long-tail recommendations —
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

**Gate-B (representation / candidate generation)** -- blocking rows are oracle-free (recall against EXPOSED holdout positives) and encode requirements: the brief's long-tail requirement and a serving budget (amended before experiment L3b, `docs/experiments/L-final.md`). Overall recall and the chance-lift rows were demoted to reporting: they were a-priori targets disclosed as miscalibrated. Every representation and retrieval hyperparameter was selected on a train-carved validation split, and the oracle-based rows below are reporting-only, computed after the selection was frozen. The oracle never touched a decision; it only ever scored the result.

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

**Two exposure logs, one primary evaluation set.** `interactions_train.parquet` uses a
popularity-biased policy; `interactions_holdout_random.parquet` uses uniform-random exposure
over a wide slate and is the **primary** evaluation population
(671 holdout trips); `interactions_holdout_logged.parquet` repeats the
biased policy over the holdout window only to compute the bias gap. Every logged interaction
carries its `p_expose`, which is what makes IPS possible.

**The bias-gap table** (every system, biased vs unbiased holdout):

| System | NDCG@10 (unbiased) | NDCG@10 (biased) | Gap |
|---|---|---|---|
| 1. Random | 0.0547 | 0.0265 | -0.0282 |
| 2. Popularity | 0.0523 | 0.1327 | +0.0804 |
| 3. Popularity + geo filter | 0.0527 | 0.1326 | +0.0799 |
| 4. Content cosine | 0.1336 | 0.0478 | -0.0858 |
| 5. Item-kNN CF | 0.0604 | 0.1363 | +0.0759 |
| 6. Logistic regression (CV-tuned L2) | 0.1196 | 0.2516 | +0.1320 |
| 7. LambdaMART (no IPS) | 0.1239 | 0.2615 | +0.1377 |
| 8. **LambdaMART + IPS (primary)** | 0.1842 | 0.1796 | -0.0046 |
| 9. Oracle (ceiling) | 0.3682 | 0.1064 | -0.2618 |

**Oracle ceiling — and why "% of ceiling" fell from 66.2% to
50.0%.** Ranking the same candidates by the true
noise-free utility gives NDCG@10 0.3682; the primary system
reaches 50.0% of it. The earlier report (DATA_CARD
history: primary 0.0875 against an oracle of 0.1322 = 66.2%) is **not comparable** to this one, for
two measured reasons, neither of which is "the model got worse" (the primary system's own NDCG@10
went *up*, from 0.0875 to 0.1842):

1. **The old oracle was crippled by the old data-generating process.** Before the simulator
   rewrite the labels were noise-dominated, so even a ranker with the true utility scored only
   0.1322 (DATA_CARD documents this: an oracle at 0.1322 is itself the evidence that labels were
   mostly noise). A model closing 66.2% of a near-noise ceiling is a small absolute number. After
   the rewrite the labels are utility-driven and the ceiling rose to
   0.3682 — the ceiling grew by a larger factor than the
   model's score, so the ratio fell even though the model improved.
2. **The candidate-relative ceiling depends on the candidate set.** The oracle re-ranks *its own*
   candidates, so a retriever that surfaces more relevant POIs raises the ideal too. Measured on
   identical holdout trips (E1), the candidate-relative oracle NDCG@10 is
   0.4037 on the old six-channel set and
   0.3682 on the learned set.

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
(the blend that was learned, and which side wins, is measured in section 3.1). Each trip's implicit block is as-of the trip's own session start
(never `start_date`: section 5.1), and 20 explicit `xf_*` traveler x POI cross features
(localness x touristiness, price gap, interest hits, compatibility sub-scores, repeat engagement,
section 5.2) are added to the ranker's frame (never to the retriever's).

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
  re-measured under the training config of that time, on the PRE-FIX frames of section 5.1 and kept as
  historical evidence): category-affinity mean delta
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
| `-IPS_weighting` | -0.0603 | 1.19e-35 | 605 |
| `-calibration` | -0.0030 | 0.0543 | 605 |
| `-MMR` | +0.0406 | 8.49e-26 | 582 |
| `-interest_channel` | +0.0001 | 0.00769 | 605 |
| `-long_tail_quota` | +0.0004 | 8.3e-06 | 605 |
| `-text_embeddings` | +0.0011 | 0.971 | 605 |
| `-implicit_taste` | +0.0120 | 7.94e-05 | 605 |
| `-explicit_interests` | +0.0050 | 0.0847 | 605 |
| `-behavioral_block` | +0.0015 | 0.971 | 605 |

Sign convention: delta = ablated minus full, so a POSITIVE delta means removing the block
*improved* NDCG@10. On the final model: raw text-embedding block +0.0011
(p=0.971), raw implicit-taste block +0.0120
(p=7.94e-05), explicit interests +0.0050
(p=0.0847), behavioural block +0.0015
(p=0.971). Only the raw implicit-taste block's removal is statistically
significant, and its size is small next to the primary score: the raw taste-vector columns add a little
noise the trees fit, while their information still reaches the model through the engineered
`interact_cos_taste_poi` and the implicit summary columns. Dropping the block is a candidate
improvement that was deliberately **not** applied: choosing it on this holdout number would be
selecting on the holdout (the validation-only sweep of section 5 measured the text block the other way
round; neither result is decisive). `-MMR` is positive and significant by design: the diversity
re-rank trades NDCG@10 for list diversity (section 4.2 and the lambda table in `docs/RESULTS.md`).

**Channel ablations.** The earlier `-CF_channel` row was a **no-op by construction** (the CF quota
is 0 in the shipped candidate generator, so removing it changed nothing and the row carried no
information); it is replaced by `-interest_channel`, a channel that is actually in the shipped
union. Both channel ablations re-score the already-trained ranker over a smaller candidate set, and
both come out *slightly positive* for NDCG@10: `-interest_channel`
+0.0001 (p=0.00769), `-long_tail_quota` +0.0004
(p=8.3e-06). The effects are far below 1% of NDCG@10 and small in absolute terms:
a plausible reading (not separately tested) is that a leaner set leaves the ranker fewer
low-relevance items to misplace. These two channels are **not** justified by top-10 relevance; they are justified by
recall and long-tail exposure (section 4), which NDCG@10 over the exposed labels does not reward.

### 3.1 How explicit and implicit signals are combined (brief section 8) — measured; for trips with history the implicit side wins

The brief asks how stated preferences and behavioural history are combined. The blend is **learned, not
hand-set**, so we measured which side won. Script `scripts/diagnose_touristiness_axis.py`, artifact
`results/parts/touristiness_axis.json`, shipped booster only, nothing re-tuned. The test case is the
touristiness preference, because it is the one stated preference whose conflict with history is easy to
read off the ranking (localness of the recommended POIs).

*What the ranker leans on.* Grouped-SHAP share of the implicit-taste group is
43.2% in the final model (the
59.4% of the pre-fix model was carried by
the leaked features and is not a current figure), against
7.0% for the
features that read `touristiness_pref` (cold-start scenario rows) and
0.9% of total split gain.
Attribution shares compare groups of very different width (100 implicit columns against 6 preference
columns), so this is corroboration, not the proof. The proof is behavioural.

*Is the preference learnable? Yes: the simulator encodes it.* The six preference-dependent columns
(`explicit_touristiness_pref`, `interact_localness_gap`, and the `xf_loc_align`, `xf_loc_gap`,
`xf_loc_x_pref`, `xf_pop_x_pref` crosses) are non-degenerate in the
486441-row training frame (NaN rate
0%, sd
0.24 for the localness x touristiness cross,
within-trip sd 0.19) and every
one is in the booster (the cross splits 1 time(s)).
The preference is spread around zero (sd 0.30 over
1875 travelers; 50%
have |pref| < 0.2, 10% have |pref| >= 0.5). And the
data carries a large behavioural effect: across the
605 holdout trips (random-exposure labels, observable localness
index only), the per-trip rank correlation between a candidate's localness and its outcome falls with the
stated preference at Spearman -0.51
(slope -0.147 per unit of preference), equally for
trips with history (-0.51) and without
(-0.55). So this is **not** a
simulator property: travelers who state a preference for local places do behave that way.

*Does the shipped ranker reproduce it? Only when it has no history.* The same statistic on the raw
ranker score, as a fraction of the label slope:

| Trips | n | Label slope | Ranker slope | Ranker / label |
|---|---|---|---|---|
| Cold-start (no history) | 65 | -0.154 | -0.128 | 83% |
| With history | 540 | -0.147 | +0.002 | -1% |

The counterfactual flip agrees: raw top-10 Jaccard 0.68
for cold-start trips in the top |pref| tercile (n=26) against
0.85 for trips with history (n=198;
same |pref| profile: mean 0.25 against
0.24; after controlling for |pref| a cold-start trip's Jaccard is
-0.10 against a warm trip's). When history
exists, the ranker's localness ordering does not measurably move with the stated preference, even though the
outcomes do. So **when the two signals conflict, implicit history dominates, and warm trips are *less*
responsive to a stated-preference flip than cold-start trips** (Jaccard
0.915 against
0.813). The
headline all-trip Jaccard of 0.904 understates the response where a
preference is actually stated: it is diluted by trips with |pref| near zero, whose flip changes almost nothing
(bottom |pref| tercile 0.973, top tercile
0.834); the negated values are inside the
observed range, so the flip is not an off-distribution probe.

*Mechanism (measured where marked).* The switch is on *whether* history exists, not on how much of it
there is: among trips with history the flip Jaccard is nearly flat in history volume (Spearman
-0.08; history terciles
0.924 /
0.912 /
0.909). That corrects the earlier wording that the
ranker learns "the blend conditioned on evidence volume": it learned a two-regime blend. The implicit
side is not earning that weight either: removing the raw implicit-taste block *improves* holdout NDCG@10
by +0.0120 (p=7.94e-05, section 3 ablation table). Why the trees
prefer history over a stated preference that the labels reward (collinearity of the two, or history's higher
signal-to-noise in training) was **not** separately tested.

*Consequence.* Taste- and history-based personalization is real: lists for different archetypes are almost
disjoint (cross-archetype Jaccard@10
0.075), but the
archetype structure is only partly recovered: the within/cross ratio is
1.20 against 1.00 for no archetype signal
and 1.89 for a perfect ranker,
i.e. 23% of the way. Responsiveness to a stated touristiness
preference is weak for every traveler with history. What would fix it, **untested**, in increasing order of
invasiveness: (i) an explicit preference-consistency term in the utility layer (section 7), tuned on
train-carved validation; (ii) preference-conditioned features whose across-traveler variance forces the
trees to split on them (for example the cross features re-expressed per traveler, or dropout applied to the
implicit block the way it already is to the behavioural block); (iii) a hard filter on the stated preference.
**For production:** on a platform whose differentiator is non-touristy local discovery, a stated preference
must be a hard filter or an explicit utility term, not something the ranker is trusted to learn.

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
DR11, frozen A3 grid at learned K=210, computed on the pre-fix features of section 5.1 and kept
as historical evidence): adding geo, semantic, CF and archetype on top of
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

**How K was chosen (E4, and why it changed).** The learned K went through three stages, each
disclosed: K=210 (a 0.90 validation-recall rule whose margin had been adjusted after seeing the
holdout gap: contaminated), then K=195 from the *pre-registered blind rule* "smallest K whose
IPS-weighted validation recall is at least 0.93" (validation = 20% of the train trips against logged
positives weighted 1/clip(p_expose); the holdout is reporting-only), then — after the feature-skew fix
of section 5.1 removed label information from the retriever's training features — the same rule
re-applied to the corrected features. The corrected retriever is honestly weaker on validation, and
the recall rule now gives **K=270**, whose validation lift over chance is
0.327: below the BLOCKING Gate-B row
(lift >= 0.35), so `make reproduce` would stop at Gate-B. No grid K satisfies both
criteria on validation (**DR13**: the two pre-registered rules are jointly infeasible because
lift over chance is recall minus a chance baseline that grows with K, so they pull in opposite
directions). Gate-B is the binding ruling, so **the shipped K is 240**, chosen by a
validation-only tie-break on the FIRST grid computation (largest K with validation lift at least the
gate plus a 0.01 margin; validation lift 0.363 at K=240 there). The grid committed here was recomputed
later on the final `candidates.parquet` and puts the validation lift at the shipped K at
0.358 (validation recall
0.911): above the gate, just under the
margin, and its margin rule would now give K=225. K was **not re-selected**
(no model changes after the pre-registered decision; the shipped pipeline meets the gate on the
holdout). This is a **deviation from the requester's E4 recall rule**, made from validation columns
only (the holdout column is printed in the same table, which is disclosed) and recorded as Amendment 2
in `docs/experiments/H-ranker-cross-features.md`. The full sweep, holdout column included:

smallest K with IPS-weighted validation overall recall >= 0.93 (val = 20% of train trips vs logged positives, weighted 1/clip(p_expose)); applied once, blind to the holdout column, which is reporting-only.

| Learned K | Val recall (IPS) | Val lift | Holdout recall (reporting-only) | Holdout long-tail | Holdout lift | Candidates/trip |
|---|---|---|---|---|---|---|
| 90 | 0.676 | +0.363 | 0.701 | 0.639 | +0.388 | 151 |
| 105 | 0.713 | +0.379 | 0.739 | 0.678 | +0.405 | 161 |
| 120 | 0.745 | +0.389 | 0.769 | 0.708 | +0.413 | 172 |
| 135 | 0.773 | +0.394 | 0.796 | 0.742 | +0.417 | 183 |
| 150 | 0.802 | +0.399 | 0.820 | 0.771 | +0.418 | 194 |
| 165 | 0.825 | +0.399 | 0.843 | 0.798 | +0.416 | 206 |
| 180 | 0.845 | +0.394 | 0.866 | 0.822 | +0.415 | 218 |
| 195 | 0.864 | +0.389 | 0.883 | 0.842 | +0.407 | 230 |
| 210 | 0.880 | +0.379 | 0.897 | 0.859 | +0.396 | 242 |
| 225 | 0.897 | +0.370 | 0.911 | 0.874 | +0.384 | 254 |
| 240 **(shipped)** | 0.911 | +0.358 | 0.924 | 0.890 | +0.371 | 267 |
| 255 | 0.923 | +0.344 | 0.936 | 0.906 | +0.356 | 280 |
| 270 **(recall rule)** | 0.933 | +0.327 | 0.947 | 0.920 | +0.340 | 292 |
| 285 | 0.944 | +0.311 | 0.957 | 0.934 | +0.323 | 306 |
| 300 | 0.956 | +0.296 | 0.964 | 0.945 | +0.304 | 318 |
| 315 | 0.964 | +0.277 | 0.970 | 0.952 | +0.282 | 332 |
| 330 | 0.973 | +0.258 | 0.976 | 0.960 | +0.260 | 345 |
| 345 | 0.980 | +0.237 | 0.980 | 0.967 | +0.237 | 358 |
| 360 | 0.984 | +0.212 | 0.985 | 0.974 | +0.213 | 372 |

At the shipped K the sweep's holdout column gives overall recall
0.924, long-tail recall
0.890, and chance-corrected lift
+0.371 (candidate set
267 POIs per trip, in the sweep's simplified
non-cross-fitted setup; the shipped pipeline's own recall rows are in the gates and the scorecard).

Recall of the candidate set against EXPOSED holdout positives. Each stratum carries its own chance baseline: the share of that stratum's destination POIs a same-size random candidate set would contain. Raw recall without this lift hid a near-chance failure for four phases.

| Stratum | Recall | Chance | Lift (abs) | Trips |
|---|---|---|---|---|
| long_tail | 0.898 | 0.481 | +0.417 | 605 |
| overall | 0.926 | 0.553 | +0.374 | 605 |
| q1_least_popular | 0.914 | 0.531 | +0.383 | 601 |
| q2 | 0.882 | 0.430 | +0.452 | 600 |
| q3 | 0.907 | 0.510 | +0.398 | 603 |
| q4_most_popular | 0.969 | 0.738 | +0.231 | 605 |

Oracle-relevance recall (top 5% of the catalog by true utility; reporting-only):
0.979 overall,
0.964 long-tail.

**Long-tail quota sweep (DR9)** — the quota barely moves ranking quality or long-tail share, so
the share is set by the ranker, not the quota:

| Long-tail quota | NDCG@10 | Long-tail share@10 | Long-tail precision@10 | Candidate recall | Long-tail recall |
|---|---|---|---|---|---|
| 0 | 0.1844 [0.1740, 0.1949] | 0.106 | 0.338 | 0.916 | 0.871 |
| 100 | 0.1838 [0.1734, 0.1943] | 0.106 | 0.336 | 0.935 | 0.921 |
| 25 | 0.1844 [0.1739, 0.1948] | 0.106 | 0.338 | 0.921 | 0.884 |
| 50 | 0.1842 [0.1738, 0.1945] | 0.106 | 0.338 | 0.926 | 0.898 |

### 4.1 Where does the gain over popularity come from? (E1, re-run after the skew fix)

**Ranking vs retrieval.** Candidate-relative NDCG@10 normalises each candidate set by its *own* ideal
ranking, so it cannot compare two candidate sets: a better retriever surfaces more positives, raising
the ideal and lowering the ratio for the same ranker. The decomposition therefore uses an **end-to-end
NDCG@10 with a fixed denominator** (the ideal over every logged label of the trip, identical for every
candidate set and system), scoring every system on both the shipped learned-retriever set and the
original six-channel set (regenerated from the pre-retriever configuration on the corrected features)
on the same holdout trips. The "retrained" row refits the ranker on the old set own training
candidates:

| End-to-end NDCG@10 (fixed denominator) | 6-channel union (190 cand/trip) | Learned retriever + long-tail + interest (266 cand/trip) |
|---|---|---|
| Random | 0.0512 [0.0460, 0.0564] | 0.0537 [0.0482, 0.0593] |
| Popularity | 0.0644 [0.0579, 0.0715] | 0.0512 [0.0457, 0.0570] |
| Content cosine | 0.1353 [0.1255, 0.1448] | 0.1315 [0.1220, 0.1410] |
| LambdaMART + IPS (shipped booster) | 0.1808 [0.1710, 0.1910] | 0.1814 [0.1712, 0.1923] |
| LambdaMART + IPS (retrained on this set) | 0.1919 [0.1812, 0.2028] | — |
| Oracle (true utility) | 0.3407 [0.3273, 0.3540] | 0.3617 [0.3475, 0.3755] |

For reference, the usual candidate-relative NDCG@10 (each set normalised by its own ideal):

| System | 6-channel union | Learned retriever |
|---|---|---|
| Random | 0.0601 [0.0542, 0.0662] | 0.0547 [0.0491, 0.0604] |
| Popularity | 0.0767 [0.0694, 0.0846] | 0.0523 [0.0469, 0.0582] |
| Content cosine | 0.1603 [0.1494, 0.1712] | 0.1336 [0.1243, 0.1432] |
| LambdaMART + IPS (shipped booster) | 0.2127 [0.2016, 0.2239] | 0.1842 [0.1738, 0.1945] |
| LambdaMART + IPS (retrained on this set) | 0.2263 [0.2141, 0.2385] | — |
| Oracle (true utility) | 0.4037 [0.3893, 0.4187] | 0.3682 [0.3547, 0.3827] |

Reading. (1) **The gain over popularity is a ranking gain.** On the shipped set the primary system is
+0.1303 above popularity end-to-end; cosine alone already gets most
of the way and the learned ranker adds the rest. (2) **The learned retriever buys recall and
long-tail coverage, not top-10 NDCG.** With a ranker retrained on each set, the six-channel set scores
*higher* end-to-end than the learned set (0.1919
vs 0.1814; paired Wilcoxon
p = 0.005), even though its
candidate recall is far lower: a smaller, taste-aligned pool is an easier ranking problem. The
learned retriever is kept because the brief's recall and long-tail requirements are gate rows
(section 4 and the scorecard) and the six-channel set fails them, and the choice is a
recall/coverage-vs-precision trade that this measurement quantifies rather than hides.

**Decomposition of the headline gain** (candidate-relative NDCG@10, shipped set, seed 42):
popularity 0.0523 → content cosine
0.1336
(**+0.0813**, *content matching*; Wilcoxon
p=7.54e-42) → LambdaMART + IPS
0.1842
(**+0.0506**, *learned ranking*;
Wilcoxon p=1.27e-14).

**Five-seed paired comparison against content cosine** (the whole pipeline regenerated per seed; H4):
the primary system is above cosine in 5 of
5 seeds, mean gap
+0.0623 NDCG@10 (sd of the per-seed gap
0.0112), paired t-test p =
0.000239, Wilcoxon signed-rank p =
0.0625 (with only 5 seeds the smallest attainable
Wilcoxon p is 0.0625, so the paired t-test and the per-trip test above carry the significance).

**Conclusion.** The learned ranker beats a well-constructed content-similarity baseline by a
substantial, significant margin once the feature skew of section 5.1 is removed; the earlier
version of this section concluded the opposite, from skewed features, and is retracted. Popularity to
cosine (content matching) is still the larger single step, and the DR2 learning curve shows the
neural alternative is not competitive at this data scale
(0.1624 for a two-tower ranker against
0.1845 for LambdaMART on the full
training set).

### 4.2 Long-tail precision at every stage (E3)

**The 0.40 long-tail precision target was miscalibrated at design time.** It was set a priori,
without reference to how often a long-tail candidate is relevant at all. The measured positive
rate among long-tail candidates in the candidate pool — what any ranker that added no signal would
deliver — is 0.097; a precision of 0.40 would
require roughly four times that. So the primary long-tail precision statistic is **lift over the
pool's base rate**, and raw precision is secondary: the raw ranker delivers
3.538x, and the served list (after the hard
gate, utility and MMR) 2.026x. Raw precision
still misses 0.40 and is reported as a miss in the scorecard, with this note.

The same audit applies to the **0.85 overall candidate-recall target**: it was also set a priori as
an absolute number, with no reference to the chance baseline, which depends on the candidate-set
size. At the shipped set size a random candidate set would already recall
0.553, so 0.85 means a lift of
+0.374 over chance. The target is met in every one of
the five replicated seeds (lowest 0.9066), but
an absolute recall number still says little without the chance baseline.

Measured at each serving stage on the same holdout trips (share = fraction of returned slots that
are long-tail; precision = fraction of those that are positives):

| Stage | Long-tail share | Long-tail precision | Lift over pool base rate |
|---|---|---|---|
| 0_candidate_pool (positive rate among long-tail candidates) | 0.434 | 0.097 | 1.000x |
| 1_raw_ranker_top10 | 0.113 | 0.343 | 3.538x |
| 2_after_hard_gate_raw_order | 0.153 | 0.268 | 2.762x |
| 3_after_utility | 0.162 | 0.262 | 2.699x |
| 4_final_after_mmr | 0.144 | 0.196 | 2.026x |

MMR lambda (diagnostic, post-hoc on the holdout -- not a selection):

| lambda | Long-tail share | Long-tail precision |
|---|---|---|
| 0.5 | 0.161 | 0.184 |
| 0.6 | 0.151 | 0.188 |
| 0.7 | 0.151 | 0.190 |
| 0.8 | 0.144 | 0.196 |
| 0.9 | 0.146 | 0.251 |
| 1 | 0.162 | 0.262 |

Reading: the raw ranker already lifts long-tail precision from the pool's base rate to
0.343
(3.538x). The serving stages then give some of it
back: the hard gate takes it to 0.268, the utility
layer is ~neutral (0.262), and the MMR diversity re-rank takes it to
0.196. The gate and MMR each cost about the same. The lambda sweep
(RESULTS.md, diversity section) is a post-hoc diagnostic on the holdout — it is not a selection and
lambda is unchanged: lambda 0.8 is a deliberate diversity-for-precision trade whose cost is
quantified there, and even with no diversity term (lambda 1) precision stays well under 0.40.

## 5. Ranking model choice — LambdaMART justified over a two-tower ranker

*The assignment (section 10): "You are not required to use a sophisticated model. A simpler model with thoughtful features, a clear learning objective, and strong evaluation is preferable to unnecessary model complexity."*

**Chosen:** LightGBM `objective="lambdarank"`, grouped by trip, IPS-weighted, behavioural-block
dropout (designed for new-POI robustness; its measured effect is in section 10).

**Learning signal, training data and inference (brief sections 5.3 and 10).** *Interaction types to a
label:* `datagen/interactions.py::INTERACTION_LABELS` maps the eight logged interaction types to a graded
relevance label: `view` and `dismiss` 0, `click` 1, `navigate`, `save` and `share` 2, `visit` and `booking`
3. A POI's label for a trip is the **maximum** over its interactions in that trip; a candidate with no
interaction row is label 0 (a true negative on the uniform-random holdout log, an unexposed candidate on
the biased training log). `dismiss` is recorded as a `hard_negative` flag in the log but is not used as a
separate training signal. *Objective:* listwise LambdaRank, one query group per trip, linear label gain
0/1/2/3 (chosen in experiment H, section 5.2), truncation level 20. *Training data:* one row per (trip,
candidate) from the same candidate generator that runs at serving time, train trips only, each row weighted
by the clipped inverse exposure propensity (section 6), 15% behavioural-block dropout, early stopping on
train-carved validation trips (never the holdout). *Inference:* candidates (learned retriever at
K=240, long-tail floor, interest channel) -> the same feature builder -> booster raw
score -> isotonic calibration -> hard-constraint gate and compatibility -> utility -> MMR (lambda
0.8) -> top-10 with `confidence`, `planner_weight` and a grouped-TreeSHAP explanation
(`eval/demo.py::recommend_for_traveler`, `scoring/output.py`).

**Measured against a two-tower neural ranker (DR2, learning curve).** A small two-tower model
(traveler tower and POI tower into a shared space, dot-product score, same IPS weights, same
train-carved early stopping) was trained on 10/25/50/100% of the training trips over three
seeds each, and scored against LambdaMART on the unbiased holdout with 2000-resample trip
bootstrap CIs. Note the two-tower can only use *separable* features; the `interact_*` pair
features cannot enter a dot product, which is part of what is being compared.

| Train fraction | Train trips | LambdaMART + IPS NDCG@10 | Two-tower NDCG@10 |
|---|---|---|---|
| 10% | 183 | 0.1736 [0.1651, 0.1830] | 0.0990 [0.0934, 0.1054] |
| 25% | 457 | 0.1790 [0.1697, 0.1882] | 0.1245 [0.1179, 0.1319] |
| 50% | 914 | 0.1810 [0.1712, 0.1907] | 0.1447 [0.1374, 0.1531] |
| 100% | 1829 | 0.1845 [0.1748, 0.1945] | 0.1624 [0.1535, 0.1717] |

Log-linear extrapolated crossover: ~4,636 training trips (2.5x the 1829 available; 4-point fit, low confidence).

There is no crossover anywhere on the measured range. The extrapolation is a four-point
log-linear fit and is an illustration of scale, not a forecast.

**Honest result on the objective (DR3).** The listwise choice is *not* supported by measurement:
on the holdout the pointwise `binary` objective scores
0.2046 against
0.1842 for `lambdarank` (the confidence intervals
just touch), and the validation-only sweep below also ranks `binary` first. `lambdarank` stays because
the assignment asks for a learning-to-rank model, and because selecting the objective on the holdout
would leak it into a decision (the validation margin, below, is under the pre-registered bar). That
choice **costs roughly two hundredths of NDCG@10** on this data; switching `objective` to `binary` in
`configs/model.yaml` is a one-line change, flagged as the first thing to revisit if the brief allows a
pointwise model.

| Variant | NDCG@10 (95% CI) |
|---|---|
| binary | 0.2046 [0.1939, 0.2159] |
| lambdarank | 0.1842 [0.1738, 0.1945] |
| rank_xendcg | 0.1950 [0.1846, 0.2051] |
| regression | 0.2033 [0.1925, 0.2142] |

**All nine systems, unbiased holdout** (671 trips, bootstrap 95% CI):

| System | NDCG@10 (95% CI) | % of oracle ceiling |
|---|---|---|
| 1. Random | 0.0547 [0.0491, 0.0604] | 14.9% |
| 2. Popularity | 0.0523 [0.0469, 0.0582] | 14.2% |
| 3. Popularity + geo filter | 0.0527 [0.0474, 0.0583] | 14.3% |
| 4. Content cosine | 0.1336 [0.1243, 0.1432] | 36.3% |
| 5. Item-kNN CF | 0.0604 [0.0542, 0.0670] | 16.4% |
| 6. Logistic regression (CV-tuned L2) | 0.1196 [0.1109, 0.1290] | 32.5% |
| 7. LambdaMART (no IPS) | 0.1239 [0.1152, 0.1335] | 33.6% |
| 8. **LambdaMART + IPS (primary)** | 0.1842 [0.1738, 0.1945] | 50.0% |
| 9. Oracle (ceiling) | 0.3682 [0.3547, 0.3827] | 100.0% |

Paired Wilcoxon: `lambdamart_ips` vs popularity p=3.54e-75
(relative lift +252.0%); vs the best baseline (content cosine)
p=1.27e-14, statistically significant at 0.05 (the simple
interest-plus-price content baseline is strong on this data; the tested wins are over popularity and
the no-IPS ranker; the logistic-regression baseline's CI sits below the primary's, but no paired
test against it is stored); `lambdamart` vs `lambdamart_ips`
p=1.19e-35.
The logistic-regression baseline is now regularised (L2 strength chosen by trip-grouped CV
log-loss: C=0.001); the original unregularised fit on 300+
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
| objective | lambdarank | 0.1662 |
| objective | binary | 0.1802 |
| objective | rank_xendcg | 0.1729 |
| IPS clip | 5.0 | 0.1849 |
| IPS clip | 10.0 | 0.1911 |
| IPS clip | 20.0 | 0.1882 |
| IPS clip | 50.0 | 0.1698 |
| IPS clip | none | 0.1315 |
| feature blocks | all_features | 0.1731 |
| feature blocks | no_raw_text_emb | 0.1743 |
| feature blocks | no_raw_taste | 0.1721 |
| feature blocks | no_raw_text_no_taste | 0.1729 |

Winner (4-seed mean 0.1944): objective binary, IPS clip 10.0, blocks no_raw_text_emb; the previously shipped config scores 0.1892.

Re-run on the final pipeline (corrected features, cross features, tuned tree parameters), the
validation winner is objective **binary**, IPS clip
10.0, blocks no_raw_text_emb
(0.1944), against
0.1892 for the shipped configuration (lambdarank,
clip 20, all features): a margin of +0.0052, **below
the +0.010 adoption bar** fixed for experiment H, so it was **not adopted**. The first plain-rule
reading of E2 ("adopt the validation winner") would adopt it; the bar was applied instead because
adopting would need a second holdout read after H, and the five finalists sit within a few
thousandths of each other (the top rows are ties across the raw-block choice). The large effects are
the IPS clip (no clipping is clearly worst) and the pointwise `binary` objective edging listwise
ones, which agrees with DR3 below; `lambdarank` stays because the assignment asks for a
learning-to-rank model, not because it won. The sweep never scored content cosine.

### 5.1 A feature-skew bug found late, and its fix

**What was wrong.** The implicit traveler block (taste vector, category distribution, interaction
counts, ...) is one row per trip and was computed as-of the trip's `start_date`. A trip's browsing
session runs 0-45 days *before* `start_date`, and its interactions are the graded labels. For
**train and validation trips** the implicit features therefore contained the very session being
predicted; for **holdout trips** they cannot (their session is not in the history pool), which is
what serving looks like. The skew is directly measurable in the committed data:

| Implicit-block statistic | Train trips (before fix) | Holdout trips (before fix) | Train trips (after fix) | Holdout trips (after fix) |
|---|---|---|---|---|
| Days since last interaction (median) | 3 | 102 | 23 | 102 |
| Interaction count (mean) | 63.0 | 32.2 | 28.0 | 32.2 |

**The fix.** The per-trip as-of cutoff is now `min(start_date, first logged impression of that trip)`
(`features/traveler_features.py`); holdout trips are unchanged. A regression test
(`tests/test_traveler_features.py::test_assemble_traveler_features_excludes_own_session_of_train_trips`)
pins it. The retriever trains on the same features, so it became honestly weaker on validation (see
the K discussion in section 4).

**Effect, measured once, on the shipped hyperparameters** (a sandbox run of the whole pipeline at seed
42 with only the fix applied and K held at the previous value; nothing was selected on the holdout; the
committed submission before the fix is commit `ec8ac4a`):

| NDCG@10, unbiased holdout | Before fix | Fix only (sandbox) |
|---|---|---|
| Primary LambdaMART + IPS | 0.1485 | 0.1856 |
| Content cosine | 0.1411 | 0.1363 |
| Popularity | 0.0629 | 0.0578 |
| Wilcoxon p, primary vs content cosine | 0.445 | 2.75e-13 |
| % of candidate-level oracle ceiling | 39.6% | 49.9% |

**This retracts the earlier conclusion.** Earlier versions of this document (and the submission as first
tagged) said the learned ranker added little over content cosine and framed that as evidence-based model
selection. That was an artifact of the skew: the ranker was fitting label-bearing features that do not
exist at serving time, so it generalised worse than it should have. With the skew removed the ranker
beats content cosine by a wide, significant margin (section 4.1).

**Why no control caught it.** Train-carved validation shared the skew (validation trips are train
trips), so validation looked *better* than holdout rather than worse; the ~2x gap between
IPS-weighted validation NDCG and holdout NDCG was attributed to the different metrics. The taste
temporal-safety test only exercised `traveler_history_before` on a hand-built fixture, not the per-trip
cutoff `assemble_traveler_features` actually applied. Both are now covered.

### 5.2 Explicit cross features and ranker tuning (experiment H)

Pre-registered in `docs/experiments/H-ranker-cross-features.md` before any run (with two amendments,
both dated and both recorded before the runs they affect). Hypothesis (from the requester): the
ranker cannot see the traveler x POI interaction terms cosine misses. All selection below is on
train-carved validation (IPS-weighted NDCG@10, 4 seeds), with the holdout read once after freezing.

**H0 diagnostics.** Grouped-SHAP share by feature group, on the model before the fix and on the final model:

| Group | Before fix | Final model |
|---|---|---|
| implicit_taste | 59.4% | 43.2% |
| interest_match | 16.5% | 16.8% |
| popularity | 6.4% | 13.0% |
| price_fit | 6.3% | 6.5% |
| localness_fit | 5.1% | 4.3% |
| geo | 1.0% | 2.6% |
| quality | 4.6% | 6.6% |
| party_fit | 0.3% | 4.6% |
| hours | 0.2% | 0.6% |
| novelty | 0.0% | 1.7% |

The requester's hypothesis is only partly supported: localness, party, price and novelty are a
minority of attribution (they were ~12% before the fix), but the dominant group before the fix
(`implicit_taste`, the group carrying the skew) was the artifact, not an unexploited signal. The
10-cross-feature-only probe scored 0.2257 against
0.3167 for the full model on the contaminated
validation (it did not match), and 0.1594 against
0.1728 on the corrected validation (a small model gets most of the way).

**Protocol consequence (Amendment 1).** The pre-registered bar (+0.010 over the shipped config) was
defined on contaminated validation, so H1-H3 were re-run on corrected frames with the bar re-baselined
*before* any candidate was scored: shipped-config corrected-validation baseline
0.1728,
bar 0.1828.

| Candidate (validation, 4-seed mean) | Val IPS-NDCG@10 | vs bar |
|---|---|---|
| H1: all 20 cross features added | 0.1827 | missed by 0.00005 |
| H1 ablation: +localness crosses only | 0.1774 | below |
| H1 ablation: +price crosses only | 0.1755 | below |
| H1 ablation: +interest crosses only | 0.1762 | below |
| H1 ablation: +compatibility sub-scores only | 0.1787 | below |
| H1 ablation: +history (repeat, dismissed) only | 0.1782 | below |
| H2: cosine `init_score`, shipped features (scale 1 / 2) | 0.1652 / 0.1366 | worse than baseline |
| H2: cosine `init_score`, + cross features (scale 1 / 2) | 0.1731 / 0.1441 | worse |
| H3: + cross, num_leaves 15 | 0.1856 | met |
| H3: + cross, leaves 15, linear label gain | **0.1892** | **met (margin 0.0064)** |
| H3: + cross, leaves 15, truncation 40 | 0.1835 | met |
| H3: shipped features + leaves 15 (attribution) | 0.1750 | below |

The H1 result alone missed the bar by a hair; `init_score` residual learning did **not** help (the
"at least as good as cosine by construction" idea hurt validation), and the adopted configuration is
the argmax of the declared joint runs: cross features + 15 leaves + linear label gain (the linear
gain deliberately decouples the training gain from the reported metric gain). Attribution on
validation: leaves alone +0.0022, cross features
on top of leaves +0.0106,
linear gain on top +0.0036.

**Holdout, read once.** Adopted configuration: primary NDCG@10
0.1842. The same pipeline without H (same K, corrected features,
seed 42): 0.1816 — a holdout
difference of +0.0026 NDCG@10, far smaller than the validation gain and inside
the bootstrap CI: the validation improvement did **not** transfer at anything like its validation size
(winner's-curse selection among ~14 validation candidates, plus IPS-weighted validation vs unweighted
holdout, are the likely reasons; not separately tested). H was adopted because the pre-registered rule
said to; the honest reading is that essentially all of the ranker's gain over cosine comes from the
skew fix, not from H.

## 6. IPS correction

Training weight `clip(1/p_expose, 1, clip_high)` renormalised per trip, unexposed candidates at
neutral weight 1.0 (dropping them would remove most negatives). Exposure rate of fit rows:
24.6%. Ablating IPS changes NDCG@10 by the `-IPS_weighting` row above.

**The clip is not NDCG-optimal on the holdout (DR7).** Holdout NDCG@10 rises with the clip up to 50
(0.2131 against 0.1842
at the shipped 20) and is lower again with no clipping (0.1891);
the validation-only sweep prefers clips of 10-20, i.e. validation and holdout disagree on where the
optimum is. Selecting the clip on the holdout would leak it into a decision, so the shipped value stays
and this is reported as an open improvement:

| Variant | NDCG@10 (95% CI) |
|---|---|
| 10 | 0.1751 [0.1658, 0.1852] |
| 20 | 0.1842 [0.1738, 0.1945] |
| 5 | 0.1504 [0.1413, 0.1601] |
| 50 | 0.2131 [0.2008, 0.2253] |
| no_ips | 0.1239 [0.1152, 0.1335] |
| none | 0.1891 [0.1783, 0.2002] |

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
| additive_no_gate (brief) | 974 | 230 / 671 | 0.8603 | 0.1740 |
| additive_with_gate_filter | 0 | 0 / 671 | 0.8708 | 0.1577 |
| multiplicative_gated (production) | 0 | 0 / 671 | 0.8420 | 0.1644 |
| multiplicative_no_gate | 1391 | 284 / 671 | 0.8209 | 0.1813 |

The production rule returns **0**
violations across 6631 recommended slots
(build-blocking test + `make audit`). Additive scoring returns violations because a high
relevance score can outvote a failed factor. NDCG does *not* show a cost for the violations
(section 0, limit 3): the additive rule's NDCG@10
(0.1740) is slightly above
the gated production rule's (0.1644),
and the ungated multiplicative variant is higher still
(0.1813) — i.e. the gate itself
costs a little NDCG on these labels, by construction, and buys the guarantee. That the labels
ignore constraints is a limitation of the simulator, not evidence for the additive rule.

**DR6 — aggregator.** Geometric mean vs min / product / arithmetic mean:

| Aggregator | NDCG@10 (gated) | Hard violations (ungated) | compat@10 (ungated) |
|---|---|---|---|
| arithmetic_mean | 0.1642 | 1476 | 0.8523 |
| geometric_mean (production) | 0.1644 | 1391 | 0.8209 |
| min | 0.1616 | 994 | 0.6193 |
| product | 0.1562 | 903 | 0.4603 |

**DR10 — α×β grid** (gated multiplicative; shipped α=1.0, β=0.7 sits on a flat surface):

| alpha | beta | NDCG@10 | compat@10 |
|---|---|---|---|
| 0.5 | 0.0 | 0.1627 | 0.8217 |
| 0.5 | 0.3 | 0.1648 | 0.8403 |
| 0.5 | 0.7 | 0.1637 | 0.8517 |
| 0.5 | 1.0 | 0.1628 | 0.8579 |
| 0.5 | 1.5 | 0.1585 | 0.8660 |
| 1.0 | 0.0 | 0.1627 | 0.8217 |
| 1.0 | 0.3 | 0.1659 | 0.8360 |
| 1.0 | 0.7 | 0.1644 | 0.8420 |
| 1.0 | 1.0 | 0.1638 | 0.8463 |
| 1.0 | 1.5 | 0.1633 | 0.8528 |
| 2.0 | 0.0 | 0.1627 | 0.8217 |
| 2.0 | 0.3 | 0.1655 | 0.8341 |
| 2.0 | 0.7 | 0.1658 | 0.8366 |
| 2.0 | 1.0 | 0.1653 | 0.8388 |
| 2.0 | 1.5 | 0.1645 | 0.8428 |

## 8. Calibration

Raw LambdaMART scores are unbounded and not comparable across travelers; the output's
`planner_weight` needs calibrated scores. Isotonic regression is fit on a calibration split
carved by trip from the fit frame (disjoint from the early-stopping validation trips and from
the holdout). Measured on the holdout: ECE 0.4711 (naive min-max) →
**0.0300** (isotonic), Brier 0.3317 →
0.0976. The `-calibration` ablation is ~0 by construction (isotonic is
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

- same-destination mean pairwise Jaccard@10: 0.0764
  (74775 pairs); all-pairs (pooled, structurally diluted):
  0.0254
- true-label within / cross Jaccard: 0.0903
  / 0.0752, ratio
  **1.20**
- reference (perfect ranker): ratio
  1.89
- K-Means proxy ratio: 1.19

**Confidence.** The confidence-decile Spearman is 0.358, below
the ≥0.6 target (a MISSED scorecard row, diagnosed there; it was
0.879 before the feature-skew fix of section 5.1).

**Cold start / new POIs / LODO.** New-POI cohort with vs without behavioural dropout: NDCG@10
0.6019 vs 0.6092
(paired Wilcoxon p=0.328: not statistically
significant at this cohort size; the point estimate is lower with dropout, so the new-POI
benefit the dropout was designed for is not demonstrated here).
Leave-one-destination-out (three retrains): see the cold-start section of `docs/RESULTS.md`.

**Scenarios** (Seoul; spec.md section 15's three required profiles plus the touristiness-flip
diagnostic, narrated as inbound personas): the flip's top-10 overlap is
0.538 (target ≤ 0.35), from candidate pools
that overlap at 0.717.

### 10.1 The two rows the skew fix regressed, diagnosed

**Scenario 4 (touristiness flip): overlap 0.176 →
0.538.** Both scenario-4 travelers are synthetic
and cold-start (zero interaction history), so this row measures how much `touristiness_pref` *alone*
moves the ranking for a brand-new user. Grouped TreeSHAP of the ranker on the scenario candidate rows,
for the model as first tagged (measured by running the same script against a checkout of commit
`ec8ac4a`) and for the final model:

| Mean absolute attribution share, scenario-1 rows | Pre-fix model | Final model |
|---|---|---|
| POI-side features | 17.5% | 35.2% |
| Stated traveler attributes (`explicit_*`) | 1.1% | 13.8% |
| Traveler-history features (`implicit_*`, all zero/absent for these travelers) | 36.2% | 21.1% |
| Pair features (`interact_*`) | 45.2% | 12.6% |
| Cross features (`xf_*`) | not in that model | 17.3% |
| **Features that read `touristiness_pref`** (preference, localness gap and the `xf_*` crosses) | **1.0%** | **7.0%** |
| `localness_fit` group | 2.0% | 8.5% |
| `interest_match` group | 9.7% | 17.2% |

What the flip actually moves (only `touristiness_pref` differs between scenarios 1 and 4; POIs in both
candidate pools): in the final model 57.2% of the
change in attribution sits in the `localness_fit` group, and the raw ranker score changes by on average
0.80 of its own spread across candidates (pre-fix model:
0.68); the two score vectors stay strongly rank-correlated
(0.933 vs 0.870), and the raw
top-10 overlap on the shared pool is 0.818 against
0.250; the two candidate pools themselves overlap at
0.717 (pre-fix 0.612).

The same flip applied as a **counterfactual to the 671 real holdout trips** (raw ranker
top-10 before the gate/utility/MMR layers, every dependent feature recomputed): mean top-10 Jaccard
0.904 (pre-fix model 0.882); trips with history
0.915, the 72 pure cold-start trips
0.813, the 67 trips with a strong
stated preference (|pref| >= 0.5) 0.822. And the personalization metrics
on real holdout travelers, with the DGP's TRUE archetype labels: cross-archetype Jaccard@10
0.075 (target <= 0.25, met) and within/cross ratio
1.20 against 1.89 for a
perfect ranker.

*Finding (measured; interpretation marked).* Personalization by taste and history is real: lists for
different archetypes are almost disjoint (cross-archetype Jaccard 0.075).
`touristiness_pref` is a **weak lever** in the final model — the features that read it carry
7.0% of attribution for a cold-start traveler, and
negating it changes the raw top-10 of a real trip by roughly one item in ten (more for cold-start and
strong-preference trips: Jaccard 0.81 and
0.82). For a brand-new traveler the ranking is driven mainly by stated
interests, popularity/quality and compatibility, not by the touristiness preference. The pre-fix value
was *not* a sign of stronger personalization: the pre-fix model put 82.5% of its
attribution on the (leak-trained) implicit-taste group and only 1.0% on
preference features, yet the flip changed about three quarters of its raw top-10 — an off-distribution
response of a model trained on features that always contained a rich (leaked) history, applied to
travelers with none (interpretation; the measured facts are the numbers above). The row stays MISSED
and is reported, not tuned away. The follow-up diagnosis (section 3.1) sharpens what it is: the
simulator does encode the preference in outcomes, the ranker reproduces
83% of that gradient for trips
without history and -1% for trips with
history, so this is a model limitation (implicit history overrides the stated preference), not a
simulator property.

**Confidence-decile Spearman: 0.879 →
0.358.** The scorecard statistic is a rank correlation over
only ten decile means (about sixty trips each). Recomputed here with the scorecard's own code for both
models, and decomposed into the five terms of the confidence score (equal weights, each
0.2):

| | Pre-fix model | Final model |
|---|---|---|
| Decile Spearman (scorecard statistic) | 0.879 | 0.358 |
| Trip-level Spearman, confidence vs per-trip NDCG@10 | 0.221 | 0.064 |
| Trip-level Spearman, traveler-evidence term vs NDCG | +0.131 | -0.040 |
| Trip-level Spearman, ensemble-agreement term vs NDCG | +0.050 | +0.156 |
| Between-trip sd of the traveler-evidence term (other terms: at most 0.095) | 0.277 | 0.277 |
| Holdout interaction count (mean; share of trips with none) | 32.5; 10.7% | 32.5; 10.7% |
| Ensemble sd over the top-10 (mean) | 0.345 | 0.201 |
| Spearman, interaction count vs NDCG | +0.131 | -0.040 |

*Mechanism (measured).* The confidence inputs on holdout did not change distribution (interaction
counts are identical, since the fix does not touch holdout features), and the traveler-evidence term
dominates the between-trip variation of confidence (its sd is several times that of any other term).
What changed is the *ranker*: before the fix its NDCG rose with the traveler's history volume
(Spearman +0.131 between interaction count and NDCG), because it had been trained on
features that leaked the answer for well-documented trips; after the fix NDCG no longer depends on
history volume (-0.040), so an evidence-volume-dominated
confidence score has nothing to track. The ensemble-disagreement term became the most informative one
(+0.156) but contributes little variance. Read plainly: the pre-fix
decile figure was a ten-point statistic of a weak trip-level relationship
(0.22) that came from the leaked features, i.e. it was
inflated; 0.358 is the honest post-fix value and the trip-level relationship is close to
nothing. Reweighting the terms towards ensemble agreement would improve the number but would be
tuning on holdout labels, so it is declined and listed as an open follow-up.

### 10.2 Three targets were miscalibrated at design time

The a-priori targets were audited alongside the results, and three were set without reference to the
data's own baselines: **long-tail precision 0.40** (the pool's positive rate among long-tail
candidates is 0.097, so the target implied a lift of about four times
that was never justified; the served list achieves 2.03x);
**candidate recall 0.85** (an absolute number with no chance baseline: a random set of the shipped size
already recalls 0.553); and the **chance-lift gate of +0.35**, which
is what made the K rule infeasible (DR13): lift over chance is recall minus a chance baseline that
grows with the set size, so "smallest K with validation recall >= 0.93" and "lift >= 0.35" pull in
opposite directions as K grows and cannot both hold at this catalog size. Of the seven MISSED
scorecard rows, **two trace to target miscalibration rather than system performance**: long-tail
precision (above) and the within/cross-archetype ratio target of 2.0 (a *perfect* ranker reaches only
1.89 in this simulator, so the target is unattainable by construction).
The targets were not relaxed to make rows pass; they are reported as miscalibrated and the measured
values stand.

### Scorecard — every miss carries a diagnosis

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

**Amendment L3a (Gate-B, written before experiment L3b ran; `docs/experiments/L-final.md`).** The chance-lift gate
is the third of the a-priori thresholds disclosed above, and it was a *blocking* rule that could decide between
retrieval designs: it made the pre-registered K rule infeasible (DR13), and any union design fails it because a
larger set raises the chance baseline the lift is measured against. Blocking gates now encode requirements only:
long-tail candidate recall at least 0.75 (the brief section 9 requirement) stays blocking, and a serving budget of
at most 300 effective candidates per trip is added; overall recall (reference 0.85) and the two chance-lift rows
(reference +0.35) are demoted to reporting and still shown for every design. Decision Register rows DR11-DR13 were
written under the four-row gate and are historical.

### 10.3 Cold start (brief section 15)

Three cases, one mechanism: every history-dependent feature is computed as-of the trip and is absent
(NaN or zero, never imputed from other travelers) when there is no history, and the ranker is trained with behavioural-block dropout so it has
seen that regime. Nothing in the ranker reads a destination id (no `dest*` column among its features;
popularity and localness are within-destination percentiles).

- **New traveler (no history).** The explicit block (interests, budget, party, mobility, touristiness
  preference) and the interest and long-tail candidate channels carry the ranking. Measured on the holdout:
  NDCG@10 0.187 for the
  72 trips with no history against
  0.182 for the
  539 trips with more than 10 interactions. Cold-start is
  *not worse* here, which is a property of this simulator (stated interests are informative by design), not
  a general claim. For the stated touristiness preference the cold-start ranker does respond (section 3.1).
- **New POI (no interactions).** The POI is represented by its text embedding, structured, category and
  geo features; behavioural aggregates are missing and dropped out in training. The
  71 POIs created after the train/holdout boundary form the cohort:
  cohort-restricted NDCG@10 0.602 with dropout,
  0.609 without (paired Wilcoxon
  p=0.328, 318 trips). The
  cohort number is computed over cohort candidates and is not comparable to the headline NDCG@10; the
  benefit of the dropout is **not demonstrated**.
- **New destination (little or no history).** Because no feature is destination-specific, a destination the
  model never saw is scored like any other. Leave-one-destination-out (three retrains): NDCG@10 of
  0.184 / 0.177 / 0.184
  (barcelona / kyoto / seoul) against
  0.178 / 0.190 / 0.185
  with the destination in training; only kyoto differs significantly
  (p=0.018). The three destinations share one
  simulator, so this is transfer between similar cities, not to a different market.

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

**Model serving.** At request time: (1) fetch the traveler's as-of history and the destination's POI
feature rows from a feature store; (2) the retriever scores the destination's POIs and the long-tail floor
and interest channel are unioned in (a few hundred candidates); (3) build the same features as in training;
(4) one booster call plus the calibrator; (5) hard gate, compatibility, utility, MMR and template
explanations on the returned rows only. Steps 3-5 are the code in `eval/demo.py::recommend_for_traveler`;
the feature store, the caching and the latency budget are prose here, not measured.

**Data freshness.** POI information (name, price, accessibility) changes rarely: nightly batch refresh of
the POI feature table. Opening hours and availability change intra-day and are read at request time, never
baked into a batch feature (the hard gate and `hours_fit` use them). Popularity and behavioural aggregates
refresh daily with time decay, so a brand-new or trending POI moves quickly without a retrain. A traveler's
own new interactions enter the implicit block on the next request.

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

### 12.1 Three decisions worth reading first (DR3, DR12, DR13)

**DR3 — the objective (a strength, not a footnote).** LightGBM `lambdarank` is the shipped objective.
The pointwise `binary` objective is better on both measurements: **+0.0052
NDCG@10 on validation** (the pre-registered adoption bar was +0.010, so it was not adopted) and
**+0.0204 on the holdout** (0.2046 against
0.1842). It was **not adopted** because adopting on the
holdout number would be selection leakage and the validation margin is under the bar; `lambdarank` is kept
as the principled listwise default when NDCG@10 is the reported metric. The cost is stated in the open, and
switching is a one-line change (`objective` in `configs/model.yaml`).

**DR12 — retrieval design.** The learned full-catalog retriever is compared with the original
six-channel union, each with a ranker retrained on it:

| | Learned retriever | Six-channel union |
|---|---|---|
| Candidate recall, overall | 0.926 | 0.596 |
| Candidate recall, long-tail | 0.898 | 0.540 |
| Lift over chance, overall / long-tail | +0.374 / +0.417 | +0.203 / +0.163 |
| Gate-B rows passed (of 4) | 4 | 0 |
| End-to-end NDCG@10, ranker retrained on the set | 0.1814 | **0.1919** |

The retriever wins recall and long-tail exposure and passes the Gate-B rows; the six-channel union wins
end-to-end NDCG@10 (paired Wilcoxon p=0.005). The
retriever is shipped and the NDCG cost is stated in the open. Switching on a holdout comparison would be
selection leakage, which is why it was not switched.

**DR13 — K selection.** The pre-registered blind rule (smallest K with validation recall >= 0.93) gives
K=270, which fails the Gate-B chance-lift row on validation
(0.327 against 0.35). The two pre-registered
rules are jointly infeasible ([] is the set of
grid K meeting both), because lift = recall - chance and chance grows with K. The validation-only
tie-break (largest K with validation lift >= 0.36) gave K=240; see the
note on the recomputed grid in section 4. The full sweep:

smallest K with IPS-weighted validation overall recall >= 0.93 (val = 20% of train trips vs logged positives, weighted 1/clip(p_expose)); applied once, blind to the holdout column, which is reporting-only.

| Learned K | Val recall (IPS) | Val lift | Holdout recall (reporting-only) | Holdout long-tail | Holdout lift | Candidates/trip |
|---|---|---|---|---|---|---|
| 90 | 0.676 | +0.363 | 0.701 | 0.639 | +0.388 | 151 |
| 105 | 0.713 | +0.379 | 0.739 | 0.678 | +0.405 | 161 |
| 120 | 0.745 | +0.389 | 0.769 | 0.708 | +0.413 | 172 |
| 135 | 0.773 | +0.394 | 0.796 | 0.742 | +0.417 | 183 |
| 150 | 0.802 | +0.399 | 0.820 | 0.771 | +0.418 | 194 |
| 165 | 0.825 | +0.399 | 0.843 | 0.798 | +0.416 | 206 |
| 180 | 0.845 | +0.394 | 0.866 | 0.822 | +0.415 | 218 |
| 195 | 0.864 | +0.389 | 0.883 | 0.842 | +0.407 | 230 |
| 210 | 0.880 | +0.379 | 0.897 | 0.859 | +0.396 | 242 |
| 225 | 0.897 | +0.370 | 0.911 | 0.874 | +0.384 | 254 |
| 240 **(shipped)** | 0.911 | +0.358 | 0.924 | 0.890 | +0.371 | 267 |
| 255 | 0.923 | +0.344 | 0.936 | 0.906 | +0.356 | 280 |
| 270 **(recall rule)** | 0.933 | +0.327 | 0.947 | 0.920 | +0.340 | 292 |
| 285 | 0.944 | +0.311 | 0.957 | 0.934 | +0.323 | 306 |
| 300 | 0.956 | +0.296 | 0.964 | 0.945 | +0.304 | 318 |
| 315 | 0.964 | +0.277 | 0.970 | 0.952 | +0.282 | 332 |
| 330 | 0.973 | +0.258 | 0.976 | 0.960 | +0.260 | 345 |
| 345 | 0.980 | +0.237 | 0.980 | 0.967 | +0.237 | 358 |
| 360 | 0.984 | +0.212 | 0.985 | 0.974 | +0.213 | 372 |

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

**DR4 (TF-IDF vs MiniLM) — read the caveat.** The synthetic corpus is generated from anchored,
synonym-rich phrase pools over latent dimensions, so its vocabulary design determines the
outcome: the conclusion is about *this dataset*, not a verdict on sentence encoders, and its
transfer to real POI text is untested. Note also that the fidelity ordering does not carry
through to the ranking metric: MiniLM's NDCG@10 is nominally *higher* than TF-IDF's with
overlapping CIs, despite much lower D11 / D9.

| Encoder | D11 ridge R2 | D9 within-trip Spearman | NDCG@10 (95% CI) |
|---|---|---|---|
| minilm_svd64 | 0.590 | 0.258 | 0.1758 [0.1656, 0.1861] |
| tfidf_svd64 | 0.865 | 0.473 | 0.1786 [0.1680, 0.1893] |

## 13. Reproduction, performance and integrity

`make reproduce` is CPU-only, single-seed, no cloud, no API keys, and does not run pytest (that
is CI). It runs both acceptance gates and ends in `compose`, the **only** writer of
`results/metrics.json` (each stage writes `results/parts/<stage>.json`; a test and `make audit`
enforce single-writer). `make reproduce-full` adds the inspection artifacts (`recommend`,
capped at a seeded 300-trip sample; `scenarios`), leave-one-destination-out, and the docs.

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

**Honest wall-clock:** `make reproduce` measured 5.6 min on the dev laptop (other
processes may have been running, so timings carry roughly 30-40% noise), which
is above the original 5-minute target and was not chased further. The two largest stages are `train` and `evaluate` (LightGBM fits, now
also computing the cross features of section 5.2); shaving them would cost fidelity (fewer
seeds/rounds) rather than remove waste.

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

Headline: primary NDCG@10 = 0.1967 ± 0.0083 (mean ± sd over 5 independently regenerated seeds; the committed seed 42, 0.1842, is the LOWEST of the five). The primary system is above content cosine in
5 of 5 seeds (mean gap +0.0623 NDCG@10). The overall candidate-recall gate (0.85) is passed in every seed, with the
lowest seed at 0.9066.
