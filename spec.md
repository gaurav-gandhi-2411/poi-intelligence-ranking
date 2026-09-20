# spec.md — Personalized POI Intelligence & Ranking System

**Owner:** GG · **Deadline:** 2026-09-20 EOD (submission 2026-09-21 21:00 KST)
**Repo:** `poi-intelligence-ranking` (public) under `ml-projects/`
**Runtime target:** full pipeline reproduces in < 5 minutes, CPU only, no GPU, no paid APIs.

---

## 0. Grading model (what we are actually optimizing)

The brief states: *"we are most interested in your problem formulation, ML reasoning, ranking approach, evaluation, and engineering decisions rather than a production-ready application."*

Therefore the ranked priorities are:

1. **Evaluation validity** — the submission must not be circular. (§2)
2. **Problem formulation** — two-factor utility, exposure bias, long-tail.
3. **Ranking approach** — justified model choice with baselines and ablations.
4. **Engineering** — one-command reproducibility, clean module boundaries, tests.
5. Code volume — explicitly *not* a goal. Do not pad.

Every claim in `docs/` must be backed by a number emitted by `evaluate.py`. **No unverified metric may appear in any document.** If a number is not produced by the pipeline, it does not go in the docs.

---

## 1. The circularity problem and our defense

Synthetic data means we author the data-generating process (DGP) *and* the model. Naively, the model just re-learns our generator and NDCG is meaningless.

### 1.1 Hard rules

- **DGP firewall.** `src/poi_rank/datagen/**` must not import from `features/`, `models/`, `candidates/`, or `scoring/`. A unit test asserts this (`tests/test_firewall.py` parses imports with `ast`).
- **Latent variables are never exported.** Traveler latent taste vectors, POI latent quality/localness, and the true utility function live only in `data/synthetic/_oracle/` which is loaded **exclusively** by `eval/oracle.py`. No feature code may read that directory. Test asserts this too.
- **Stated interests are a lossy projection of latent taste**: top-k of the latent vector, with label noise (10% of stated interests are drawn at random; 20% of genuinely high-latent interests are omitted). This guarantees explicit features alone cannot reach the ceiling — implicit behavior must add real information.

### 1.2 DGP design (`datagen/`)

Latent traveler utility for POI *p*:

```
u(t,p) = w_taste · cos(taste_t, poi_semantic_p)
       + w_cat  · taste_t[category_p]
       + w_local· localness_p · touristiness_pref_t        # signed
       + w_qual · latent_quality_p                          # not = rating
       + w_party· party_fit(p, party_type_t)
       + w_price· price_fit(p, budget_t)
       + w_novel· novelty_t(p)                              # repeat-visit decay
       + ε,  ε ~ N(0, σ)                                    # irreducible noise
```

- `latent_quality_p` is observed only *noisily* through `rating` and `review_count` — so the model can never fully recover it. This creates a genuine Bayes ceiling.
- Choice is stochastic: **Plackett–Luce over softmax(u/τ)** within an impression slate, not argmax.
- Travelers are drawn from **8 archetype mixtures** (local-food explorer, history/architecture, family-activities, nightlife, nature/outdoor, shopping, luxury-gastronomy, budget-backpacker) with per-traveler Dirichlet mixing + noise, so archetypes are soft, not classes.

### 1.3 Exposure simulation (the key asset)

Two logs are generated:

| Log | Exposure policy | Use |
|---|---|---|
| `interactions_train.parquet` | **Popularity-biased logging policy**: `p(expose) ∝ popularity^1.5 × geo_prox`, top-20 slates | Training + behavioral features |
| `interactions_holdout_random.parquet` | **Uniform-random slates** over the destination catalog | **Primary unbiased evaluation set** |
| `interactions_holdout_logged.parquet` | Same biased policy, later time window | Secondary; demonstrates the bias gap |

Logged propensities `p_expose` are stored, enabling IPS.

**This is the single highest-value element of the submission.** It lets us report:
- NDCG on biased holdout vs unbiased holdout → the *bias gap* for a popularity baseline vs our model.
- IPS-corrected training lift as a measured delta.

### 1.4 Oracle ceiling

`eval/oracle.py` ranks the unbiased holdout by true `u(t,p)` (noise-free component). Report:

```
NDCG@10:  model 0.XXX   |  oracle 0.YYY  |  model = ZZ% of ceiling
```

All headline numbers in `docs/RESULTS.md` are reported as absolute **and** as % of ceiling.

---

## 2. Synthetic dataset spec

Committed to `data/synthetic/`. Target total size < 25 MB.

### 2.1 Scale

| Entity | Count |
|---|---|
| Destinations | 3 — Seoul, Kyoto, Barcelona |
| POIs | ~500 per destination (~1,500 total) |
| Travelers | 600 (200 per destination home-market mix) |
| Trips | ~800 (some travelers take 2) |
| Impressions (train log) | ~120,000 |
| Positive interactions | ~25,000 |
| Unbiased holdout impressions | ~30,000 |

Split is **temporal**: train window = first 80% of timeline, holdouts = last 20%. No random splits anywhere (prevents behavioral-feature leakage).

### 2.2 POI schema

`poi_id, destination, name, category, subcategory, lat, lon, description, tags[], price_level(1-4), rating, review_count, foreign_review_ratio, popularity_raw, expected_duration_min, opening_hours (7×open/close, may be null/irregular), reservation_required, reservation_lead_days, accessibility{wheelchair, stroller, kid_friendly}, indoor_outdoor, seasonality[], avg_crowd_by_hour[24], created_at`

**Deliberate dirtiness to demonstrate §6.1 awareness** (documented in `docs/DATA_CARD.md`):
- ~4% near-duplicate POIs (name variants + <50 m apart)
- ~8% missing `price_level`, ~5% missing `expected_duration`, ~12% missing/irregular `opening_hours`
- Inconsistent category strings (`"Museum"`, `"museums"`, `"Museum / Gallery"`)
- ~15% sparse POIs (`review_count < 10`)
- ~5% brand-new POIs (`created_at` inside holdout window, zero interactions) → new-POI cold-start cohort
- Long-tail review_count distribution (log-normal), so popularity bias is real

### 2.3 Traveler / trip schema

`traveler_id, home_market, trip_id, destination, start_date, end_date, trip_duration_days, party_type{solo,couple,family_young_kids,family_teens,friends}, party_size, budget{low,medium,high}, mobility{walk,public_transport,car,mixed}, interests[], touristiness_pref ∈ [-1,1], pace{relaxed,moderate,packed}, accessibility_needs[], stay_lat, stay_lon, explicit_preferences (free text), dietary[]`

### 2.4 Interaction schema

`traveler_id, trip_id, poi_id, slate_id, interaction_type, timestamp, position_in_slate, p_expose`

Taxonomy and graded label mapping:

| Type | Label | Notes |
|---|---|---|
| `booking` | 3 | strongest intent |
| `visit` | 3 | |
| `navigate` | 2 | |
| `save` | 2 | |
| `share` | 2 | |
| `click` | 1 | weak positive |
| `view` (impression only) | 0 | **true negative** — this is why impressions matter |
| `dismiss` | 0 + hard-negative flag | separate feature + used in traveler negative taste vector |

Rationale documented: because we log impressions, we have *observed* negatives and do not need negative sampling — this materially changes the training-data construction story.

---

## 3. Pipeline architecture

```
datagen ──► data prep ──► feature store (parquet)
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
      POI representation  Traveler repr.  Behavioral aggregates
             └────────────────┬────────────────┘
                              ▼
                   Candidate generation (design-time: 6 channels; SHIPPED: learned + long-tail + interest, see docs/TECHNICAL.md s4)
                              ▼
                   LambdaMART ranker (IPS-weighted)
                              ▼
                   Calibration (isotonic) ──► preference_score
                              ▼
                   Compatibility scoring (multiplicative + hard gates)
                              ▼
                   Utility = rel^α × compat^β ──► MMR diversify
                              ▼
                   Explanations (grouped TreeSHAP → templates)
                              ▼
                   Ranked, weighted POI output (JSON)
```

---

## 4. Data preparation (`data/`)

- **Dedup**: blocking on destination + H3 cell, then `rapidfuzz` token_set_ratio ≥ 88 AND haversine < 50 m → merge (sum review_count, weighted rating).
- **Category canonicalization**: mapping table → 12 canonical categories × subcategories. Unmapped → `other`, logged.
- **Rating shrinkage** (empirical Bayes): `r_shrunk = (v·R + m·C)/(v+m)`, `C` = destination mean, `m` = destination median review_count. Prevents 5.0-with-3-reviews from dominating.
- **Popularity**: `pop_pct = percentile_rank(log1p(review_count) × r_shrunk)` **within destination**. All popularity-derived features are within-destination percentiles — this is what enables new-destination transfer.
- **Localness index** (composite, *not* inverse popularity):
  `localness = z(-pop_pct) · 0.35 + z(-foreign_review_ratio) · 0.30 + z(haversine_to_tourist_centroid) · 0.20 + z(local_tag_hits) · 0.15`
  Tourist centroid = review-count-weighted centroid of top-decile POIs.
  **Validation**: report Spearman ρ between `localness` and the DGP's latent localness. Target ρ > 0.6. This is a measured number, printed by `evaluate.py`.
- **Opening hours** → 168-bit weekly mask + `hours_missing` flag; imputation by category median schedule.
- **Missing values**: median-by-(destination, category) + explicit `_was_missing` indicator columns (LightGBM handles NaN natively, but indicators are kept for explainability).
- **Geo**: haversine helpers, H3 resolution 8 cells, distance-to-stay, distance-to-nearest-transit-node (synthetic transit graph per destination).

---

## 5. POI representation (`features/poi_features.py`)

Hybrid, justified in `docs/TECHNICAL.md`:

| Block | Features |
|---|---|
| **Text** | `all-MiniLM-L6-v2` embedding of `name + description + tags` (384d) → TruncatedSVD to 64d fitted on the catalog. **Cached to `artifacts/poi_emb.npy` and committed** (~1.5k × 384 fp16 ≈ 1.1 MB). Deterministic fallback: TF-IDF(1,2-gram) → SVD-64 if `sentence-transformers` unavailable or offline. Fallback path is tested in CI. |
| **Numeric** | shrunk rating, log review_count, pop_pct, localness, price_level, expected_duration, open_hours_per_week, reservation_lead_days, crowd_index |
| **Categorical** | canonical category, subcategory, indoor/outdoor, price_level (LightGBM native categorical) |
| **Geo** | lat/lon, H3 cell, dist_to_tourist_centroid, dist_to_transit, POI density in 500 m |
| **Behavioral** (train-window only) | impressions, smoothed CTR, save_rate, visit_rate, dismiss_rate, unique_travelers, archetype affinity profile (8d: normalized engagement share by traveler archetype) |

Why hybrid: text embeddings separate *"Small local neighborhood experience"* from *"Highly popular tourist attraction"* within the same category (§7 of the brief) where structured features cannot; behavioral features capture what text cannot; structured features carry constraint semantics. Ablation (§10) quantifies each block's contribution.

---

## 6. Traveler representation (`features/traveler_features.py`)

**Explicit block:** interests multi-hot (14d), budget ordinal, party_type one-hot, mobility one-hot, `touristiness_pref`, pace, accessibility needs, trip_duration, season, days_remaining.

**Implicit block:** taste vector in POI embedding space:

```
taste_t = normalize( Σ_i  w(type_i) · exp(-Δt_i / τ) · emb(poi_i) )
τ = 180 days half-life
w: booking 1.0, visit 1.0, navigate 0.7, save 0.6, share 0.5, click 0.2, view 0.05, dismiss -0.8
```

Plus: implicit category distribution, implicit mean price_level, implicit mean localness, implicit mean pop_pct, interaction count, days since last interaction, distinct categories engaged (breadth).

**Combination of explicit and implicit — the answer to brief §8:**
Both blocks are fed to the ranker as features, plus interaction features (`cos(taste_t, emb_p)`, `|implicit_localness − touristiness_pref|`, `interest_match_score`, `price_gap`). **The ranker learns the blend conditioned on evidence volume**, which is strictly better than a hand-tuned α because the optimal weighting is not constant across travelers.

A confidence shrinkage `α_t = n_t / (n_t + k)`, `k = 5`, is used **only** in the cold-start fallback path and in the confidence score. Documented explicitly as *two different mechanisms for two different jobs*.

**Ablation required:** explicit-only vs implicit-only vs both. If "both" does not beat "explicit-only" on the unbiased holdout, the implicit representation is not earning its place and we say so.

---

## 7. Candidate generation (`candidates/`)

From ~500 POIs/destination → ~250 candidates. Union of channels with **hard quotas**:

| Channel | Method | Quota |
|---|---|---|
| Geo | H3 k-ring from stay point, radius by mobility (walk 2 km, transit 8 km, car 25 km) | 60 |
| Interest/category | POIs in stated interest categories, ranked by shrunk rating | 50 |
| Semantic | brute-force cosine(taste_t, poi_emb) top-k (1.5k POIs → ~2 ms, no faiss) | 50 |
| Collaborative | item-item kNN on normalized co-interaction matrix, seeded by traveler history | 40 |
| **Long-tail exploration** | `pop_pct < 40` AND semantic_sim > threshold; ε-greedy sampled | **50 (hard floor)** |
| Archetype prior | nearest archetype centroid's top POIs (cold-start path) | 30 |

**Brief §9 explicitly asks how we avoid killing the long tail.** Our answer, and it is measurable:
1. Hard long-tail quota that popularity channels cannot cannibalize.
2. Semantic channel is popularity-agnostic by construction.
3. **Metric:** `candidate_recall@250` computed *separately for the bottom-50% popularity stratum*. Report per-channel **marginal** recall (leave-one-channel-out), not just union recall — a channel that adds no marginal recall gets deleted.

Target: overall candidate recall@250 ≥ 0.90, long-tail candidate recall ≥ 0.80.

---

## 8. Ranking model (`models/`)

**Chosen: LightGBM `objective="lambdarank"` (LambdaMART), grouped by `trip_id`, `lambdarank_truncation_level=20`, label_gain for graded 0–3.**

Justification to be written in `docs/TECHNICAL.md` and defended:
- Data scale (~800 trips × ~250 candidates ≈ 200k rows) is squarely tabular-GBM territory; a two-tower or neural ranker has more parameters than useful signal and will overfit.
- Listwise objective directly optimizes the reported metric (NDCG), unlike pointwise/pairwise surrogates.
- Native NaN + categorical handling removes preprocessing failure modes.
- TreeSHAP gives exact, fast per-prediction attributions → explainability comes free rather than bolted on.
- **Explicitly document what would change the decision**: >10M interactions, need for real-time embedding retrieval, or multi-task objectives → then two-tower retrieval + GBM re-rank.

**IPS correction:** training example weight `w_i = clip(1 / p_expose_i, 1, 20)`, normalized per group. Report NDCG on the unbiased holdout with and without IPS as a measured ablation.

**New-POI robustness:** during training, randomly mask the entire behavioral feature block for 15% of rows (set to NaN). Forces the model to maintain a content-only pathway → graceful degradation for cold POIs. Measured on the new-POI cohort.

**Baselines (all implemented, all in the results table):**

1. Random
2. Popularity (destination popularity ranking)
3. Popularity + geo filter
4. Content cosine (explicit interests only)
5. Item-kNN collaborative filtering
6. Logistic regression, pointwise, same features
7. **LambdaMART (proposed)**
8. LambdaMART + IPS (proposed, primary)
9. Oracle (latent utility) — ceiling

**Statistics:** NDCG@10 per traveler → 2,000-sample bootstrap 95% CI; **paired Wilcoxon signed-rank** of (8) vs (2) and (8) vs (4). Report p-values and effect size. Point estimates without CIs are not acceptable output.

---

## 9. Scoring layer (`scoring/`)

### 9.1 Two-factor utility — deviation from the brief

The brief's §11 example is additive. **We use multiplicative with hard gates and we justify the deviation**, because additive scoring allows `preference 0.95 + availability 0.00` to survive — which is exactly the failure the brief itself describes.

```
hard_gate     = 0 if (closed_entire_trip) or (accessibility_need unmet) or (unreachable by mobility) else 1
compatibility = (budget_fit · mobility_fit · hours_fit · reservation_fit · party_fit · duration_fit) ^ (1/6)   # geometric mean
utility       = hard_gate · relevance^α · compatibility^β         # α=1.0, β=0.7 (swept, sensitivity reported)
```

Each sub-score in [0,1], each defined in `docs/TECHNICAL.md` with its formula:
- `budget_fit`: 1 − normalized |price_level − budget_target|, asymmetric (over-budget penalized ~2× under-budget)
- `mobility_fit`: decaying function of travel time under the traveler's mobility mode
- `hours_fit`: fraction of trip days × plausible visit windows the POI is open
- `reservation_fit`: 1 if `reservation_lead_days ≤ days_until_trip` else steep decay
- `party_fit`: kid/stroller/accessibility compatibility
- `duration_fit`: `expected_duration` vs remaining daily time budget given `pace`

### 9.2 Calibration

Raw LambdaMART scores are not probabilities and are not comparable across travelers — which breaks a downstream planner that consumes them as *weights*. Fit **isotonic regression** on a calibration split mapping score → P(positive engagement). Report **ECE (15 bins)**, Brier score, and a reliability plot. This is the brief's §14 "Calibration" item done properly.

### 9.3 Confidence

```
confidence = g( n_interactions_traveler, poi_impression_count, review_count,
                ensemble_std (5 seeds), calibration_bin_width )
```

**Validation, not assertion:** bin recommendations into confidence deciles and report NDCG@10 per decile. Confidence is only meaningful if that curve is monotone increasing — report Spearman ρ of (confidence decile, NDCG). If it isn't monotone, the confidence formula is wrong and we fix it.

### 9.4 Diversity

MMR re-rank over top-50: `argmax [ λ·utility − (1−λ)·max_sim_to_selected ]`, similarity = 0.6·cosine(emb) + 0.4·same-category indicator. Sweep λ ∈ {0.5…1.0} and report the **NDCG-vs-diversity trade-off curve**. Default λ = 0.8.

### 9.5 Output schema

```json
{
  "trip_id": "T0042",
  "traveler_id": "U0117",
  "destination": "seoul",
  "generated_at": "...",
  "model_version": "lambdamart-ips-v1",
  "recommendations": [
    {
      "poi_id": "P0311",
      "rank": 1,
      "utility": 0.912,
      "planner_weight": 0.078,
      "preference_score": 0.94,
      "context_compatibility": 0.89,
      "compatibility_breakdown": {
        "budget_fit": 0.95, "mobility_fit": 0.88, "hours_fit": 1.0,
        "reservation_fit": 1.0, "party_fit": 0.90, "duration_fit": 0.82
      },
      "confidence": 0.86,
      "hard_constraints_ok": true,
      "expected_duration_min": 90,
      "diversity_group": "food_local",
      "popularity_percentile": 22,
      "localness_index": 0.81,
      "top_signals": [
        {"feature_group": "interest_match_local_food", "contribution": 0.21},
        {"feature_group": "implicit_taste_similarity", "contribution": 0.17},
        {"feature_group": "localness_vs_preference", "contribution": 0.11}
      ],
      "explanation": [
        "Strong match with your stated interest in local food",
        "Similar to neighborhood eateries you saved on past trips",
        "Less touristy than 78% of comparable POIs in Seoul",
        "12 min from your stay by metro — fits your public-transport preference",
        "Within your medium budget"
      ]
    }
  ]
}
```

`planner_weight` = utility normalized to sum 1 over the returned set — the explicit contract with the downstream itinerary planner (brief §12, §19).

---

## 10. Explainability (`explain/`)

- **Grouped TreeSHAP**: raw SHAP values aggregated into ~10 semantic feature groups (interest match, implicit taste, localness fit, popularity, price fit, geo, hours, party fit, quality, novelty). Per-POI group contributions exported in the JSON.
- **Template layer**: each group maps to a natural-language template with the numeric fill-in. No LLM — deterministic, reproducible, and LLM assistants are explicitly out of scope per brief §4.
- **Counterfactual line** where a compatibility sub-score is the binding constraint: *"Would rank #3 instead of #11 if your trip included a weekday."*
- Global feature-group importance plot in `docs/RESULTS.md`.

---

## 11. Evaluation (`eval/`) — the centerpiece

`make evaluate` emits `results/metrics.json`, `results/figures/*.png`, and regenerates `docs/RESULTS.md` from a template. **Docs are generated, never hand-written numbers.**

### 11.1 Primary — ranking quality on the **unbiased random-exposure holdout**

NDCG@{5,10,20}, Precision@{5,10}, Recall@{10,20}, MAP, MRR — for all 9 systems, with bootstrap 95% CIs and paired Wilcoxon vs popularity.

Plus the **bias-gap table**: each system's NDCG@10 on the biased holdout vs the unbiased holdout. Expectation to verify: popularity baseline shows a large positive gap (flattered by biased logs); our IPS model shows a small gap.

Plus **% of oracle ceiling** for every system.

### 11.2 Personalization

- Mean pairwise Jaccard@10 across all traveler pairs (lower = more personalized).
- **Within-archetype vs cross-archetype Jaccard@10.** This is the correct test: high within, low across. Low overlap everywhere would just mean noise. Report both numbers and the ratio.
- Rank-biased overlap (RBO, p=0.9) as a rank-aware complement.
- Per-scenario qualitative diff table for the 3 required profiles.

### 11.3 Coverage

Catalog coverage@10 across all trips; Gini coefficient and entropy of the recommendation frequency distribution; compared against popularity baseline.

### 11.4 Long-tail / local discovery

- Share of top-10 recommendations in the bottom-50% popularity stratum.
- **Long-tail precision**: among recommended long-tail POIs, what fraction are genuinely relevant per the unbiased holdout / oracle utility. Coverage without precision is just noise injection — both numbers are reported together.

### 11.5 Constraint compatibility

- % of top-10 with `compatibility ≥ 0.7`.
- **Hard-constraint violation rate in top-10 must be 0.000** — this is a correctness assertion, enforced by a test that fails the build if violated.

### 11.6 Diversity

Category entropy @10, intra-list mean cosine distance, and the λ trade-off curve.

### 11.7 Calibration

ECE (15 bins), Brier, reliability diagram; before vs after isotonic.

### 11.8 Cold-start cohorts

NDCG@10 by traveler interaction count: 0 / 1–3 / 4–10 / >10.
NDCG@10 for new POIs (created inside holdout window, zero interactions).
**Leave-one-destination-out**: train on 2 destinations, evaluate on the 3rd. This turns "new destination" from prose into a measured result.

### 11.9 Ablations (each a single row with Δ NDCG@10 and CI)

`−text embeddings` · `−implicit taste` · `−explicit interests` · `−behavioral block` · `−CF channel` · `−long-tail quota` · `−IPS weighting` · `−calibration` · `−MMR`

### 11.10 Success criteria (stated up front in the docs, then measured)

| Metric | Target |
|---|---|
| NDCG@10 (unbiased holdout) vs popularity baseline | ≥ +40% relative, Wilcoxon p < 0.01 |
| % of oracle ceiling | ≥ 70% |
| Candidate recall@250 (long-tail stratum) | ≥ 0.80 |
| Cross-archetype Jaccard@10 | ≤ 0.25 |
| Within/cross Jaccard ratio | ≥ 2.0 |
| Hard-constraint violations in top-10 | 0 |
| ECE after calibration | ≤ 0.05 |
| Confidence-decile NDCG monotonicity (Spearman) | ≥ 0.7 |
| Long-tail share of top-10 | ≥ 0.25 with long-tail precision ≥ 0.5 |

**If a target is missed, report it honestly with a diagnosis in `docs/RESULTS.md`.** Do not tune to hit the number and do not quietly drop the metric. An honest miss with a root-cause analysis scores better than a suspiciously perfect table.

---

## 12. Cold start (brief §15)

| Case | Approach | Status |
|---|---|---|
| **New traveler** | Explicit-only path; assign to nearest archetype centroid by stated interests; archetype-prior candidate channel; raised exploration quota; confidence down-weighted via `α_t` | **Implemented + measured** (0-interaction cohort) |
| **New POI** | Content-only features (text + structured); behavioral-block feature dropout at train time; UCB exploration bonus `+ c·sqrt(log(N)/n_p)`; popularity prior from category × destination | **Implemented + measured** (new-POI cohort) |
| **New destination** | All popularity/behavioral features are within-destination percentiles ⇒ model transfers; category and semantic priors carry over; bootstrap with editorial/tag priors | **Implemented + measured** (leave-one-destination-out) |

All three get real numbers. The brief allows prose for these — delivering measurements instead is a differentiator.

---

## 13. Production considerations (`docs/TECHNICAL.md` §11 — prose, no implementation)

- **Scale**: candidate generation via ANN index (ScaNN/faiss IVF-PQ) + H3 geo shards; ranker scores ≤ 500 candidates. Latency budget: candidate gen ≤ 20 ms, feature hydration ≤ 15 ms, LightGBM scoring 500×~120 features ≤ 10 ms, p99 target < 120 ms.
- **Offline vs online features**: offline nightly (embeddings, SVD, popularity percentiles, localness, item-item CF, archetype affinities); online at request (geo distance from current location/stay, open-now, availability, days-until-trip, session context). Feature store with a shared transformation library to prevent train/serve skew; skew monitored by logging serving features and diffing against offline recomputation on a sample.
- **Freshness**: CDC stream for POI attribute and hours changes; streaming counters with exponential decay for popularity/CTR; availability from partner APIs at request time with a short TTL cache; embeddings recomputed only when description/tags hash changes.
- **Retraining**: weekly full retrain; daily refresh of behavioral aggregates; automatic promotion gated on offline NDCG on a fresh unbiased slice plus an interleaving experiment.
- **Feedback loop**: log propensities from the serving policy so all offline evaluation is IPS-correctable; reserve 2–5% exploration traffic for unbiased data collection (this is exactly what our `holdout_random` simulates); team-draft interleaving for fast online comparison before full A/B.
- **Monitoring**: PSI feature drift, NDCG on interleaved traffic, calibration drift (ECE), catalog coverage and Gini (guard against popularity collapse), long-tail share, hard-constraint violation rate (must stay 0), cold-start traffic share, p99 latency, candidate-recall proxy.

---

## 14. Repository structure

```
poi-intelligence-ranking/
├─ README.md                      # clone → install → run → expected numbers
├─ Makefile                       # make reproduce  (single entry point)
├─ pyproject.toml                 # pinned deps
├─ configs/
│  ├─ datagen.yaml  ├─ features.yaml  ├─ model.yaml  ├─ scoring.yaml  └─ eval.yaml
├─ src/poi_rank/
│  ├─ datagen/      # FIREWALLED: archetypes, catalog, travelers, exposure, interactions
│  ├─ data/         # prepare, dedup, hours, geo, categories
│  ├─ features/     # poi_features, traveler_features, text, behavioral, interactions
│  ├─ candidates/   # channels, union, quotas
│  ├─ models/       # lambdamart, baselines, calibration, ips
│  ├─ scoring/      # compatibility, utility, confidence, diversity
│  ├─ explain/      # shap_groups, templates
│  ├─ eval/         # metrics, oracle, personalization, coverage, calibration_eval, ablations, report
│  └─ cli.py        # typer: generate | prepare | features | train | evaluate | recommend | scenarios
├─ data/synthetic/  # committed (+ _oracle/ used only by eval)
├─ artifacts/       # poi_emb.npy, model.txt, calibrator.pkl (committed, small)
├─ results/         # metrics.json, scenarios/*.json, figures/
├─ docs/
│  ├─ TECHNICAL.md  # the 11 required sections
│  ├─ RESULTS.md    # GENERATED from results/metrics.json
│  └─ DATA_CARD.md  # DGP description, dirtiness catalog, limitations
├─ notebooks/       # 01_data_exploration.ipynb (optional, executed + committed)
└─ tests/           # firewall, no-leakage, hard-constraint, determinism, schema, metrics unit tests
```

### Reproducibility contract

- `make reproduce` = generate → prepare → features → train → evaluate → scenarios → render docs. Wall clock < 5 min on a laptop CPU.
- Global seed in `configs/`; `PYTHONHASHSEED=0`; SHA256 of each generated dataset printed and asserted in `tests/test_determinism.py`.
- Two runs must produce byte-identical `results/metrics.json`.
- Dependency set kept minimal and pinned: `numpy, pandas, pyarrow, scikit-learn, lightgbm, shap, scipy, rapidfuzz, h3, typer, pyyaml, matplotlib, sentence-transformers (optional extra)`.
- README states expected headline numbers so a reviewer can verify their run matches.

---

## 15. Three required scenarios (brief §17)

`make scenarios` writes `results/scenarios/{1,2,3}.json` and a comparison table into `docs/RESULTS.md`:

1. **Local Experience** — interests: local food, neighborhoods; `touristiness_pref = −0.8`; budget medium; mobility public transport; solo.
2. **History & Architecture** — interests: history, architecture, museums; `touristiness_pref = +0.4`; budget high; mobility public transport; couple.
3. **Family with young children** — interests: activities, parks, interactive experiences; party `family_young_kids`; stroller accessibility required; pace relaxed; budget medium.

Plus a **4th diagnostic scenario**: the *same* traveler profile with `touristiness_pref` flipped +0.8 → −0.8, holding everything else constant. Shows the ranking is driven by the preference signal, not by profile confounds. Report top-10 overlap between the two (should be low).

For each: top-10 table (poi_id, name, category, utility, preference, compatibility, confidence, popularity percentile, localness), top signals, explanations, and the overlap matrix across scenarios.

---

## 16. Build plan (2 days, P0 = must ship)

**Day 1**
- P0 · datagen (archetypes, catalog with dirtiness, travelers, biased + random exposure logs, oracle export) + firewall test
- P0 · data prep (dedup, categories, shrinkage, hours, geo, localness + ρ validation)
- P0 · features (text with committed cache + fallback, POI, traveler explicit/implicit, interaction features)
- P0 · candidate generation with quotas + candidate-recall metrics (overall + long-tail + per-channel marginal)
- P0 · baselines 1–6 + metrics module with bootstrap CIs

**Day 2**
- P0 · LambdaMART + IPS + behavioral dropout; calibration; oracle ceiling
- P0 · compatibility, utility, confidence, MMR, output JSON schema
- P0 · TreeSHAP grouped explanations + templates
- P0 · full evaluation suite + generated `docs/RESULTS.md`
- P0 · 3 + 1 scenarios
- P0 · `docs/TECHNICAL.md` (all 11 sections), README, DATA_CARD
- P1 · ablation table (all 9), leave-one-destination-out, confidence-decile validation
- P2 · exploration notebook, λ sweep figure, feature-importance figure

**Cut order if time-constrained:** notebook → λ sweep → some ablations → LODO. **Never cut**: unbiased holdout, oracle ceiling, baselines + CIs, hard-constraint test, generated RESULTS.md.

---

## 17. Audit checklist (GG reviews CC output against this)

- [ ] `make reproduce` runs clean from a fresh clone in < 5 min
- [ ] Two runs → identical `metrics.json`
- [ ] `tests/test_firewall.py` passes (datagen imports nothing from model/feature code)
- [ ] No number appears in `docs/` that is not produced by `results/metrics.json`
- [ ] Every baseline present, every metric has a CI, Wilcoxon p-values reported
- [ ] Bias-gap table present; oracle ceiling present; % of ceiling reported
- [ ] Hard-constraint violation rate = 0, enforced by a failing test
- [ ] Long-tail share **and** long-tail precision both reported
- [ ] Within- vs cross-archetype Jaccard both reported
- [ ] Confidence-decile NDCG monotonicity reported
- [ ] Missed targets are reported honestly with diagnosis, not hidden
- [ ] TECHNICAL.md covers all 11 required sections, and explicitly justifies (a) LambdaMART over two-tower, (b) multiplicative over additive utility
- [ ] README lists expected headline numbers for verification
