# spec-v2-remediation.md — POI Intelligence & Ranking System

**Supersedes conflicting parts of `spec.md`. Everything in `spec.md` not contradicted here still holds.**

**Context:** P0 is built (10 phases, 398 tests, clean determinism/firewall/lint). The engineering is sound. The science is not: 5 of 9 §11.10 targets missed, and all were attributed to an "information-theoretic ceiling." That attribution is wrong. This document identifies three root causes, specifies the fixes, adds a DGP acceptance gate that should have existed before Phase 2, and adds a Decision Register that replaces prose justification with measured evidence.

**Time budget:** ~2 days. Submission 2026-09-21 21:00 KST.
**Recipient context:** konnect.kr — Seoul-based platform for foreigners visiting/living in South Korea (inbound travel, moving toward an integrated tourist/resident lifestyle platform). Competitors (VisitKorea, Triple Korea, Creatrip) already rank by popularity. **Local / long-tail discovery is their product wedge.** Frame results accordingly.

---

## 0. The falsifiable claim to test FIRST

Before changing any code, run the diagnostic harness in §1. It must either confirm or refute:

> **Claim:** The 9 missed targets are not an information ceiling. They are caused by (RC1) an evaluation-set mismatch, (RC2) utility-term scale imbalance causing noise domination in the DGP, and (RC3) 79% cold-start trips.

Supporting arithmetic already in the repo's own reports:

- catalog/destination ≈ 482 POIs; mean candidates/trip = 186.7 ⇒ **38.7% of catalog**
- measured random-selection recall = **0.3889** (≈ 38.7%, as expected)
- measured 6-channel recall = 0.4413 ⇒ **six engineered channels add 5.2 points over blind sampling**
- random ranker NDCG@10 = 0.0304; **oracle NDCG@10 = 0.1322**

Under RC1 alone, with *deterministic* argmax choice inside a 20-item random slate, the chosen POI sits at ≈ the 95th global utility percentile (≈ rank 23 of 482) and the oracle would score NDCG@10 ≈ 0.5. Observed 0.1322 ⇒ choice is near-uniform ⇒ **RC2 is real and primary.**

If the §1 diagnostics refute this, report that immediately with the numbers and stop — do not proceed to §2 fixes.

---

## 1. DGP diagnostic harness (new: `src/poi_rank/eval/dgp_diagnostics.py`)

New CLI: `poi_rank.cli diagnose-dgp`. Writes `results/parts/dgp_diagnostics.json`. All reads of `_oracle/` here are eval-only and permitted.

Compute and report:

| ID | Diagnostic | Interpretation |
|---|---|---|
| D1 | **Variance decomposition of `u(t,p)`**: share of total variance from each of the 7 deterministic terms + `ε`, computed across all (trip, POI-in-catalog) pairs | If `Var(ε)/Var(u) > 0.40` → noise-dominated. If any term's share < 0.02 → that term is inert |
| D2 | **Distribution of `cos(taste_t, poi_semantic_p)`**: mean, sd, 5th/95th pct | Phase 3 reported pairwise cosine 0.004–0.45. If sd < 0.05 the taste term is structurally inert regardless of `w_taste` |
| D3 | **Spearman(u, realized label)** across all holdout impressions | < 0.35 ⇒ labels are noise-dominated |
| D4 | **Choice sharpness**: mean global utility percentile of the chosen POI within each slate; and mean within-slate utility rank of the chosen POI (1 = argmax) | Within-slate mean rank near 10/20 ⇒ uniform choice |
| D5 | **Oracle NDCG@10 at the slate level** (rank only the 20 exposed items) vs **at candidate level** (rank ~187) | Quantifies how much of the deflation is RC1 |
| D6 | **Cold-start share**: % of trips with zero prior engaged interactions available at their as-of cutoff | Currently 79% |
| D7 | **Spearman(latent_localness, POI lat/lon-derived geo features)** | Phase 2 found these are independent — confirms the geo/localness generation bug |
| D8 | **Bias gap**: popularity baseline NDCG@10 on biased holdout minus on random holdout | The submission's stated spine. Currently unmeasured |

---

## 2. Root-cause fixes

### RC1 — Evaluation set mismatch (structural)

Labels come from a 20-item uniform-random slate; ranking was evaluated over ~187 candidates and the full catalog. This is off-policy evaluation without the correction.

**Fixes:**
1. **Primary ranking metric becomes slate-level**: for each holdout slate, rank exactly the exposed items and compute NDCG@{5,10}/P@K/MAP/MRR over them. Oracle should reach ≥0.75 here by construction — if it does not, something else is wrong.
2. **Raise random-slate size from 20 to 70** in `configs/datagen.yaml` so exposed set ≈ candidate set and full-catalog metrics become meaningful.
3. **Keep full-catalog recall/NDCG as a clearly-labelled secondary metric** with the exposure caveat stated inline in RESULTS.md, not hidden.
4. **Candidate recall is re-defined against the slate-exposed positives**, and additionally reported against an *oracle-relevance* set (top-5% of catalog by true utility per trip) — the latter is the honest measure of "did candidate generation find what matters," and is not conditioned on exposure at all.

### RC2 — Utility term scale imbalance → noise domination (primary)

The 7 terms are on incompatible natural scales. `cos(taste, poi)` has near-zero variance, so `w_taste` is a lie; `ε` dominates.

**Fixes (in `datagen/`, firewall preserved):**
1. **Standardize every utility term to zero mean / unit variance across the destination catalog before applying weights.** Weights then mean what they say.
2. **Respecify weights as explicit variance shares** summing to 1.0 in `configs/datagen.yaml`. Proposed starting point (tune to hit the §3 gate):
   `taste 0.24, category 0.16, localness×pref 0.18, latent_quality 0.14, party_fit 0.08, price_fit 0.08, novelty 0.04, ε 0.08`
   Note localness×pref gets a high share deliberately — it is the axis konnect.kr's product depends on and the axis scenario 4 tests.
3. **Sharpen the taste term**: compute `cos` on centered embeddings (subtract the catalog mean vector before normalizing) so similarity spreads across [-1, 1] instead of collapsing near 0. Apply the same centering in `features/` so train and DGP see comparable geometry.
4. **Tune `τ`** so within-slate choice sharpness (D4) puts the chosen POI in the top 3 of 20 on average.
5. **Fix RC2b — geo/localness independence** (Phase 2's finding): sample POI lat/lon *conditioned* on `latent_localness` — local POIs cluster away from the tourist centroid, touristy ones cluster toward it, with overlap. This is realism, not leakage; the localness *index* still only ever sees observable columns.

### RC3 — 79% cold-start

Implicit taste — the brief's §8 centerpiece — is exercised on 21% of trips.

**Fixes:**
1. **Per-impression leave-one-out as-of cutoff** within a trip's browsing session (the "future work" deferred in Phase 3). Traveler state at impression *i* uses impressions `< t_i` including earlier ones in the same trip. This is how real planning sessions work and it's what makes implicit taste load-bearing.
2. **Seed every traveler with pre-trip history**: 5–40 interactions in the 0–180 days before trip start, drawn from their latent taste (with the same noise model). Some travelers keep 0 history on purpose — that is the *deliberate* cold-start cohort, not an accident.
3. **Target: cold-start share ≤ 25%**, with an explicit 0-interaction cohort of ~15% retained for §15 cold-start evaluation.
4. Scale trips **800 → 2,500** (holdout ≈ 500) so CIs are usable. Runtime must stay under the 5-minute budget; profile and report.

### RC4 — Personalization measured against the wrong labels

Within/cross-archetype Jaccard used K-Means clusters derived from the same features being evaluated. Ratio→1 is an artifact.

**Fix:** use the DGP's **true archetype mixture labels** (eval-only `_oracle/` read, same permission as the oracle ceiling). Define within-archetype pairs as travelers whose dominant archetype matches and whose mixture cosine > 0.8. Report the K-Means-proxy version alongside, labelled as a proxy, to show why it was misleading.

### RC5 — Missing bias-gap table

The stated differentiator was never measured across 10 phases. **Required table in RESULTS.md:** every system × {biased holdout NDCG@10, random holdout NDCG@10, gap, gap as % of biased}. Expected shape to verify: popularity shows a large positive gap; LambdaMART+IPS shows a small one.

### Secondary fixes

| # | Issue | Fix |
|---|---|---|
| S1 | MiniLM silently replaced by TF-IDF; stated reason (90MB / 5-min budget) is false since the cache is committed | Run `all-MiniLM-L6-v2` once, commit `artifacts/poi_emb_minilm.npy`, make it the canonical path; keep TF-IDF as the tested offline fallback; report both as a Decision Register row |
| S2 | LR baseline unregularized, 308 features / 200 trips | CV-tuned L2 + standardization. A crippled baseline invalidates the headline win |
| S3 | `metrics.json` read-modify-write clobbered twice | Each stage writes `results/parts/<stage>.json`; new `poi_rank.cli compose` merges into `metrics.json`; `make reproduce` always ends with compose → docs. Add a test that `metrics.json` is never written by more than one module |
| S4 | Verifier returns PASS while admitting priority checks never ran | Verifier PASS is **invalid** if any priority check is PENDING/PARTIAL. Every check must emit a computed artifact (a number or a file path), not a code-reading opinion. Missing artifact ⇒ FAIL |
| S5 | `xfail(strict=True)` used for missed metric targets | Remove. Targets live in the RESULTS scorecard. Build-blocking tests reserved for invariants only: hard-constraint=0, firewall, `_oracle` isolation, determinism, no-leakage |
| S6 | Confidence ρ = −0.382 | Re-measure after RC1–RC3. If still non-monotone, redesign: confidence should be dominated by (a) evidence volume for the *traveler*, (b) calibration-bin width at the predicted score, (c) seed-ensemble sd. Report the positive control that was already built |

---

## 3. DGP acceptance gate (new, blocking)

The original build spent 8 phases on an unvalidated generator. Add `poi_rank.cli gate-dgp`, run immediately after `generate`, before `prepare`. **`make reproduce` fails if the gate fails.**

| Gate | Threshold |
|---|---|
| Oracle NDCG@10, slate-level | ≥ 0.70 |
| Oracle NDCG@10, candidate-level | ≥ 0.35 |
| Spearman(u, label) | ≥ 0.40 |
| `Var(ε)/Var(u)` | ≤ 0.20 |
| Min term variance share | ≥ 0.02 (no inert term) |
| sd of `cos(taste, poi)` | ≥ 0.12 |
| Cold-start trip share | ≤ 0.25 |
| Spearman(latent_localness, observable localness inputs) | ≥ 0.55 |
| Popularity bias gap (D8) | ≥ 0.02 absolute |

Tuning the DGP to pass this gate is legitimate and must be documented in DATA_CARD.md — it is calibrating the *simulator*, not the model. **The model, features, candidate channels and ranker must not be touched while tuning the gate**, and the gate must be frozen before any post-fix model training. State this explicitly in TECHNICAL.md; it is the argument that the improved numbers are not circular.

---

## 4. Post-fix §11.10 scorecard (re-baselined)

| Target | Old | New target |
|---|---|---|
| Slate-level NDCG@10 vs popularity | n/a | ≥ +40% relative, Wilcoxon p < 0.01 |
| % of oracle ceiling (slate-level) | 66.2% | ≥ 70% |
| Candidate recall@250 vs oracle-relevance set, overall | 0.4413 | ≥ 0.85 |
| Candidate recall, long-tail stratum | 0.3936 | ≥ 0.75 |
| Cross-archetype Jaccard@10 (true labels) | 0.0404 | ≤ 0.25 |
| Within/cross Jaccard ratio (true labels) | 1.08 | ≥ 2.0 |
| Hard-constraint violations in top-10 | 0 | 0 (unchanged, blocking) |
| ECE after calibration | 0.0457 | ≤ 0.05 |
| Confidence-decile Spearman | −0.382 | ≥ 0.6 |
| Long-tail share / precision @10 | 0.2338 / 0.0678 | ≥ 0.25 / ≥ 0.40 |
| Scenario-4 (touristiness flip) top-10 overlap | 0.5385 | ≤ 0.35 |
| Localness index Spearman ρ | 0.4757 | ≥ 0.60 |

Honest-miss discipline still applies — but a miss is only reportable as a ceiling **after** the §1 diagnostics have ruled out a mechanism, with the diagnostic number cited.

---

## 5. Decision Register (new — `docs/TECHNICAL.md` §12)

Every design choice not dictated by the assignment must carry measured evidence for why the alternative was rejected. Prose justification is not acceptable. Table columns: **Decision | Alternative | Experiment | Metric | Result | Verdict**.

| # | Decision | Alternative to disprove | Experiment |
|---|---|---|---|
| DR1 | Multiplicative utility `rel^α · compat^β` with hard gates | The brief's own **additive** formula | Implement additive; report top-10 hard-constraint violation count, mean compatibility@10, NDCG@10 for both. Prediction: additive > 0 violations |
| DR2 | LightGBM LambdaMART | **Two-tower neural ranker** | Implement a small two-tower (shared MLP, dot-product scoring). **Learning curve**: NDCG@10 vs training-set fraction {10,25,50,100%} with bootstrap CIs for both models. Show no crossover in our data regime, and state the extrapolated crossover scale |
| DR3 | Listwise (`lambdarank`) | Pointwise (`binary`), pairwise, `rank_xendcg` | 4-way LightGBM objective comparison, same features/splits |
| DR4 | MiniLM sentence embeddings | TF-IDF→SVD-64 | Full-pipeline ablation, both paths, NDCG + localness-discrimination |
| DR5 | Brute-force cosine retrieval | faiss / hnswlib | Latency + recall@k table at 1.5k **and 1M** synthetic vectors. Justifies "no ANN at this scale" and shows we know when it flips |
| DR6 | Geometric-mean compatibility | `min()`, plain product | 3-way sweep on NDCG + violation rate + compat@10 |
| DR7 | IPS clip = 20 | {5, 10, 50, ∞} | Sweep on random-holdout NDCG |
| DR8 | 180-day half-life, interaction weights | ±2× half-life; uniform weights | Sensitivity sweep |
| DR9 | Long-tail quota = 50 | {0, 25, 50, 100} | NDCG vs long-tail-precision frontier plot |
| DR10 | α=1.0, β=0.7 | grid | Re-sweep post-fix |
| DR11 | 6 candidate channels | leave-one-out | Already built — carry forward with post-fix numbers |

Every row must cite a number produced by the pipeline. A row whose experiment was not run must say **NOT RUN**, not be omitted.

---

## 6. GCP usage

Critical path stays local: `make reproduce` is CPU-only, single-seed, < 5 minutes, no cloud dependency, no `ANTHROPIC_API_KEY`, nothing a reviewer must configure.

GCP is used only to produce artifacts that get committed:

| Job | Why GCP |
|---|---|
| MiniLM embedding generation | one-time, committed as `.npy` |
| Two-tower training (DR2 learning curve) | parallel across 4 training fractions × 5 seeds |
| **5-seed replication of the full pipeline** | seed-variance table in RESULTS.md — a rigor signal reviewers notice |
| DGP calibration sweep for §3 gate | parallel parameter search |
| ANN benchmark at 1M vectors (DR5) | memory |

One preemptible `n2-standard-16`, expected < $5. Add `scripts/gcp_sweep.sh` and document in TECHNICAL.md that cloud is used for evidence generation only, never for reproduction.

---

## 7. Konnect-specific framing (docs only, no model change)

- **Seoul is the lead destination** in RESULTS.md and all scenario write-ups.
- Personas reframed as **inbound foreign travelers**: first-time K-culture visitor; repeat visitor seeking local/non-touristy; family with young children; food-led traveler. Keep spec.md §15's three required profiles intact — add the Konnect framing as the narrative layer, not as a substitute.
- **Lead RESULTS.md with the long-tail / local-discovery result**, not with NDCG. Competitors already rank by popularity; the differentiating claim is "we surface the genuinely relevant non-obvious POI," and it must be backed by long-tail *precision*, not just share.
- One short paragraph in TECHNICAL.md's production section on what changes for an inbound-travel platform specifically: heavy new-traveler cold start (most users are first-time), language/locale features, seasonality, and event/pop-up POIs with short lifecycles.

---

## 8. Execution order

**Block A (today, blocking everything):** §1 diagnostics → report numbers → §2 RC1/RC2/RC3 fixes → §3 gate passes → regenerate data.
**Block B:** re-run Phases 2–9 end-to-end on fixed data; produce new scorecard (§4), bias-gap table (RC5), true-archetype personalization (RC4), S1–S6.
**Block C:** Decision Register experiments (DR1–DR11), GCP-parallel where useful.
**Block D:** regenerate TECHNICAL.md / RESULTS.md / README with Konnect framing; final determinism + gate + test pass; tag release.

Cut order if time runs short: DR5 → DR8 → DR7 → 5-seed replication → DR3. **Never cut:** §1 diagnostics, §3 gate, RC1–RC5 fixes, DR1, DR2, bias-gap table.

---

## 9. Audit checklist (GG)

- [ ] §1 diagnostics reported as numbers before any fix was written
- [ ] §3 gate is blocking in `make reproduce` and passes
- [ ] DGP tuning is documented and was frozen before model retraining (stated in TECHNICAL.md)
- [ ] Oracle slate-level NDCG@10 ≥ 0.70
- [ ] Bias-gap table present for all systems
- [ ] Personalization uses true archetype labels, K-Means version labelled as proxy
- [ ] MiniLM is the canonical embedding path
- [ ] LR baseline is regularized
- [ ] `metrics.json` written only by `compose`; enforced by test
- [ ] No `xfail` used for metric targets
- [ ] Decision Register has all 11 rows, each with a number or explicit NOT RUN
- [ ] DR1 shows additive formula produces > 0 hard-constraint violations
- [ ] DR2 learning curve present with CIs
- [ ] Every remaining miss cites the diagnostic that ruled out a mechanism
- [ ] `make reproduce` < 5 min, CPU, no cloud, byte-identical twice
