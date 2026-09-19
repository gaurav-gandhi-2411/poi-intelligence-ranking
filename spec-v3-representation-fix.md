# spec-v3-representation-fix.md

**Supersedes conflicting parts of `spec-v2-remediation.md`. Everything not contradicted still holds.**

Repo: `poi-intelligence-ranking` under `ml-projects/`. Deadline 2026-09-21 21:00 KST. Recipient: konnect.kr (Seoul, inbound-travel platform for foreigners in Korea).

---

## 0. State of play

Block A rewrote the DGP. 6 of 9 gates now pass, including the two worst pre-fix numbers:

| Gate | Before | After | Threshold | Result |
|---|---|---|---|---|
| Spearman(latent_localness, geo) | 0.012 | **0.693** | ≥0.55 | PASS |
| Cold-start trip share | 0.79 | **0.103** | ≤0.25 | PASS |
| traveler_dependent_variance_share | 0.463 | **0.772** | ≥0.75 | PASS |
| Spearman(u, label) | 0.290 | **0.456** | ≥0.40 | PASS |
| Var(ε)/Var(u) | 0.305 | **0.110** | ≤0.15 | PASS |
| min non-ε term share (novelty) | 0.0044 | **0.057** | ≥0.02 | PASS |
| D10 description conditioning | 0.208 | 0.452 | ≥0.50 | FAIL |
| Oracle NDCG@10, candidate-level | 0.132 | 0.432 | ≥0.45 | FAIL |
| D9 semantic fidelity | 0.02 | 0.187 | ≥0.50 | FAIL |

Other state: 455 passed / 1 failed / 2 xfailed at 2,500 trips. The one failure is Phase 6 calibration ECE 0.0567 (was 0.0457) — expected drift from the new data distribution, correctly left for Block B. Wall clock for generate+prepare+features+candidates = 271.65s, with candidates scaling superlinearly (4.7× time for 3.1× trips).

## 1. Diagnosis — the three failures are one root cause

**D9 is not an independent failure.** D9 measures the fidelity of the taste-match signal end to end, and it decomposes multiplicatively:

```
D9 ≈ (text-channel fidelity) × (taste-estimator fidelity)
0.187 ≈ 0.452 × 0.41
```

**Oracle candidate-level NDCG (0.432) is capped by candidate recall (~0.53).** Against a candidate budget of roughly 40–45% of the catalog, 0.53 is still close to chance — the identical pre-fix failure mode — because every semantic candidate channel reads the same broken representation.

**Single root cause: the observable POI representation does not recover the latent semantic space.** Fixing it moves D9, candidate recall, and oracle candidate-level NDCG together. Do not treat these as three separate honest misses.

## 2. Two claims from the Block A report that are overruled

**2.1 "D10 has a hard asymptote near 0.46; closing it trades against lexical variety."** This is false. The asymptote is an artifact of a thin flavor vocabulary, not an information-theoretic limit. Fidelity and lexical variety are not in tension when the vocabulary is wide:

- Give each latent semantic dimension a pool of **15–25 distinct phrases/lexemes** (not 2–4).
- For each POI, sample 3–6 phrases with probability proportional to that POI's loading on each dimension, plus a small uniform background rate so no phrase is deterministic.
- Vary surface realization (word order, connectives, sentence templates) independently of which phrases are chosen — that is where lexical variety belongs, not in weakening the semantic link.

**Target D10 ≥ 0.65.** Report the vocabulary-size sweep as evidence.

**2.2 Embedding centering.** I removed centering from the DGP correctly (D2 showed the latent taste term was never inert). I was wrong to remove it from the **features** side. It belongs there — see §3.2.

## 3. The fix

### 3.1 Text-channel fidelity (datagen, Gate-A scope)

Implement §2.1. Re-measure D10. Do not touch features/, models/, candidates/, scoring/ during this pass.

### 3.2 Taste-estimator fidelity (features, Gate-B scope)

The current taste vector is a plain interaction-weighted mean of engaged-POI embeddings. Three defects, all fixable:

1. **No centering.** Every traveler's vector is dominated by the shared catalog-mean direction, so cosines compress and differences between travelers shrink. Subtract the catalog mean vector before normalizing, on both the POI side and the taste side.
2. **No IDF weighting.** Ubiquitous embedding dimensions dominate. Weight dimensions by inverse catalog variance.
3. **Exposure bias.** Engagement is logged under a popularity-biased policy, so the plain mean is a biased estimator of taste. Apply the same clipped IPS weights already used in training when forming the taste vector.

Additionally, replace the plain mean with a **ridge-regularized least-squares estimate** of the taste direction over engaged POIs, shrinking toward the archetype-prior centroid with strength `k/(n_t + k)`. This is the correct estimator under sparse history and it makes the existing cold-start shrinkage principled rather than bolted on.

### 3.3 Behavioral item embedding — the missing component

Text can only carry so much; nobody has tried the standard remedy. Add a **behavioral item embedding**:

- Fit **item2vec (skip-gram on co-engagement sequences)** or **implicit-ALS item factors**, 32 dimensions, on **train-window interactions only**.
- Concatenate with the 64-d text SVD → 96-d POI representation. Both blocks feed the ranker and the semantic candidate channel.
- Zero circularity: this uses only the interaction log, which is exactly what a production system has. It must not read `_oracle/`. The firewall test must cover it.
- With cold-start now at 10% and 2,500 trips, there is finally enough co-engagement signal for this to work. This was not viable pre-Block-A; it is now.
- New POIs have no behavioral factor — fall back to the text block, which the existing 15% behavioral-dropout training already handles gracefully. Verify on the new-POI cohort.

### 3.4 Correct statistic for representation quality — new D11

Pairwise-cosine Spearman (D9/D10) is a weak, indirect proxy sensitive to global embedding geometry. Add **D11**:

> Out-of-sample **ridge regression R²** and **first canonical correlation (CCA)** from `poi_emb → poi_semantic`, 5-fold, fit only on observable columns.

This directly answers "how much of the latent semantic space is recoverable from what the model can see." Report D11 for text-only, behavioral-only, and concatenated representations. Keep D9/D10 as secondary.

## 4. Gate restructure

The current single gate mixes simulator quality with representation quality, which produced a frozen-code confound (oracle candidate-level NDCG failing on untouched `candidates/`). Split it.

**Gate-A — simulator quality (`datagen/` only).** Freeze before any features/model work.

| Gate | Threshold | Current |
|---|---|---|
| traveler_dependent_variance_share | ≥ 0.75 | 0.772 PASS |
| Spearman(u, label) | ≥ 0.40 | 0.456 PASS |
| Var(ε)/Var(u) | ≤ 0.15 | 0.110 PASS |
| min non-ε term share | ≥ 0.02 | 0.057 PASS |
| Spearman(latent_localness, geo) | ≥ 0.55 | 0.693 PASS |
| D10 description conditioning | ≥ 0.65 | 0.452 FAIL |
| **Oracle NDCG@10 over the FULL CATALOG** (no candidate generation) | ≥ 0.55 | not yet measured |
| Cold-start trip share | ≤ 0.25 | 0.103 PASS |

Full-catalog oracle NDCG replaces candidate-level oracle NDCG here — it is a pure DGP property with no downstream code in the path.

**Gate-B — representation quality (`features/`, `candidates/`; run only after Gate-A is frozen).**

| Gate | Threshold |
|---|---|
| D11 ridge R² (`poi_emb → poi_semantic`), concatenated | ≥ 0.45 |
| D9 semantic fidelity | ≥ 0.45 |
| Candidate recall@K vs oracle-relevance set, overall | ≥ 0.85 |
| Candidate recall, long-tail stratum | ≥ 0.75 |
| Candidate recall lift over the chance baseline (K/catalog) | ≥ +0.35 absolute |
| Oracle NDCG@10, candidate-level | ≥ 0.50 |

The chance-baseline lift row is mandatory. Raw recall without it hid the original failure for four phases.

Both gates blocking in `make reproduce`. Gate-A frozen before Gate-B work begins; that sequencing is stated in TECHNICAL.md as the non-circularity argument.

## 5. Wall-clock

271.65s for generate→candidates, with candidates at 4.7× time for 3.1× trips. Superlinearity here is a per-trip Python loop, not inherent complexity.

1. **Vectorize the semantic channel**: one `(n_trips × n_pois)` matmul. 2,500 × 482 ≈ 1.2M cosines — milliseconds, not minutes.
2. **Precompute H3 k-rings once per destination**, not per trip.
3. **Precompute the item-item CF neighbor table once**, look up per trip.
4. **joblib-parallelize** per-trip candidate assembly across cores.
5. Target: candidates < 20s; full `make reproduce` < 5 min including train/evaluate/scenarios.

Do not reduce trip count until 1–4 are done. If still over budget after vectorization, drop to 1,800 trips (holdout ≈ 360, CIs still usable) — that is the last resort, not the first.

## 6. Housekeeping

- Regenerate the stale `code_reading_answer` field in `results/parts/dgp_diagnostics.json` — it still asserts "no dependency edge" post-fix. It violates the no-stale-numbers rule and would feed the Decision Register.
- Phase 6 calibration ECE drifted to 0.0567. Refit isotonic against the new data in Block B; target ≤0.05.
- TF-IDF (0.187) currently beats MiniLM (0.133) on D9. Keep this as a DR4 row, but TECHNICAL.md **must state** that this is an artifact of templated synthetic descriptions and would likely reverse on real POI text. Do not claim TF-IDF is generally superior. Re-measure both after §3.1 and §3.3.

## 7. Execution order

- **A2** — §3.1 text vocabulary widening. Re-measure D10 + full-catalog oracle NDCG. **Gate-A must pass.** Freeze.
- **A3** — §3.2 taste estimator + §3.3 behavioral item embedding + §3.4 D11. **Gate-B must pass.**
- **A4** — §5 vectorization. Wall clock under budget.
- **B** — re-run Phases 2–9 on the fixed data: recalibrate, full scorecard, bias-gap table (prominent — the measurement exists from Phase 8, it was never surfaced), true-archetype personalization, regularized LR baseline, MiniLM re-measure, `results/parts/` + `compose`.
- **C** — Decision Register DR1–DR11. Uncuttable: DR1 (additive utility → show >0 hard-constraint violations) and DR2 (two-tower learning curve with CIs).
- **D** — docs regeneration with Konnect framing, final determinism + both gates + full suite, tag.

Cut order if time runs short: DR5 → DR8 → DR7 → 5-seed replication → DR3. Never cut: both gates, §3 fixes, DR1, DR2, bias-gap table.

## 8. Standing rules (unchanged)

- Verifier PASS is invalid if any priority check is PENDING or PARTIAL. Every check emits a computed artifact — a number or a file path — not a code-reading opinion.
- `_oracle/` is readable only by `eval/` code. The behavioral embedding must not read it; extend the firewall test.
- No hand-typed numbers in `docs/`. `metrics.json` has exactly one writer (`compose`).
- `make reproduce`: local, CPU, single-seed, < 5 min, byte-identical twice, no cloud, no API keys.
- GCP (one preemptible n2-standard-16) for evidence generation only: MiniLM, DR2 two-tower learning curve, 5-seed replication, 1M-vector ANN benchmark, parameter sweeps. Commit the artifacts.
- A miss is reportable as a ceiling only after a diagnostic has ruled out a mechanism, citing the number.
