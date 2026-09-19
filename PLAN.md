# PLAN.md — poi-intelligence-ranking execution tracker

Deadline: 2026-09-20 EOD (submission 2026-09-21 21:00 KST). Repo:
https://github.com/gaurav-gandhi-2411/poi-intelligence-ranking

Workflow: orchestrator (this session) dispatches an `executor` subagent per phase
with the full spec context, then an independent `verifier` subagent re-derives the
key claims from the actual repo state (never trusts the executor's self-report),
then the orchestrator commits+pushes. Every commit leaves `make reproduce` green.

## Done (Day 1 P0, spec.md §16 — all committed + independently verified)

1. **Scaffold** (`a166233`) — package layout, `pyproject.toml` (uv, ruff, mypy
   strict), Makefile, isolated `.venv` (never conda base).
2. **Datagen + firewall** (`7880254`) — 8-archetype Dirichlet travelers, latent
   utility DGP, Plackett-Luce slate choice, popularity-biased train log +
   uniform-random primary holdout + biased secondary holdout w/ propensities,
   catalog dirtiness. `test_firewall.py`, `test_oracle_isolation.py`,
   `test_determinism.py` all pass.
3. **Data prep** (`e21abc5`) — dedup (3.60%, matches 3.8% injected), category
   canonicalization (0% unmapped), EB rating shrinkage, within-destination
   pop_pct, localness index, hours/geo prep. **Localness Spearman ρ = 0.4757 vs
   spec target 0.6 — honest miss**, `xfail(strict=True)` (spec frames this as a
   reported metric, not a build-blocking invariant — verifier agreed).
4. **Features** (`0d56690`) — TF-IDF→SVD-64 text embeddings (canonical path;
   sentence-transformers is opt-in only, not installed), POI/traveler feature
   tables (99 / 139 cols, block-prefixed for later ablations), temporal
   as-of-cutoff on implicit taste (635/800 trips correctly cold-start),
   POI-id reconciliation across the Phase-2 dedup merge (self-discovered fix).
5. **Candidate generation** (`d0a98a5`) — 6 channels w/ hard quotas, union to
   ~187 candidates/trip. **candidate_recall@250 = 0.4413 vs target 0.90
   (long-tail 0.3936 vs 0.80) — honest miss**, root-caused to a genuine
   information ceiling (2 of 7 DGP utility terms have zero observable proxy;
   stated interests are a deliberately lossy/noisy projection by design) and
   verified NOT a bug (naive random-k baseline = 0.3889, real system only
   +13.4% relative over chance; verifier independently confirmed via direct
   oracle read for verification purposes only).
6. **Baselines + metrics** (`773e24e`) — all 6 baselines (random, popularity,
   popularity+geo, content-cosine, item-kNN CF, logistic regression) + oracle
   ceiling, `eval/metrics.py` (NDCG@{5,10,20}, P@{5,10}, R@{10,20}, MAP, MRR,
   2000-resample bootstrap 95% CI, paired Wilcoxon), `results/metrics.json`.
   NDCG@10: oracle 0.1322 (ceiling) > content-cosine 0.0807 (61% of ceiling,
   beats popularity p=0.035) > popularity 0.0622 > logistic regression 0.0603
   (honest miss — likely overfits 308 features on ~200 holdout trips) > item-kNN
   CF 0.0543 > popularity+geo 0.0514 > random 0.0304. **Every number here is
   mechanically bounded by item 5's ~0.44 candidate-recall ceiling** — stated in
   `metrics.json`'s own `meta.note`.

**Full test suite at this checkpoint: 187 passed, 1 xfailed, 0 failed. ruff/mypy
clean throughout. Determinism byte-identical, verified at every phase.**

## Known, already-honestly-documented misses (do NOT re-litigate, do NOT tune to hide)

- Localness Spearman ρ = 0.4757 (target 0.6) — `docs/DATA_CARD.md` #13.
- candidate_recall@250 = 0.4413 overall / 0.3936 long-tail (targets 0.90/0.80) —
  `docs/DATA_CARD.md` ~#31, `results/metrics.json` meta.note. **This bounds every
  ranking metric downstream.** UPDATE (Phase 5, `docs/DATA_CARD.md` #47): measured
  outcome was a partial miss, not the full miss originally expected — lambdamart_ips
  MET the NDCG@10-vs-popularity target (+40.6% relative, Wilcoxon p=0.0042, both legs
  of "≥+40%, p<0.01" satisfied) but MISSED the ≥70%-of-oracle-ceiling target (66.2%
  measured), root-caused to this same candidate-recall ceiling as predicted. Not
  tuned to hit either number.
- Logistic regression baseline underperforms popularity — `docs/DATA_CARD.md`
  Phase 4b section.
- Confidence-decile NDCG monotonicity Spearman ρ = -0.382 (target ≥0.7) —
  `docs/DATA_CARD.md` #58. Root-caused to the same candidate-recall ceiling above:
  every one of spec's 5 named confidence inputs is individually uncorrelated with
  per-trip NDCG on this dataset (measured, not assumed); a positive control using
  other model-internal signals confirms the measurement methodology itself is
  sound. `xfail(strict=True)`, not tuned to hide.

## Done (Day 2 P0, Phase 5, spec.md §16 item 7 — committed, independently verified)

7. **LambdaMART + IPS** (`models/lambdamart.py`, `eval/new_poi_cohort.py`) —
   LightGBM `lambdarank`, grouped by `trip_id`, `lambdarank_truncation_level=20`,
   `label_gain=[0,1,3,7]` (matches `dcg_at_k`'s exponential gain exactly), native
   `category`-dtype categoricals, native NaN (no imputation). Train/val split
   carved from TRAIN by trip_id (never the holdout being reported on) for early
   stopping. `poi_rank.cli train` fits + persists 3 boosters to `artifacts/*.txt`
   (system 7 `model_no_ips.txt`, system 8/primary `model.txt`, and a
   dropout-ablation-only `model_ips_no_dropout.txt`); `poi_rank.cli evaluate`
   only ever LOADS them. **IPS**: `clip(1/p_expose, 1, 20)`, unexposed train rows
   (73.5% of train candidates, measured) get neutral weight 1.0 (docs/DATA_CARD.md
   #41), normalized per trip-group to sum to the group's own row count (#42).
   **Behavioral dropout**: 15% of FIT-split rows, `behav_*` block → NaN,
   independent per row, applied identically to systems 7/8 (#44). **New-POI
   cohort**: 68/1446 catalog POIs (#43), 44/202 holdout trips with a relevant
   cohort candidate. **Determinism**: `deterministic=True` + `force_row_wise=True`
   + `num_threads=1`, verified byte-identical across 2 full `train`+`evaluate`
   runs on every artifact (#46).

   **Results** (`results/metrics.json`, `docs/DATA_CARD.md` "Measured results,
   systems 7/8"): lambdamart_ips (system 8, primary) NDCG@10 = **0.0875**
   [0.0715, 0.1042], **66.2% of oracle ceiling**, **+40.6% relative vs popularity,
   Wilcoxon p=0.0042** — the §11.10 NDCG-vs-popularity target (≥+40%, p<0.01) is
   **MET**; the ≥70%-of-ceiling target is **MISSED** (66.2%), root-caused to the
   same already-documented ~0.44 `candidate_recall@250` ceiling (#47) — the
   oracle itself only reaches 0.1322 NDCG@10 on this same candidate-recall-capped
   set, so 66.2% of an already-capped ceiling is consistent with, not
   contradictory to, that known constraint. IPS ablation: lambdamart (no IPS) =
   0.0545 (UNDERPERFORMS popularity, 0.0622) vs lambdamart_ips = 0.0875 — IPS
   correction is a highly significant, large improvement (p=1.80e-05). New-POI
   dropout ablation: 0.5208 with dropout vs 0.5075 without (n=44 trips,
   p=0.994) — directionally consistent with the intended robustness effect but
   NOT statistically significant at this cohort size, reported honestly as such.

   Test suite at this checkpoint: 203 passed, 1 xfailed, 0 failed (`tests/
   test_lambdamart.py` new: IPS-weight clip/normalization on a hand-computed
   example, behavioral-dropout masking correctness/seeding, real-fixture-chain
   training + no-NaN scoring, booster-level determinism; `tests/
   test_new_poi_cohort.py` new: cohort identification against real data, NDCG@10
   evaluation helper on hand-built frames). ruff/mypy clean throughout.

## Done (Day 2 P0, Phase 6, spec.md §16 item 8 — committed, independently verified)

8. **Scoring layer** (`scoring/compatibility.py`, `utility.py`, `calibration.py`,
   `confidence.py`, `diversity.py`, `output.py`, `configs/scoring.yaml`) —
   multiplicative `utility = hard_gate * relevance^alpha * compatibility^beta`
   (α=1.0, β=0.7 default, swept {0.3,0.5,0.7,0.9,1.1}, deviation from the brief's
   additive §11 example justified per `docs/DATA_CARD.md` #54); hard gate
   (`closed_entire_trip` / `accessibility_need unmet` / `unreachable by mobility`,
   reusing `candidates/channels.py`'s/`models/baselines.py`'s own geo radius
   logic); 6-term geometric-mean compatibility (`budget_fit` asymmetric ~2x
   over-budget penalty, `mobility_fit` exponential half-life decay, `hours_fit`
   fraction-of-trip-days-in-plausible-window, `reservation_fit` steep decay past
   an assumed 21-day planning lead time, `party_fit` independent observable
   kid/accessibility computation, `duration_fit` pace-conditioned per-POI time
   budget — each documented in `docs/DATA_CARD.md` #48-53); isotonic calibration
   on a calibration split carved from LightGBM's own `fit_frame` (disjoint from
   both val and holdout by construction, #56); confidence `g(...)` over spec's 5
   named inputs incl. a genuine 5-seed LightGBM ensemble std (#58); MMR diversity
   re-rank (`argmax[λ·utility − (1−λ)·max_sim]`, similarity =
   0.6·cosine+0.4·same-category, λ swept {0.5..1.0}, default 0.8, #60); full
   spec.md §9.5 output JSON schema (`top_signals`/`explanation`/`diversity_group`
   are documented placeholders pending the `explain/` phase, #63).
   `poi_rank.cli recommend` (+ `make recommend`) is the entry point — scores the
   primary unbiased holdout trip set (202 trips), NOT wired into `make reproduce`
   (spec.md §14's reproducibility contract does not list `recommend`).

   **Hard-constraint violation rate in top-10 = 0** (spec.md §11.5, build-blocking
   — `tests/test_hard_constraints.py`, genuinely passes, never `xfail`): hard-gate
   filtering happens BEFORE ranking (not just relying on `utility==0`), plus a
   defensive runtime assertion in `scoring/output.py::assemble_output_payload`
   before any row is ever serialized.

   **Results** (`docs/DATA_CARD.md` "Measured results, Phase 6", real committed
   dataset, 202 holdout trips, `uv run python -m poi_rank.cli recommend` ≈53s):
   ECE after calibration = **0.0457** (target ≤0.05, **MET**; before=0.3601,
   Brier before=0.2149→after=0.0521). Confidence-decile monotonicity Spearman
   rho = **-0.382** (target ≥0.7, **MISSED** — genuine, measured honest miss:
   every one of spec's 5 named confidence inputs individually shows no
   significant correlation with per-trip NDCG@10 on this dataset, |rho|<0.09,
   p>0.24; 2 different combination functions tried; a positive control using
   other model-internal signals (not among spec's 5 inputs) DOES show weak
   significant correlation, proving the measurement methodology detects real
   signal when present — root-caused to the same already-documented
   candidate-recall ceiling, `docs/DATA_CARD.md` #58, `xfail(strict=True)`).
   β-sensitivity: flat curve, 0.0686-0.0693 across the sweep (#55). λ-sweep:
   NDCG@10 rises 0.1247→0.1439 and mean intra-list similarity rises
   0.0822→0.1681 as λ rises 0.5→1.0, the expected trade-off direction (#60,
   figure `results/figures/mmr_lambda_sweep.png`).

   Test suite at this checkpoint: 272 passed, 2 xfailed (localness-ρ +
   confidence-decile-monotonicity), 0 failed. 70 new tests this phase across
   `tests/test_firewall_scoring.py` (5), `tests/test_compatibility.py` (26, every
   sub-score + hard gate hand-computed), `tests/test_calibration.py` (9, incl. a
   synthetic isotonic-improves-ECE-vs-naive proof), `tests/test_confidence.py`
   (7), `tests/test_diversity.py` (7, incl. MMR similarity hand-checked pair +
   λ=1.0 degenerate-case proof), `tests/test_utility.py` (5),
   `tests/test_hard_constraints.py` (2, build-blocking), and
   `tests/test_scoring_output.py` (9, full output-schema validation against
   spec.md §9.5's exact field set + the confidence-decile xfail). ruff/mypy
   clean throughout. Self-caught and fixed one real correctness bug during
   development (`docs/DATA_CARD.md` #60): the MMR λ-sweep's NDCG@k was initially
   computed with IDCG derived from only the already-MMR-selected top-k rows
   rather than the trip's full candidate pool, silently inflating NDCG
   ~4-6x — caught by sanity-checking against Phase 5's own system-8 NDCG@10
   before it ever reached a committed test assertion.

## Done (Day 2 P0, Phase 7, spec.md §16 item 9 — committed, independently verified)

9. **Explainability** (`explain/shap_groups.py`, `templates.py`,
   `counterfactual.py`, `output_enrichment.py`) — grouped TreeSHAP
   (`shap.TreeExplainer` against `artifacts/model.txt`, exact not sampled, verified
   `sum(SHAP)+expected_value == booster.predict()` to `atol=1e-6` across all 37,874
   holdout candidate rows), all 237 real feature columns (233 numeric + 4
   categorical) mapped to exactly one of spec's ~10 named groups
   (`docs/DATA_CARD.md` #64 has the full table). **`novelty` is a genuinely empty
   group** (zero observable proxy for the DGP's latent novelty term, per Phase 4a's
   own already-measured finding) — contribution exactly 0.0 always, not
   fabricated; `quality` maps to `rating_shrunk`/`review_count` per spec.md
   §2.2's own "observed only noisily through rating and review_count" text.
   Deterministic NL template layer (`templates.py`, no LLM anywhere per spec §10/
   brief §4) with a genuine cold-start-safe `implicit_taste` variant (never claims
   "past trips" for a traveler with zero as-of-safe interaction history — verified
   against every real cold-start recommendation, not just a hand-built example).
   Genuine counterfactual re-score/re-rank (`counterfactual.py`, reuses the real
   `scoring.compatibility`/`scoring.utility` functions, 2 documented judgment-call
   thresholds for "binding constraint") — 613/2,020 real recommendations carry one.
   **Architecture**: `scoring.output.run_recommend` gained a generic
   `payload_enricher: Callable[[dict[str, Any]], dict[str, Any]] | None` hook
   (zero import of `explain/` inside `scoring/`, preserving Phase 6's firewall);
   `poi_rank.cli recommend` is the only place that wires `explain/` in, via
   `explain.output_enrichment.build_payload_enricher`. `diversity_group` left as
   Phase 6's existing heuristic (task-sanctioned optional polish, not pursued —
   no measured improvement to justify a second computation path).

   **Results** (`docs/DATA_CARD.md` #64-68, real committed dataset, 202 holdout
   trips): `uv run python -m poi_rank.cli recommend` now **1m54s** (up from Phase
   6's ~53s, still well within the <5min budget). Two independent full runs
   produce **byte-identical** `recommendations.json` content once the
   spec-mandated wall-clock `generated_at` field is excluded (verified directly,
   202/202 trips). Every Phase 6 number (ECE, beta-sensitivity, lambda-sweep,
   confidence-decile rho) reproduced EXACTLY — confirms Phase 7 only appends
   `top_signals`/`explanation` content, never touches the scoring math.

   Test suite at this checkpoint: **325 passed, 2 xfailed** (localness-rho +
   confidence-decile-monotonicity, unchanged pre-existing honest misses), **0
   failed** — 53 new tests (`tests/test_firewall_explain.py`,
   `tests/test_shap_groups.py`, `tests/test_templates.py`,
   `tests/test_counterfactual.py`, `tests/test_explain_integration.py`). ruff/mypy
   clean throughout.

   **Orchestrator-caught bug, fixed before commit** (`docs/DATA_CARD.md` #69):
   `_weak_compat_line`'s budget_fit branch always said "Priced above what's
   typical for your {budget} budget" regardless of the actual gap direction —
   factually wrong on 535/766 (70%) of real generated lines (the POI was actually
   CHEAPER than the traveler's target, not pricier). Neither the executor's own
   verification nor the independently-dispatched verifier subagent (across two
   full dispatch rounds) actually completed the data-level spot-check both were
   explicitly told was "the highest-value check in this phase" — both reported
   PASS while explicitly noting that exact check was "blocked by SHAP I/O timing"
   and treating the gap as non-blocking. The orchestrator ran the check directly
   against the real `results/recommendations.json` and found the bug. Fixed:
   `ExplanationContext` gained `price_level`/`budget_target_price_level` fields
   threaded from the same source `scoring/compatibility.py`'s own `budget_fit_score`
   already uses; the line now checks the sign correctly. Re-verified on the real
   regenerated dataset: 231/231 "Priced above" + 535/535 "More budget-friendly"
   lines now factually correct, zero mismatches. Two regression tests added.
   Scoring-layer numbers unchanged (confirmed byte/value-identical) — the fix only
   touches explanation text, never the ranking/scoring computation itself.

## Done (Day 2 P0, Phase 8, spec.md §16 item 10 — the full eval suite)

10. **Full eval suite** (`eval/personalization.py`, `eval/coverage.py`,
    `eval/longtail.py`, `eval/constraints.py`, `eval/cold_start.py`,
    `eval/ablations.py`, `eval/report.py`) — personalization (mean pairwise
    Jaccard@10 + RBO p=0.9 + within/cross archetype-proxy Jaccard, archetype proxy
    reused from Phase 3's `assign_traveler_segments`, never the oracle), coverage
    (catalog coverage@10, Gini, entropy, vs the popularity baseline), long-tail
    (share + precision reported together per spec's "coverage without precision is
    noise" instruction), constraint compatibility (% ≥0.7 + the real hard-violation
    count), diversity (category entropy + intra-list distance, reusing Phase 6's
    lambda sweep), calibration/confidence-decile/beta-sensitivity (Phase 6 numbers,
    now persisted to `results/metrics.json` for the first time, not just CLI
    stdout), cold-start (interaction-count buckets + new-POI cohort cross-
    reference + leave-one-destination-out), and all 9 ablations (`-IPS_weighting`,
    `-calibration`, `-MMR`, `-CF_channel`, `-long_tail_quota` cheap/no-retrain;
    `-text_embeddings`, `-implicit_taste`, `-explicit_interests`,
    `-behavioral_block` one full LightGBM retrain each — **none skipped**, all 9
    measured on the real dataset). `eval/run.py` now also computes a bias-gap
    table (spec §1.3's "single highest-value element," reusing every system's
    already-computed score against a new `models.ranking_data
    .load_holdout_biased_evaluation_frame`, row-order-aligned to the primary frame
    by construction — no rescoring) and persists `candidate_recall@250` (Phase
    4a's number, previously CLI-only) to `results/metrics.json` for the first
    time. **`poi_rank.cli lodo`** (`eval/cold_start.py::run_lodo`) is a genuinely
    separate command — 3 full destination-held-out LightGBM retrains — that
    merges a `"lodo"` key into the already-written `results/metrics.json`, kept
    OUT of `make reproduce`'s default chain per spec §16's own cut-order but run
    and included in this submission anyway (36.4s wall-clock, cheap enough).
    **`eval/report.py`** generates `docs/RESULTS.md` from `results/metrics.json`
    via pure f-string interpolation — zero hand-typed numbers — enforced by
    `tests/test_report.py::test_every_number_in_results_md_traces_to_metrics_json`,
    a real grep-and-cross-check against the source JSON (not a manual promise).

    **Results** (real committed dataset, 202 holdout trips, `uv run python -m
    poi_rank.cli evaluate` = **2m39.8s**, `uv run python -m poi_rank.cli lodo` =
    **36.4s**, both well within the <5 min per-command budget; full determinism
    verified directly — two independent `run_evaluate` calls on the real dataset
    produce byte-identical `results/metrics.json`, 39,636 bytes each). Systems
    table reproduces Phase 5 exactly (lambdamart_ips NDCG@10 = 0.0875, 66.2% of
    ceiling, +40.6% vs popularity p=0.0042). **New Phase 8 honest misses**:
    within/cross archetype Jaccard ratio = 1.08 (target ≥2.0, **MISSED** —
    root-caused to the K-Means archetype PROXY being too coarse relative to what
    actually drives personalization, per-trip stay location + implicit taste +
    semantic candidates; cross-archetype Jaccard itself, 0.0404, **MET** the ≤0.25
    target in isolation); long-tail share = 0.2338 (target ≥0.25, **MISSED**, a
    near-miss) with precision = 0.0678 (target ≥0.5, **MISSED** — but honestly
    NOT a long-tail-specific collapse: it sits close to the primary system's
    OVERALL precision@10 of 8.4%, tracing to the same already-documented
    candidate-recall ceiling, not a defect unique to long-tail POIs). **New Phase
    8 successes** (not named §11.10 targets but measured anyway): coverage@10 =
    27.4% vs the popularity baseline's 4.7% (5.8x more of the catalog, lower
    Gini, higher entropy — a clear, measured "no popularity monoculture" result);
    bias-gap for lambdamart_ips (+0.0828) is smaller than both popularity's
    (+0.1099) and plain lambdamart-without-IPS's (+0.1721), confirming IPS
    correction measurably reduces exposure-bias inflation. **LODO**: every
    destination shows lower NDCG@10 held out vs full training (barcelona 0.0700
    vs 0.1040, p=0.048 — the only one reaching significance at this
    per-destination trip count; kyoto 0.0575 vs 0.0776 p=0.12; seoul 0.0634 vs
    0.0836 p=0.12) — directionally consistent, honestly reported as
    only-partially-significant. **Ablations**: `-IPS_weighting` is the largest,
    most significant contributor (Δ=-0.0329, p=1.8e-05); `-CF_channel`
    (Δ=+0.0017, p=0.013) and `-long_tail_quota` (Δ=+0.0077, p=3.3e-07) both
    slightly IMPROVE raw NDCG@10 when removed — reported as a genuine, honest
    trade-off (both channels exist for coverage/long-tail objectives, not NDCG,
    and Phase 4a already showed both contribute real positive marginal candidate
    recall), never spun as "the channel is useless"; the 4 feature-block
    ablations all point the expected direction (removing hurts) but none reaches
    p<0.05 individually at this holdout size. Full numbers, every diagnosis, and
    the complete generated table: `docs/RESULTS.md`, `docs/DATA_CARD.md` #70-77.

    Test suite at this checkpoint: **381 tests collected (379 passed), 2 xfailed**
    (pre-existing, unchanged — localness-rho, confidence-decile-monotonicity),
    **0 failed**, confirmed via multiple independent full `uv run pytest` runs
    (all exit code 0). New test files: `tests/test_personalization.py` (12),
    `tests/test_coverage.py` (14, incl. the determinism regression test below),
    `tests/test_longtail.py` (4), `tests/test_constraints.py` (2),
    `tests/test_cold_start.py` (5), `tests/test_ablations.py` (9),
    `tests/test_report.py` (4, incl. the grep-based no-hand-typed-numbers
    enforcement test), `tests/test_firewall_eval.py` (3, the "nothing imports
    eval/" mirror-image firewall). Extended `tests/test_ranking_data.py` (the
    biased/unbiased frame row-order-alignment invariant) and
    `tests/test_evaluate.py` (the full Phase 8 payload shape). ruff/mypy clean
    throughout.

    **Orchestrator-caught genuine byte-determinism failure, fixed before commit**
    (`docs/DATA_CARD.md` #78): the verifier ran `uv run python -m poi_rank.cli
    evaluate` twice directly (no `make`, no externally-set `PYTHONHASHSEED`) and
    got two DIFFERENT `results/metrics.json` SHA256 hashes — this project's hard
    determinism requirement genuinely violated, not accepted as "fine for
    numerical computing." Root cause: `eval/coverage.py::recommendation_frequency`
    iterated the raw `catalog_poi_ids` `set[str]` directly to build a dict whose
    `.values()` fed `shannon_entropy`'s floating-point summation — `set` iteration
    order for `str` depends on per-process hash randomization, so the summation
    order (and therefore the exact float, ~2e-15 off between runs) varied on every
    invocation that didn't share a `PYTHONHASHSEED`. Same failure CLASS already
    documented elsewhere in this project for `candidates/channels.py`'s
    epsilon-greedy sampling (fixed there via a SHA256-derived seed, never Python's
    `hash()`) — hash-order-dependent iteration reaching a numeric reduction.
    Fixed: `recommendation_frequency` now iterates `sorted(catalog_poi_ids)`,
    deterministic independent of `PYTHONHASHSEED` entirely (more robust than
    relying on the Makefile's `export PYTHONHASHSEED := 0`, which only covers
    invocations that go through `make`). Re-verified: ran `evaluate` twice with
    `PYTHONHASHSEED` explicitly unset (the exact failing scenario) — byte-identical
    SHA256 both times. Regression test added. No other `set[str]`-into-numeric-
    reduction pattern found elsewhere in the new Phase 8 modules (checked
    `personalization.py`'s Jaccard/RBO specifically — both are exact-cardinality
    operations, order-independent by construction).

## Done (Phase 9, spec.md section 15 — the 3+1 required scenarios)

11. **Scenarios** (`eval/scenarios.py`, `poi_rank.cli scenarios` / `make
    scenarios`) — 4 hand-specified SYNTHETIC traveler/trip profiles (not real
    dataset travelers), run through the EXACT SAME live pipeline `recommend`
    uses for real holdout trips (candidate generation → LambdaMART+IPS scoring
    + calibration + confidence → compatibility/hard gates → multiplicative
    utility → MMR → grouped-TreeSHAP explanations), never a second parallel
    scoring implementation. 4 small, all-`None`-by-default override parameters
    added to 3 existing modules for this (`scoring/output.py
    ::run_scoring_pipeline`, `explain/output_enrichment.py
    ::enrich_recommend_result`, `candidates/union.py::generate_candidates`'s
    `segments_override`) — zero behavior change for every existing caller
    (`docs/DATA_CARD.md` #83). New `features/traveler_features.py
    ::assign_traveler_segments_out_of_sample` gives synthetic travelers a
    K-Means archetype segment that means the SAME thing as
    `poi_features.parquet`'s already-persisted affinity columns, avoiding a
    real relabeling bug a naive joint refit would have introduced
    (`docs/DATA_CARD.md` #82). Destination: seoul, all 4 scenarios (documented
    choice, #80). Scenario interests mapped from spec.md's plain English onto
    the real `INTEREST_LABELS` vocabulary (#79). Diagnostic scenario 4 bases on
    Scenario 1 (#81), `touristiness_pref` flipped -0.8 → +0.8, everything else
    held constant (verified via a field-by-field row diff test).

    **Results** (real committed dataset, `uv run python -m poi_rank.cli
    scenarios` = **44.8s**, well within the <5min budget, byte-deterministic
    across 2 runs): all 4 scenarios produced real top-10 recommendations with
    genuine grouped-TreeSHAP explanations (not placeholders). **Honest miss,
    root-caused, not hidden** (`docs/DATA_CARD.md` #85): the scenario-4-vs-base
    top-10 Jaccard overlap = **0.5385**, NOT low as spec.md section 15 expects
    (using this project's own established ≤0.25 "low" bar) — root-caused to
    Scenarios 1/4's candidate pools being **89.8%** Jaccard-identical (both
    cold-start travelers land in the SAME K-Means archetype segment, since
    `touristiness_pref` is only 1 of ~38 clustering dimensions and doesn't gate
    any of the other 5 candidate channels), so `touristiness_pref` can only
    reorder the final ranking through 2 of >230 real feature columns
    (`explicit_touristiness_pref`, `interact_localness_gap`) — `scoring/
    compatibility.py`'s 6 sub-scores carry no localness term at all. Not tuned
    to force a lower number.

    **Orchestrator-caught pre-existing repo inconsistency, fixed before commit**
    (`docs/DATA_CARD.md` #85): regenerating `docs/RESULTS.md` for this phase's
    scenarios section would have silently REVERTED the already-committed LODO
    section to "not yet run" — the committed `results/metrics.json` lacked the
    `"lodo"` key even though the committed `docs/RESULTS.md` already showed
    real LODO numbers (a pre-existing gap from a prior session, not caused by
    this phase). Caught by diffing the regenerated file against git before
    committing. Fixed by re-running the already-built `poi_rank.cli lodo`
    command (no new code) — numbers reproduced exactly, only wall-clock
    differed (36.4s → 22.8s, a genuine re-measurement).

    **Orchestrator caught a RECURRENCE of the exact same issue before commit**
    (`docs/DATA_CARD.md` #85 addendum): `lodo`'s merge into `metrics.json` is a
    read-modify-write that does not survive a later `poi_rank.cli evaluate`
    call (which writes a fresh file with no `lodo` key). After the executor's
    own fix above, the dispatched verifier's backward-compatibility check
    (CHECK 8) re-ran `evaluate` to diff against Phase 8's baseline — legitimate
    on its own terms, but it silently re-wiped the just-restored `lodo` key,
    undetected until the orchestrator re-checked `results/metrics.json`'s
    actual keys immediately before staging (not `docs/RESULTS.md`, which still
    looked fine — it was generated from a snapshot taken before the re-wipe).
    Fixed by re-running the correct final sequence once more — `evaluate` →
    `lodo` → `scenarios` → `docs` — and confirming `metrics.json` genuinely has
    the key and every RESULTS.md LODO number matches it, immediately before
    commit. **Lesson for future phases**: any downstream step that re-runs
    `evaluate` (a re-verification, a fresh reproduce, a backward-compat diff)
    can silently undo `lodo`'s hand-run merge with zero visible symptom in the
    generated doc. Never trust "it worked earlier in this session" — check the
    actual current file's keys right before every commit.

    Test suite at this checkpoint: **all tests passed** — new
    `tests/test_scenarios.py` (profile-field correctness, scenario-4-vs-base
    diff invariant, overlap-matrix symmetry/diagonal/genuinely-computed
    invariants, full real-fixture-chain determinism); `tests/test_report.py`
    extended (not a parallel mechanism) with the scenarios section's own
    number-tracing enforcement. ruff/mypy clean throughout.

## Next (Day 2 P0, spec.md §16)

7. ~~**LambdaMART + IPS** (`models/lambdamart.py`) — `lightgbm` `lambdarank`,
   grouped by `trip_id`, `lambdarank_truncation_level=20`, graded labels 0-3.
   IPS weights `clip(1/p_expose, 1, 20)` normalized per group — report NDCG
   with/without IPS as an ablation. 15% behavioral-feature-block dropout for
   new-POI robustness (measure on the new-POI cohort). This becomes systems 7
   (LambdaMART) + 8 (LambdaMART+IPS, primary) in `results/metrics.json`, which
   is already structured to accept them without rework (`eval/run.py`).~~
   **DONE — see "Done (Day 2 P0, Phase 5)" above.**
8. ~~**Scoring layer** (`scoring/`) — multiplicative utility (deviation from
   brief's additive formula, justify in TECHNICAL.md), hard gates, 6-term
   geometric-mean compatibility, isotonic calibration (ECE/Brier/reliability),
   confidence (validate monotonicity via Spearman, §9.3), MMR diversity (λ
   sweep). Output JSON schema per §9.5.~~
   **DONE — see "Done (Day 2 P0, Phase 6)" below.**
9. ~~**Explainability** (`explain/`) — grouped TreeSHAP, template layer, no LLM.~~
   **DONE — see "Done (Day 2 P0, Phase 7)" above.**
10. ~~**Full eval suite** (`eval/`) — personalization (Jaccard within/cross
    archetype), coverage (Gini/entropy), long-tail precision, constraint
    compatibility (hard-violation-rate=0 test, build-blocking per spec §11.5),
    diversity, calibration, cold-start cohorts, leave-one-destination-out,
    ablations (9 rows), then **generate `docs/RESULTS.md` from
    `results/metrics.json`** — no hand-typed numbers anywhere in `docs/`.~~
    **DONE — see "Done (Day 2 P0, Phase 8)" below. All 9 ablations AND LODO
    completed (not deferred to P1) — real wall-clock made both affordable
    within budget.**
11. ~~3+1 required scenarios (`make scenarios`).~~
    **DONE — see "Done (Phase 9, spec.md section 15)" below.**
12. ~~`docs/TECHNICAL.md` (11 sections, incl. explicit LambdaMART-vs-two-tower and
    multiplicative-vs-additive justifications), README, finalize DATA_CARD.~~
    **DONE — Phase 10 (final).** `docs/TECHNICAL.md` (701 lines, all 11
    sections: problem formulation, DGP/circularity defense, feature
    architecture, candidate generation, ranking model choice (LambdaMART vs
    two-tower, explicit), IPS correction, scoring layer (multiplicative vs
    additive, explicit, tied to the real 0-violation hard-constraint test),
    calibration, explainability, evaluation methodology, production
    considerations) + closing honest-misses summary (**5 missed, 4 met** of
    spec.md §11.10's 9 targets, independently recounted by the verifier
    directly from `docs/RESULTS.md`'s table, not just trusted). `README.md`
    (183 lines) — quick-start, expected headline numbers, honest misses,
    architecture/repo-layout notes. Every number in both traces to a real
    artifact: independently spot-verified by the verifier against 27 sampled
    claims (15 from TECHNICAL.md, 12 from README.md), zero untraced. `make` is
    not installed in this dev environment (documented, `docs/DATA_CARD.md`
    #87) — README's primary quick-start is the manual 9-command
    `uv run python -m poi_rank.cli ...` sequence, actually executed end-to-end
    (generate→prepare→features→candidates→train→evaluate→lodo→scenarios→report),
    every command's exit code confirmed, resulting diff hand-verified as only
    `generated_at` timestamps + one LODO wall-clock re-measurement (no
    substantive drift) before reverting by hand (never `git checkout`/`reset`
    per rule 55a) and confirming `git status --short` empty. New
    `tests/test_docs_numbers.py` (extends the number-tracing enforcement to
    these two hand-written files) and `tests/test_readme_quickstart.py`
    (genuinely exercises the CLI via `typer.testing.CliRunner`, not mocked).

    Test suite at this checkpoint: **398 passed, 2 xfailed** (pre-existing,
    unchanged — localness-rho, confidence-decile-monotonicity), **0 failed**,
    400 collected — independently re-run by both the verifier and the
    orchestrator. ruff/mypy clean throughout.

    **This closes out every P0 item in spec.md §16's build plan.** Remaining
    work is P1/P2 only (exploration notebook, feature-importance figure — both
    explicitly optional per spec's own cut-order, and the λ sweep figure is
    already done from Phase 6).
13. ~~P1: full ablation table, LODO, confidence-decile validation (if time).~~
    **DONE early — see item 10 above (all 9 ablations + LODO landed in Phase 8
    itself); confidence-decile validation was already measured in Phase 6 and
    is now also persisted to `results/metrics.json`.**
14. P2 (cut first if short on time): notebook, λ sweep figure, feature-importance
    figure. (λ sweep figure already done, Phase 6.)

## Diagnostic-only harness (spec-v2-remediation.md section 0/section 1, D1-D10)

15. **DGP diagnostic harness** (`src/poi_rank/eval/dgp_diagnostics.py`, new CLI
    `poi_rank.cli diagnose-dgp`) — measurement only, no fix code, per explicit
    task scope. Computes D1 (variance decomposition of `u(t,p)`, 7 deterministic
    terms + epsilon), D2 (taste-cosine distribution), D3 (Spearman(u, label)),
    D4 (choice sharpness), D5 (oracle NDCG@10, slate- vs candidate-level), D6
    (cold-start share), D7 (Spearman(latent_localness, geo feature)), D8
    (popularity bias gap, cross-checked against the already-committed Phase 8
    number). Writes `results/parts/dgp_diagnostics.json`. Small additions to
    `eval/oracle.py` (`load_traveler_taste`,
    `validate_geo_feature_against_latent_localness`) — no other module touched.
    `tests/test_dgp_diagnostics.py` (20 tests: hand-built arithmetic per
    diagnostic + a real-fixture-chain correctness invariant). Full numbers and
    the report on this harness are reserved for the orchestrator/GG per task
    scope — not interpreted here.

    **Extended with D9 (semantic-space fidelity) + D10 (is POI text generation
    conditioned on `poi_semantic`?)** — same file, same firewall discipline, no
    new `eval/oracle.py` reads needed (D9 reuses D1/D2's already-computed
    `taste_sim`; D10 reuses `load_poi_latent`, already permitted). D9 compares
    `Spearman(cos_DGP(taste,poi_semantic), cos_observable(taste_feature,
    poi_emb))` for both the canonical TF-IDF→SVD-64 path (reads the
    already-committed `poi_features.parquet`/`traveler_features.parquet`
    directly) and a one-time `all-MiniLM-L6-v2` measurement (`sentence-
    transformers` installed via `uv sync --extra text`; `features/text_embed.py`'s
    existing `embed_text_sentence_transformer` reused unchanged, run in an
    isolated subprocess — see `_run_minilm_encode_in_subprocess`'s docstring for
    why: a genuine, verified Windows DLL-ordering conflict on this dev machine
    between `pandas`/`pyarrow` and `torch` when `pandas` loads first in the same
    process, which it always does by the time this diagnostic runs via the CLI
    or pytest). MiniLM artifact written to `results/parts/
    poi_emb_minilm_diagnostic.npy` (diagnostic-only, never `artifacts/
    poi_emb.npy`). D10 gives a direct code-reading answer (with file:line
    citations into `datagen/catalog.py`/`text_templates.py`) plus an independent
    empirical Spearman over a seeded 5,000-pair within-destination sample of the
    full ~1,446-POI catalog. 9 new tests in `tests/test_dgp_diagnostics.py` (29
    total in that file); full suite 427 passed, 2 xfailed, 0 failed (427 = 418
    prior baseline + 9 new). `ruff`/`mypy` clean. Numbers not interpreted here —
    reserved for the orchestrator/GG per task scope.

## DGP remediation, Block A (spec-v2-remediation.md + orchestrator's superseding numbers)

16. **RC1/RC2/RC3 fixes + `gate-dgp` acceptance gate** — full detail in
    `docs/DATA_CARD.md` "DGP remediation, Block A" section (before/after D1-D10
    table, exact config diffs, wall-clock, freeze-discipline confirmation). One-line
    summary: RC1 (random-holdout slate 20->150), RC2a (z-score-before-weight
    standardization + reweighting, `datagen/utility.py` rewritten), RC2b (POI geo
    generation conditioned on `latent_localness`, confirmed-bug fix, D7 0.012->0.693),
    RC2c (POI text generation conditioned on `poi_semantic`, confirmed-bug fix, D10
    0.208->0.452), RC3.1 (per-impression as-of cutoff, `features/traveler_features.py`
    simplified to a pure timestamp filter — the one sanctioned `features/` change),
    RC3.2 (pre-trip synthetic history, new `interactions_pretrip.parquet`, cold-start
    67.8%->10.3% holdout), RC3.4 (scale 800->2,500 trips). New `poi_rank.cli
    gate-dgp` command + `src/poi_rank/eval/gate_dgp.py`, reusing `diagnose-dgp`'s
    already-computed D1-D10 payload, applying the orchestrator's exact 9-threshold
    table. **Gate result: 6/9 PASS** (`results/parts/dgp_gate.json`) — FAILs: D9
    semantic fidelity (0.187 vs 0.50, largely a `features/`-pipeline-chain property
    out of this task's scope), D10 description conditioning (0.452 vs 0.50, close,
    genuine effort, hit a measured structural asymptote), D5 candidate-level oracle
    NDCG@10 (0.432 vs 0.45, close, root-caused to candidate-recall ceiling, a
    `candidates/` property out of scope). Freeze discipline held throughout:
    `candidates/`, `models/`, `scoring/` untouched; `features/` touched only at the
    one sanctioned call site. **Full pipeline wall-clock flagged as a real risk**:
    `generate`+`prepare`+`features`+`candidates` alone now take ~4m32s at the new
    scale (candidates dominates at ~3m9s, superlinear vs the 3.125x trip scale-up) —
    inserting `gate-dgp` (~60s, requires prepare/features/candidates to already
    exist per its own documented precondition) pushes this close to or over the
    5-minute `make reproduce` budget BEFORE Block B's train/evaluate/lodo/recommend/
    scenarios stages are added back — flagged for Block B/D's attention, not fixed
    here (candidate-generation performance is out of this task's scope). New tests:
    `tests/test_gate_dgp.py` (16 cases), plus additions to `tests/
    test_datagen_schema.py` (RC1/RC2b/RC2a/RC3 mechanism tests), `tests/
    test_traveler_features.py` (per-impression cutoff + pretrip-integration tests),
    `tests/test_dgp_diagnostics.py` (updated for the standardization-aware
    `compute_utility_term_components`), and `tests/test_localness_oracle.py`'s
    `xfail` reason updated to the new measured number (0.4757 -> 0.579, still a
    genuine miss, not XPASS). **Explicitly stops here — Block B (re-running Phases
    2-9 downstream against the fixed data) is out of scope for this task.**

## Process notes for future sessions

- Every phase: dispatch executor (fresh, full context in prompt) → dispatch
  verifier (independent, told to distrust executor's self-report, re-derive key
  numbers itself) → orchestrator reviews both, fixes small things directly
  (e.g. one mypy annotation, one xfail conversion) → stage explicit paths (never
  `git add -A`) → verify `git diff --cached --stat` matches intent → commit → push.
- One executor (Phase 4a) hit a session rate limit mid-task; its work was
  actually complete (confirmed via git status + re-running tests/lint/mypy
  myself) — it just never sent its final report. Lesson: on a "failed" task
  notification, check actual repo state before assuming lost work.
- `git status --short` after every dispatch, before staging — catches leftover
  scratch files (e.g. verifier's manual `_run1` comparison copies after Phase 3,
  deleted before commit).

## A2 -- text vocabulary widening + Gate-A restructure (spec-v3 sections 2.1/3.1/4)

- Done: per-dimension synonym-rich phrase pools (`datagen/text_templates.py`, config `text:`),
  loading-proportional (systematic) sampling, surface realization from an independent stream;
  D10 `raw_tfidf` (Gate-A) 0.4607 -> 0.6892, `canonical_svd64` 0.4522 -> 0.7102; Gate-A 8/8 PASS
  (`results/parts/dgp_gate.json`). Evidence: `results/parts/d10_vocab_sweep.json`;
  write-up: docs/DATA_CARD.md "A2".
- Finding to carry forward: pool size (3..25) does not move D10 for a bag-of-words reader
  (0.7097 -> 0.6900 raw); anchoring, systematic sampling, 10-14 mentions/POI and a small surface
  budget did. 10-14 mentions is a deviation from spec-v3's 3-6 (3-6 reached only 0.5540).
- DR4 caveat: the synonym-rich, anchored vocabulary design determines the TF-IDF-vs-MiniLM
  outcome; DR4's conclusion is about this dataset, not encoders in general.
- Full-catalog oracle NDCG@10 (0.3554) is a diagnostic only (exposure-capped; exposed fraction
  0.4931). D9 and candidate-level oracle NDCG moved to Gate-B (A3); still computed.
- Gotchas: text now uses per-POI seeded streams, so the main RNG stream shifted and every
  downstream quantity is a new realization (holdout trips 620 -> 671); committed
  `data/synthetic/*`, `artifacts/poi_emb.npy`, and `results/parts/*` were regenerated, but
  `artifacts/model*.txt`, `calibrator.pkl`, `results/metrics.json`, figures are STALE until
  Block B.


## Session 2 (2026-09-20) — A3 / A4 / B / C / D

Tier: T1 portfolio project (take-home for konnect.kr; due 2026-09-21 21:00 KST).

Decisions taken autonomously (each with the number behind it; details in `docs/DATA_CARD.md`
"Post-A2 record", `results/parts/*.json` and RESULTS.md):

- **A3 step 0 -> behavioural item embedding SKIPPED.** Text D11 ridge R² 0.865 (>= 0.45); text-alone
  within-trip D9 0.927; taste-estimator-alone 0.572; shipped chain 0.473. Candidate recall was the
  real failure (6-channel union 0.596 at ~40% of the catalog, chance 0.394; true-utility top-K
  would recall 0.991) -> learned cross-fitted retriever. Gate-B blocking rows pass.
- **A4** reproduce (data -> gates -> train -> evaluate -> compose) and reproduce-full timings are in
  `results/parts/timings*.json`; the 5-minute reproduce target was NOT met on this laptop (see
  README/RESULTS for the measured numbers). Biggest wins: persisted confidence ensemble, TreeSHAP on
  returned rows, vectorised MMR. Concurrent LightGBM fits did not help.
- **B**: parts + compose; LR CV-L2; personalization same-destination + true archetype labels
  (+ perfect-ranker reference); per-stratum chance lift; Gate-B; make audit; xfail removed (S5).
- **C**: DR1-DR4, DR6-DR11 measured; DR5 (ANN benchmark) NOT RUN; 5-seed replication run
  (`scripts/seed_replication.py`).
- **D**: RESULTS.md leads with local/long-tail discovery + bias gap; TECHNICAL.md and README.md are
  now templates (`*.tmpl`) whose numbers resolve from metrics.json.

Open items / follow-ups (not blocking): DR3 and DR7 suggest a pointwise objective and less IPS
clipping may do better on this data (not adopted: selecting on the holdout would leak); the
localness index weights predate the geo fix and are now below the best single input (not re-tuned:
would leak the oracle); long-tail precision 0.4 target unmet (hypothesis untested).
