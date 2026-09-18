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
9. **Explainability** (`explain/`) — grouped TreeSHAP, template layer, no LLM.
10. **Full eval suite** (`eval/`) — personalization (Jaccard within/cross
    archetype), coverage (Gini/entropy), long-tail precision, constraint
    compatibility (hard-violation-rate=0 test, build-blocking per spec §11.5),
    diversity, calibration, cold-start cohorts, leave-one-destination-out,
    ablations (9 rows), then **generate `docs/RESULTS.md` from
    `results/metrics.json`** — no hand-typed numbers anywhere in `docs/`.
11. 3+1 required scenarios (`make scenarios`).
12. `docs/TECHNICAL.md` (11 sections, incl. explicit LambdaMART-vs-two-tower and
    multiplicative-vs-additive justifications), README, finalize DATA_CARD.
13. P1: full ablation table, LODO, confidence-decile validation (if time).
14. P2 (cut first if short on time): notebook, λ sweep figure, feature-importance
    figure.

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
