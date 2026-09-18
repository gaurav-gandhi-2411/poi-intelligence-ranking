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
  ranking metric downstream — expect LambdaMART's NDCG@10 vs popularity and %-of-
  ceiling targets in §11.10 to likely miss too. Report honestly with the same
  root-cause discipline, do not tune candidate generation post-hoc to inflate it.**
- Logistic regression baseline underperforms popularity — `docs/DATA_CARD.md`
  Phase 4b section.

## Next (Day 2 P0, spec.md §16)

7. **LambdaMART + IPS** (`models/lambdamart.py`) — `lightgbm` `lambdarank`,
   grouped by `trip_id`, `lambdarank_truncation_level=20`, graded labels 0-3.
   IPS weights `clip(1/p_expose, 1, 20)` normalized per group — report NDCG
   with/without IPS as an ablation. 15% behavioral-feature-block dropout for
   new-POI robustness (measure on the new-POI cohort). This becomes systems 7
   (LambdaMART) + 8 (LambdaMART+IPS, primary) in `results/metrics.json`, which
   is already structured to accept them without rework (`eval/run.py`).
8. **Scoring layer** (`scoring/`) — multiplicative utility (deviation from
   brief's additive formula, justify in TECHNICAL.md), hard gates, 6-term
   geometric-mean compatibility, isotonic calibration (ECE/Brier/reliability),
   confidence (validate monotonicity via Spearman, §9.3), MMR diversity (λ
   sweep). Output JSON schema per §9.5.
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
