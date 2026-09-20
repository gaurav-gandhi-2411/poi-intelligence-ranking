# Experiment H — can explicit traveler x POI cross features make the ranker beat content cosine?

**Status:** pre-registered 2026-09-20, BEFORE any H experiment was run. Branch
`feat/h-ranker-cross-features`. Parent commit: `ec8ac4a` (tag `v1.0-konnect-submission`).

## Question and hypothesis

The shipped ranker beats content cosine by only +0.0074 NDCG@10 (seed 42, p=0.445; 5-seed mean gap
+0.0110, above cosine in 4 of 5 seeds). Hypothesis (from the requester, tested here, not assumed):
the ranker cannot see the traveler-dependent terms cosine misses because they live in
traveler x POI interactions, which a GBDT recovers poorly from raw factors (traveler features are
constant within a trip group, so splits on them waste depth), and the shipped feature set has few
explicit crosses (`interact_localness_gap` uses an unbounded z-score against a [-1, 1] preference).

## H0 — confirm or refute before building (measured, reported either way)

- (a) Grouped-SHAP attribution by feature group on the shipped model (train-carved validation
  trips; descriptive, not a selection). Report the share carried by localness / party / price /
  novelty groups.
- (b) Probe: a ranker trained on ONLY 7-10 hand-built cross features, compared with the shipped
  237-feature ranker on validation IPS-weighted NDCG@10. If the small model matches or beats the big
  one, the feature set (not the model class) is the bottleneck.

## Selection metric and protocol (fixed now)

- Metric: IPS-weighted validation NDCG@10 on the train-carved validation split, exactly as
  `eval/ranker_sweep.py` (eval clip 20, same fit/val split, same IPS weights), mean over seeds
  {42, 7, 11, 13}. Oracle-free; the holdout is NOT read during selection and is reported ONCE after
  the configuration is frozen.
- Baseline (shipped, re-measured on the K=195 candidates): 0.3167 (4-seed mean,
  `results/parts/ranker_sweep.json`, `previous_shipped_config_4_seed`).
- **Adoption bar: validation >= 0.3267** (= shipped + 0.010 absolute, the requester's stated intent).
  The requester's literal figure was 0.3226 against a shipped 0.3126; 0.3126 was the K=210 measurement
  and the shipped value on the current K=195 candidates is 0.3167, so the literal figure would be a
  +0.006 bar. The stricter 0.3267 is used (it also satisfies 0.3226).
- If the best configuration lands below the bar: DO NOT merge; keep the shipped model; write up the H0
  numbers as a measured finding about where the remaining signal is.
- If it clears the bar: merge only if Gate-A, Gate-B and `audit --deep` all still pass; then re-run
  the 5-seed replication, the affected Decision Register rows, and re-render all docs.

## Candidate changes (H1-H3), all selected on validation only

- H1: cross features from OBSERVABLE columns only (stated traveler attributes and derived POI
  columns the production system has; nothing from `_oracle/`): localness x touristiness and gap,
  category/tag/interest affinity, signed and asymmetric price gap, party fit, novelty (time-decayed
  prior engagement), travel time under the mobility mode and its ratio to the mode's tolerance,
  cosine to the traveler's dismissed-POI centroid. These are the compatibility dimensions the
  assignment's section 11 requires; they happen to align with the simulator, which is standard
  travel-domain feature engineering, not leakage. The firewall test is extended to cover them.
- H2: LightGBM `init_score` = calibrated content-cosine score (learn the residual). Adopt only if it
  wins on validation.
- H3: `lambdarank_truncation_level`, `label_gain`, `min_data_in_leaf`, `num_leaves` (validation only).
- H4: report the 5-seed PAIRED comparison against content cosine (mean gap, sd, paired t and Wilcoxon
  across seeds), not only the single-seed per-trip Wilcoxon.

## Non-goals

No retriever change (cross features are excluded from the retriever so candidate sets and the K=195
rule are untouched), no data regeneration, no holdout-driven selection.
