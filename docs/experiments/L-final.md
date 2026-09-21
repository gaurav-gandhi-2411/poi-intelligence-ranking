# Experiment L — final scoring-layer and retrieval decisions (pre-registered)

Written 2026-09-21, **before** any L run. Standing rules: every selection is made on the train-carved
VALIDATION split (`models/lambdamart.py::train_val_split_by_trip`, the same split that early-stops the
booster; the booster is not refit), the holdout is read ONCE per decision after the selection is frozen,
and the oracle (`_oracle/`) is never in a selection path. No numeric target is set from prior belief;
every constraint below is derived from the data. Nothing is retrained in L1/L2.

## L1 + L2 — stated preference steers the ranking; MMR lambda is selected

### Why (K1)

`docs/TECHNICAL.md` section 3.1: the simulator encodes the stated touristiness preference in outcomes
(trip-level Spearman -0.51) and the shipped ranker reproduces 83% of that gradient for cold-start trips
but about 0% for trips with history. In training data history and stated preference agree, so the trees
route touristiness through history-derived features. A stated "prefer less touristy" is a per-trip
instruction, not a long-term taste estimate, and the brief's section 11 puts trip context in the scoring
layer. The fix is therefore a scoring-layer factor, not a retrain.

### The change

```
utility = hard_gate * relevance^alpha * compatibility^beta * pref_align^gamma
pref_align(t, p) = sigmoid( localness_centered_p * (-touristiness_pref_t) / s )
localness_centered_p = num_localness_p - mean(num_localness over the TRAIN ranking frame rows)
s = population sd of  localness_centered * (-touristiness_pref)  over the TRAIN ranking frame rows
```

Sign convention: the resolved DGP convention (negative preference = prefers local; `docs/DATA_CARD.md`
ambiguity 1). Only the observable localness index (`num_localness`) is used, never `_oracle/`. At
`gamma = 0` the utility is unchanged. `pref_align` for a neutral traveler (pref = 0) is the constant 0.5, so
it cannot reorder that traveler's candidates. The two constants (`center`, `s`) are label-free functions of
the train frame; they are computed once by the selection script, written to `configs/scoring.yaml` and
guarded by a test that recomputes them. `pref_align` is added to the output schema (a field on each
recommendation and an entry of `compatibility_breakdown`) and an explanation template is added (the brief's
section 13 example includes "lower tourist concentration than comparable POIs").

### MMR lambda

`lambda = 0.8` was a default written into the spec and was never selected by any procedure. E3
(`results/parts/longtail_stages.json`) measured MMR as the largest long-tail precision loss (3.54x the
pool base rate at the raw ranker, 2.03x served). Selecting lambda on validation is not re-tuning: it is the
selection that was skipped.

### The pre-registered joint rule

Grid: `gamma in {0, 0.25, 0.5, 1, 2, 4}` x `lambda in {0.6, 0.7, 0.8, 0.9, 1.0}` (30 configs; re-scoring only).

All quantities are computed on the validation trips through the whole serving path (shipped booster ->
calibrator -> hard gate -> compatibility -> utility with `pref_align^gamma` -> MMR over the top-50 -> top-10):

* **V-NDCG@10** = IPS-weighted NDCG@10 of the served top-10 (`eval/ranker_sweep.py::ips_weighted_ndcg10`, the
  metric every validation-side selection in this project uses).
* **V-entropy** = Shannon entropy (bits) of the category distribution over all served top-10 slots
  (the holdout `category_entropy_at_10_bits` definition, `eval/run.py::_diversity_payload`).
* **Flip-overlap** = mean over validation trips of the Jaccard overlap of the served top-10 with the served
  top-10 obtained after negating the trip's `touristiness_pref` and recomputing every dependent quantity
  (`explicit_touristiness_pref`, `interact_localness_gap`, all `xf_*` columns, the ranker score, the
  calibrated relevance, `pref_align`). It uses no label, so measuring it on validation leaks nothing.
  Also reported split into cold-start trips (no history) and trips with history.
* **V-LT-precision** = share of long-tail served items with label >= 1 (the scorecard definition).

Constraints (both on validation), relative to the shipped configuration `(gamma = 0, lambda = 0.8)`:

1. `V-NDCG@10 >= V-NDCG@10(shipped) - 0.003`. The tolerance is data-referenced: the holdout NDCG@10
   bootstrap CI half-width is 0.0104 (`docs/RESULTS.md`), and 0.003 is under a third of it, so a config
   passing (i) is not distinguishable from the shipped one at the resolution of the holdout metric.
2. `V-entropy >= 0.9 x V-entropy(shipped)`.

Objective: among configs satisfying both constraints, **minimise validation flip-overlap**. Tie-break:
configs whose flip-overlap is within one standard error of the mean flip-overlap (across validation trips,
taken at the minimising config) of the minimum are tied, and the tie is broken by higher V-LT-precision.

If no config satisfies both constraints, `gamma = 0` and `lambda = 0.8` stay. The constraints are not relaxed.

Reported for the winner AND every grid point: V-NDCG@10, V-entropy, V-LT-precision, flip-overlap overall,
cold-start and with-history. The success signal is that the with-history responsiveness moves materially off
about 0% (K1's statistic: the ranker-score slope of per-trip Spearman(localness, score) on stated
preference, as a share of the label slope), reported before and after for both regimes on the holdout, once,
after the selection is frozen.

### What is not claimed

Minimising flip-overlap alone would drive `gamma` upward without bound; the two constraints are what stop
it, and the objective is deliberately not "maximise NDCG". If the winner sits on a constraint boundary that
is reported. Validation labels come from the popularity-biased training log, so V-NDCG is IPS-weighted and
V-LT-precision is exposure-biased: both are used only to rank configs against each other.

## Result of L1 + L2 (validation only; the holdout is not read here)

Run: `scripts/l12_select.py` -> `results/parts/l12_selection.json` (274 validation
trips, 274 scored; `pref_align` constants center -0.0474, scale
0.2398, from the train frame). Shipped configuration on validation: V-NDCG@10
0.1463, V-entropy 3.247 bits, flip-overlap
0.847 (cold-start 0.719, with history
0.866). Constraint floors: V-NDCG@10 >= 0.1433,
V-entropy >= 2.922. 9 of 30 configurations are feasible.

| gamma | lambda | V-NDCG@10 (IPS) | V-entropy (bits) | V-LT precision | flip-overlap | cold-start | with history | feasible |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6 | 0.1402 | 3.281 | 0.287 | 0.796 | 0.677 | 0.813 | no |
| 0 | 0.7 | 0.1413 | 3.283 | 0.288 | 0.825 | 0.695 | 0.844 | no |
| 0 | 0.8 | 0.1463 | 3.247 | 0.302 | 0.847 | 0.719 | 0.866 | yes |
| 0 | 0.9 | 0.1556 | 3.110 | 0.302 | 0.877 | 0.769 | 0.893 | yes |
| 0 | 1 | 0.1630 | 2.938 | 0.291 | 0.893 | 0.780 | 0.910 | yes |
| 0.25 | 0.6 | 0.1389 | 3.271 | 0.254 | 0.494 | 0.422 | 0.505 | no |
| 0.25 | 0.7 | 0.1428 | 3.274 | 0.251 | 0.514 | 0.444 | 0.524 | no |
| 0.25 | 0.8 | 0.1485 | 3.259 | 0.278 | 0.539 | 0.460 | 0.550 | yes |
| 0.25 | 0.9 | 0.1556 | 3.143 | 0.298 | 0.585 | 0.475 | 0.601 | yes |
| 0.25 | 1 | 0.1634 | 2.932 | 0.283 | 0.659 | 0.587 | 0.669 | yes |
| 0.5 | 0.6 | 0.1285 | 3.293 | 0.220 | 0.380 | 0.310 | 0.390 | no |
| 0.5 | 0.7 | 0.1337 | 3.300 | 0.235 | 0.393 | 0.331 | 0.402 | no |
| 0.5 | 0.8 | 0.1427 | 3.285 | 0.247 | 0.410 | 0.354 | 0.418 | no |
| 0.5 | 0.9 | 0.1504 | 3.163 | 0.275 | 0.439 | 0.379 | 0.447 | yes |
| 0.5 | 1 | 0.1590 | 2.930 | 0.258 | 0.514 | 0.443 | 0.525 | yes |
| 1 | 0.6 | 0.1191 | 3.313 | 0.168 | 0.262 | 0.205 | 0.270 | no |
| 1 | 0.7 | 0.1233 | 3.312 | 0.178 | 0.267 | 0.220 | 0.274 | no |
| 1 | 0.8 | 0.1256 | 3.313 | 0.193 | 0.283 | 0.233 | 0.290 | no |
| 1 | 0.9 | 0.1394 | 3.236 | 0.215 | 0.303 | 0.268 | 0.308 | no |
| 1 | 1 | 0.1480 | 2.933 | 0.218 | 0.357 | 0.294 | 0.366 | yes |
| 2 | 0.6 | 0.1027 | 3.371 | 0.107 | 0.177 | 0.139 | 0.182 | no |
| 2 | 0.7 | 0.1089 | 3.374 | 0.124 | 0.182 | 0.129 | 0.190 | no |
| 2 | 0.8 | 0.1097 | 3.371 | 0.128 | 0.189 | 0.141 | 0.195 | no |
| 2 | 0.9 | 0.1162 | 3.332 | 0.135 | 0.192 | 0.166 | 0.195 | no |
| 2 | 1 | 0.1344 | 2.975 | 0.153 | 0.229 | 0.174 | 0.237 | no |
| 4 | 0.6 | 0.0825 | 3.441 | 0.071 | 0.145 | 0.119 | 0.149 | no |
| 4 | 0.7 | 0.0831 | 3.440 | 0.073 | 0.144 | 0.121 | 0.148 | no |
| 4 | 0.8 | 0.0867 | 3.438 | 0.077 | 0.139 | 0.114 | 0.142 | no |
| 4 | 0.9 | 0.0925 | 3.417 | 0.085 | 0.138 | 0.105 | 0.143 | no |
| 4 | 1 | 0.1126 | 3.052 | 0.085 | 0.155 | 0.115 | 0.161 | no |

**Winner: gamma = 1, lambda = 1.** V-NDCG@10
0.1480 (+0.0017 against shipped), flip-overlap
0.357 (cold-start 0.294, with history
0.366) against 0.847. Tie band (one SE of the mean
flip-overlap at the minimum) 0.020; 1 configuration inside it.

Reported, as the pre-registration requires:

* The winner sits **on the entropy constraint**: V-entropy 2.933 against a floor of
  2.922. lambda = 1 is the MMR penalty switched off (the top-50 re-ranked by utility alone).
* **Validation long-tail precision falls** from 0.302 to 0.218
  and the long-tail share of the served lists rises from 0.137 to
  0.196. Long-tail precision was the tie-break (no tie occurred: one configuration was inside the tie band), not a
  constraint. The validation labels are the popularity-biased training log, so this number is used only to
  compare configurations; the holdout (uniform-random exposure) is reported below and points the other way.
  The long-tail cutoff is the scorecard one (`eval.yaml`, bottom 50%); an earlier run of this script used the
  candidate channel's 0.40 floor by mistake, which changed only these two reported columns, not the selection.
* Among the feasible configurations gamma = 0 is never the minimiser; the selection moved the flip-overlap from
  0.847 to 0.357 at a validation NDCG cost of
  +0.0017 (a gain, inside the pre-registered tolerance either way).

Applied as selected: `configs/scoring.yaml` `utility.gamma: 1.0`, `diversity.lambda_default: 1.0`.

## L3a — Gate-B amendment (written and committed BEFORE any L3b run)

The chance-lift gate (+0.35 absolute, overall and long-tail) is one of the three a-priori thresholds already
disclosed as miscalibrated (`docs/TECHNICAL.md` section 10.2), and it is now a blocking rule that decides
between designs: it made the pre-registered K rule infeasible (DR13), and any union design fails it because a
larger candidate set raises the chance baseline the lift is measured against. A blocking gate should encode a
requirement, not a guess. Amendment, fixed here so the L3b decision cannot depend on its outcome:

| Row | Before | After | Reason |
|---|---|---|---|
| Long-tail candidate recall | blocking, >= 0.75 | **blocking, >= 0.75** | the brief section 9 requirement (do not eliminate relevant long-tail POIs) |
| Effective candidates per trip | not gated | **blocking, <= 300** | the serving budget the lift gate was proxying: the ranker scores a few hundred candidates (`TECHNICAL.md` section 11) |
| Overall candidate recall | blocking, >= 0.85 | reporting (reference 0.85) | a-priori target disclosed as miscalibrated |
| Recall lift over chance, overall | blocking, >= +0.35 | reporting (reference +0.35) | same |
| Recall lift over chance, long-tail | blocking, >= +0.35 | reporting (reference +0.35) | same |

Chance-lift and overall recall are still computed and shown for every design, beside their old reference
thresholds. `gate-representation` and `make reproduce` follow the amended rows
(`src/poi_rank/eval/gate_representation.py`); the Decision Register rows written under the four-row gate
(DR11-DR13) are historical and say so.

## L3b — retrieval design, decided on validation (pre-registered)

DR12's 0.1919 vs 0.1814 is a **holdout** comparison and cannot drive a switch; it was never run on validation.
The comparison below is run on the train-carved validation trips only.

Designs (same catalog, same features, same ranker recipe; only the candidate set differs):

* **(A)** the shipped set: learned retriever at K = 240 + long-tail floor + interest channel
  (`data/synthetic/candidates.parquet`; the retriever scores for train trips are cross-fitted by construction).
* **(B)** the legacy six-channel union (`candidates_legacy6.parquet`).
* **(C)** A union B, deduplicated per trip.

For each design the ranker is **retrained on that design's own train candidates** (system-8 recipe: LambdaRank,
IPS weights, behavioural dropout, early stopping on the train-carved validation trips, `configs/model.yaml`) so a
design is never penalised for a train/serve candidate mismatch; the retrieval K is not reopened. Four seeds
(42, 7, 11, 13, the protocol of experiment H); the reported value is the seed mean.

Measured on the validation trips (identical for every design):

* **V-NDCG@10, fixed denominator** = IPS-weighted NDCG@10 of the raw ranker top-10, whose ideal DCG is computed
  over **every** logged (exposed) label of the trip, not just the design's candidates, so retrieval quality is
  not normalised away (`eval/decomposition.py::end_to_end_ndcg10`, with the IPS gain weights of
  `eval/ranker_sweep.py`).
* **V-long-tail recall** and **V-overall recall** = per-trip IPS-weighted recall of exposed positives (label >= 1)
  by the candidate set, mean over trips with a positive (the validation-recall protocol of the E4 sweep);
  long-tail = popularity percentile below the project cutoff.
* **Chance-lift** = recall minus the share of the stratum's destination POIs the set contains (reporting).
* **Effective K** = mean distinct candidates per validation trip.

Rule: choose the design that **maximises V-NDCG@10 subject to V-long-tail recall >= 0.75 and effective K <= 300**.
**Adoption bar:** switch away from (A) only if the winner beats (A) by **>= +0.010 V-NDCG@10**, the same bar used
throughout this project (experiment H, the E2 sweep). If nothing clears it, (A) stays and DR12 stands as written.
If (A) itself fails a constraint, the best feasible design is adopted without the bar. All three designs are
reported on validation (V-NDCG@10, overall and long-tail recall, chance-lift, effective K). The holdout is read
once, for the adopted design only, in the final pipeline run. Time-box: 3 hours including retraining; if
undecided, (A) stays.

## Result of L3b (validation only; the holdout is read once, for the adopted design, in the final pipeline run)

Run: `scripts/l3b_designs.py` -> `results/parts/l3b_designs.json` (274 validation trips;
ranker retrained per design, seeds 42/7/11/13, seed mean).

| Design | V-NDCG@10 (fixed denominator, IPS) | overall recall | long-tail recall | overall lift | long-tail lift | effective K | feasible |
|---|---|---|---|---|---|---|---|
| A learned K=240 (shipped) | 0.1851 (sd 0.0012) | 0.918 | 0.908 | +0.365 | +0.401 | 267 | yes |
| B legacy six-channel | 0.1745 (sd 0.0021) | 0.601 | 0.592 | +0.171 | +0.177 | 207 | no |
| C = A union B | 0.1845 (sd 0.0015) | 0.942 | 0.940 | +0.297 | +0.344 | 311 | no |

**Decision: design (A) stays.** No feasible design beats the incumbent by the adoption bar. Against (A): (B) -0.0106 V-NDCG@10
and it fails the long-tail recall requirement (0.592 < 0.75); (C) -0.0007 and it
exceeds the serving budget (311 > 300 effective candidates per trip). DR12 stands as
written.

Reported, because it changes how DR12 should be read: on validation, with the ranker retrained per design, the
legacy six-channel union is **not** ahead of the learned retriever (B is 0.0106 behind on the seed mean, and the four
seeds of the two designs do not overlap; 274 validation trips, so trip-sampling noise is not captured by the seed spread). DR12's holdout comparison
(legacy 0.1919 against learned 0.1814) pointed the other way. The two comparisons differ in the population
(exposed, popularity-biased training log against the uniform-random holdout), in the NDCG denominator, and in
the ranker that scores the learned set (the shipped booster against a retrained one), so this does not overturn DR12's measurement, but it does mean the holdout
advantage of the legacy union is not a stable property of the design. The chance-lift gate was not consulted:
the amended Gate-B rows are the constraints above.

## Holdout, read once (after L1, L2 and L3 were frozen)

`scripts/l_holdout_report.py` -> `results/parts/l_holdout_report.json`, plus the full `reproduce-full` pipeline run
(671 unbiased-exposure holdout trips). An earlier run of the report script used the candidate channel's 0.40
long-tail cutoff instead of the scorecard's 0.5 for its long-tail columns; the definition was corrected and the script re-run (same
trips, same frozen configuration; no selection depended on it).

| | shipped (gamma 0, lambda 0.8) | selected (gamma 1, lambda 1) |
|---|---|---|
| Served top-10 flip-overlap, all trips | 0.837 | 0.353 |
| ... cold-start (72 trips) | 0.697 | 0.293 |
| ... with history | 0.854 | 0.360 |
| Gradient reproduction by the served utility, cold-start | 82% | 1345% |
| Gradient reproduction by the served utility, with history | -5% | 1056% |
| Served NDCG@10 (project metric, 671 trips) | 0.1345 | 0.1636 |
| Category entropy@10 (bits) | 3.257 | 2.940 |
| Long-tail precision / share (scorecard definition) | 0.196 / 0.144 | 0.266 / 0.220 |

The ranker score alone (K1's basis, every candidate row) reproduces 83% of the outcome gradient for cold-start trips and
-1% for trips with history: the same figures as `results/parts/touristiness_axis.json`. The served utility at gamma = 1
reproduces 1056% / 1345% (with history / cold-start): the factor **overshoots** the outcome gradient about
11x, because the pre-registered rule minimised flip-overlap under two constraints instead of matching the slope. On validation the long-tail
precision fell under the selection, on the holdout it rose: the validation labels are the popularity-biased training log.

Scenario artifacts (`results/scenarios/`): scenario 4 (touristiness flipped) against scenario 1 top-10 overlap 0.538 -> 0.000 (target at most 0.35);
local-experience against history 0.111 -> 0.000. Gate-B (amended): long-tail recall 0.898 (>= 0.75), effective candidates per trip
266 (<= 300): both pass. Scorecard: no MET row degraded; the scenario-4 row moved MISSED -> MET (six MISSED rows remain); the confidence-decile
Spearman (a MISSED row) fell 0.358 -> 0.236.
