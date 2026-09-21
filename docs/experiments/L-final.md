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
| 0 | 0.6 | 0.1402 | 3.281 | 0.305 | 0.796 | 0.677 | 0.813 | no |
| 0 | 0.7 | 0.1413 | 3.283 | 0.302 | 0.825 | 0.695 | 0.844 | no |
| 0 | 0.8 | 0.1463 | 3.247 | 0.317 | 0.847 | 0.719 | 0.866 | yes |
| 0 | 0.9 | 0.1556 | 3.110 | 0.314 | 0.877 | 0.769 | 0.893 | yes |
| 0 | 1 | 0.1630 | 2.938 | 0.309 | 0.893 | 0.780 | 0.910 | yes |
| 0.25 | 0.6 | 0.1389 | 3.271 | 0.275 | 0.494 | 0.422 | 0.505 | no |
| 0.25 | 0.7 | 0.1428 | 3.274 | 0.268 | 0.514 | 0.444 | 0.524 | no |
| 0.25 | 0.8 | 0.1485 | 3.259 | 0.288 | 0.539 | 0.460 | 0.550 | yes |
| 0.25 | 0.9 | 0.1556 | 3.143 | 0.315 | 0.585 | 0.475 | 0.601 | yes |
| 0.25 | 1 | 0.1634 | 2.932 | 0.315 | 0.659 | 0.587 | 0.669 | yes |
| 0.5 | 0.6 | 0.1285 | 3.293 | 0.245 | 0.380 | 0.310 | 0.390 | no |
| 0.5 | 0.7 | 0.1337 | 3.300 | 0.256 | 0.393 | 0.331 | 0.402 | no |
| 0.5 | 0.8 | 0.1427 | 3.285 | 0.266 | 0.410 | 0.354 | 0.418 | no |
| 0.5 | 0.9 | 0.1504 | 3.163 | 0.299 | 0.439 | 0.379 | 0.447 | yes |
| 0.5 | 1 | 0.1590 | 2.930 | 0.287 | 0.514 | 0.443 | 0.525 | yes |
| 1 | 0.6 | 0.1191 | 3.313 | 0.187 | 0.262 | 0.205 | 0.270 | no |
| 1 | 0.7 | 0.1233 | 3.312 | 0.195 | 0.267 | 0.220 | 0.274 | no |
| 1 | 0.8 | 0.1256 | 3.313 | 0.207 | 0.283 | 0.233 | 0.290 | no |
| 1 | 0.9 | 0.1394 | 3.236 | 0.233 | 0.303 | 0.268 | 0.308 | no |
| 1 | 1 | 0.1480 | 2.933 | 0.246 | 0.357 | 0.294 | 0.366 | yes |
| 2 | 0.6 | 0.1027 | 3.371 | 0.118 | 0.177 | 0.139 | 0.182 | no |
| 2 | 0.7 | 0.1089 | 3.374 | 0.137 | 0.182 | 0.129 | 0.190 | no |
| 2 | 0.8 | 0.1097 | 3.371 | 0.141 | 0.189 | 0.141 | 0.195 | no |
| 2 | 0.9 | 0.1162 | 3.332 | 0.146 | 0.192 | 0.166 | 0.195 | no |
| 2 | 1 | 0.1344 | 2.975 | 0.172 | 0.229 | 0.174 | 0.237 | no |
| 4 | 0.6 | 0.0825 | 3.441 | 0.079 | 0.145 | 0.119 | 0.149 | no |
| 4 | 0.7 | 0.0831 | 3.440 | 0.080 | 0.144 | 0.121 | 0.148 | no |
| 4 | 0.8 | 0.0867 | 3.438 | 0.084 | 0.139 | 0.114 | 0.142 | no |
| 4 | 0.9 | 0.0925 | 3.417 | 0.091 | 0.138 | 0.105 | 0.143 | no |
| 4 | 1 | 0.1126 | 3.052 | 0.092 | 0.155 | 0.115 | 0.161 | no |

**Winner: gamma = 1, lambda = 1.** V-NDCG@10
0.1480 (+0.0017 against shipped), flip-overlap
0.357 (cold-start 0.294, with history
0.366) against 0.847. Tie band (one SE of the mean
flip-overlap at the minimum) 0.020; 1 configuration inside it.

Reported, as the pre-registration requires:

* The winner sits **on the entropy constraint**: V-entropy 2.933 against a floor of
  2.922. lambda = 1 is the MMR penalty switched off (the top-50 re-ranked by utility alone).
* **Validation long-tail precision falls** from 0.317 to 0.246
  and the long-tail share of the served lists rises from 0.126 to
  0.162. Long-tail precision was the tie-break, not a constraint; it is the price of steering by
  each traveler's stated preference (a traveler who states a preference for famous landmarks is now served fewer
  long-tail POIs). Validation labels are the popularity-biased training log, so this number is used only to
  compare configurations.
* Among the feasible configurations gamma = 0 is never the minimiser; the selection moved the flip-overlap from
  0.847 to 0.357 at a validation NDCG cost of
  +0.0017 (a gain, inside the pre-registered tolerance either way).

Applied as selected: `configs/scoring.yaml` `utility.gamma: 1.0`, `diversity.lambda_default: 1.0`.
