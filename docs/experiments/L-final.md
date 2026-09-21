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
