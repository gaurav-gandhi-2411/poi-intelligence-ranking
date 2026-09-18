# Data Card — `poi-intelligence-ranking` synthetic dataset

This document describes the data-generating process (DGP) in `src/poi_rank/datagen/`,
the deliberate dirtiness injected into the catalog, and every point at which the spec
as originally written was ambiguous or self-contradictory and had to be resolved.
Grows across build phases. Phase 1 (datagen) is covered first; Phase 2 (data prep,
`src/poi_rank/data/`) follows below.

## Resolved ambiguities

### 1. Sign of the localness term in `u(t,p)`

`spec.md` section 1.2, as literally written:

```
u(t,p) = ... + w_local * localness_p * touristiness_pref_t + ...
```

`touristiness_pref ∈ [-1, 1]` is defined as **negative = prefers local/non-touristy,
positive = prefers touristy**. Taken literally, the formula above rewards a
local-preferring traveler (negative `touristiness_pref`) for visiting a **touristy**
(low-localness) POI whenever the sign arithmetic works out that way — backwards from
the intended semantics.

**Resolution implemented in `src/poi_rank/datagen/utility.py`:**

```
u(t,p) = ... + w_local * localness_p * (-touristiness_pref_t) + ...
```

so a local-preferring traveler (`touristiness_pref < 0`) gets **positive** utility
contribution from a **high-localness** POI. Verified by
`tests/test_datagen_schema.py::TestUtilitySignConvention`.

### 2. `party_fit`/`price_fit`/`novelty_t` are latent, not the same code as `scoring/`'s
observable compatibility functions

A later `scoring/` phase computes an *observable* `party_fit`/`budget_fit` from
visible fields only, for a completely different purpose (context-compatibility
scoring shown to users). The DGP's `party_fit`/`price_fit`/`novelty_t` in
`datagen/utility.py` share names with those (spec.md explicitly uses the same names in
its two different formula blocks) but are deliberately separate functions in a
separate, firewalled module — never shared code, never imported across the boundary.

### 3. `poi_semantic` and traveler `taste_t` share one latent space

spec.md leaves the choice of `poi_semantic`'s latent space open ("this can be the same
embedding space the text fields are lexically consistent with, or a separate latent
vector that loosely correlates with category/tags"). Implemented choice: both live in
one `TASTE_DIM = len(CATEGORIES) + len(TAGS) = 32`-dimensional space
(`src/poi_rank/datagen/taxonomy.py`) — the first 12 dims are a category axis (used
directly by the `w_cat * taste_t[category_p]` term), the remaining 20 are a tag-affinity
axis. This keeps the `w_taste * cos(taste_t, poi_semantic_p)` and
`w_cat * taste_t[category_p]` terms internally consistent with each other rather than
sampling two unrelated latent spaces that could disagree about what a traveler likes.

### 4. Missingness in the exported catalog represents an incomplete *listing*, not an
absent real-world attribute

`price_level`, `expected_duration_min`, and `opening_hours` are generated with true
values first; a configured fraction of rows then have the **exported** field nulled
(not a sentinel value — an actual parquet null). The DGP's true utility computation
(`datagen/utility.py`, called from `datagen/pipeline.py`) always uses the true,
pre-nulling values — exactly the same pattern as `rating`/`review_count` being noisy
*observations* of `latent_quality`, never overwritten with the nulled/noisy export
value. This is a deliberate, documented modeling choice: a POI's price doesn't stop
existing because a listing forgot to record it.

### 5. `interactions_holdout_logged.parquet` size target

spec.md's scale table (section 2.1) gives explicit targets for POIs, travelers, trips,
train impressions, and unbiased-holdout impressions, but not for the secondary
biased-holdout log. Sized at ~half the random-holdout session density per trip
(`target_holdout_logged_impressions: 15000` in `configs/datagen.yaml`, vs 30,000 for
the random holdout over the same ~160-200 holdout trips) — enough to compute a
meaningful bias-gap comparison in a later eval phase without doubling the impression
volume of the primary unbiased set.

### 6. Oracle export scope

`data/synthetic/_oracle/holdout_utility_true.parquet` exports noise-free `u(t,p)` for
every (holdout trip, eligible POI) pair — not the full trip x catalog cross product.
This is exactly the population a future `eval/oracle.py` needs ("ranks the unbiased
holdout by true u(t,p)") and keeps the oracle export a few hundred KB instead of
multiple MB, well inside the `data/synthetic/` < 25 MB budget.

### 7. Oracle isolation test's write-path exemption

`tests/test_oracle_isolation.py` enforces that no file outside `eval/oracle.py` (the
future reader) references the `_oracle` subdirectory — but spec.md section 8 also
requires *something* in `datagen/` to write there. Resolving this: the `_oracle`
subdirectory name is confined to exactly one additional module,
`datagen/oracle_export.py` (`ORACLE_SUBDIR_NAME` / `oracle_dir_from_output`); every
other caller (`datagen/pipeline.py` included) receives an already-resolved `Path` and
never spells out the subdirectory name itself. The isolation test allowlists exactly
these two files (one reader, one writer) and nothing else.

## DGP structure

- **Traveler taste**: 8 archetypes (`datagen/archetypes.py`), each a prototype vector
  in the 32-dim taste space plus typical touristiness/budget/party-type distributions.
  Each traveler draws a Dirichlet(α=0.7) mixture weight over the 8 archetypes, blends
  the prototype vectors, adds per-traveler Gaussian noise (σ=0.15), and L2-normalizes
  — archetypes are a generative prior, never a stored hard label.
- **Stated interests**: a genuinely lossy projection of the latent taste vector — top-k
  labels (categories ∪ tags, k=6) independently omitted w.p. 20%, independently
  replaced with an unrelated label w.p. 10%. This is the mechanism that keeps
  explicit-only features from reaching the oracle ceiling.
- **POI catalog**: 12 canonical categories, category-conditioned generative priors for
  every field (popularity, price, duration, hours, accessibility, crowd curve). Near-
  duplicates are generated as full independent catalog rows (own `poi_id`, jittered
  <50m, altered name, reduced review_count) cloned from a random base POI, not a
  display-only artifact.
- **Exposure**: popularity-biased slates use `popularity_raw^1.5 * geo_proximity`
  weighted sampling without replacement (`np.random.Generator.choice`); `p_expose` is
  the standard PPS-sampling approximation `min(1, k*w_i/Σw)`. Uniform-random slates use
  exact `p_expose = k/n` (every eligible POI equally likely). "Eligible" = same
  destination and `created_at <= trip.start_date`.
- **Choice simulation**: Plackett-Luce draws (not argmax) select which slate POIs get
  engaged (mean `engage_lambda=3.0` per slate) vs dismissed (mean `dismiss_lambda=0.5`,
  weighted toward low utility) vs left as a bare `view` (the default, label 0). Engaged
  POIs' interaction type severity decays with PL draw rank (`rank_decay=2.0`) — earlier
  draws skew toward `booking`/`visit`, later draws toward `click`.
- **Novelty**: computed once per trip from the traveler's engagement history in
  *earlier* trips only (never later trips, never the trip's own in-progress sessions).
  First trip for any traveler: novelty ≈ 1.0 for everything.
- **Temporal split**: one global timeline (`configs/datagen.yaml` `timeline.start_date`
  / `end_date`), split at `train_fraction=0.80` of the span. A trip is a train trip iff
  `start_date < split`; new POIs are guaranteed `created_at >= split`, so they
  mechanically get zero train interactions.

## Dirtiness catalog (all rates configured in `configs/datagen.yaml` `dirtiness:`)

| Dirtiness | Rate | Mechanism |
|---|---|---|
| Near-duplicate POIs | 4% | full cloned catalog row, name variant, <50m jitter |
| Missing `price_level` | 8% | true value generated, export field nulled |
| Missing `expected_duration_min` | 5% | true value generated, export field nulled |
| Missing `opening_hours` | 12% | export field set to `None` (JSON-encoded on non-null rows) |
| Inconsistent `category` string | 6% | export replaced with a messy variant (e.g. `"museums"`, `"Museum / Gallery"`); canonical form used internally throughout |
| Sparse POIs (`review_count < 10`) | ~15% | forced on a sampled subset; natural long-tail also contributes |
| New POIs (`created_at` inside holdout window) | ~5% | zero train interactions by construction |

## File layout

```
data/synthetic/
├── destinations.parquet
├── pois.parquet                        # exported (dirty) catalog -- datagen output, Phase 1
├── pois_prepared.parquet               # cleaned/enriched catalog -- data prep output, Phase 2
├── travelers.parquet
├── trips.parquet
├── interactions_train.parquet          # biased policy, train window
├── interactions_holdout_random.parquet # uniform-random policy, holdout window (PRIMARY eval set)
├── interactions_holdout_logged.parquet # biased policy, holdout window (secondary, bias-gap)
└── _oracle/                            # read ONLY by eval/oracle.py
    ├── traveler_taste.parquet          # true latent taste vectors
    ├── poi_latent.parquet              # true latent_quality, latent_localness, poi_semantic
    └── holdout_utility_true.parquet    # noise-free u(t,p) for every (holdout trip, eligible POI) pair
```

`opening_hours` in `pois.parquet` is a JSON-encoded string (or null) rather than a
nested parquet struct column — sidesteps struct-nullability edge cases in
`pyarrow`/pandas round-tripping while still representing "field entirely missing" as
an actual null, not a sentinel.

---

## Phase 2 — data prep (`src/poi_rank/data/`)

Reads `data/synthetic/pois.parquet` (the dirty, exported catalog only — never
`_oracle/`) and writes `data/synthetic/pois_prepared.parquet`: dedup → category
canonicalization → rating shrinkage → popularity percentile → localness index →
opening-hours mask → numeric imputation → geo (H3 cell + synthetic transit graph).
Config in `configs/features.yaml`. CLI: `python -m poi_rank.cli prepare` /
`make prepare`.

### Resolved ambiguities / design choices

#### 8. `distance-to-stay`'s source field lives on `trips.parquet`, not `travelers.parquet`

The task description for this phase says "distance-to-stay (per trip, from
travelers.parquet's stay_lat/stay_lon)". Checked directly against
`datagen/travelers.py::generate_trips`: `stay_lat`/`stay_lon` are trip-level fields
(a traveler's stay location differs per trip, e.g. their two trips can be to two
different destinations), written into `trips.parquet`, not `travelers.parquet`.
`data/geo_prep.py::distance_to_point_km` is documented to read from `trips.parquet`.
Also: distance-to-stay is inherently trip-conditional (it depends on *which* trip's
stay point a candidate POI is being compared against), so it is **not** materialized
as a static column on `pois_prepared.parquet` — it stays a plain callable helper,
applied per (trip, candidate POI) pair at the later candidate-generation/feature
stage, which is the earliest point a "trip" is actually in scope.

#### 9. Haversine math duplicated in `data/geo_prep.py`, not imported from `datagen/geo.py`

`datagen/geo.py::haversine_km` is pure trig with no downstream dependencies, so
importing it from `data/` would not technically violate the DGP firewall (firewall is
one-directional: `datagen/**` must never import downstream, but nothing stops
downstream importing `datagen/`). Chose to duplicate the ~10 lines instead, to keep
`data/`'s module boundary unambiguous rather than relying on a reader correctly
reasoning through the firewall's directionality every time this file is touched.

#### 10. Synthetic transit-node graph: origin and placement

`datagen/` does not generate a transit network, and spec.md section 4 requires
"distance-to-nearest-transit-node (synthetic transit graph per destination)" as a
prep-stage artifact without specifying how the nodes should be placed. Implemented in
`data/geo_prep.py::synthesize_transit_nodes`: `n` nodes per destination (config:
`geo.transit_nodes_per_destination`, default 15) scattered with Gaussian noise
(config: `geo.transit_node_spread_deg`) around that destination's *actual* POI
centroid (computed from the data at hand, not a hardcoded city-center constant) — an
observable-infrastructure-metadata choice, not a latent variable, so synthesizing it
at the prep stage (rather than in `datagen/`) is legitimate. Seeded from
`configs/features.yaml`'s `seed` via an independent `np.random.Generator` stream —
never reuses or re-derives datagen's own seeded stream.

#### 11. Category canonicalization mapping table is hand-authored, not imported from `datagen/`

`data/categories.py`'s ~36-entry dirty-string → canonical-category lookup table was
built by directly inspecting `datagen/taxonomy.py::CATEGORY_STRING_VARIANTS`, but is
re-typed as its own standalone table rather than importing that dict. A real
production category-cleaning pipeline would not have access to its training-data
generator's internals; duplicating this small table keeps `data/` self-contained and
realistic. Consequence, measured: the mapping is currently *exhaustive* for every
string Phase 1 actually generates, so the "other" bucket is 0 rows / 0.0% on the
committed dataset (`n_other_category=0` in the `make prepare` summary) — reported
honestly rather than assumed nonzero, per spec.md's requirement to log (not assume) the
other-rate.

#### 12. Median-by-(destination, category) imputation, spec.md's two-part sentence

spec.md section 4 pairs "median-by-(destination,category) [imputation]" with
"explicit `_was_missing` indicator columns... LightGBM handles NaN natively, but
indicators are kept for explainability" — a sentence that, read literally, asks for
both an actual imputed value *and* for the raw NaN to still reach a later model
untouched. Resolved in `data/impute.py`: the original column (real NaNs intact) is
never modified; for each imputed field, two *new* columns are added —
`<field>_imputed` (median-by-(destination, category) filled value, falling back to
destination-median then global-median for any empty group) and
`<field>_was_missing` (boolean indicator). Three columns per field, so neither half
of spec.md's sentence is silently dropped: a later model phase can consume the raw
column directly (NaN and all) for LightGBM's native handling, or the `_imputed`
column where an always-present number is needed (e.g. non-tree-based baselines,
explainability displays).

#### 13. Localness index weight reweighting — an honest miss against the ρ > 0.6 target

spec.md section 4 states the localness composite's weights literally:
`0.35·z(-pop_pct) + 0.30·z(-foreign_review_ratio) + 0.20·z(dist_to_tourist_centroid)
+ 0.15·z(local_tag_hits)`, validated by Spearman ρ against the DGP's true
`latent_localness` (target: ρ > 0.6, measured only in `eval/oracle.py`, never inside
`data/localness.py`).

**Measured, with the literal spec.md weights:** ρ ≈ 0.372 (`n=1446`). Below target.

Per-component diagnostic (each z-scored signal alone vs. `latent_localness`,
Spearman ρ, computed via a throwaway analysis script against `eval/oracle.py` — never
baked into `data/localness.py`'s runtime code path):

| Component | ρ alone |
|---|---|
| `z(-foreign_review_ratio)` | 0.50 |
| `z(-pop_pct)` | 0.19 |
| `z(local_tag_hits)` | 0.05 |
| `z(dist_to_tourist_centroid)` | 0.01 |

Root cause: `dist_to_tourist_centroid_km` carries almost no signal for this DGP
realization because POI `lat`/`lon` are generated as Gaussian jitter around each
destination's center **independent of `latent_localness`** (`datagen/catalog.py`
never uses `latent_localness` when sampling `lat`/`lon`) — geographic clustering
relative to the tourist centroid is a plausible real-world signal in general, but
this particular synthetic catalog doesn't encode it. `local_tag_hits` is weak because
tag sampling weights (`CATEGORY_TAG_AFFINITY`) are category-conditioned, not directly
tied to the localness dimension. `foreign_review_ratio` is the dominant carrier since
Phase 1's DGP formula gives it the most direct dependence on `latent_localness`.

This is exactly the "is `local_tag_hits` actually informative" diagnostic + "weight
tuning" iteration spec.md section 4 invites. **What was changed:** default weights in
`configs/features.yaml` moved to `(pop=0.20, foreign=0.60, geo=0.10, tag=0.10)` —
up-weighting the empirically strongest term, down-weighting (not zeroing — a
composite index should stay a composite) the two demonstrably weak ones. **What was
deliberately not done:** no coefficient was derived by regressing directly against
`latent_localness` (an in-sample OLS fit of `[z(-foreign_review_ratio),
z(-log1p(popularity_raw))]` against `latent_localness` reaches ρ ≈ 0.61 in this
throwaway analysis, clearing 0.6 — but adopting those exact fitted coefficients would
be calibrating the observable index directly against the oracle, the precise
circularity spec.md section 1 exists to prevent, even though `data/localness.py`
itself would still never *import* `_oracle/`). `compute_localness`'s own function
defaults remain the literal spec.md weights (0.35/0.30/0.20/0.15) for direct
spec-reference use in isolation; only the production `configs/features.yaml` pipeline
config carries the tuned weights.

**Result after honest reweighting:** ρ ≈ 0.4757 (`n=1446`, p≈1.6e-82) — still below
the 0.6 target. `tests/test_localness_oracle.py::test_localness_spearman_rho_against_oracle`
asserts the literal target and is **expected to fail** — the miss is surfaced, not
hidden, per spec.md's explicit instruction not to silently lower a missed bar.
