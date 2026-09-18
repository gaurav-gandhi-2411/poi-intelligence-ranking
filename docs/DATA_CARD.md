# Data Card — `poi-intelligence-ranking` synthetic dataset

This document describes the data-generating process (DGP) in `src/poi_rank/datagen/`,
the deliberate dirtiness injected into the catalog, and every point at which the spec
as originally written was ambiguous or self-contradictory and had to be resolved.
Grows across build phases; this entry covers Phase 1 (datagen) only.

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
├── pois.parquet                        # exported (dirty) catalog
├── travelers.parquet
├── trips.parquet
├── interactions_train.parquet          # biased policy, train window
├── interactions_holdout_random.parquet # uniform-random policy, holdout window (PRIMARY eval set)
├── interactions_holdout_logged.parquet # biased policy, holdout window (secondary, bias-gap)
└── _oracle/                            # read ONLY by eval/oracle.py (not built yet)
    ├── traveler_taste.parquet          # true latent taste vectors
    ├── poi_latent.parquet              # true latent_quality, latent_localness, poi_semantic
    └── holdout_utility_true.parquet    # noise-free u(t,p) for every (holdout trip, eligible POI) pair
```

`opening_hours` in `pois.parquet` is a JSON-encoded string (or null) rather than a
nested parquet struct column — sidesteps struct-nullability edge cases in
`pyarrow`/pandas round-tripping while still representing "field entirely missing" as
an actual null, not a sentinel.
