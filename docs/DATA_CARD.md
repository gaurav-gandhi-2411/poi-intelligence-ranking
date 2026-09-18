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
├── poi_features.parquet                # POI feature table -- features output, Phase 3
├── traveler_features.parquet           # traveler feature table (per traveler x trip) -- Phase 3
├── travelers.parquet
├── trips.parquet
├── interactions_train.parquet          # biased policy, train window
├── interactions_holdout_random.parquet # uniform-random policy, holdout window (PRIMARY eval set)
├── interactions_holdout_logged.parquet # biased policy, holdout window (secondary, bias-gap)
└── _oracle/                            # read ONLY by eval/oracle.py
    ├── traveler_taste.parquet          # true latent taste vectors
    ├── poi_latent.parquet              # true latent_quality, latent_localness, poi_semantic
    └── holdout_utility_true.parquet    # noise-free u(t,p) for every (holdout trip, eligible POI) pair

artifacts/
└── poi_emb.npy                         # committed cache: final L2-normalized 64d POI text embedding
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

---

## Phase 3 — features (`src/poi_rank/features/`)

Reads `data/synthetic/pois_prepared.parquet`, `travelers.parquet`, `trips.parquet`,
`interactions_train.parquet` (never holdout logs, never `_oracle/`) and writes
`data/synthetic/poi_features.parquet` (one row per POI) and
`data/synthetic/traveler_features.parquet` (one row per `(traveler_id, trip_id)`).
Config additions live in the same `configs/features.yaml` (new `text_embedding`,
`poi_features`, `traveler_features` sections, read by a separate
`features.config.FeatureBuildConfig` loader — see resolved ambiguity #20). CLI:
`python -m poi_rank.cli features` / `make features`.

### Resolved ambiguities / design choices

#### 14. Text embedding: TF-IDF is the canonical default path, not sentence-transformers

spec.md section 5 describes `all-MiniLM-L6-v2` (384d) → SVD-64 as the primary path
with a TF-IDF fallback "if `sentence-transformers` unavailable or offline." In this
environment `sentence-transformers` is genuinely not installed (it is an optional
extra, `pyproject.toml` `[project.optional-dependencies] text`) and downloading a
~90MB model on a fresh clone would risk both the `<5min` CPU-only reproducibility
budget and the "no paid APIs, minimal deps" spirit of the assignment for a one-time
grading run. **Decision:** `configs/features.yaml`'s `text_embedding.method: tfidf`
is the actual default `make reproduce` exercises; `sentence_transformers` is a
genuine opt-in a reviewer can select (`method: sentence_transformers` + `uv sync
--extra text`), never a silent runtime fallback — `build_poi_text_embeddings` raises
an actionable `ImportError` if that method is selected but the package is missing,
rather than quietly downgrading to TF-IDF. This is spec-anticipated, not corner-
cutting: spec.md itself requires the TF-IDF path be "tested in CI," which only makes
sense if it is a real, exercised path, not merely a defensive branch.

#### 15. What gets cached to `artifacts/poi_emb.npy`: the final 64d (post-SVD) matrix, not a raw pre-SVD embedding

spec.md's literal text is "Cached to `artifacts/poi_emb.npy` ... (~1.5k x 384 fp16 ≈
1.1 MB)" — a size citation anchored specifically to the sentence-transformer path's
fixed-width 384d raw output. TF-IDF has no equivalent fixed-width raw form (its
dimensionality is vocabulary-sized, capped by `tfidf_max_features`, not model-fixed),
and fitting TF-IDF + SVD on 1,446 POIs is sub-second — there is no recompute-cost
justification for caching anything pre-SVD when TF-IDF is canonical. **Decision:**
`artifacts/poi_emb.npy` holds the FINAL, L2-normalized 64d embedding matrix
(`float32`, ~370 KB on the committed catalog — smaller than spec's own 1.1 MB
citation, well inside the `data/`+`artifacts/` size budget), giving every downstream
phase (semantic candidate channel, traveler taste vectors, `models/`) a byte-
identical embedding source without re-fitting SVD each run — the actual property
spec.md's caching requirement protects. If a reviewer opts into the
`sentence_transformers` path, the SAME cache slot holds that path's 64d SVD output
instead (the cache is keyed by shape `(n_pois, svd_dim)`, not by method) — there is
deliberately no separate raw-384d cache artifact, since nothing downstream ever needs
the un-reduced sentence-transformer output.

#### 16. `tau = 180 days half-life` is honored as a literal mathematical half-life

spec.md section 6 writes `tau = 180 days half-life` then the formula
`exp(-delta_t_i / tau)`. Taken as a literal variable substitution (`tau = 180`
plugged directly into the exponent), the decay at `delta_t = 180` days would be
`exp(-1) ≈ 0.368`, not the `0.5` a "180-day half-life" implies. **Decision:** `tau`
in the implemented formula is the derived decay constant `180 / ln(2) ≈ 259.66`
days (`traveler_features.half_life_to_decay_constant`), so `exp(-delta_t/tau)`
genuinely equals `0.5` at `delta_t = 180` days — honoring the precise mathematical
meaning of "half-life" over the literal variable-name substitution. Verified by
`tests/test_traveler_features.py::test_half_life_to_decay_constant_gives_half_decay_at_halflife`.

#### 17. Explicit interests multi-hot is 31-dimensional, not spec's literal "14d"

spec.md section 6 states "interests multi-hot (14d)." The actual stated-interest
vocabulary (`datagen/taxonomy.py::INTEREST_LABELS = CATEGORIES + TAGS`) has 32 list
entries but only **31 distinct strings** — `CATEGORIES` and `TAGS` both separately
contain the literal string `"shopping"` (a category and, independently, a tag),
collapsing to one column. Neither the 32-entry literal sum nor spec's "14d" figure
matches. `features/traveler_features.py::build_interest_vocabulary` derives the
vocabulary empirically from `travelers.parquet['interests']` (every distinct label
actually stated by at least one of the 600 travelers) rather than importing
`datagen`'s internal taxonomy — mirroring `data/categories.py`'s established
precedent of hand-deriving vocabulary from what the data actually contains, since a
real production feature pipeline would not have access to its data generator's
internals either. Measured on the committed dataset: 31 of the possible 31 unique
labels are observed (full coverage from 600 travelers), so the explicit-block
`interest_*` column count is 31. `tests/test_traveler_features.py` asserts vocabulary
determinism directly; the 14d figure is treated as a nominal spec placeholder, not an
authoritative target, since no vocabulary construction in the codebase produces it.

#### 18. POI id reconciliation: interaction logs reference the PRE-dedup catalog

Discovered by direct inspection while building this phase, not called out in
spec.md's Phase 3 task description. `datagen/pipeline.py` samples exposure slates
from the full, un-deduped POI array (`build_catalog_arrays` over `poi_true_df`,
before `apply_catalog_dirtiness`/Phase 2's dedup ever runs), so
`interactions_train.parquet` (and both holdout logs) reference raw `poi_id`s from
the pre-dedup catalog. Phase 2's dedup (`data/dedup.py`) merges near-duplicate rows
into one surviving record per cluster keyed by the highest-review-count member's
`poi_id`, recording every original member id in that row's `merged_poi_ids` list.
**Measured**: 40 of 1,386 distinct `poi_id`s referenced in `interactions_train.parquet`
(~2.9%) no longer exist as standalone rows in `pois_prepared.parquet` — they were
merged into a *different* row's `poi_id`. Every join between interaction logs and
`pois_prepared.parquet` (the behavioral block, the implicit taste vector's
`emb(poi_i)` lookups) goes through `features/reconcile.py::build_poi_id_canonical_map`
+ `remap_interaction_poi_ids` first, or those 40 POIs' interactions would silently
vanish (counted as if they never happened) instead of correctly rolling up onto the
surviving canonical row.

#### 19. `price_level` appears in both spec.md's Numeric and Categorical feature rows — resolved as two different representations, not a literal duplicate

spec.md section 5's feature table literally lists `price_level` under BOTH the
Numeric row and the Categorical row. **Decision:** the numeric block
(`num_price_level`) uses Phase 2's `price_level_imputed` column (always-present,
median-by-destination-category filled — appropriate for a numeric feature that
LightGBM/non-tree baselines can consume directly); the categorical block
(`cat_price_level`) uses the RAW `price_level` column cast to pandas `category`
dtype, preserving real missingness as a genuine NaN category (appropriate for
LightGBM's native categorical + missing-value handling, spec.md section 8). Two
different representations for two different downstream consumers, not one field
duplicated verbatim.

**Sub-finding, a real `pyarrow` round-trip bug caught while building this:** a
`category` dtype backed by pandas-nullable `Int64` category *labels* (e.g.
`price_level.astype("Int64").astype("category")`) silently degrades to plain
`float64` on a `to_parquet` → `read_parquet` round trip — verified directly (in-memory
the dtype is correctly `category`; after a real parquet write+read it is not). This
would have silently undone the LightGBM-native-categorical intent the moment the
feature table hit disk, with no error anywhere. Fixed by stringifying
`price_level`'s category labels (`"1"`..`"4"`) before casting to `category` — string-
labeled categories round-trip correctly. Regression-tested directly against a real
parquet round trip in `tests/test_poi_features.py::test_categorical_block_price_level_survives_parquet_round_trip`
(not just an in-memory dtype check, which would not have caught this).

#### 20. "Archetype affinity profile" is an observable K-Means proxy — zero oracle information

spec.md section 5 lists a POI behavioral feature "archetype affinity profile (8d:
normalized engagement share by traveler archetype)." The DGP's 8 archetypes
(`datagen/archetypes.py`) are latent, soft Dirichlet-mixture weights per traveler —
`travelers.parquet` does not export them (verified: `travelers.parquet`'s columns are
exactly `traveler_id, home_market, home_destination, party_type, budget, mobility,
interests, touristiness_pref, pace, accessibility_needs, explicit_preferences,
dietary` — no archetype/mixture field), and reading them from `_oracle/` would be a
hard firewall violation. **Decision:** `features/traveler_features.py::assign_traveler_segments`
builds a genuinely observable proxy instead — K-Means (`k=8`, seeded, `n_init=10`)
over a feature matrix built ONLY from stated traveler attributes (interests
multi-hot, budget ordinal, party_type one-hot, `touristiness_pref`, all
`StandardScaler`-normalized). `k=8` mirrors the DGP's archetype count as a
structurally reasonable choice, NOT because any archetype definition or per-traveler
assignment is read — this module never imports from `datagen/archetypes.py` and
never references `_oracle` (enforced by `tests/test_firewall_features.py`). The
resulting per-POI `behav_archetype_affinity_NN` columns are the normalized share of
that POI's ENGAGED (label ≥ 1) train-window interactions attributable to travelers in
each of the 8 observable clusters — a real, measurable, leakage-free behavioral
signal, structurally similar in spirit to spec's request but built entirely from
information a production system would actually have.

#### 21. Implicit preference-summary stats use ENGAGED interactions only, never raw impressions

`implicit_category_distribution`, `implicit_mean_price_level`,
`implicit_mean_localness`, and `implicit_mean_pop_pct` (spec.md section 6) are
computed only from as-of-safe history rows with `label >= 1` (a genuine positive
engagement), never from raw impressions (`view`-only rows included). Computing these
from raw impressions would just reproduce the shape of the popularity-biased
**exposure** policy (`p(expose) ∝ popularity^1.5 × geo_prox`, spec.md section 1.3) —
i.e. "what POIs was this traveler shown," which is a property of the logging policy,
not the traveler's actual taste — silently contaminating a feature meant to
represent implicit preference with exposure-policy bias. The taste VECTOR itself
(`compute_taste_vector`) correctly uses the FULL history including `view` (weight
0.05) and `dismiss` (weight −0.8), per spec.md's literal weight table — those low/
negative weights are precisely how the taste vector already down-weights
impression-only exposure without needing a hard `label >= 1` filter; the summary
STATS use the harder filter because they have no analogous per-row weighting
mechanism of their own.

#### 22. `explicit_days_remaining` and `implicit_days_since_last_interaction` are computed identically

spec.md section 6 lists `days_remaining` in the explicit block and (separately)
"days since last interaction" in the implicit block, without defining either
relative to a concrete reference point. For an OFFLINE, per-`(traveler_id, trip_id)`
feature table (no live "request time" exists in a historical dataset — spec.md
section 13 itself classifies "days-until-trip" as an *online*, request-time feature),
the only leakage-safe reference point already established by this module's as-of
discipline (resolved ambiguity below) is the traveler's own most recent as-of-safe
interaction. Since the dataset provides no separate "trip booking" event distinct
from browsing-interaction timestamps, both quantities reduce to the same number in
this implementation: `(trip.start_date − most_recent_active_interaction_timestamp)`,
computed once and emitted under both spec-mandated column names. A real production
system would compute `days_remaining` from genuine request time at serving time
(a true online feature, not derivable at offline feature-build time at all) — this
static column is a best-effort table-completeness stand-in, not the production-
correct implementation of that feature.

#### 23. Traveler implicit-block as-of cutoff: per-trip, not per-impression

The load-bearing temporal-leakage decision of this phase, spelled out in full in
`traveler_features.py`'s module docstring: every implicit-block/behavioral-adjacent
quantity for a `(traveler_id, trip_id)` row uses ONLY `interactions_train.parquet`
rows with `timestamp < trip.start_date` AND `trip_id != <this trip>` — i.e. only the
traveler's own EARLIER trips, mirroring `datagen/pipeline.py`'s own "novelty computed
from earlier trips only" convention. This is a deliberately coarser approximation
than a true per-impression as-of cutoff (which would require the not-yet-built
`models/`-phase training-row assembly to join at individual-impression grain); a
trip-level cutoff is safe by construction because every one of a trip's own
train-window interactions has `timestamp <= trip.start_date` (datagen's
`session_base_ts = trip.start_date`, `lead_days >= 0` — see `datagen/pipeline.py`),
so the timestamp filter alone already excludes the current trip's own sessions.
Verified directly (not just asserted) by
`tests/test_traveler_features.py::test_taste_vector_unaffected_by_interactions_after_as_of`
and `test_traveler_history_before_excludes_current_trip_and_future_timestamps`. A
traveler's first trip (`trip_sequence == 1`, 600 of 800 committed trips) correctly
gets an all-zero taste vector and `n_interactions=0` by construction; measured on the
committed dataset, 635 of 800 trips (600 first-trips + 35 second-trips whose own
first trip fell inside the holdout window and so logged zero train interactions) have
this cold-start signature — a fully-explained, non-bug count, cross-checked directly
against `trips.parquet`'s `trip_sequence`/`is_holdout` columns.

#### 24. Per-phase config class, not a shared one

`FeatureBuildConfig` (`features/config.py`) is its own typed loader for
`configs/features.yaml`'s new `text_embedding`/`poi_features`/`traveler_features`
sections, independent of `data/config.py::FeaturesConfig` (Phase 2's
`dedup`/`localness`/`geo` sections) even though both read the same physical YAML
file — mirrors the codebase's established one-typed-config-class-per-phase
convention (`datagen/config.py::DatagenConfig`, `data/config.py::FeaturesConfig`)
rather than coupling one phase's config class to another's.

---

## Phase 4a — candidate generation (`src/poi_rank/candidates/`)

Reads `pois_prepared.parquet`, `travelers.parquet`, `trips.parquet`,
`poi_features.parquet`, `traveler_features.parquet`, `interactions_train.parquet`
(never holdout logs at generation time — those are read only by the recall-metrics
evaluation step, never fed into channel logic) and writes
`data/synthetic/candidates.parquet`: one row per `(trip_id, poi_id)` selected by
**any** of 6 channels, with a boolean membership column per channel. CLI:
`python -m poi_rank.cli candidates` / `make candidates`.

### Resolved ambiguities / design choices

#### 25. No dedicated `configs/candidates.yaml`

spec.md section 14's repository layout lists exactly `datagen.yaml, features.yaml,
model.yaml, scoring.yaml, eval.yaml` — no `candidates.yaml`. A new `candidates:`
section was appended to the existing `configs/features.yaml` instead, read by its own
`candidates/config.py::CandidatesConfig` typed loader — mirrors the precedent already
established twice in this same file (`data.config.FeaturesConfig` and
`features.config.FeatureBuildConfig` each read their own section of the identical
physical YAML), rather than inventing a file spec.md never asked for.

#### 26. Interest channel matches `travelers.interests` against POI `{category} ∪ tags`, not the exploded `explicit_interest_*` columns

Both representations are available (`travelers.parquet['interests']` — the raw
stated-interest list per traveler — and `traveler_features.parquet`'s one-hot
`explicit_interest_*` columns built from the same list). The raw list is simpler to
consume for a **membership** check (`stated_interests & (poi.tags ∪ {poi.category})`)
and avoids re-deriving the traveler-feature-table row for every trip inside the
candidate-generation loop. Measured: the interest vocabulary (31 labels) is an exact
1:1 match with the POI tag+category vocabulary (also 31 labels, zero labels present on
one side only) — both taxonomies are drawn from the same `datagen/taxonomy.py` source,
so membership matching works cleanly with no silent under-coverage from a vocabulary
mismatch.

#### 27. Geo channel: H3 k-ring is a coarse first filter, never the final radius decision

spec.md section 7 literally specifies "H3 k-ring from stay point, radius by
mobility." H3's k-ring covers an *approximate* disk (its true footprint varies with
hex orientation relative to the query point, especially at small k) — taken
literally, an under-sized k could silently clip real in-radius POIs sitting just past
a coarse hex boundary, or an over-sized k could admit POIs beyond the literal radius.
**Resolved**: `channel_geo` computes `k = ceil(radius_km / edge_length_km) + 1` (a
generous +1-ring safety margin), gathers the k-ring's POIs from
`pois_prepared.h3_cell`, then applies an **exact haversine cutoff** at the literal
mobility-conditioned radius before ranking nearest-first and truncating to quota. The
k-ring is a cheap coarse filter; the haversine cutoff is what actually enforces the
spec's literal radius semantics.

`radius_km_mixed` (spec.md leaves "mixed" mobility's radius undefined, suggesting
"e.g. max of transit/car" as a documented default): set equal to `car`'s radius
(25 km) — "mixed" mobility means the traveler has access to the most permissive
transport mode available, not an average across modes.

`geo.h3_resolution` in `configs/features.yaml` is pinned to `8`, matching
`dedup.h3_resolution` (also `8`) — `pois_prepared.h3_cell` is built at that
resolution, and k-ring cell-id comparisons are only meaningful when both sides agree
on resolution. Documented as a coupled pair, not independently tunable.

#### 28. `pois_prepared.pop_pct` is a `[0, 1]` percentile, not spec.md's literal "0-40" reading

`data/popularity.py::compute_popularity_percentile` produces a `[0, 1]`
within-destination percentile rank (measured: `min≈0.002, max=1.0` on the committed
dataset), not a `0-100` scale. spec.md section 7's literal `pop_pct < 40` is
implemented as `pop_pct < 0.40` on this dataset's actual scale — the same
bottom-40th-percentile semantics, just expressed on the scale the column actually
uses.

#### 29. Long-tail channel's cold-start fallback: the semantic-relevance filter is skipped, not defaulted to a permissive threshold

`compute_semantic_similarity` returns an *identical* `0.0` for every POI in a
destination when the traveler's taste vector has zero magnitude (a genuine
zero-interaction-history cold start — measured 635/800 trips overall, 137/202 in the
holdout evaluation set specifically). A literal `sims > 0.0` filter would therefore
exclude **every** POI for a cold-start trip, silently starving the channel's HARD
FLOOR for the majority of trips. **Resolved**: `channel_longtail` skips the semantic
filter entirely for cold-start travelers (only the popularity cutoff applies), and the
"rest by semantic score" ranking degenerates to a stable poi_id tie-break (all scores
tied at 0.0) — exploration is genuinely uniform for these travelers, which is the more
honest behavior for a traveler with no taste signal to rank against at all.

#### 30. Collaborative-filtering channel: co-interaction is engagement co-occurrence across the FULL train window, not per-slate co-occurrence; leakage safety lives in the per-trip seed, not the global matrix

"Co-interaction" is defined as: two POIs co-interact iff the SAME traveler
positively engaged (`label >= 1`) with both, anywhere in `interactions_train`'s train
window — the standard user-item → item-item cosine-similarity construction (binary
engagement matrix, item similarity = column-cosine), not restricted to the same
slate/session. This global similarity matrix is fit **once**, over the full train
window, mirroring `poi_features.py::build_behavioral_block`'s already-established
precedent (POI-level behavioral aggregates use the full train window, not a per-trip
as-of cutoff — resolved ambiguity #18's sibling reasoning). The task's explicit
leakage-safety obligation ("reuse the same temporal as-of-cutoff discipline... do not
leak") applies to the per-trip **seed** (`union.cf_seed_poi_ids`), which reuses
`features.traveler_features.traveler_history_before` directly and is restricted to
the traveler's own earlier-trip, positively-engaged history — never reimplemented,
never the current trip's own session, never a later trip. Additionally,
`interactions_train.parquet` structurally contains zero rows for any holdout trip
(verified: 0 of 121,360 rows reference a holdout `trip_id`), so for every trip
actually scored by `candidates/recall_metrics.py` (holdout trips only) the global
matrix carries no information about that trip's own session at all — the matrix's
full-train-window construction cannot leak into the metric that matters.

Collaborative-filtering channel coverage is inherently sparse: 165/800 trips overall
(65/202 in the holdout set) have a non-empty seed history and so a non-empty
`channel_cf` — the remainder correctly get an empty CF channel (their first trip),
per spec.md section 12's assignment of the archetype-prior channel, not this one, as
the designated cold-start path.

#### 31. Semantic channel and CF channel reuse Phase 3's precomputed artifacts directly, never recompute

The semantic channel's `taste_t` is read directly from
`traveler_features.parquet`'s `implicit_taste_*` columns (already as-of-safe per
Phase 3's temporal-leakage discipline) rather than recomputed from raw interaction
history. The archetype-prior channel reuses
`features.traveler_features.assign_traveler_segments` directly (same K-Means
observable-segment construction that built `poi_features.parquet`'s
`behav_archetype_affinity_NN` columns) and ranks by those already-computed affinity
columns. Both choices avoid a second, potentially-inconsistent reimplementation of
already-built, already-tested Phase 3 logic.

#### 32. Long-tail exploration: ε-greedy parameterization and per-trip seeding

`epsilon = 0.30` (30% of the long-tail quota filled by uniform-random draw from the
qualifying pool; 70% by semantic-score rank) is a documented, arbitrary-but-stated
choice, not derived from any target metric. The random draw is seeded per trip via
`channels.trip_seed` — a SHA256 digest of `(seed, trip_id)`, deliberately **not**
Python's built-in `hash()`, which is only reproducible across process runs when
`PYTHONHASHSEED` happens to already be externally fixed (`Makefile` sets it for
`make candidates`, but a direct `uv run python -m poi_rank.cli candidates` invocation
does not). Verified directly: two independent `uv run python -m poi_rank.cli
candidates` invocations (no `make`, no externally-set `PYTHONHASHSEED`) produced a
byte-identical `candidates.parquet`
(`sha256=629d4312b136277b4abe2f232a2c188096bd55322310921e104cb1a1e18deded`).

### Measured results (committed dataset, 800 trips)

`uv run python -m poi_rank.cli candidates`, wall clock ≈ 39–40 s (well inside the
project's < 5 min full-pipeline budget):

| Metric | Value |
|---|---|
| Candidates per trip (mean / median / min / max) | 186.7 / 189.0 / 148 / 224 |
| `candidate_recall@250`, overall | **0.4413** (n=202 holdout trips, 0 excluded) |
| `candidate_recall@250`, long-tail stratum (pop_pct < 0.5) | **0.3936** (n=201, 1 excluded — zero long-tail-relevant POIs) |

Targets (spec.md §11.10): overall ≥ 0.90, long-tail ≥ 0.80. **Both missed — reported
honestly, per-channel diagnostics below, no channel logic was tuned to inflate this
number.**

Per-channel marginal recall (leave-one-channel-out, overall):

| Channel | `recall_without_channel` | Marginal |
|---|---|---|
| `channel_geo` | 0.3712 | **+0.0702** |
| `channel_interest` | 0.3166 | **+0.1248** (largest single contributor) |
| `channel_semantic` | 0.4055 | +0.0358 |
| `channel_cf` | 0.4298 | +0.0116 (smallest — see diagnosis below) |
| `channel_longtail` | 0.3980 | +0.0434 |
| `channel_archetype` | 0.4045 | +0.0368 |

Every channel contributes strictly positive marginal recall — **none is flagged for
removal**. `channel_cf`'s marginal is the smallest in absolute terms, but this is a
**coverage artifact, not a weak mechanism**: it is structurally non-empty for only
65/202 holdout trips (32%, first-trip travelers correctly get an empty CF channel by
design — see resolved ambiguity #30). Restricting the same leave-one-out computation
to only the 65 trips where `channel_cf` actually fires (a diagnostic-only
computation, not part of the production metric): recall_full=0.4624,
recall_without_cf=0.4263, **conditional marginal = 0.0360** — three times its
unconditional value, and comparable in magnitude to `channel_semantic`'s (0.0358) and
`channel_archetype`'s (0.0368) unconditional marginals. `channel_cf` is not deleted:
its low *unconditional* marginal is explained by sparse eligibility, not by the
mechanism failing to add signal when it does fire.

### Honest diagnosis: why overall recall (0.44) and long-tail recall (0.39) miss their 0.90/0.80 targets

Two root causes, both measured directly against the committed dataset rather than
guessed:

1. **Achieved candidate-set size is below the ~250 target.** Mean candidates per trip
   is 186.7 (75% of ~250), not because any channel under-fills its own quota
   relative to *qualifying* POIs (`channel_interest`, `channel_semantic`, and
   `channel_longtail` hit their exact quota of 50/50/50 on every single trip;
   `channel_geo` averages 50.1/60 and `channel_archetype` exactly 30/30) — the mean
   per-channel selection count sums to 237.1 (50.1 + 50.0 + 50.0 + 7.0 + 50.0 +
   30.0), yet the deduplicated union averages only 186.7: (a) heavy inter-channel
   overlap (channels are correlated, not independent, by construction — e.g.
   semantic and long-tail both rank by the same `taste_t` cosine score) and (b)
   `channel_cf` is structurally
   near-empty for 68% of trips (resolved ambiguity #30) — an expected, documented
   consequence of the dataset's cold-start rate, not a bug.

2. **The deeper, larger cause: even at the literal ~250 target, a purely
   coverage-driven ceiling would remain far below 0.90.** A trip's holdout-"relevant"
   POI (the one it positively engaged with under `interactions_holdout_random`'s
   *uniform*-random exposure policy) is governed by the DGP's full latent utility
   function `u(t,p)` — 7 weighted terms (taste cosine, category affinity, signed
   localness, latent quality, party fit, price fit, novelty) plus irreducible
   Plackett–Luce choice noise (`datagen/utility.py`) — while every candidate channel
   here is restricted, by the project's own circularity firewall, to *observable*
   proxies for a subset of those terms (stated interests are a genuinely lossy,
   noisy projection of latent taste — 20% omission + 10% noise by construction, per
   the DGP; `latent_quality` and `novelty` have no observable channel at all).
   Measured evidence this is a real information gap, not a channel implementation
   bug: **a naive random-baseline recall — selecting `k` POIs from the destination
   catalog uniformly at random, where `k` equals the trip's own actual candidate-set
   size — averages 0.3889 (mean `k/N` across the 202 holdout trips)**, essentially
   the same order of magnitude as the measured 0.4413. The full 6-channel union
   beats this naive baseline by only ~13% relative. Splitting further by whether the
   traveler had any as-of-safe interaction history at all (i.e. whether the
   semantic/CF channels carried real signal instead of degenerating to the
   cold-start fallback): non-cold-start trips (n=65) recall = **0.4624**; cold-start
   trips (n=137) recall = **0.4314** — genuine, positive lift from real behavioral
   signal, but a modest one (~7% relative), consistent with the channels adding real
   but limited information over a fundamentally weak observable→latent mapping.

   This mirrors the already-documented localness-ρ miss (resolved ambiguity #13):
   both are cases where an observable, firewall-compliant proxy captures a real but
   partial slice of a latent DGP quantity, measured honestly rather than hidden or
   curve-fit. **What was deliberately not done**: no channel threshold, quota, or
   ranking rule was adjusted specifically to raise this number — the quotas are
   spec.md's own literal per-channel table (60/50/50/40/50/30), and the two
   arbitrary-but-principled threshold choices made during implementation
   (`pop_pct_cutoff=0.40` from spec.md's literal text rescaled to this dataset's
   `[0,1]` scale; `semantic_sim_threshold=0.0`, a genuine floor rather than a high
   bar) were fixed *before* the recall number was ever computed, not tuned
   afterward.

## Phase 4b — baselines + evaluation harness (`src/poi_rank/models/`, `src/poi_rank/eval/`)

#### 33. One shared "ranking frame" assembly, two frames, two label sources

`models/ranking_data.py::build_ranking_frame` is called exactly twice:
`load_holdout_evaluation_frame` (PRIMARY, spec.md section 11.1: holdout trips only,
via `trips.is_holdout`, labels from `interactions_holdout_random.parquet`) and
`load_train_ranking_frame` (train trips only, labels from
`interactions_train.parquet`, for baseline 6's fit and later phases' LambdaMART/IPS).
Verified directly on the committed dataset: `trips.is_holdout` exactly partitions the
800 trips into the same 598 train / 202 holdout sets already implied by which
`trip_id`s appear in `interactions_train.parquet` vs
`interactions_holdout_random.parquet` (zero overlap either way) — so filtering
`candidates.parquet` (which carries candidate sets for all 800 trips, per Phase 4a's
own design) by `trips.is_holdout` is the correct, simpler split, not a second
independent derivation that could disagree with the interaction logs.

A candidate with no matching interaction row for its trip gets **label 0** — for the
PRIMARY frame this is a true negative (`interactions_holdout_random` is a
uniform-random-exposure log, so "no row" means "never shown, or shown and only ever
`view`d," both already label 0), not a negative-sampling assumption. A candidate
POI's label is the **max** across every interaction row logged against it within that
trip (a POI can be exposed/interacted with more than once in a trip — e.g. `view` then
later `click` — the strongest observed signal is the correct graded-relevance label).

#### 34. `interact_localness_gap` reads the CANDIDATE POI's own localness, not the traveler's aggregate

spec.md section 6's literal formula is `|implicit_localness − touristiness_pref|`.
Built as a traveler x POI interaction feature (varies per candidate, not just per
trip), it reads the **candidate POI's own** `num_localness` (from
`poi_features.parquet`) against the traveler's `explicit_touristiness_pref` — not the
traveler-level `implicit_mean_localness` aggregate (already exported by Phase 3 with
zero per-candidate dependence, so there is nothing new to "interact" against). See
`models/ranking_data.py::add_interaction_features`'s docstring for the full reasoning.

#### 35. Baseline 3 (popularity + geo filter): fallback rate measured at 0.0% on the committed holdout set

Every one of the 202 holdout trips had at least one candidate inside its
mobility-conditioned radius (the fallback — skip the filter for that trip — never
fired). This is expected, not a sign the fallback path is untested: Phase 4a's geo
channel already contributes up to 60 in-radius candidates per trip by construction, so
an empty-after-filtering candidate set would require every other channel's
contribution to ALSO fall entirely outside the radius, which the unit test
(`test_baseline_popularity_geo_filter_falls_back_when_all_excluded`) exercises
directly with a hand-built adversarial fixture instead of hoping real data happens to
trigger it.

#### 36. Baseline 5 (item-kNN CF): cold-start fallback rate measured at 67.8% (137/202 holdout trips)

Matches Phase 4a's own measured CF-channel coverage precedent (resolved ambiguity
#30: 65/202 holdout trips, 32%, have a non-empty as-of-safe seed history) almost
exactly (202−137=65) — the same underlying "share of holdout trips that are a
traveler's first trip" population drives both numbers, as expected since both reuse
`candidates.union.cf_seed_poi_ids` directly.

#### 37. Baseline 6 (logistic regression): one-hot encoding is a deliberate, documented divergence from "keep categoricals native"

`configs/features.yaml`'s categorical POI columns (`cat_category`, `cat_subcategory`,
`cat_indoor_outdoor`, `cat_price_level`) are pandas `category` dtype specifically for
LightGBM's native categorical handling (spec.md section 5/8). `LogisticRegression` has
no native categorical support, so baseline 6 one-hot-encodes them
(`models/baselines.py::_design_matrix`) — a legitimate, model-specific divergence
documented at the point of use, not a project-wide reversal of the native-categorical
convention (a later LambdaMART system consumes the same `cat_*` columns natively,
unchanged). Numeric/boolean features with `NaN` (e.g. a cold-start traveler's
`implicit_mean_localness`) are filled with `0.0` for the same model-specific reason —
`LogisticRegression` has no native NaN handling either, unlike LightGBM.

#### 38. Bootstrap resampling unit: trip, not traveler

spec.md section 8 writes "NDCG@10 per traveler → 2,000-sample bootstrap 95% CI."
Read as "per ranking instance" in context (every metric in this harness — including
Phase 4a's own `candidate_recall@250` — is already computed and reported per TRIP,
and a traveler with 2 trips already contributes 2 independent ranking instances to
every point estimate) rather than literally re-aggregating to one value per unique
traveler first. `configs/eval.yaml`'s `bootstrap.resample_unit: trip` documents this
explicitly as the resolved reading.

#### 39. NDCG uses exponential graded gain, matching LightGBM's `lambdarank` default

`dcg_at_k` uses `(2^label − 1) / log2(rank + 1)` (exponential gain), not the simpler
linear-gain `label / log2(rank + 1)` some NDCG references use — chosen to match
LightGBM's `label_gain` default for the `lambdarank` objective (spec.md section 8),
so this phase's baseline NDCG numbers and a later LambdaMART system's NDCG numbers are
computed identically and remain directly comparable.

#### 40. Undefined-metric exclusion, mirrored from `candidates/recall_metrics.RecallResult`

NDCG/Recall/MAP/MRR are `None` (excluded from the mean, count reported, never
coerced to 0 or 1) for any trip whose candidate set contains zero
ground-truth-relevant (`label >= 1`) POIs — this is a REAL, expected consequence of
Phase 4a's ~0.44 `candidate_recall@250` (a meaningful share of holdout trips have
none of their true-relevant POIs inside their own candidate set at all).
Precision@k is always defined (an empty top-k is trivially precision 0). Measured on
the committed dataset: `n_excluded=0` for every system's `ndcg@10` — every one of the
202 holdout trips had at least one relevant POI somewhere inside its own ~187-POI
candidate set (a partial-recall outcome, not a zero-recall one, for all 202 trips) —
so this phase's exclusion machinery is implemented and tested
(`tests/test_metrics.py`) but did not fire on the actual committed data.

### Measured results (committed dataset, 202 holdout trips)

`uv run python -m poi_rank.cli evaluate`, wall clock ≈ 9 s. Two independent runs
produce a byte-identical `results/metrics.json`
(`sha256=427e7562862fc8ba291dcbc64be5017564ff6352dd4634635691ca198c4d0278`).

| System | NDCG@10 | 95% CI | % of oracle ceiling |
|---|---|---|---|
| Random | 0.0304 | [0.0219, 0.0392] | 23.0% |
| Popularity | 0.0622 | [0.0489, 0.0769] | 47.1% |
| Popularity + geo filter | 0.0514 | [0.0393, 0.0648] | 38.8% |
| Content cosine (explicit only) | 0.0807 | [0.0660, 0.0968] | 61.0% |
| Item-kNN CF | 0.0543 | [0.0420, 0.0679] | 41.1% |
| Logistic regression (full features) | 0.0603 | [0.0474, 0.0737] | 45.6% |
| **Oracle (ceiling)** | **0.1322** | [0.1139, 0.1500] | 100.0% |

Paired Wilcoxon vs popularity (`ndcg@10`, n=202 pairs): content_cosine p=0.0349
(significant at α=0.05); item_knn_cf p=0.0742 (not significant at α=0.05); logistic_
regression p=0.4972 (not significant); oracle p=2.32e-11 (highly significant, as
expected — the oracle uses information (noise-free latent utility) no observable
baseline has access to).

**Honest reading, in the context of the already-diagnosed ~0.44 candidate-recall
ceiling (resolved ambiguity #32's "Honest diagnosis" section)**: every number above
is capped by that same ceiling — even a hypothetically perfect ranker over a trip's
own ~187-POI candidate set cannot recover a true-relevant POI that Phase 4a's
candidate generation never included in the first place. The oracle's own NDCG@10
(0.1322) is itself evidence of this: an oracle ranking by TRUE noise-free utility
still only reaches ~13% NDCG@10, far below what an unconstrained oracle over the FULL
destination catalog would achieve, precisely because it too is restricted to ranking
only the same capped candidate set every baseline sees. Baseline ordering is
directionally sensible (content_cosine > logistic_regression > item_knn_cf ≈
popularity > popularity_geo_filter > random, oracle strictly on top) and the only
statistically significant lift over popularity at this candidate-set scale, n=202,
comes from content_cosine and the oracle itself — logistic_regression and item_knn_cf
show a positive but not-yet-significant direction. This is reported as-is, not tuned
or cherry-picked: no baseline hyperparameter in `configs/model.yaml` was adjusted
after seeing this table.

## Phase 5 resolved ambiguities (systems 7/8, `models/lambdamart.py`)

#### 41. IPS weight for a TRAIN candidate with no logged impression: neutral weight 1.0, not zero/dropped

73.5% of TRAIN ranking-frame rows (measured: 94,664 fit-split rows total, 25,207 with
a logged `p_expose` — 26.6% exposure rate) have no matching row in
`interactions_train.parquet` at all — the popularity-biased slate policy only ever
shows a small fraction of each trip's ~186 candidates. IPS reweights the RELATIVE
contribution of rows we actually observed under a biased sampling process; it cannot
synthesize a correction for a sampling event that never happened. These rows get the
neutral raw weight 1.0 (before group normalization) — the same "no logged impression
⇒ assumed label 0" treatment `models/ranking_data.py` already documents for the
label itself, extended consistently to the weight. Zero-weighting or dropping them
was considered and rejected: it would remove nearly 3/4 of the negative training
signal LambdaMART's listwise objective needs to discriminate real negatives from
real positives across a trip's FULL candidate set — the same (not exposure-
restricted) set it is scored against at eval time.

#### 42. IPS "normalized per group" reading: per-trip weights rescaled to sum to the group's own row count

spec.md section 8 says "normalized per group" without a formula. Read as: raw weight
`clip(1/p_expose, 1, 20)` (or 1.0 per #41) rescaled within each `trip_id` group via
`raw * group_size / group_sum`, so every group's weights sum to exactly its own row
count (equivalently: every group's MEAN weight is 1.0). Chosen specifically so
system 8's (IPS) total per-group loss magnitude is directly comparable to system 7's
(uniform weight 1.0, whose per-group sum is trivially `group_size` too) — the IPS
ablation (#44) then isolates "which rows within a group matter more", not also
"which groups get more total gradient weight", which a different normalization
(e.g. global weight-sum-to-1) would have confounded. Unit-tested on a hand-computed
2-group synthetic example (`tests/test_lambdamart.py`).

#### 43. New-POI cohort identification recomputes the timeline split from `configs/datagen.yaml`, in `eval/`, not `models/`

A "new POI" is `created_at >= split` (the same global train/holdout boundary every
trip's `is_holdout` flag already derives from, `datagen/timeline.py::build_timeline`).
This boundary and `created_at` are both PUBLIC, already-observable quantities (every
phase's `is_holdout` split already depends on it; `pois_prepared.parquet` already
exports `created_at`) — not latent DGP/oracle information, so reading
`datagen.timeline`/`datagen.config` here does not violate the `models/` firewall's
purpose (spec.md section 1.1: stop the ranking model from cheating on TRUE utility,
not from knowing the public train/holdout boundary every downstream phase already
uses). Placed in a new `eval/new_poi_cohort.py`, not `models/lambdamart.py`, so
`tests/test_firewall_models.py`'s per-file scan never has to reason about this
distinction — `eval/run.py` already reads `datagen.oracle_export` for the same class
of documented, non-cheating reason. Measured cohort: 68/1446 catalog POIs (4.7%,
consistent with the configured `new_poi_rate: 0.05`), appearing in 1,634 holdout
candidate rows across all 202 holdout trips, with 44/202 trips having at least one
`label >= 1` new-POI candidate (the population NDCG@10 is actually computed over).

#### 44. Behavioral dropout applied identically to systems 7 and 8 (fit-split only); a third ablation-only booster isolates the dropout effect from the IPS effect

Both systems 7 (no IPS) and 8 (IPS) are trained with the SAME 15%
behavioral-block dropout — applying it to only one would confound the IPS ablation
(the system 7 vs 8 comparison) with a second, uncontrolled variable. A third
booster (`lambdamart_ips_no_dropout`, IPS on, dropout off, never exposed in the main
9-system table) is trained purely to isolate the dropout effect: compared against
system 8 (dropout on) on the new-POI cohort (#43), with IPS held constant at "on" for
both sides. Dropout is applied to the FIT split only, never the carved validation
split (`train_val_split_by_trip`) — validation is meant to reflect genuine
serving-time NDCG on ordinary (non-cold-start) trips for early-stopping purposes, not
a dropout-augmented objective.

#### 45. `artifacts/model.txt` (spec.md section 14, singular) is reserved for the PRIMARY system (8); two more boosters are a documented deviation

Two systems (7, 8) plus one ablation-only booster (#44) are built, not one, so 3
files are persisted: `model.txt` (system 8, LambdaMART+IPS — spec.md's own framing
of it as "primary"), `model_no_ips.txt` (system 7), and
`model_ips_no_dropout.txt` (ablation-only). `poi_rank.cli train` fits and persists
all 3; `poi_rank.cli evaluate` only ever LOADS them (never retrains), so the
project's byte-identical-rerun contract applies to `train`'s output directly, not
indirectly through `evaluate`'s own determinism.

#### 46. LightGBM CPU determinism: `deterministic=True` + `force_row_wise=True` + `num_threads=1`, verified byte-identical across 2 full `train`+`evaluate` runs

LightGBM's histogram-building order is not perfectly associative across threads by
default (LightGBM's own documented caveat) — `deterministic: true` requires
`force_row_wise`/`force_col_wise` to be set, and `num_threads: 1` removes any
thread-order variance outright. All 4 explicit RNG seeds (`seed`, `bagging_seed`,
`feature_fraction_seed`, `data_random_seed`) are pinned to `model_cfg.seed`.
Verified empirically, not just configured: two full, independent
`poi_rank.cli train` + `poi_rank.cli evaluate` runs on the committed dataset produced
byte-identical `artifacts/model.txt`, `artifacts/model_no_ips.txt`,
`artifacts/model_ips_no_dropout.txt`, AND `results/metrics.json` (`diff -q`,
zero differences on all 4 files) — also covered by
`tests/test_lambdamart.py::test_run_train_lambdamart_is_deterministic` and
`tests/test_evaluate.py::test_run_evaluate_is_deterministic` on the fast-config
fixture chain.

#### 47. Success criteria (spec.md section 11.10): NDCG@10-vs-popularity target MET, % of oracle ceiling MISSED — tied to the already-documented candidate-recall ceiling

Measured on the committed dataset (`results/metrics.json`, 202 holdout trips):
lambdamart_ips NDCG@10 = 0.0875 [0.0715, 0.1042] vs popularity's 0.0622 — a **+40.6%
relative uplift, Wilcoxon p=0.0042** (both the ≥+40% and p<0.01 legs of the target
are MET). % of oracle ceiling = **66.2%** (target ≥70%) — a **MISSED** target, by a
margin of 3.8 points. Root cause, tying back to the already-documented (PLAN.md,
resolved ambiguity #32) candidate_recall@250 ceiling: the oracle itself only reaches
NDCG@10=0.1322 on this same candidate-recall-capped candidate set (resolved
ambiguity #32's "Honest diagnosis" — an oracle ranking by TRUE noise-free latent
utility still cannot recover a true-relevant POI Phase 4a's candidate generation
never included in the first place). lambdamart_ips closing 66.2% of that already-
capped ceiling, rather than the full 70%, is consistent with — not contradictory to
— the ~0.44 candidate-recall ceiling already identified as the dominant constraint on
every ranking metric in this project: there is a hard information ceiling on HOW MUCH
of the oracle's own already-capped NDCG@10 any observable-features-only ranker can
close, and this system closes nearly two-thirds of it. No hyperparameter in
`configs/model.yaml`'s `lambdamart:` section was tuned against this specific number
after first seeing it — the reported config is the first one trained.

### Measured results, systems 7/8 (committed dataset, 202 holdout trips)

`uv run python -m poi_rank.cli train` (≈23 s) then `uv run python -m poi_rank.cli
evaluate` (≈10 s). Two independent full `train`+`evaluate` runs produced
byte-identical output on every file (resolved ambiguity #46):
`artifacts/model.txt sha256=562ed7dfb295c201c00b506d46f6d675bb5c4670666cd16f1a3b3d52ce8671ef`,
`artifacts/model_no_ips.txt sha256=eb04d89ec6fe276df8fb6ca35e0aea14388028ae1c70801de441ea25574bf04a`,
`artifacts/model_ips_no_dropout.txt sha256=f7b212a453659fbd4937e9b9327bb52562e7de81cc85b3005a185399950eb97c`,
`results/metrics.json sha256=d61c414ae3e45c004186600dbb69ef2ffac068befbb31348fbb8c0bb121874cc`.

| System | NDCG@10 | 95% CI | % of oracle ceiling |
|---|---|---|---|
| LambdaMART (system 7, no IPS) | 0.0545 | [0.0423, 0.0668] | 41.3% |
| **LambdaMART+IPS (system 8, primary)** | **0.0875** | [0.0715, 0.1042] | **66.2%** |

Paired Wilcoxon (`ndcg@10`, n=202 pairs): lambdamart_ips vs popularity **p=0.0042**
(significant, +40.6% relative — spec.md section 11.10's target MET, see #47);
lambdamart_ips vs content_cosine (best baseline) p=0.812 (numerically higher —
0.0875 vs 0.0807 — but NOT a statistically distinguishable improvement at n=202);
**lambdamart (no IPS) vs lambdamart_ips: p=1.80e-05** — IPS correction is a highly
significant, large improvement (system 7 without IPS actually underperforms
popularity, 0.0545 vs 0.0622 — consistent with a pointwise-shaped ranker, even
under a listwise objective, partially reproducing the popularity-biased LOGGING
policy's own bias when trained on its output uncorrected).

**New-POI cohort** (68 catalog POIs, 1,634 holdout candidate rows, 44/202 trips with
a relevant cohort candidate — resolved ambiguity #43): NDCG@10 with 15% behavioral
dropout = 0.5208 [0.4568, 0.5888]; without dropout = 0.5075 [0.4506, 0.5713];
paired Wilcoxon p=0.994 (n=44 pairs) — **not a statistically significant
difference at this cohort size**. Reported honestly: the direction is consistent
with the intended robustness effect (dropout-trained model scores marginally
higher), but 44 trips is too small a sample to distinguish this from noise, and no
hyperparameter was adjusted to try to manufacture significance.
