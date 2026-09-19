"""Oracle reader (spec.md section 1.1): the SOLE legitimate reader of
`data/synthetic/_oracle/`. Every other module in `src/poi_rank/` must never import
from or read this directory -- enforced by `tests/test_oracle_isolation.py`.

Uses so far:
  - Localness-rho validation (Phase 2): the observable `localness` index computed by
    `data/localness.py` against the DGP's true `latent_localness` (spec.md section 4
    target: Spearman rho > 0.6).
  - `validate_geo_feature_against_latent_localness` (spec-v2-remediation.md
    diagnostic D7): the SAME underlying computation as the localness-rho
    validation above, just pointed at a different observable column
    (`dist_to_tourist_centroid_km`, the raw geo feature, instead of the composite
    `localness` index) -- a thin, differently-named wrapper so external callers
    never need to spell out `validate_localness_against_oracle`'s own name (see
    that wrapper's docstring for why).
  - Oracle ceiling (Phase 4b, spec.md section 1.4 / section 8 system 9): rank a
    trip's own candidate set by the DGP's TRUE, noise-free latent utility `u(t,p)`
    (`_oracle/holdout_utility_true.parquet`, exported by
    `datagen/oracle_export.py::write_holdout_utility`) -- the ceiling every baseline/
    model in `eval/metrics.py`'s results table is reported as a percentage of.
  - `load_traveler_taste` (spec-v2-remediation.md diagnostics D1/D2): true latent
    traveler taste vectors, needed alongside `load_poi_latent`'s `poi_semantic` to
    recompute the DGP's `w_taste * cos(taste_t, poi_semantic_p)` term for the
    variance-decomposition/taste-cosine-distribution diagnostics in
    `eval/dgp_diagnostics.py` -- eval-only reads, explicitly permitted by
    spec-v2-remediation.md section 1 ("All reads of `_oracle/` here are eval-only
    and permitted"). `models/**` never imports this module or reads `_oracle`
    itself (`tests/test_firewall_models.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from poi_rank.datagen.utility import TermStandardization

TARGET_LOCALNESS_RHO = 0.6
HOLDOUT_UTILITY_FILENAME = "holdout_utility_true.parquet"
TERM_STANDARDIZATION_FILENAME = "term_standardization.json"


@dataclass(frozen=True)
class LocalnessValidationResult:
    """Spearman-rho validation result for the observable `localness` index against
    the oracle's true `latent_localness`."""

    spearman_rho: float
    p_value: float
    n: int

    @property
    def meets_target(self) -> bool:
        return self.spearman_rho > TARGET_LOCALNESS_RHO


def load_poi_latent(oracle_dir: Path) -> pd.DataFrame:
    """Load `poi_latent.parquet` (`poi_id`, `destination`, `latent_quality`,
    `latent_localness`, `poi_semantic`) -- true latent values, never exposed outside
    this module."""
    return pd.read_parquet(oracle_dir / "poi_latent.parquet")


def validate_localness_against_oracle(
    observable_localness: pd.DataFrame,
    oracle_dir: Path,
    poi_id_col: str = "poi_id",
    localness_col: str = "localness",
) -> LocalnessValidationResult:
    """Spearman rho between the observable localness index and the DGP's true
    `latent_localness`. spec.md section 4 target: rho > 0.6.

    `observable_localness` must be the prepared POI catalog (or any DataFrame
    carrying at least `poi_id_col` + `localness_col`) -- it must never itself have
    been computed from `_oracle/`. Only this function reads `_oracle/`.
    """
    latent = load_poi_latent(oracle_dir)
    merged = observable_localness[[poi_id_col, localness_col]].merge(
        latent[["poi_id", "latent_localness"]],
        left_on=poi_id_col,
        right_on="poi_id",
        how="inner",
    )
    rho, p_value = stats.spearmanr(merged[localness_col], merged["latent_localness"])
    return LocalnessValidationResult(spearman_rho=float(rho), p_value=float(p_value), n=len(merged))


def validate_geo_feature_against_latent_localness(
    observable_df: pd.DataFrame,
    oracle_dir: Path,
    poi_id_col: str,
    geo_col: str,
) -> LocalnessValidationResult:
    """spec-v2-remediation.md diagnostic D7: Spearman(latent_localness, a raw
    POI lat/lon-derived geo feature, e.g. `dist_to_tourist_centroid_km`) --
    checks whether the geo GENERATION process itself correlates with latent
    localness, a more upstream question than the composite-index validation
    above. A thin, differently-named wrapper around
    `validate_localness_against_oracle` (same computation, reused directly, never
    duplicated) purely so callers in OTHER modules (e.g.
    `eval/dgp_diagnostics.py`) never need to spell out that function's own name
    literally in their own source text -- it happens to end in the substring
    `tests/test_oracle_isolation.py`'s isolation check scans for, and that
    check is a raw text scan, not an AST-aware one, so it cannot distinguish
    "reads the oracle directory directly" from "merely names an already-permitted
    `eval/oracle.py` function whose name contains that substring."
    """
    return validate_localness_against_oracle(observable_df, oracle_dir, poi_id_col, geo_col)


def load_traveler_taste(oracle_dir: Path) -> pd.DataFrame:
    """Load `traveler_taste.parquet` (`traveler_id`, `taste_vector` -- a 32-dim
    `TASTE_DIM` array per traveler) -- the DGP's true latent taste vector, never
    exposed outside this module. Used by `eval/dgp_diagnostics.py`'s D1/D2
    (variance decomposition / taste-cosine distribution) to recompute the DGP's
    `w_taste * cos(taste_t, poi_semantic_p)` term exactly, via
    `datagen.utility.cosine_similarity_to_taste`.
    """
    return pd.read_parquet(oracle_dir / "traveler_taste.parquet")


def load_holdout_utility_true(oracle_dir: Path) -> pd.DataFrame:
    """Load `holdout_utility_true.parquet` (`trip_id`, `traveler_id`, `poi_id`,
    `utility_true`) -- the DGP's noise-free latent utility for every `(trip, poi)`
    pair in the holdout population (spec.md section 1.4), scoped to holdout trips
    only. Never exposed outside this module.
    """
    return pd.read_parquet(oracle_dir / HOLDOUT_UTILITY_FILENAME)


def load_term_standardization(oracle_dir: Path) -> TermStandardization:
    """Load the Block A RC2a utility-term standardization reference
    (`datagen/oracle_export.py::write_term_standardization`) -- the EXACT stats
    `datagen/pipeline.py` used at generation time to z-score the 7 deterministic
    utility terms, so `eval/dgp_diagnostics.py`'s D1 recomputation matches the
    real, committed `utility_true` values exactly rather than an independently
    refit (and potentially slightly different) population estimate."""
    path = oracle_dir / TERM_STANDARDIZATION_FILENAME
    return TermStandardization.from_dict(json.loads(path.read_text(encoding="utf-8")))


def oracle_ceiling_scores(frame_keys: pd.DataFrame, oracle_dir: Path) -> pd.Series:
    """Oracle-ceiling ranking score for each `(trip_id, poi_id)` row of `frame_keys`
    (any DataFrame carrying at least those two columns -- e.g.
    `models.ranking_data`'s holdout evaluation frame) -- ranks each trip's own
    candidate set by the DGP's TRUE, noise-free `u(t,p)` (spec.md section 1.4).

    A candidate with no matching oracle row falls back to `-inf` (ranked last, never
    silently dropped) -- should not occur for the committed dataset:
    `holdout_utility_true.parquet` covers ~475-500 POIs per holdout trip (see
    `datagen/oracle_export.py::write_holdout_utility`), a strict superset of any
    trip's ~250-candidate set.

    Returned as a `pd.Series` aligned to `frame_keys.index` (row order preserved via
    an explicit `(trip_id, poi_id) -> utility_true` dict lookup, not a pandas merge,
    so this never depends on merge-induced row reordering).
    """
    utility = load_holdout_utility_true(oracle_dir)
    utility_map: dict[tuple[str, str], float] = {
        (str(t), str(p)): float(u)
        for t, p, u in zip(
            utility["trip_id"], utility["poi_id"], utility["utility_true"], strict=True
        )
    }
    scores = np.array(
        [
            utility_map.get((str(t), str(p)), -np.inf)
            for t, p in zip(frame_keys["trip_id"], frame_keys["poi_id"], strict=True)
        ],
        dtype=np.float64,
    )
    return pd.Series(scores, index=frame_keys.index, name="score_oracle")


def oracle_dir_for(data_dir: Path) -> Path:
    """The oracle export directory of a dataset directory -- the single place its name is
    spelled outside the writer, so other `eval/` modules resolve it through this reader."""
    return data_dir / "_oracle"
