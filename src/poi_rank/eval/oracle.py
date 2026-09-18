"""Oracle reader (spec.md section 1.1): the SOLE legitimate reader of
`data/synthetic/_oracle/`. Every other module in `src/poi_rank/` must never import
from or read this directory -- enforced by `tests/test_oracle_isolation.py`.

Two uses so far:
  - Localness-rho validation (Phase 2): the observable `localness` index computed by
    `data/localness.py` against the DGP's true `latent_localness` (spec.md section 4
    target: Spearman rho > 0.6).
  - Oracle ceiling (Phase 4b, spec.md section 1.4 / section 8 system 9): rank a
    trip's own candidate set by the DGP's TRUE, noise-free latent utility `u(t,p)`
    (`_oracle/holdout_utility_true.parquet`, exported by
    `datagen/oracle_export.py::write_holdout_utility`) -- the ceiling every baseline/
    model in `eval/metrics.py`'s results table is reported as a percentage of. This
    is the ONLY oracle-touching code this phase adds; `models/**` never imports this
    module or reads `_oracle` itself (`tests/test_firewall_models.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TARGET_LOCALNESS_RHO = 0.6
HOLDOUT_UTILITY_FILENAME = "holdout_utility_true.parquet"


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


def load_holdout_utility_true(oracle_dir: Path) -> pd.DataFrame:
    """Load `holdout_utility_true.parquet` (`trip_id`, `traveler_id`, `poi_id`,
    `utility_true`) -- the DGP's noise-free latent utility for every `(trip, poi)`
    pair in the holdout population (spec.md section 1.4), scoped to holdout trips
    only. Never exposed outside this module.
    """
    return pd.read_parquet(oracle_dir / HOLDOUT_UTILITY_FILENAME)


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
