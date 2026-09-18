"""Oracle reader (spec.md section 1.1): the SOLE legitimate reader of
`data/synthetic/_oracle/`. Every other module in `src/poi_rank/` must never import
from or read this directory -- enforced by `tests/test_oracle_isolation.py`.

This phase's first use: validate the observable `localness` index computed by
`data/localness.py` against the DGP's true `latent_localness` (spec.md section 4
target: Spearman rho > 0.6). Later phases extend this module for the oracle-ceiling
NDCG computation (spec.md section 1.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

TARGET_LOCALNESS_RHO = 0.6


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
