"""Write latent/ground-truth data to `data/synthetic/_oracle/`.

**Isolation contract (spec.md section 1.1, enforced by tests/test_oracle_isolation.py):**
this directory must be read by NOTHING except a future `eval/oracle.py`. This module
is the sole designated WRITER — the only other file allowed to reference the
`_oracle` subdirectory name at all. Every other caller (pipeline.py included) gets a
fully-formed `Path` back from `oracle_dir_from_output` and never spells out the
subdirectory name itself, so the isolation test's "only eval/oracle.py may reference
_oracle" check has exactly one documented, necessary exception (the designated writer)
rather than being silently violated by the write path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.datagen.archetypes import ARCHETYPE_NAMES
from poi_rank.datagen.utility import TermStandardization

ORACLE_SUBDIR_NAME = "_oracle"


def oracle_dir_from_output(output_dir: Path) -> Path:
    """Resolve the oracle subdirectory path from the top-level output directory."""
    return output_dir / ORACLE_SUBDIR_NAME


def write_traveler_taste(
    output_dir: Path, taste_vectors: dict[str, npt.NDArray[np.float64]]
) -> Path:
    """Write true traveler taste vectors: traveler_id -> latent taste vector (list col)."""
    df = pd.DataFrame(
        {
            "traveler_id": list(taste_vectors.keys()),
            "taste_vector": [v.tolist() for v in taste_vectors.values()],
        }
    )
    path = output_dir / "traveler_taste.parquet"
    df.to_parquet(path, index=False)
    return path


def write_traveler_archetype(
    output_dir: Path, mixtures: dict[str, npt.NDArray[np.float64]]
) -> Path:
    """Write each traveler's TRUE archetype mixture (Dirichlet weights over the 8 archetypes) and
    its dominant archetype name. Eval-only: `eval/personalization.py` groups travelers by this,
    instead of a K-Means clustering of the very features being evaluated."""
    names = list(ARCHETYPE_NAMES)
    df = pd.DataFrame(
        {
            "traveler_id": list(mixtures.keys()),
            "archetype_mixture": [np.asarray(m).tolist() for m in mixtures.values()],
            "dominant_archetype": [names[int(np.argmax(m))] for m in mixtures.values()],
        }
    )
    path = output_dir / "traveler_archetype.parquet"
    df.to_parquet(path, index=False)
    return path


def write_poi_latent(output_dir: Path, poi_true_df: pd.DataFrame) -> Path:
    """Write per-POI latent quality/localness/semantic vector."""
    df = pd.DataFrame(
        {
            "poi_id": poi_true_df["poi_id"],
            "destination": poi_true_df["destination"],
            "latent_quality": poi_true_df["latent_quality"],
            "latent_localness": poi_true_df["latent_localness"],
            "poi_semantic": [v.tolist() for v in poi_true_df["poi_semantic"]],
        }
    )
    path = output_dir / "poi_latent.parquet"
    df.to_parquet(path, index=False)
    return path


def write_holdout_utility(output_dir: Path, holdout_utility_df: pd.DataFrame) -> Path:
    """Write noise-free true utility u(t,p) for every (trip, poi) pair `eval/oracle.py`
    needs to rank the unbiased/logged holdout candidates by.

    Scoped to holdout trips only (not the full trip x catalog cross product) — this is
    the only population spec.md's oracle-ceiling computation actually needs, and
    keeping the export scoped to it keeps `data/synthetic/` well under the 25 MB
    budget (documented in docs/DATA_CARD.md).
    """
    path = output_dir / "holdout_utility_true.parquet"
    holdout_utility_df.to_parquet(path, index=False)
    return path


def write_term_standardization(output_dir: Path, standardization: TermStandardization) -> Path:
    """Write the Block A RC2a utility-term standardization reference (per-destination
    mean/std for 6 terms + the global `novelty` a priori reference,
    `datagen/utility.py::compute_term_standardization`) so `eval/dgp_diagnostics.py`
    can recompute the EXACT SAME standardized+weighted terms `datagen/pipeline.py`
    used at generation time, without recomputing (and risking a mismatched)
    population-level fit itself. These are aggregate mean/std scalars derived from
    oracle-only latent fields (`poi_semantic`, `latent_quality`, `latent_localness`,
    traveler taste vectors) -- never per-POI/per-traveler values -- so this lives in
    `_oracle/` under the same isolation contract as every other file here."""
    path = output_dir / "term_standardization.json"
    path.write_text(
        json.dumps(standardization.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path
