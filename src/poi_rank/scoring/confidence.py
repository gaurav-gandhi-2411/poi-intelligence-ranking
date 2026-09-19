"""Confidence (spec.md section 9.3):

```
confidence = g( n_interactions_traveler, poi_impression_count, review_count,
                ensemble_std (5 seeds), calibration_bin_width )
```

`g` is left abstract by spec.md -- this project's own design (documented in
`configs/scoring.yaml`): an evidence-volume shrinkage term per raw-count input
(mirrors `features/traveler_features.py`'s own `alpha_t = n_t / (n_t + k)`
cold-start-shrinkage convention, same shape, independently parameterized -- see
that module's own "two different mechanisms for two different jobs" framing,
extended here to a THIRD, distinct mechanism/job), plus two stability terms that
DECREASE confidence as ensemble disagreement / calibration-bin coarseness grows.
Combined as a weighted arithmetic mean, clipped to `[0, 1]`.

**Validation, not assertion** (spec.md section 9.3): `scoring/output.py` bins
per-trip confidence into deciles and reports NDCG@10 per decile plus the Spearman
rho of (confidence decile, NDCG@10) -- see that module for the decile-binning
resolution (spec.md leaves "recommendations" ambiguous between per-item and
per-trip-list; NDCG is inherently a per-list metric, so this project reads it as
per-trip, documented in docs/DATA_CARD.md).

**Ensemble std**: 5 LightGBM models, same IPS-weighted, dropout-augmented training
recipe as system 8's own primary booster, reusing `models.lambdamart
.fit_lambdamart_booster` directly (module docstring: "reuse lambdamart.py's
training function with a seed override, don't duplicate the training logic") --
only the LightGBM boosting/bagging seed varies across the 5 (train/val split and
dropout mask stay pinned to `configs/model.yaml`'s own seeds), so the resulting
std measures genuine training-stochasticity disagreement.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.features.config import FeatureBuildConfig
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import LambdaMartConfig, ModelConfig
from poi_rank.models.lambdamart import (
    apply_behavioral_dropout,
    attach_train_ips_weight,
    fit_lambdamart_booster,
    score_booster,
    train_val_split_by_trip,
)
from poi_rank.models.ranking_data import load_train_ranking_frame
from poi_rank.scoring.config import ConfidenceConfig, ScoringConfig

FloatArray = npt.NDArray[np.float64]


# -----------------------------------------------------------------------------------
# Ensemble std (5 seeds)
# -----------------------------------------------------------------------------------


def train_ensemble_boosters(
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    lm_cfg: LambdaMartConfig,
    seeds: tuple[int, ...],
) -> list[lgb.Booster]:
    """Fit `len(seeds)` LightGBM boosters with system 8's exact recipe (IPS-weighted,
    behavioral-dropout-augmented), varying only the LightGBM training seed (module
    docstring). Reuses `models.lambdamart`'s own split/weight/dropout machinery
    directly -- never reimplemented."""
    numeric_columns = numeric_feature_columns(train_frame)
    categorical_columns = categorical_feature_columns(train_frame)
    fit_frame, val_frame = train_val_split_by_trip(
        train_frame, lm_cfg.val_fraction, lm_cfg.val_split_seed
    )
    ips_weight = attach_train_ips_weight(fit_frame, interactions_train, pois_df, lm_cfg)
    fit_frame_dropout, _ = apply_behavioral_dropout(
        fit_frame, lm_cfg.behavioral_dropout_rate, lm_cfg.behavioral_dropout_seed
    )
    return [
        fit_lambdamart_booster(
            fit_frame_dropout,
            val_frame,
            numeric_columns,
            categorical_columns,
            lm_cfg,
            seed,
            sample_weight=ips_weight,
        )
        for seed in seeds
    ]


def run_train_ensemble(
    data_dir: Path,
    artifacts_dir: Path,
    model_cfg: ModelConfig,
    feature_cfg: FeatureBuildConfig,
    scoring_cfg: ScoringConfig,
) -> list[Path]:
    """Fit + persist the confidence ensemble (`poi_rank.cli train`'s second half)."""
    train_frame = load_train_ranking_frame(
        data_dir, feature_cfg.traveler_features.budget_target_price_level
    )
    boosters = train_ensemble_boosters(
        train_frame,
        pd.read_parquet(data_dir / "interactions_train.parquet"),
        pd.read_parquet(data_dir / "pois_prepared.parquet"),
        model_cfg.lambdamart,
        scoring_cfg.confidence.ensemble_seeds,
    )
    return save_ensemble(boosters, scoring_cfg.confidence.ensemble_seeds, artifacts_dir)


ENSEMBLE_FILENAME_TEMPLATE = "ensemble_seed{seed}.txt"


def save_ensemble(
    boosters: list[lgb.Booster], seeds: tuple[int, ...], artifacts_dir: Path
) -> list[Path]:
    """Persist the ensemble at `poi_rank.cli train` time. Training 5 extra LightGBM boosters
    inside every `evaluate` / `recommend` / `scenarios` run (the original design) was ~80 s of
    pure repeated work per invocation -- the ensemble depends only on the train frame and the
    seeds, never on what is being scored."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = [artifacts_dir / ENSEMBLE_FILENAME_TEMPLATE.format(seed=s) for s in seeds]
    for booster, path in zip(boosters, paths, strict=True):
        booster.save_model(str(path))
    return paths


def load_ensemble(seeds: tuple[int, ...], artifacts_dir: Path) -> list[lgb.Booster]:
    paths = [artifacts_dir / ENSEMBLE_FILENAME_TEMPLATE.format(seed=s) for s in seeds]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"confidence-ensemble boosters missing from {artifacts_dir}: {missing}. "
            "Run `poi_rank.cli train` first (it fits and saves them)."
        )
    return [lgb.Booster(model_file=str(p)) for p in paths]


def ensemble_std_scores(
    boosters: list[lgb.Booster],
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> FloatArray:
    """Std, per row, of `frame`'s predicted score across `boosters` -- the
    "ensemble_std (5 seeds)" confidence input (spec.md section 9.3)."""
    scores = np.stack(
        [
            score_booster(b, frame, numeric_columns, categorical_columns).to_numpy(dtype=np.float64)
            for b in boosters
        ],
        axis=1,
    )
    result: FloatArray = scores.std(axis=1)
    return result


# -----------------------------------------------------------------------------------
# g(...)
# -----------------------------------------------------------------------------------


def evidence_shrinkage(x: FloatArray, k: float) -> FloatArray:
    """`x / (x + k)` -- an evidence-volume shrinkage term in `[0, 1)`, 0.5 at
    `x == k` (module docstring)."""
    result: FloatArray = x / (x + k)
    return result


def ensemble_agreement(std: FloatArray, tau: float) -> FloatArray:
    """`exp(-std / tau)` -- high ensemble agreement (low std) -> high confidence."""
    result: FloatArray = np.exp(-std / tau)
    return result


def calibration_stability(bin_width: FloatArray, tau: float) -> FloatArray:
    """`exp(-bin_width / tau)` -- a narrow isotonic calibration segment -> high
    confidence."""
    result: FloatArray = np.exp(-bin_width / tau)
    return result


def compute_confidence(
    n_interactions_traveler: FloatArray,
    poi_impression_count: FloatArray,
    review_count: FloatArray,
    ensemble_std: FloatArray,
    calibration_bin_width_values: FloatArray,
    cfg: ConfidenceConfig,
) -> FloatArray:
    """`g(...)` (spec.md section 9.3, module docstring): weighted arithmetic mean of
    5 `[0, 1]`-valued terms, clipped defensively to `[0, 1]`."""
    traveler_term = evidence_shrinkage(n_interactions_traveler, cfg.traveler_evidence_k)
    poi_term = evidence_shrinkage(poi_impression_count, cfg.poi_impression_k)
    review_term = evidence_shrinkage(np.log1p(review_count), cfg.review_count_k)
    ensemble_term = ensemble_agreement(ensemble_std, cfg.ensemble_std_tau)
    calib_term = calibration_stability(calibration_bin_width_values, cfg.calibration_bin_width_tau)

    weights = np.array(
        [
            cfg.weight_traveler,
            cfg.weight_poi,
            cfg.weight_reviews,
            cfg.weight_ensemble,
            cfg.weight_calibration,
        ],
        dtype=np.float64,
    )
    weights = weights / weights.sum()  # defensive normalization -- see configs/scoring.yaml

    combined = (
        weights[0] * traveler_term
        + weights[1] * poi_term
        + weights[2] * review_term
        + weights[3] * ensemble_term
        + weights[4] * calib_term
    )
    result: FloatArray = np.clip(combined, 0.0, 1.0)
    return result
