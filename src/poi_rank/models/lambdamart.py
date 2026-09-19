"""Systems 7/8 (spec.md section 8): LightGBM `objective="lambdarank"` (LambdaMART),
grouped by `trip_id`, `lambdarank_truncation_level=20`, graded 0-3 relevance labels.

Two systems, one training routine, three booster artifacts:
  - **System 7 "lambdamart"** (`booster_lambdamart`): uniform per-row training weight
    (no IPS).
  - **System 8 "lambdamart_ips"** (`booster_lambdamart_ips`): IPS-weighted training,
    `w_i = clip(1 / p_expose_i, 1, 20)`, normalized per group -- **the primary
    proposed system** (spec.md section 8).
  - **`booster_lambdamart_ips_no_dropout`**: identical to system 8 except the
    behavioral-dropout augmentation (below) is OFF. Exists ONLY to isolate the
    dropout ablation (`eval/new_poi_cohort.py`) from the IPS ablation -- comparing
    system 7 vs 8 directly would confound "IPS on/off" with "dropout on/off" if
    dropout were applied asymmetrically between them, so systems 7 and 8 both get
    the SAME dropout treatment (on) and this third booster is the dedicated
    dropout-off counterpart to system 8, never exposed in the main 9-system table.

**New-POI robustness** (spec.md section 8): during training, the entire `behav_*`
feature block (14 POI-level behavioral-aggregate columns -- impressions, CTR,
save/visit/dismiss rate, per-archetype affinity; see `features/poi_features.py`) is
masked to `NaN` for `behavioral_dropout_rate` (15%) of FIT-SPLIT training rows,
independently per row (not per-POI-globally -- a given POI can appear masked in one
trip's row and unmasked in another's). Forces the model to maintain a content-only
scoring pathway for POIs with no behavioral history, exactly the situation a genuine
new POI presents at serving time. LightGBM's native NaN handling (learns an optimal
per-split missing-value direction) needs no imputation -- that is the entire point of
using masking instead of, say, zero-filling. Applied to the FIT split only, never the
validation split: validation is meant to reflect genuine serving-time NDCG on
ordinary (non-cold-start) trips for early-stopping purposes, not a dropout-augmented
objective.

**IPS correction** (system 8): `p_expose` is joined from `interactions_train.parquet`
per `(trip_id, poi_id)` (verified constant across a pair's duplicate interaction rows
-- same POI, same trip, same popularity/geo weight -- so `.mean()` is a safe, exact
aggregation, not an approximation). A TRAIN candidate with **no logged impression at
all** (73.5% of train candidate rows, measured -- the biased slate policy only shows
a small fraction of each trip's ~186 candidates) has no propensity to correct: IPS
reweights the RELATIVE contribution of rows we observed under a biased sampling
process, it cannot synthesize a correction for rows we never observed sampling. These
rows get the neutral weight 1.0 (same treatment as every other unexposed-implies-
label-0 assumption `models/ranking_data.py` already documents) rather than being
zero-weighted or dropped -- dropping them would remove nearly 3/4 of the "negative"
training signal LambdaMART's listwise objective needs to discriminate real
negatives from real positives within a trip's full (not just biased-exposed)
candidate set, which is exactly the set it is scored against at eval time.

**Weight normalization ("normalized per group", spec.md section 8)**: raw weights
`clip(1/p_expose, 1, 20)` (or 1.0 for unexposed rows) are rescaled within each
trip-group so the group's weights sum to its own row count (`raw_w * group_size /
group_sum`) -- i.e. every group's MEAN weight is exactly 1.0. This keeps system 8's
total per-group loss magnitude directly comparable to system 7's uniform-weight-1.0
training (whose per-group weight sum is trivially `group_size` too), so the IPS
ablation isolates "which rows within a group matter more" rather than also
confounding it with "which groups get more total gradient weight overall."

**Firewall**: like every other `models/` file, must never import from
`poi_rank.scoring` and must never reference the oracle-only export directory
(`tests/test_firewall_models.py` covers this file automatically via its
`MODELS_DIR.rglob("*.py")` scan). Legitimately imports from `poi_rank.data`,
`poi_rank.features`, `poi_rank.candidates`, and `poi_rank.models.baselines` (reusing
`numeric_feature_columns`/`categorical_feature_columns` directly rather than
reimplementing feature-column selection).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.reconcile import build_poi_id_canonical_map, remap_interaction_poi_ids
from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns
from poi_rank.models.config import LambdaMartConfig, ModelConfig
from poi_rank.models.ranking_data import load_train_ranking_frame

FloatArray = npt.NDArray[np.float64]
BehavioralMask = npt.NDArray[np.bool_]

BEHAVIORAL_PREFIX = "behav_"

INTERACTIONS_TRAIN_FILENAME = "interactions_train.parquet"
POIS_PREPARED_FILENAME = "pois_prepared.parquet"

MODEL_IPS_FILENAME = "model.txt"  # primary (spec.md section 14 names this file singular)
MODEL_NO_IPS_FILENAME = "model_no_ips.txt"
MODEL_IPS_NO_DROPOUT_FILENAME = "model_ips_no_dropout.txt"  # dropout-ablation counterpart only


# -----------------------------------------------------------------------------------
# Behavioral-block dropout (new-POI robustness)
# -----------------------------------------------------------------------------------


def behavioral_feature_columns(frame: pd.DataFrame) -> list[str]:
    """The 14 POI-level `behav_*` columns (`features/poi_features.py`) -- the
    "entire behavioral feature block" spec.md section 8 says to mask."""
    return sorted(c for c in frame.columns if c.startswith(BEHAVIORAL_PREFIX))


def apply_behavioral_dropout(
    frame: pd.DataFrame, dropout_rate: float, seed: int
) -> tuple[pd.DataFrame, BehavioralMask]:
    """Mask the entire `behav_*` block to `NaN` for `dropout_rate` of ROWS,
    independently per row (module docstring). Returns the augmented frame (copy) and
    the boolean mask actually applied (for tests / diagnostics), deterministic given
    `seed`."""
    cols = behavioral_feature_columns(frame)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < dropout_rate
    out = frame.copy()
    out.loc[mask, cols] = np.nan
    return out, mask


# -----------------------------------------------------------------------------------
# IPS weights
# -----------------------------------------------------------------------------------


def p_expose_by_trip_poi(interactions: pd.DataFrame, canonical_map: dict[str, str]) -> pd.Series:
    """`{(trip_id, poi_id): p_expose}` -- canonical-remaps `interactions` (same
    discipline as `models.ranking_data.label_by_trip_poi`) then averages `p_expose`
    across a pair's duplicate interaction rows. `p_expose` is verified constant
    across duplicates for the same `(trip_id, poi_id)` (same POI, same trip, same
    exposure weight regardless of which slate/session it appeared in), so `.mean()`
    is an exact aggregation, not an approximation -- chosen over `.first()`/`.max()`
    only for robustness to any future DGP change that might break that invariant."""
    remapped = remap_interaction_poi_ids(interactions, canonical_map)
    return remapped.groupby(["trip_id", "poi_id"])["p_expose"].mean()


def compute_ips_weights(
    trip_ids: pd.Series, p_expose: pd.Series, clip_low: float, clip_high: float
) -> pd.Series:
    """Pure, testable IPS-weight computation (module docstring): `clip(1/p_expose,
    clip_low, clip_high)` for rows with an observed `p_expose` (NaN treated as "no
    logged impression" -- gets the neutral raw weight 1.0 instead, see module
    docstring), then rescaled per `trip_ids` group so each group's weights sum to its
    own row count. `p_expose` must be index-aligned to `trip_ids`."""
    observed = p_expose.notna().to_numpy()
    inv = np.clip(1.0 / p_expose.fillna(1.0).to_numpy(dtype=np.float64), clip_low, clip_high)
    raw = np.where(observed, inv, 1.0)

    grouping = pd.DataFrame({"trip_id": trip_ids.to_numpy(), "raw": raw})
    group_sum = grouping.groupby("trip_id")["raw"].transform("sum").to_numpy()
    group_size = grouping.groupby("trip_id")["raw"].transform("size").to_numpy()
    normalized = raw * (group_size / group_sum)
    return pd.Series(normalized, index=trip_ids.index, name="ips_weight")


def train_frame_p_expose(
    train_frame: pd.DataFrame, interactions_train: pd.DataFrame, pois_df: pd.DataFrame
) -> pd.Series:
    """`p_expose_by_trip_poi`'s `{(trip_id, poi_id): p_expose}` lookup, joined onto
    every row of `train_frame` (index-aligned) -- `NaN` for a candidate with no
    logged impression in `interactions_train.parquet` (module docstring). Exposed
    separately from `attach_train_ips_weight` so callers can measure exposure
    coverage directly (`.notna()`) rather than inferring it from the post-
    normalization weight, which is NOT 1.0 for unexposed rows either once group
    renormalization is applied."""
    canonical_map = build_poi_id_canonical_map(pois_df)
    p_expose = p_expose_by_trip_poi(interactions_train, canonical_map)
    p_expose_df = p_expose.rename("p_expose").reset_index()
    joined = train_frame[["trip_id", "poi_id"]].merge(
        p_expose_df, on=["trip_id", "poi_id"], how="left"
    )
    joined.index = train_frame.index
    return joined["p_expose"]


def attach_train_ips_weight(
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    cfg: LambdaMartConfig,
) -> pd.Series:
    """`compute_ips_weights` wired up against the real TRAIN ranking frame + the
    real `interactions_train.parquet` propensity log (module docstring's "no logged
    impression" handling included)."""
    p_expose = train_frame_p_expose(train_frame, interactions_train, pois_df)
    return compute_ips_weights(
        train_frame["trip_id"], p_expose, cfg.ips_clip_low, cfg.ips_clip_high
    )


# -----------------------------------------------------------------------------------
# Train/validation split (train-only, never touches the holdout being reported on)
# -----------------------------------------------------------------------------------


def train_val_split_by_trip(
    frame: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation split from `frame` BY TRIP (never splitting a trip's
    candidate rows across both sides -- would leak group structure), for LightGBM
    early stopping. `frame` must be the TRAIN ranking frame; the returned `val_frame`
    is still a train-population split, never the holdout evaluation frame."""
    trip_ids = np.sort(frame["trip_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(trip_ids)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = set(shuffled[:n_val].tolist())
    is_val = frame["trip_id"].isin(val_ids)
    fit_frame = frame.loc[~is_val].sort_values(["trip_id", "poi_id"]).reset_index(drop=True)
    val_frame = frame.loc[is_val].sort_values(["trip_id", "poi_id"]).reset_index(drop=True)
    return fit_frame, val_frame


# -----------------------------------------------------------------------------------
# LightGBM dataset / training / scoring
# -----------------------------------------------------------------------------------


def _feature_matrix(
    frame: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    """Numeric block cast to float64 WITHOUT `fillna` (LightGBM's native missing-
    value handling is the point -- unlike baseline 6's sklearn design matrix).
    Categorical block passed through as-is, preserving the `category` dtype
    (`configs/features.yaml` already exports `cat_*` columns that way) so LightGBM
    auto-detects + natively splits on them, no one-hot expansion."""
    # float32: LightGBM bins features into <=255 histogram buckets anyway, so the extra
    # mantissa bits change nothing material but double Dataset-construction time/memory.
    numeric = frame[numeric_columns].astype(np.float32)
    categorical = frame[categorical_columns]
    return pd.concat([numeric, categorical], axis=1)


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    """Per-trip row counts in FRAME ROW ORDER (LightGBM's `group` param: contiguous
    group sizes matching physical row order -- both ranking-frame builders already
    sort by `trip_id`/`poi_id`, and `train_val_split_by_trip` re-sorts each split the
    same way, so `sort=False` here just preserves that existing contiguous order)."""
    sizes: list[int] = frame.groupby("trip_id", sort=False).size().astype(int).tolist()
    return sizes


def _lgb_params(cfg: LambdaMartConfig, seed: int) -> dict[str, Any]:
    """LightGBM `lambdarank` params. `deterministic=True` + `force_row_wise=True` +
    `num_threads=1` are LightGBM's own documented recipe for byte-identical reruns
    on CPU (histogram-building order otherwise introduces float non-associativity
    across threads) -- verified empirically in `tests/test_lambdamart.py`, since the
    project's hard determinism requirement applies here too, not just to datagen."""
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": list(cfg.eval_ndcg_at),
        "lambdarank_truncation_level": cfg.lambdarank_truncation_level,
        "label_gain": list(cfg.label_gain),
        "num_leaves": cfg.num_leaves,
        "max_bin": cfg.max_bin,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": cfg.num_threads,
        "verbosity": -1,
    }


def fit_lambdamart_booster(
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    cfg: LambdaMartConfig,
    seed: int,
    sample_weight: pd.Series | None = None,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    """Fit one LambdaMART booster on `fit_frame`, early-stopping on `val_frame`
    (spec.md section 8). `num_boost_round`, if given, trains exactly that many rounds with no
    validation set / early stopping (used for the confidence-ensemble members, whose round
    count is taken from the primary booster's early-stopped `best_iteration`).
    `sample_weight`, if given, must be index-aligned to `fit_frame` (system 8's
    `attach_train_ips_weight` output); `None` means system 7's uniform per-row weight
    (LightGBM's own default)."""
    x_fit = _feature_matrix(fit_frame, numeric_columns, categorical_columns)
    x_val = _feature_matrix(val_frame, numeric_columns, categorical_columns)
    weight = sample_weight.to_numpy(dtype=np.float64) if sample_weight is not None else None

    train_set = lgb.Dataset(
        x_fit,
        label=fit_frame["label"].to_numpy(dtype=np.float64),
        group=_group_sizes(fit_frame),
        weight=weight,
        categorical_feature=categorical_columns,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        x_val,
        label=val_frame["label"].to_numpy(dtype=np.float64),
        group=_group_sizes(val_frame),
        reference=train_set,
        categorical_feature=categorical_columns,
        free_raw_data=False,
    )

    if num_boost_round is not None:
        return lgb.train(_lgb_params(cfg, seed), train_set, num_boost_round=num_boost_round)

    booster = lgb.train(
        _lgb_params(cfg, seed),
        train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster


def score_booster(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.Series:
    """Score any ranking frame with a fitted (or freshly-loaded, `models/lambdamart
    .load_boosters`) booster -- returns a `BaselineResult.score`-compatible
    `pd.Series` aligned to `frame.index`."""
    x = _feature_matrix(frame, numeric_columns, categorical_columns)
    preds = booster.predict(x)
    scores = np.asarray(preds, dtype=np.float64)
    return pd.Series(scores, index=frame.index, name="score_lambdamart")


# -----------------------------------------------------------------------------------
# Full training orchestration (`poi_rank.cli train`)
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaMartArtifacts:
    """Fitted boosters + the exact feature-column layout they were trained on (so
    `evaluate` can reconstruct an identical design matrix), plus training
    diagnostics (best_iteration, val NDCG@10, row/weight summaries) for the CLI
    report and `docs/DATA_CARD.md`."""

    booster_lambdamart: lgb.Booster
    booster_lambdamart_ips: lgb.Booster
    booster_lambdamart_ips_no_dropout: lgb.Booster
    numeric_columns: list[str]
    categorical_columns: list[str]
    diagnostics: dict[str, Any]


def train_lambdamart_systems(
    train_frame: pd.DataFrame,
    interactions_train: pd.DataFrame,
    pois_df: pd.DataFrame,
    model_cfg: LambdaMartConfig,
    seed: int,
) -> LambdaMartArtifacts:
    """Fit systems 7, 8, and the dropout-ablation-only booster (module docstring)
    from the TRAIN ranking frame. `train_frame` is `models.ranking_data
    .load_train_ranking_frame`'s output; never the holdout frame."""
    numeric_columns = numeric_feature_columns(train_frame)
    categorical_columns = categorical_feature_columns(train_frame)

    fit_frame, val_frame = train_val_split_by_trip(
        train_frame, model_cfg.val_fraction, model_cfg.val_split_seed
    )
    p_expose_fit = train_frame_p_expose(fit_frame, interactions_train, pois_df)
    ips_weight_fit = compute_ips_weights(
        fit_frame["trip_id"], p_expose_fit, model_cfg.ips_clip_low, model_cfg.ips_clip_high
    )

    fit_frame_dropout, dropout_mask = apply_behavioral_dropout(
        fit_frame, model_cfg.behavioral_dropout_rate, model_cfg.behavioral_dropout_seed
    )

    booster_no_ips = fit_lambdamart_booster(
        fit_frame_dropout, val_frame, numeric_columns, categorical_columns, model_cfg, seed
    )
    booster_ips = fit_lambdamart_booster(
        fit_frame_dropout,
        val_frame,
        numeric_columns,
        categorical_columns,
        model_cfg,
        seed,
        sample_weight=ips_weight_fit,
    )
    booster_ips_no_dropout = fit_lambdamart_booster(
        fit_frame,
        val_frame,
        numeric_columns,
        categorical_columns,
        model_cfg,
        seed,
        sample_weight=ips_weight_fit,
    )

    diagnostics = {
        "n_fit_rows": int(len(fit_frame)),
        "n_val_rows": int(len(val_frame)),
        "n_fit_trips": int(fit_frame["trip_id"].nunique()),
        "n_val_trips": int(val_frame["trip_id"].nunique()),
        "n_dropout_rows": int(dropout_mask.sum()),
        "dropout_rate_actual": float(dropout_mask.mean()),
        "n_numeric_columns": len(numeric_columns),
        "n_categorical_columns": len(categorical_columns),
        "best_iteration_lambdamart": int(booster_no_ips.best_iteration),
        "best_iteration_lambdamart_ips": int(booster_ips.best_iteration),
        "best_iteration_lambdamart_ips_no_dropout": int(booster_ips_no_dropout.best_iteration),
        "best_score_ndcg10_lambdamart": float(
            booster_no_ips.best_score["val"][f"ndcg@{model_cfg.eval_ndcg_at[0]}"]
        ),
        "best_score_ndcg10_lambdamart_ips": float(
            booster_ips.best_score["val"][f"ndcg@{model_cfg.eval_ndcg_at[0]}"]
        ),
        "n_ips_exposed_rows": int(p_expose_fit.notna().sum()),
        "ips_exposure_rate": float(p_expose_fit.notna().mean()),
        "mean_ips_weight": float(ips_weight_fit.mean()),
    }

    return LambdaMartArtifacts(
        booster_lambdamart=booster_no_ips,
        booster_lambdamart_ips=booster_ips,
        booster_lambdamart_ips_no_dropout=booster_ips_no_dropout,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        diagnostics=diagnostics,
    )


def save_boosters(artifacts: LambdaMartArtifacts, artifacts_dir: Path) -> dict[str, Path]:
    """Persist the 3 boosters as LightGBM's native `.txt` format (spec.md section
    14: `artifacts/model.txt` committed). `model.txt` (singular, per spec) is
    reserved for the PRIMARY system (8, LambdaMART+IPS) -- `model_no_ips.txt` and
    `model_ips_no_dropout.txt` are documented deviations (`docs/DATA_CARD.md`) since
    two systems, plus one ablation-only booster, are built, not one."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "lambdamart_ips": artifacts_dir / MODEL_IPS_FILENAME,
        "lambdamart": artifacts_dir / MODEL_NO_IPS_FILENAME,
        "lambdamart_ips_no_dropout": artifacts_dir / MODEL_IPS_NO_DROPOUT_FILENAME,
    }
    artifacts.booster_lambdamart_ips.save_model(str(paths["lambdamart_ips"]))
    artifacts.booster_lambdamart.save_model(str(paths["lambdamart"]))
    artifacts.booster_lambdamart_ips_no_dropout.save_model(str(paths["lambdamart_ips_no_dropout"]))
    return paths


def load_boosters(artifacts_dir: Path) -> dict[str, lgb.Booster]:
    """Load the 3 persisted boosters for `poi_rank.cli evaluate` -- scoring only,
    never retraining, so `evaluate` reruns are cheap and `train`'s own determinism
    is the only thing gating byte-identical `model.txt` across reruns."""
    return {
        "lambdamart_ips": lgb.Booster(model_file=str(artifacts_dir / MODEL_IPS_FILENAME)),
        "lambdamart": lgb.Booster(model_file=str(artifacts_dir / MODEL_NO_IPS_FILENAME)),
        "lambdamart_ips_no_dropout": lgb.Booster(
            model_file=str(artifacts_dir / MODEL_IPS_NO_DROPOUT_FILENAME)
        ),
    }


def run_train_lambdamart(
    data_dir: Path, artifacts_dir: Path, model_cfg: ModelConfig, feature_cfg: FeatureBuildConfig
) -> dict[str, Any]:
    """`poi_rank.cli train` entry point: builds the TRAIN ranking frame, fits
    systems 7/8 + the dropout-ablation booster, persists all 3 to `artifacts_dir`
    (spec.md section 14). Returns a summary dict (`paths`, `diagnostics`) for the
    CLI report and tests."""
    train_frame = load_train_ranking_frame(
        data_dir, feature_cfg.traveler_features.budget_target_price_level
    )
    interactions_train = pd.read_parquet(data_dir / INTERACTIONS_TRAIN_FILENAME)
    pois_df = pd.read_parquet(data_dir / POIS_PREPARED_FILENAME)

    artifacts = train_lambdamart_systems(
        train_frame, interactions_train, pois_df, model_cfg.lambdamart, model_cfg.seed
    )
    paths = save_boosters(artifacts, artifacts_dir)
    return {"paths": paths, "diagnostics": artifacts.diagnostics}


def load_and_score_holdout(
    artifacts_dir: Path, holdout_frame: pd.DataFrame
) -> dict[str, pd.Series]:
    """Load the 2 primary-table boosters (systems 7/8) and score the holdout
    evaluation frame -- `eval/run.py`'s entry point into this module. The
    dropout-ablation booster is scored separately, only against the new-POI cohort
    frame (`eval/new_poi_cohort.py`), never against the full holdout table."""
    numeric_columns = numeric_feature_columns(holdout_frame)
    categorical_columns = categorical_feature_columns(holdout_frame)
    boosters = load_boosters(artifacts_dir)
    return {
        "lambdamart": score_booster(
            boosters["lambdamart"], holdout_frame, numeric_columns, categorical_columns
        ),
        "lambdamart_ips": score_booster(
            boosters["lambdamart_ips"], holdout_frame, numeric_columns, categorical_columns
        ),
    }
