"""Attach the explicit traveler x POI cross features (`xf_*`, experiment H) to a ranking frame.

`features/cross_features.py` (+ `features/compat_scores.py`) compute them; this module only loads
the dataset tables and constants they need. The retriever never calls it (it builds its own frame
with `build_ranking_frame`, which has no `xf_` columns), so candidate sets are unaffected.

`xf_reservation_fit` is deliberately not emitted: it is exactly 1.0 on every row of the committed
dataset (reservation lead times never exceed the assumed planning lead), so it carries no signal.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from poi_rank.features.compat_scores import CompatParams
from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.features.cross_features import CrossFeatureContext, add_cross_features

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCORING_CONFIG_PATH = REPO_ROOT / "configs" / "scoring.yaml"
PRETRIP_FILENAME = "interactions_pretrip.parquet"


def attach_cross_features(
    frame: pd.DataFrame,
    data_dir: Path,
    budget_target: BudgetTargetPriceLevel,
    scoring_config_path: Path | None = None,
    *,
    extra_trips_df: pd.DataFrame | None = None,
    extra_travelers_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return `frame` plus every `xf_*` column, built from the dataset in `data_dir`.

    `extra_trips_df` / `extra_travelers_df` add trips/travelers that are not in the dataset (the
    scenario layer synthetic profiles); they have no logged history, so their history features
    are 0.

    `POI_RANK_DISABLE_XF=1` returns `frame` unchanged: the ablation switch used only by
    `scripts/h_holdout_ablation.py` to measure the holdout effect of experiment H on its own.
    """
    if os.environ.get("POI_RANK_DISABLE_XF") == "1":
        return frame
    trips_df = pd.read_parquet(data_dir / "trips.parquet")
    travelers_df = pd.read_parquet(data_dir / "travelers.parquet")
    if extra_trips_df is not None:
        trips_df = pd.concat([trips_df, extra_trips_df], ignore_index=True)
    if extra_travelers_df is not None:
        travelers_df = pd.concat([travelers_df, extra_travelers_df], ignore_index=True)
    pois_df = pd.read_parquet(data_dir / "pois_prepared.parquet")
    poi_feat = pd.read_parquet(data_dir / "poi_features.parquet")
    train_ix = pd.read_parquet(data_dir / "interactions_train.parquet")
    parts = [train_ix]
    if (data_dir / PRETRIP_FILENAME).exists():
        parts.append(pd.read_parquet(data_dir / PRETRIP_FILENAME))
    history = pd.concat(parts, ignore_index=True)[
        ["traveler_id", "poi_id", "interaction_type", "label", "timestamp"]
    ]
    ctx = CrossFeatureContext(
        trips_df=trips_df,
        session_start=train_ix.groupby("trip_id")["timestamp"].min(),
        poi_localness_reference=poi_feat["num_localness"].to_numpy(dtype=float),
        history=history,
        budget_target_price_level=budget_target,
        travelers_df=travelers_df,
        pois_df=pois_df,
        compat=CompatParams.from_scoring_yaml(scoring_config_path or DEFAULT_SCORING_CONFIG_PATH),
    )
    return add_cross_features(frame, ctx)
