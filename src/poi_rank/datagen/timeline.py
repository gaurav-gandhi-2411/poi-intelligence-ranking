"""Shared global timeline: one `start`/`end`/`split` triple used for POI `created_at`,
trip dates, and impression timestamps alike, so temporal splitting is consistent
across every entity the DGP produces (spec.md: "no random splits anywhere").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from poi_rank.datagen.config import DatagenConfig


@dataclass(frozen=True)
class Timeline:
    start: pd.Timestamp
    end: pd.Timestamp
    split: pd.Timestamp  # train (< split) / holdout (>= split) boundary


def build_timeline(cfg: DatagenConfig) -> Timeline:
    """Compute the global start/end/split timestamps from `configs/datagen.yaml`."""
    start = pd.Timestamp(cfg.timeline.start_date)
    end = pd.Timestamp(cfg.timeline.end_date)
    span = end - start
    split = start + span * cfg.split.train_fraction
    return Timeline(start=start, end=end, split=split)
