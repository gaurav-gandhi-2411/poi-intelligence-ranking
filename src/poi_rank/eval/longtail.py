"""Long-tail / local-discovery metrics (spec.md section 11.4): share of top-10
recommendations in the bottom-50%-within-destination-popularity stratum, AND
long-tail PRECISION (among recommended long-tail POIs, what fraction are genuinely
relevant per the unbiased holdout) -- reported together, per spec.md's own
instruction ("coverage without precision is just noise injection"). Targets:
long-tail share >= 0.25 AND long-tail precision >= 0.5.

**Popularity-stratum cutoff and "relevant" ground truth are reused, not
re-derived**: the `pop_pct < cutoff` long-tail definition is the SAME
`long_tail_pop_pct_cutoff` (0.5) `configs/eval.yaml` already defines and
`candidates/recall_metrics.py` already uses for `candidate_recall@250`'s long-tail
stratum -- consistency across every long-tail metric in this project, not two
independently-tuned thresholds. "Genuinely relevant" is read as `label >= 1` against
`interactions_holdout_random.parquet` (the unbiased holdout, the SAME ground truth
`eval/metrics.py`'s own NDCG/Recall/MAP/MRR already use) -- spec.md section 11.4
explicitly permits "unbiased holdout / oracle utility"; this module uses the
holdout-label reading for consistency with every other precision-style metric in
this codebase, never the oracle (this module reads no oracle-only data).
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_LONGTAIL_SHARE = 0.25
TARGET_LONGTAIL_PRECISION = 0.5


@dataclass(frozen=True)
class LongTailReport:
    n_total_recommended: int
    n_longtail_recommended: int
    share: float
    n_longtail_relevant: int
    precision: float | None  # None if n_longtail_recommended == 0 (undefined)

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "n_total_recommended": self.n_total_recommended,
            "n_longtail_recommended": self.n_longtail_recommended,
            "share": self.share,
            "n_longtail_relevant": self.n_longtail_relevant,
            "precision": self.precision,
            "share_target_met": bool(self.share >= TARGET_LONGTAIL_SHARE),
            "precision_target_met": bool(
                self.precision is not None and self.precision >= TARGET_LONGTAIL_PRECISION
            ),
        }


def longtail_share_and_precision(
    lists_by_trip: dict[str, list[str]],
    pop_pct_by_poi: dict[str, float],
    cutoff: float,
    label_by_trip_poi: dict[tuple[str, str], int],
) -> LongTailReport:
    """`lists_by_trip`: `{trip_id: [poi_id, ...]}` top-10 recommendation lists
    (module docstring). `pop_pct_by_poi`: within-destination popularity percentile
    per POI (`pois_prepared.parquet`'s `pop_pct`). `label_by_trip_poi`: graded
    0-3 relevance label per `(trip_id, poi_id)` pair from the unbiased holdout,
    missing pairs treated as label 0 (the same "no logged interaction -> true
    negative" convention `models/ranking_data.py` already establishes for this
    exact evaluation population)."""
    total = 0
    longtail = 0
    longtail_relevant = 0
    for trip_id, recs in lists_by_trip.items():
        for poi_id in recs:
            total += 1
            pop_pct = pop_pct_by_poi.get(poi_id)
            if pop_pct is not None and pop_pct < cutoff:
                longtail += 1
                if label_by_trip_poi.get((trip_id, poi_id), 0) >= 1:
                    longtail_relevant += 1

    share = longtail / total if total > 0 else 0.0
    precision = longtail_relevant / longtail if longtail > 0 else None
    return LongTailReport(
        n_total_recommended=total,
        n_longtail_recommended=longtail,
        share=share,
        n_longtail_relevant=longtail_relevant,
        precision=precision,
    )
