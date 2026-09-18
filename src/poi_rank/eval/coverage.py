"""Coverage metrics (spec.md section 11.3): catalog coverage@10 across all trips,
Gini coefficient and Shannon entropy of the recommendation-frequency distribution --
compared against the popularity baseline's own coverage/Gini/entropy (`eval/run.py`
calls this module twice: once for the primary system's top-10s, once for baseline
2's, per spec.md's explicit "compared against popularity baseline" requirement).
This IS the measurable answer to "does the ranker collapse to a popularity
monoculture" -- a system recommending the same handful of popular POIs to everyone
has LOW coverage and HIGH Gini; one that spreads recommendations across the catalog
has HIGH coverage and LOW Gini.

**Frequency distribution includes every catalog POI, including ones recommended
zero times** -- not just the POIs that happen to appear at least once. A
zero-frequency POI is real information about concentration (Gini/entropy computed
only over the POIs that DID appear would systematically understate concentration by
construction, since it would silently drop every POI a monoculture system never
recommends at all -- exactly the failure mode this metric exists to catch).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def recommendation_frequency(
    lists_by_trip: dict[str, list[str]], catalog_poi_ids: set[str]
) -> dict[str, int]:
    """`{poi_id: n_times_recommended}` for EVERY `poi_id` in `catalog_poi_ids`
    (module docstring) -- zero for a POI never recommended in any trip's top-10.

    Iterates `sorted(catalog_poi_ids)`, never the raw `set` directly: Python's
    `set` iteration order for `str` depends on per-process hash randomization
    (`PYTHONHASHSEED`), so building this dict (and therefore `coverage_report`'s
    downstream `freq_arr = np.array(list(freq_map.values()))`) straight off the
    set gives `shannon_entropy`'s floating-point summation a different term order
    on every process invocation that doesn't happen to share a `PYTHONHASHSEED` --
    caught as a genuine (if tiny, ~2e-15) `results/metrics.json` byte-determinism
    failure across two direct `uv run python -m poi_rank.cli evaluate` invocations
    (the Makefile's `export PYTHONHASHSEED := 0` masks this when running via
    `make`, but the project's own reproducibility contract must hold for a direct
    invocation too -- the exact class of gap already flagged in docs/DATA_CARD.md
    for `candidates/channels.py`'s SHA256-seeded RNG, fixed the same way there:
    never let hash-randomization-dependent ordering reach a floating-point
    reduction)."""
    counts = Counter(pid for recs in lists_by_trip.values() for pid in recs)
    return {pid: counts.get(pid, 0) for pid in sorted(catalog_poi_ids)}


def catalog_coverage_at_10(lists_by_trip: dict[str, list[str]], catalog_poi_ids: set[str]) -> float:
    """Fraction of `catalog_poi_ids` appearing in AT LEAST ONE trip's top-10."""
    if not catalog_poi_ids:
        return 0.0
    recommended: set[str] = {pid for recs in lists_by_trip.values() for pid in recs}
    return len(recommended & catalog_poi_ids) / len(catalog_poi_ids)


def catalog_coverage_by_destination(
    lists_by_trip: dict[str, list[str]],
    trip_destination: dict[str, str],
    poi_destination: dict[str, str],
) -> dict[str, float]:
    """Per-destination catalog coverage@10 -- a trip's top-10 only ever draws from
    its own destination's catalog (`candidates/` is destination-scoped by
    construction), so this restricts both the "recommended" set and the "catalog"
    denominator to one destination at a time."""
    catalog_by_dest: dict[str, set[str]] = {}
    for pid, dest in poi_destination.items():
        catalog_by_dest.setdefault(dest, set()).add(pid)

    recs_by_dest: dict[str, set[str]] = {}
    for trip_id, recs in lists_by_trip.items():
        trip_dest = trip_destination.get(trip_id)
        if trip_dest is None:
            continue
        recs_by_dest.setdefault(trip_dest, set()).update(recs)

    out: dict[str, float] = {}
    for dest, catalog in catalog_by_dest.items():
        recommended = recs_by_dest.get(dest, set())
        out[dest] = len(recommended & catalog) / len(catalog) if catalog else 0.0
    return out


def gini_coefficient(frequencies: FloatArray) -> float:
    """Standard Gini coefficient over a non-negative frequency array (0 = perfectly
    equal distribution, 1 = maximally concentrated in one item). All-zero input
    (degenerate: nothing ever recommended) is defined as `0.0` (no inequality to
    measure), not `NaN`/division-by-zero."""
    if len(frequencies) == 0:
        return 0.0
    total = float(frequencies.sum())
    if total <= 0.0:
        return 0.0
    sorted_freq = np.sort(frequencies)
    n = len(sorted_freq)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * sorted_freq) - (n + 1.0) * total) / (n * total))


def shannon_entropy(frequencies: FloatArray, base: float = 2.0) -> float:
    """Shannon entropy (bits, `base=2` default) of the frequency distribution
    normalized to probabilities. Zero-frequency items contribute `0` (the standard
    `0 * log(0) := 0` convention), never `NaN`/`-inf`."""
    total = float(frequencies.sum())
    if total <= 0.0:
        return 0.0
    probs = frequencies[frequencies > 0] / total
    return float(-np.sum(probs * (np.log(probs) / np.log(base))))


@dataclass(frozen=True)
class CoverageReport:
    catalog_coverage_at_10: float
    catalog_coverage_by_destination: dict[str, float]
    gini: float
    entropy_bits: float
    n_catalog_pois: int
    n_pois_ever_recommended: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_coverage_at_10": self.catalog_coverage_at_10,
            "catalog_coverage_by_destination": self.catalog_coverage_by_destination,
            "gini": self.gini,
            "entropy_bits": self.entropy_bits,
            "n_catalog_pois": self.n_catalog_pois,
            "n_pois_ever_recommended": self.n_pois_ever_recommended,
        }


def coverage_report(
    lists_by_trip: dict[str, list[str]],
    catalog_poi_ids: set[str],
    trip_destination: dict[str, str],
    poi_destination: dict[str, str],
) -> CoverageReport:
    """Full coverage report (module docstring) for one system's top-10 lists."""
    freq_map = recommendation_frequency(lists_by_trip, catalog_poi_ids)
    freq_arr = np.array(list(freq_map.values()), dtype=np.float64)
    return CoverageReport(
        catalog_coverage_at_10=catalog_coverage_at_10(lists_by_trip, catalog_poi_ids),
        catalog_coverage_by_destination=catalog_coverage_by_destination(
            lists_by_trip, trip_destination, poi_destination
        ),
        gini=gini_coefficient(freq_arr),
        entropy_bits=shannon_entropy(freq_arr, base=2.0),
        n_catalog_pois=len(catalog_poi_ids),
        n_pois_ever_recommended=int((freq_arr > 0).sum()),
    )
