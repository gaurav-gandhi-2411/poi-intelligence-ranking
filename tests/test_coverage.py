"""`eval/coverage.py` tests: hand-computed Gini/entropy examples (spec.md section
11.3)."""

from __future__ import annotations

import numpy as np
import pytest

from poi_rank.eval.coverage import (
    catalog_coverage_at_10,
    catalog_coverage_by_destination,
    coverage_report,
    gini_coefficient,
    recommendation_frequency,
    shannon_entropy,
)


def test_recommendation_frequency_includes_zero_frequency_pois() -> None:
    lists_by_trip = {"T1": ["A", "B"], "T2": ["A"]}
    freq = recommendation_frequency(lists_by_trip, {"A", "B", "C"})
    assert freq == {"A": 2, "B": 1, "C": 0}


def test_recommendation_frequency_key_order_is_sorted_not_set_iteration_order() -> None:
    """Regression: `coverage_report` builds `freq_arr = np.array(list(freq_map
    .values()))` and feeds it straight into `shannon_entropy`'s floating-point
    summation. If `recommendation_frequency` iterated the raw `catalog_poi_ids`
    `set` (hash-order-dependent, varies with `PYTHONHASHSEED` across process
    invocations) instead of `sorted(catalog_poi_ids)`, `entropy_bits` would get a
    different summation order -- and therefore a different float -- on every run,
    which is exactly what caused a genuine `results/metrics.json` byte-determinism
    failure across two direct `uv run python -m poi_rank.cli evaluate` invocations
    (docs/DATA_CARD.md). Two differently-constructed-but-equal sets must produce
    identical dict key order."""
    catalog_a: set[str] = {"C", "A", "B"}
    catalog_b: set[str] = set()
    for pid in ["B", "C", "A"]:  # different insertion order, same final set
        catalog_b.add(pid)
    assert catalog_a == catalog_b  # same set, by value

    lists_by_trip = {"T1": ["A"]}
    freq_a = recommendation_frequency(lists_by_trip, catalog_a)
    freq_b = recommendation_frequency(lists_by_trip, catalog_b)
    assert list(freq_a.keys()) == ["A", "B", "C"]
    assert list(freq_a.keys()) == list(freq_b.keys())


def test_catalog_coverage_at_10_hand_computed() -> None:
    lists_by_trip = {"T1": ["A", "B"], "T2": ["A"]}
    # {A,B} recommended, catalog has 4 POIs -> coverage = 2/4 = 0.5
    assert catalog_coverage_at_10(lists_by_trip, {"A", "B", "C", "D"}) == pytest.approx(0.5)


def test_catalog_coverage_at_10_empty_catalog_is_zero() -> None:
    assert catalog_coverage_at_10({}, set()) == 0.0


def test_catalog_coverage_by_destination_hand_computed() -> None:
    lists_by_trip = {"T1": ["A"], "T2": ["C"]}
    trip_destination = {"T1": "seoul", "T2": "kyoto"}
    poi_destination = {"A": "seoul", "B": "seoul", "C": "kyoto", "D": "kyoto"}
    result = catalog_coverage_by_destination(lists_by_trip, trip_destination, poi_destination)
    assert result["seoul"] == pytest.approx(0.5)  # {A} / {A,B}
    assert result["kyoto"] == pytest.approx(0.5)  # {C} / {C,D}


def test_gini_coefficient_hand_computed() -> None:
    # sorted freq = [0, 0, 1, 3], n=4, sum=4.
    # G = (2*sum(rank*freq) - (n+1)*sum) / (n*sum)
    #   = (2*(1*0+2*0+3*1+4*3) - 5*4) / (4*4) = (2*15 - 20) / 16 = 10/16 = 0.625
    freq = np.array([0.0, 0.0, 1.0, 3.0])
    assert gini_coefficient(freq) == pytest.approx(0.625)


def test_gini_coefficient_perfect_equality_is_zero() -> None:
    freq = np.array([2.0, 2.0, 2.0, 2.0])
    assert gini_coefficient(freq) == pytest.approx(0.0)


def test_gini_coefficient_all_zero_is_zero_not_nan() -> None:
    freq = np.array([0.0, 0.0, 0.0])
    assert gini_coefficient(freq) == pytest.approx(0.0)


def test_gini_coefficient_maximal_concentration() -> None:
    # All mass on one item -> Gini approaches (n-1)/n for n items.
    freq = np.array([0.0, 0.0, 0.0, 4.0])
    assert gini_coefficient(freq) == pytest.approx(0.75)


def test_shannon_entropy_hand_computed() -> None:
    # freq = [1, 3] -> probs = [0.25, 0.75]
    # H = -(0.25*log2(0.25) + 0.75*log2(0.75)) = 0.8112781...
    freq = np.array([1.0, 3.0])
    assert shannon_entropy(freq, base=2.0) == pytest.approx(0.8112781, abs=1e-6)


def test_shannon_entropy_uniform_distribution_is_log2_n() -> None:
    freq = np.array([1.0, 1.0, 1.0, 1.0])
    assert shannon_entropy(freq, base=2.0) == pytest.approx(2.0)  # log2(4) = 2 bits


def test_shannon_entropy_degenerate_single_item_is_zero() -> None:
    freq = np.array([0.0, 0.0, 5.0])
    assert shannon_entropy(freq, base=2.0) == pytest.approx(0.0)


def test_shannon_entropy_all_zero_is_zero_not_nan() -> None:
    assert shannon_entropy(np.array([0.0, 0.0]), base=2.0) == pytest.approx(0.0)


def test_coverage_report_end_to_end() -> None:
    lists_by_trip = {"T1": ["A", "B"], "T2": ["A"]}
    catalog = {"A", "B", "C", "D"}
    trip_destination = {"T1": "seoul", "T2": "seoul"}
    poi_destination = {"A": "seoul", "B": "seoul", "C": "seoul", "D": "seoul"}
    report = coverage_report(lists_by_trip, catalog, trip_destination, poi_destination)
    assert report.catalog_coverage_at_10 == pytest.approx(0.5)
    assert report.n_catalog_pois == 4
    assert report.n_pois_ever_recommended == 2
    d = report.to_dict()
    assert set(d) == {
        "catalog_coverage_at_10",
        "catalog_coverage_by_destination",
        "gini",
        "entropy_bits",
        "n_catalog_pois",
        "n_pois_ever_recommended",
    }
