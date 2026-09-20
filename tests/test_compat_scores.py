"""`features/compat_scores.py` must reproduce `scoring/compatibility.py` value-for-value (two
implementations of the same formulas, because `features/` may not import `scoring/`; this test is
what keeps them from drifting). Runs on a sample of the committed dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poi_rank.features import compat_scores as cs
from poi_rank.features.config import BudgetTargetPriceLevel
from poi_rank.scoring import compatibility as compat
from poi_rank.scoring.config import ScoringConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
SCORING = ROOT / "configs" / "scoring.yaml"
BUDGET = BudgetTargetPriceLevel(low=1.3, medium=2.5, high=3.7)

pytestmark = pytest.mark.skipif(not (DATA / "candidates.parquet").exists(), reason="dataset absent")


@pytest.fixture(scope="module")
def contexts() -> tuple[pd.DataFrame, pd.DataFrame, ScoringConfig, cs.CompatParams]:
    trips = pd.read_parquet(DATA / "trips.parquet")
    travelers = pd.read_parquet(DATA / "travelers.parquet")
    pois = pd.read_parquet(DATA / "pois_prepared.parquet")
    cands = pd.read_parquet(DATA / "candidates.parquet")
    keep = sorted(cands["trip_id"].unique())[::40][:25]
    pairs = cands.loc[cands["trip_id"].isin(keep), ["trip_id", "poi_id"]].reset_index(drop=True)
    ref = compat.build_compatibility_context(pairs, set(keep), trips, travelers, pois)
    ref = pairs.merge(ref, on=["trip_id", "poi_id"], how="left")
    mine = cs.build_context(pairs, trips, travelers, pois)
    return ref, mine, ScoringConfig.from_yaml(SCORING), cs.CompatParams.from_scoring_yaml(SCORING)


def test_budget_fit_matches_scoring_layer(contexts: tuple) -> None:  # type: ignore[type-arg]
    ref, mine, scfg, p = contexts
    expected = compat.budget_fit_score(ref, BUDGET, scfg.compatibility.budget_fit)
    np.testing.assert_allclose(cs.budget_fit(mine, BUDGET, p), expected, rtol=0, atol=1e-12)


def test_mobility_fit_and_travel_time_match_scoring_layer(contexts: tuple) -> None:  # type: ignore[type-arg]
    ref, mine, scfg, p = contexts
    expected = compat.mobility_fit_score(ref, scfg.compatibility.mobility_fit)
    np.testing.assert_allclose(cs.mobility_fit(mine, p), expected, rtol=0, atol=1e-12)


def test_hours_fit_matches_scoring_layer(contexts: tuple) -> None:  # type: ignore[type-arg]
    ref, mine, scfg, p = contexts
    expected = compat.hours_fit_score(ref, scfg.compatibility.hours_fit)
    np.testing.assert_allclose(cs.hours_fit(mine, p), expected, rtol=0, atol=1e-12)


def test_party_fit_matches_scoring_layer(contexts: tuple) -> None:  # type: ignore[type-arg]
    ref, mine, scfg, p = contexts
    expected = compat.party_fit_score(ref, scfg.compatibility.party_fit)
    np.testing.assert_allclose(cs.party_fit(mine, p), expected, rtol=0, atol=1e-12)


def test_duration_fit_matches_scoring_layer(contexts: tuple) -> None:  # type: ignore[type-arg]
    ref, mine, scfg, p = contexts
    expected = compat.duration_fit_score(ref, scfg.compatibility.duration_fit)
    np.testing.assert_allclose(cs.duration_fit(mine, p), expected, rtol=0, atol=1e-12)
