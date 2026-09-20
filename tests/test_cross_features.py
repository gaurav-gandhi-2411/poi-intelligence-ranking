"""Unit tests for `features/cross_features.py` (experiment H): value semantics of the cross
features, the no-own-session-leak discipline of the history features, and the firewall."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poi_rank.features import cross_features as xf
from poi_rank.features.config import BudgetTargetPriceLevel

BUDGET = BudgetTargetPriceLevel(low=1.3, medium=2.5, high=3.7)


def _frame() -> pd.DataFrame:
    emb = {f"text_emb_{i:02d}": v for i, v in enumerate([1.0, 0.0])}
    rows = []
    for trip, pref, budget in [("T1", -1.0, "medium"), ("T2", 1.0, "low")]:
        for poi, loc, price, cat, tags in [
            ("P1", 1.5, 1.0, "restaurant", ["local", "cafe"]),
            ("P2", -1.5, 4.0, "museum", ["touristy"]),
        ]:
            rows.append(
                {
                    "trip_id": trip,
                    "traveler_id": "U1",
                    "poi_id": poi,
                    "explicit_touristiness_pref": pref,
                    "num_localness": loc,
                    "num_pop_pct": 0.5,
                    "num_price_level": price,
                    "budget": budget,
                    "interests": ["restaurant", "local"],
                    "poi_category_raw": cat,
                    "poi_tags_raw": tags,
                    **(emb if poi == "P1" else {"text_emb_00": 0.0, "text_emb_01": 1.0}),
                }
            )
    return pd.DataFrame(rows)


def _ctx(session_start: dict[str, pd.Timestamp], history: pd.DataFrame) -> xf.CrossFeatureContext:
    return xf.CrossFeatureContext(
        trips_df=pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "start_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-12-01")],
            }
        ),
        session_start=pd.Series(session_start, dtype="datetime64[ns]"),
        poi_localness_reference=np.linspace(-2.0, 2.0, 101),
        history=history,
        budget_target_price_level=BUDGET,
    )


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["traveler_id", "poi_id", "interaction_type", "label", "timestamp"]
    ).astype({"timestamp": "datetime64[ns]"})


def test_localness_alignment_sign_follows_stated_preference() -> None:
    out = xf.add_cross_features(_frame(), _ctx({}, _empty_history())).set_index(
        ["trip_id", "poi_id"]
    )
    # T1 wants LESS touristy (pref -1): local P1 aligns positively, touristy P2 negatively.
    assert out.loc[("T1", "P1"), "xf_loc_align"] > 0 > out.loc[("T1", "P2"), "xf_loc_align"]
    # T2 wants MORE touristy (pref +1): the reverse.
    assert out.loc[("T2", "P2"), "xf_loc_align"] > 0 > out.loc[("T2", "P1"), "xf_loc_align"]
    assert out.loc[("T1", "P1"), "xf_loc_gap"] < out.loc[("T1", "P2"), "xf_loc_gap"]


def test_price_over_and_under_are_asymmetric_pair_of_the_signed_gap() -> None:
    out = xf.add_cross_features(_frame(), _ctx({}, _empty_history())).set_index(
        ["trip_id", "poi_id"]
    )
    signed = out["xf_price_signed"]
    assert (out["xf_price_over"] - out["xf_price_under"]).to_numpy() == pytest.approx(
        signed.to_numpy()
    )
    assert out.loc[("T1", "P2"), "xf_price_over"] > 0 == out.loc[("T1", "P2"), "xf_price_under"]


def test_interest_hits_count_stated_interests_against_tags_and_category() -> None:
    out = xf.add_cross_features(_frame(), _ctx({}, _empty_history())).set_index(
        ["trip_id", "poi_id"]
    )
    assert out.loc[("T1", "P1"), "xf_interest_tag_hits"] == 1.0  # "local"
    assert out.loc[("T1", "P1"), "xf_interest_cat_hit"] == 1.0  # category "restaurant"
    assert out.loc[("T1", "P2"), "xf_interest_cat_hit"] == 0.0


def test_prior_engagement_ignores_the_trips_own_session() -> None:
    """A trip session is dated BEFORE its start_date: history must be cut at the session start,
    so the labelled session never feeds its own feature (the leak the first version had)."""
    hist = pd.DataFrame(
        [
            # earlier-history engagement with P1 (before T1 session) -> visible
            ["U1", "P1", "visit", 3, pd.Timestamp("2025-02-01")],
            # T1 own session engagement with P1 (before T1 start_date, after session start)
            ["U1", "P1", "visit", 3, pd.Timestamp("2025-05-25")],
        ],
        columns=["traveler_id", "poi_id", "interaction_type", "label", "timestamp"],
    )
    ctx = _ctx({"T1": pd.Timestamp("2025-05-20")}, hist)  # T1 session starts 2025-05-20
    out = xf.add_cross_features(_frame(), ctx).set_index(["trip_id", "poi_id"])
    one_visible = 0.5 ** ((pd.Timestamp("2025-05-20") - pd.Timestamp("2025-02-01")).days / 90.0)
    assert out.loc[("T1", "P1"), "xf_prior_poi_engaged"] == pytest.approx(one_visible, rel=1e-3)
    # T2 (no session rows, cutoff = start_date 2025-12-01) sees BOTH earlier engagements, each
    # decayed from ITS cutoff.
    cutoff = pd.Timestamp("2025-12-01")
    both = sum(0.5 ** ((cutoff - t).days / 90.0) for t in hist["timestamp"])
    assert out.loc[("T2", "P1"), "xf_prior_poi_engaged"] == pytest.approx(both, rel=1e-3)


def test_dismissed_cosine_is_zero_without_history_and_positive_toward_dismissed_taste() -> None:
    none = xf.add_cross_features(_frame(), _ctx({}, _empty_history()))
    assert (none["xf_cos_dismissed"] == 0.0).all()
    hist = pd.DataFrame(
        [["U1", "P1", "dismiss", 0, pd.Timestamp("2025-02-01")]],
        columns=["traveler_id", "poi_id", "interaction_type", "label", "timestamp"],
    )
    out = xf.add_cross_features(_frame(), _ctx({}, hist)).set_index(["trip_id", "poi_id"])
    assert out.loc[("T1", "P1"), "xf_cos_dismissed"] == pytest.approx(1.0)  # same embedding
    assert out.loc[("T1", "P2"), "xf_cos_dismissed"] == pytest.approx(0.0)  # orthogonal


def test_cross_feature_module_never_touches_the_oracle_or_datagen() -> None:
    src = Path(xf.__file__).read_text(encoding="utf-8")
    for forbidden in ("datagen", "_oracle", "oracle_export", "latent_"):
        assert forbidden not in src, forbidden
