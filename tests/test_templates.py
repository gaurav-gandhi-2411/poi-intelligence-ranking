"""`explain/templates.py`: deterministic template rendering (spec.md section 10) --
`top_signals` ranking, per-group renderers, cold-start gating, and the combined
`explanation` assembly. Pure unit tests against hand-built contexts (no LLM, no
real-data dependency needed for these -- `test_explain_integration.py` covers the
real-fixture-chain end-to-end path)."""

from __future__ import annotations

from poi_rank.explain.shap_groups import FEATURE_GROUPS, NOVELTY
from poi_rank.explain.templates import (
    ExplanationContext,
    build_explanation,
    top_signals,
)

_BASE_CONTRIBUTIONS: dict[str, float] = dict.fromkeys(FEATURE_GROUPS, 0.0)


def _contributions(**overrides: float) -> dict[str, float]:
    out = dict(_BASE_CONTRIBUTIONS)
    out.update(overrides)
    return out


def _context(**overrides: object) -> ExplanationContext:
    base: dict[str, object] = {
        "poi_category": "restaurant",
        "destination": "seoul",
        "budget": "medium",
        "party_type": "solo",
        "mobility": "public_transport",
        "pop_pct": 0.22,
        "rating_shrunk": 4.3,
        "review_count": 128.0,
        "budget_fit": 0.95,
        "mobility_fit": 0.88,
        "hours_fit": 1.0,
        "party_fit": 0.9,
        "travel_min": 12.0,
        "implicit_interaction_count": 0.0,
        "interests": frozenset({"local_food"}),
        "poi_terms": frozenset({"restaurant", "local_food"}),
        "price_level": 2.0,
        "budget_target_price_level": 2.5,
    }
    base.update(overrides)
    return ExplanationContext(**base)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------------
# top_signals
# -----------------------------------------------------------------------------------


def test_top_signals_sorted_by_contribution_descending() -> None:
    contributions = _contributions(interest_match=0.21, implicit_taste=0.17, localness_fit=0.11)
    signals = top_signals(contributions, top_n=3)
    groups = [s["feature_group"] for s in signals]
    assert groups == ["interest_match", "implicit_taste", "localness_fit"]
    assert [s["contribution"] for s in signals] == [0.21, 0.17, 0.11]


def test_top_signals_ties_broken_by_group_name_ascending() -> None:
    contributions = _contributions(popularity=0.3, geo=0.3, hours=0.3)
    signals = top_signals(contributions, top_n=3)
    assert [s["feature_group"] for s in signals] == ["geo", "hours", "popularity"]


def test_top_signals_excludes_novelty() -> None:
    contributions = _contributions(novelty=999.0, interest_match=0.01)
    signals = top_signals(contributions, top_n=3)
    assert all(s["feature_group"] != NOVELTY for s in signals)


# -----------------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------------


def test_templates_are_deterministic() -> None:
    contributions = _contributions(interest_match=0.21, implicit_taste=0.17, localness_fit=0.11)
    compat_breakdown = {
        "budget_fit": 0.95,
        "mobility_fit": 0.88,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.82,
    }
    ctx = _context(implicit_interaction_count=3.0)
    first = build_explanation(contributions, compat_breakdown, ctx)
    second = build_explanation(contributions, compat_breakdown, ctx)
    assert first == second


# -----------------------------------------------------------------------------------
# Cold-start gating (task instruction)
# -----------------------------------------------------------------------------------


def test_implicit_taste_line_never_claims_past_trips_for_cold_start_traveler() -> None:
    contributions = _contributions(implicit_taste=0.5)
    compat_breakdown = {
        "budget_fit": 0.95,
        "mobility_fit": 0.88,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.82,
    }
    ctx = _context(implicit_interaction_count=0.0)
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert not any("past trips" in line for line in lines)


def test_implicit_taste_line_mentions_past_trips_for_non_cold_start_traveler() -> None:
    contributions = _contributions(implicit_taste=0.5)
    compat_breakdown = {
        "budget_fit": 0.95,
        "mobility_fit": 0.88,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.82,
    }
    ctx = _context(implicit_interaction_count=5.0)
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert any("past trips" in line for line in lines)


# -----------------------------------------------------------------------------------
# Binding vs. non-binding compatibility line selection
# -----------------------------------------------------------------------------------


def test_weak_compatibility_line_appears_when_a_subscore_is_clearly_weakest() -> None:
    contributions = _contributions(interest_match=0.1)
    compat_breakdown = {
        "budget_fit": 0.95,
        "mobility_fit": 0.30,  # clearly weakest, well below 0.85
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.95,
    }
    ctx = _context(mobility_fit=0.30, travel_min=55.0)
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert any("trek" in line for line in lines)


def test_weak_budget_fit_line_says_above_when_poi_pricier_than_target() -> None:
    contributions = _contributions(interest_match=0.1)
    compat_breakdown = {
        "budget_fit": 0.5,  # clearly weakest
        "mobility_fit": 0.9,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.95,
    }
    ctx = _context(budget_fit=0.5, price_level=3.5, budget_target_price_level=1.5)
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert any("Priced above" in line for line in lines)
    assert not any("budget-friendly" in line for line in lines)


def test_weak_budget_fit_line_says_below_when_poi_cheaper_than_target() -> None:
    # Regression: a prior version of `_weak_compat_line` always said "Priced above"
    # regardless of the actual gap direction, which was factually wrong on 535/766
    # (70%) of the real "weak budget fit" explanation lines this pipeline generated
    # on the committed dataset -- a low `budget_fit` can equally come from the POI
    # being CHEAPER than the traveler's target (asymmetric penalty still applies,
    # just less steeply), not only from it being pricier.
    contributions = _contributions(interest_match=0.1)
    compat_breakdown = {
        "budget_fit": 0.5,  # clearly weakest
        "mobility_fit": 0.9,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.95,
    }
    ctx = _context(budget_fit=0.5, price_level=2.0, budget_target_price_level=3.5)
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert any("budget-friendly" in line for line in lines)
    assert not any("Priced above" in line for line in lines)


def test_confirmation_lines_appear_when_compatibility_uniformly_high() -> None:
    contributions = _contributions(interest_match=0.1)
    compat_breakdown = {
        "budget_fit": 0.95,
        "mobility_fit": 0.88,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.95,
    }
    ctx = _context()
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert any("budget" in line.lower() for line in lines)
    assert any("min from your stay" in line for line in lines)


# -----------------------------------------------------------------------------------
# interest_match: None when no genuine overlap
# -----------------------------------------------------------------------------------


def test_interest_match_line_omitted_when_no_textual_overlap() -> None:
    contributions = _contributions(interest_match=0.3)
    compat_breakdown = {
        "budget_fit": 0.95,
        "mobility_fit": 0.88,
        "hours_fit": 1.0,
        "reservation_fit": 1.0,
        "party_fit": 0.9,
        "duration_fit": 0.95,
    }
    ctx = _context(
        interests=frozenset({"nightlife"}), poi_terms=frozenset({"museum", "historic_site"})
    )
    lines = build_explanation(contributions, compat_breakdown, ctx)
    assert not any("Strong match" in line for line in lines)
