"""Experiment L1/L2: the `pref_align` factor, its explanation line and the selection rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poi_rank.eval.l_select import (
    ENTROPY_FLOOR_RATIO,
    NDCG_TOLERANCE,
    jaccard,
    pref_align_constants,
    select_config,
)
from poi_rank.explain.templates import ExplanationContext, build_explanation
from poi_rank.scoring.config import ScoringConfig
from poi_rank.scoring.utility import compute_utility, pref_align

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pref_align_is_neutral_without_a_stated_preference() -> None:
    loc = np.array([-2.0, 0.0, 3.0])
    out = pref_align(loc, np.zeros(3), center=0.0, scale=0.5)
    assert np.allclose(out, 0.5)  # a constant cannot reorder that traveler's candidates


def test_pref_align_sign_follows_the_resolved_dgp_convention() -> None:
    loc = np.array([1.0, -1.0])  # a local and a touristy POI (centered)
    wants_local = pref_align(loc, np.array([-0.8, -0.8]), center=0.0, scale=0.5)
    wants_famous = pref_align(loc, np.array([0.8, 0.8]), center=0.0, scale=0.5)
    assert wants_local[0] > 0.5 > wants_local[1]  # negative pref = prefers local
    assert wants_famous[1] > 0.5 > wants_famous[0]
    assert np.allclose(wants_local, wants_famous[::-1])  # exact mirror


def test_gamma_zero_is_exactly_the_two_factor_utility() -> None:
    gate = np.array([1.0, 1.0, 0.0])
    rel = np.array([0.2, 0.9, 0.9])
    comp = np.array([0.5, 0.7, 0.9])
    base = compute_utility(gate, rel, comp, alpha=1.0, beta=0.7)
    with_factor = compute_utility(gate, rel, comp, 1.0, 0.7, 0.0, np.array([0.1, 0.9, 0.5]))
    assert np.array_equal(base, with_factor)


def test_gamma_scales_utility_by_the_factor_and_never_revives_a_gated_poi() -> None:
    gate = np.array([1.0, 0.0])
    rel = np.array([0.5, 0.5])
    comp = np.array([1.0, 1.0])
    factor = np.array([0.8, 0.8])
    out = compute_utility(gate, rel, comp, 1.0, 0.7, 2.0, factor)
    assert out[0] == pytest.approx(0.5 * 0.8**2)
    assert out[1] == 0.0


def test_nonzero_gamma_without_the_factor_is_an_error() -> None:
    with pytest.raises(ValueError, match="pref_align_factor"):
        compute_utility(np.ones(1), np.ones(1), np.ones(1), 1.0, 0.7, gamma=1.0)


def _ctx(pref_align_value: float, pref: float) -> ExplanationContext:
    return ExplanationContext(
        poi_category="restaurant",
        destination="seoul",
        budget="medium",
        party_type="couple",
        mobility="public_transport",
        pop_pct=0.2,
        rating_shrunk=4.2,
        review_count=120.0,
        budget_fit=0.95,
        mobility_fit=0.95,
        hours_fit=0.95,
        party_fit=1.0,
        travel_min=12.0,
        implicit_interaction_count=0.0,
        interests=frozenset({"foodie"}),
        poi_terms=frozenset({"restaurant"}),
        price_level=2.0,
        budget_target_price_level=2.0,
        pref_align=pref_align_value,
        touristiness_pref=pref,
    )


_BREAKDOWN = {
    "budget_fit": 0.95,
    "mobility_fit": 0.95,
    "hours_fit": 0.95,
    "reservation_fit": 1.0,
    "party_fit": 1.0,
    "duration_fit": 1.0,
    "pref_align": 0.05,  # lowest value: must NOT be picked as the "weakest compatibility" term
}


def test_explanation_states_the_stated_preference_only_when_the_poi_is_on_its_side() -> None:
    local = build_explanation({"popularity": 0.1}, _BREAKDOWN, _ctx(0.8, -0.6))
    assert any("Lower tourist concentration than comparable POIs" in ln for ln in local)
    famous = build_explanation({"popularity": 0.1}, _BREAKDOWN, _ctx(0.8, 0.6))
    assert any("famous landmarks" in ln for ln in famous)
    neutral = build_explanation({"popularity": 0.1}, _BREAKDOWN, _ctx(0.5, -0.6))
    assert not any("fits your preference" in ln for ln in neutral)


def test_pref_align_is_not_treated_as_a_compatibility_subscore() -> None:
    lines = build_explanation({"popularity": 0.1}, _BREAKDOWN, _ctx(0.5, 0.0))
    assert all("pref_align" not in ln for ln in lines)


def test_select_config_rule_constraints_objective_and_fallback() -> None:
    def row(g: float, lam: float, ndcg: float, ent: float, flip: float, ltp: float) -> dict:
        return {
            "gamma": g,
            "lambda": lam,
            "v_ndcg10_ips": ndcg,
            "v_category_entropy_bits": ent,
            "flip_overlap": flip,
            "flip_overlap_se": 0.01,
            "v_longtail_precision": ltp,
        }

    shipped = row(0.0, 0.8, 0.15, 3.0, 0.9, 0.3)
    ok = row(1.0, 0.8, 0.15 - NDCG_TOLERANCE + 1e-6, 3.0 * ENTROPY_FLOOR_RATIO + 1e-6, 0.4, 0.2)
    bad_ndcg = row(2.0, 0.8, 0.15 - NDCG_TOLERANCE - 1e-3, 3.0, 0.1, 0.9)
    bad_entropy = row(4.0, 0.8, 0.2, 3.0 * ENTROPY_FLOOR_RATIO - 1e-3, 0.05, 0.9)
    tied = row(
        0.5, 0.9, 0.16, 3.1, 0.405, 0.35
    )  # within one SE of `ok`, higher long-tail precision
    picked = select_config([shipped, ok, bad_ndcg, bad_entropy, tied])
    assert picked["winner"] == {"gamma": 0.5, "lambda": 0.9}
    only_shipped = select_config([shipped, bad_ndcg, bad_entropy])
    assert only_shipped["winner"] == {"gamma": 0.0, "lambda": 0.8} or only_shipped["winner"] is None


def test_jaccard() -> None:
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


@pytest.mark.slow
def test_configured_constants_match_the_train_frame() -> None:
    from poi_rank.features.config import FeatureBuildConfig
    from poi_rank.models.ranking_data import load_train_ranking_frame

    feature_cfg = FeatureBuildConfig.from_yaml(REPO_ROOT / "configs" / "features.yaml")
    frame = load_train_ranking_frame(
        REPO_ROOT / "data" / "synthetic", feature_cfg.traveler_features.budget_target_price_level
    )
    consts = pref_align_constants(frame)
    cfg = ScoringConfig.from_yaml(REPO_ROOT / "configs" / "scoring.yaml").utility
    assert cfg.pref_align_center == pytest.approx(consts["center"], rel=1e-9)
    assert cfg.pref_align_scale == pytest.approx(consts["scale"], rel=1e-9)
