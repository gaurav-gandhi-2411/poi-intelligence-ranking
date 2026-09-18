"""Template layer (spec.md section 10): each grouped-TreeSHAP feature group maps to
a deterministic natural-language template with numeric fill-ins from the REAL scoring
pipeline output -- never a generated string, never an LLM call (spec.md section 10 /
brief section 4: LLM assistants are explicitly out of scope). Same input always
produces the same explanation text (`tests/test_templates.py
::test_templates_are_deterministic`).

**`implicit_taste`'s two variants** (task instruction, spec.md section 10): a
traveler with zero as-of-safe interaction history (`implicit_interaction_count ==
0`, `features/traveler_features.py`'s documented cold-start case) never gets the
"past trips" phrasing -- that would be a false claim about history that does not
exist. Cold-start travelers get an honest alternative sentence instead, grounded in
the POI's own `behav_archetype_affinity_*` collaborative signal (a real, POI-level
"do travelers with your inferred profile engage with this" quantity that requires
no personal history at all) -- never simply omitted, since it is a TRUE statement
either way, just sourced from a different (POI-level, not traveler-level) signal.

**`top_signals`**: top-N groups by RAW (signed) contribution, descending -- i.e. the
most positively-contributing groups lead, matching spec.md section 9.5's own example
(3 positive contributions, largest first). Ties broken by group name ascending for
full determinism. `novelty` (`shap_groups.py`: structurally always exactly 0.0) is
excluded from consideration -- it can never be an informative top signal.

**`explanation`**: up to 3 SHAP-group lines (only for groups with a genuinely
positive contribution AND a renderable template -- e.g. `interest_match` renders
`None` when the traveler's stated interests do not actually overlap this POI's
category/tags, which can happen for a candidate the ranker liked for other reasons)
PLUS exactly one compatibility-derived line: the weakest of the 6
`compatibility_breakdown` sub-scores if it is below `NEAR_BINDING_THRESHOLD` (a
genuine near-binding constraint worth surfacing), else two positive confirmation
lines (mobility + budget) mirroring spec.md section 9.5's own 5-line example, whose
compatibility is uniformly high and shows exactly that shape (2 SHAP-adjacent lines
+ mobility + budget, no binding-constraint warning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poi_rank.explain.shap_groups import (
    FEATURE_GROUPS,
    GEO,
    HOURS,
    IMPLICIT_TASTE,
    INTEREST_MATCH,
    LOCALNESS_FIT,
    NOVELTY,
    PARTY_FIT,
    POPULARITY,
    PRICE_FIT,
    QUALITY,
)

TOP_N_SIGNALS = 3
CONTRIBUTION_EPSILON = 1e-9
NEAR_BINDING_THRESHOLD = 0.85  # judgment call, documented in docs/DATA_CARD.md

FAMILY_PARTY_TYPES: tuple[str, ...] = ("family_young_kids", "family_teens")

MOBILITY_LABEL: dict[str, str] = {
    "walk": "walking",
    "public_transport": "public transport",
    "car": "car",
    "mixed": "mixed transport",
}

SUBSCORE_LABEL: dict[str, str] = {
    "budget_fit": "budget",
    "mobility_fit": "travel distance",
    "hours_fit": "opening hours",
    "reservation_fit": "reservation lead time",
    "party_fit": "party/accessibility fit",
    "duration_fit": "visit-length fit",
}


@dataclass(frozen=True)
class ExplanationContext:
    """Every raw value a template renderer needs for one recommendation row --
    assembled once by `explain/output_enrichment.py` from the scoring pipeline's own
    fully-joined `full` frame plus a genuinely-recomputed `travel_min` (never a
    fabricated number -- see `output_enrichment.py`'s module docstring)."""

    poi_category: str
    destination: str
    budget: str
    party_type: str
    mobility: str
    pop_pct: float
    rating_shrunk: float
    review_count: float
    budget_fit: float
    mobility_fit: float
    hours_fit: float
    party_fit: float
    travel_min: float
    implicit_interaction_count: float
    interests: frozenset[str]
    poi_terms: frozenset[str]
    price_level: float
    budget_target_price_level: float


def _category_label(category: str) -> str:
    return category.replace("_", " ")


# -----------------------------------------------------------------------------------
# top_signals (spec.md section 9.5)
# -----------------------------------------------------------------------------------


def top_signals(
    group_contributions: dict[str, float], top_n: int = TOP_N_SIGNALS
) -> list[dict[str, Any]]:
    """Top-`top_n` groups by raw (signed) contribution, descending, `novelty`
    excluded (module docstring). Deterministic tie-break: group name ascending."""
    candidates = [(g, float(v)) for g, v in group_contributions.items() if g != NOVELTY]
    candidates.sort(key=lambda gv: (-gv[1], gv[0]))
    return [{"feature_group": g, "contribution": round(v, 6)} for g, v in candidates[:top_n]]


# -----------------------------------------------------------------------------------
# Per-group SHAP-driven explanation renderers -- each returns `None` when the group
# has nothing genuinely renderable for this row (never a placeholder string).
# -----------------------------------------------------------------------------------


def _interest_match_line(ctx: ExplanationContext) -> str | None:
    matched = ctx.interests & ctx.poi_terms
    if not matched:
        return None
    label = _category_label(sorted(matched)[0])
    return f"Strong match with your stated interest in {label}"


def _implicit_taste_line(ctx: ExplanationContext) -> str:
    if ctx.implicit_interaction_count > 0:
        return (
            f"Similar to {_category_label(ctx.poi_category)} POIs you've engaged with on past trips"
        )
    # Cold-start (module docstring): a TRUE, differently-sourced statement, never a
    # false claim about nonexistent personal history.
    return "Popular with travelers who share your interests"


def _localness_fit_line(ctx: ExplanationContext) -> str:
    touristy_pct = 100.0 * (1.0 - ctx.pop_pct)
    return f"Less touristy than {touristy_pct:.0f}% of comparable POIs in {ctx.destination.title()}"


def _popularity_line(ctx: ExplanationContext) -> str:
    return (
        f"Popular choice -- busier than {ctx.pop_pct * 100:.0f}% of comparable POIs "
        f"in {ctx.destination.title()}"
    )


def _price_fit_line(ctx: ExplanationContext) -> str:
    return f"Matches your stated {ctx.budget} budget"


def _mobility_confirmation_line(ctx: ExplanationContext) -> str:
    mode_label = MOBILITY_LABEL.get(ctx.mobility, ctx.mobility)
    return (
        f"{ctx.travel_min:.0f} min from your stay by {mode_label} -- "
        f"fits your {mode_label} preference"
    )


def _geo_line(ctx: ExplanationContext) -> str:
    return _mobility_confirmation_line(ctx)


def _hours_line(ctx: ExplanationContext) -> str:
    return f"Open {ctx.hours_fit * 100:.0f}% of your trip's plausible visiting hours"


def _party_fit_line(ctx: ExplanationContext) -> str | None:
    if ctx.party_type not in FAMILY_PARTY_TYPES or ctx.party_fit < 0.8:
        return None
    return "Family/kid-friendly, matching your travel party"


def _quality_line(ctx: ExplanationContext) -> str:
    return f"Highly rated ({ctx.rating_shrunk:.1f}/5 from {int(ctx.review_count)} reviews)"


_GROUP_RENDERERS: dict[str, Any] = {
    INTEREST_MATCH: _interest_match_line,
    IMPLICIT_TASTE: _implicit_taste_line,
    LOCALNESS_FIT: _localness_fit_line,
    POPULARITY: _popularity_line,
    PRICE_FIT: _price_fit_line,
    GEO: _geo_line,
    HOURS: _hours_line,
    PARTY_FIT: _party_fit_line,
    QUALITY: _quality_line,
}
assert set(_GROUP_RENDERERS) == set(FEATURE_GROUPS) - {NOVELTY}  # module-load invariant


# -----------------------------------------------------------------------------------
# Compatibility-derived lines
# -----------------------------------------------------------------------------------


def _weak_compat_line(subscore_name: str, subscore_value: float, ctx: ExplanationContext) -> str:
    label = SUBSCORE_LABEL[subscore_name]
    if subscore_name == "budget_fit":
        # `budget_fit` is weak from EITHER direction of the gap (over-budget is
        # penalized harder, but a large under-budget gap can also pull it below
        # NEAR_BINDING_THRESHOLD) -- the sentence must check the actual sign of
        # `price_level - budget_target_price_level`, not assume "over budget"
        # unconditionally. A prior version hardcoded "Priced above ..." regardless
        # of direction; verified wrong on 535/766 (70%) of real generated lines
        # against the committed dataset (POI cheaper than the traveler's target,
        # not pricier) before being fixed here -- see docs/DATA_CARD.md.
        if ctx.price_level > ctx.budget_target_price_level:
            return f"Priced above what's typical for your {ctx.budget} budget"
        if ctx.price_level < ctx.budget_target_price_level:
            return f"More budget-friendly than typical for your {ctx.budget} budget"
        return f"Weaker fit on {label} than most of your other options"
    if subscore_name == "mobility_fit":
        mode_label = MOBILITY_LABEL.get(ctx.mobility, ctx.mobility)
        return f"{ctx.travel_min:.0f} min from your stay by {mode_label} -- a bit of a trek"
    if subscore_name == "hours_fit":
        return f"Only open {subscore_value * 100:.0f}% of your trip's plausible visiting hours"
    return f"Weaker fit on {label} than most of your other options"


def _budget_confirmation_line(ctx: ExplanationContext) -> str:
    if ctx.budget_fit >= 0.9:
        return f"Within your {ctx.budget} budget"
    return f"Reasonably close to your {ctx.budget} budget"


def build_explanation(
    group_contributions: dict[str, float],
    compatibility_breakdown: dict[str, float],
    ctx: ExplanationContext,
) -> list[str]:
    """Assemble `explanation` (spec.md section 9.5): up to 3 SHAP-driven lines +
    exactly 1-2 compatibility-driven lines (module docstring). Does NOT include the
    counterfactual line -- `explain/output_enrichment.py` appends that separately
    (`explain/counterfactual.py`) only when a genuine binding constraint exists."""
    signals = top_signals(group_contributions, TOP_N_SIGNALS)
    lines: list[str] = []
    for signal in signals:
        contribution = signal["contribution"]
        if contribution <= CONTRIBUTION_EPSILON:
            continue
        renderer = _GROUP_RENDERERS[signal["feature_group"]]
        line = renderer(ctx)
        if line is not None:
            lines.append(line)

    weakest_name = min(compatibility_breakdown, key=lambda name: compatibility_breakdown[name])
    weakest_value = compatibility_breakdown[weakest_name]
    if weakest_value < NEAR_BINDING_THRESHOLD:
        lines.append(_weak_compat_line(weakest_name, weakest_value, ctx))
    else:
        lines.append(_mobility_confirmation_line(ctx))
        lines.append(_budget_confirmation_line(ctx))

    return lines
