"""Input handling and formatting of `poi_rank.cli demo` (the live pipeline itself is exercised by
the scenarios/recommend tests; the demo reuses that exact code path)."""

from __future__ import annotations

import pytest

from poi_rank.eval.demo import (
    DemoInputError,
    build_profile,
    format_recommendations,
    parse_interests,
)

VOCAB = {"local", "foodie", "authentic", "museum", "nightlife", "cafe"}


def test_aliases_expand_to_real_interest_labels() -> None:
    assert parse_interests("local_food,neighborhoods", VOCAB) == ("local", "foodie", "authentic")


def test_raw_labels_are_accepted_and_deduplicated() -> None:
    assert parse_interests("cafe, cafe ,museum", VOCAB) == ("cafe", "museum")


def test_unknown_interest_lists_the_valid_choices() -> None:
    with pytest.raises(DemoInputError, match="unknown interest 'karaoke'.*or an alias"):
        parse_interests("karaoke", VOCAB)


def test_empty_interests_is_an_error() -> None:
    with pytest.raises(DemoInputError, match="INTERESTS is empty"):
        parse_interests(" , ", VOCAB)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"budget": "lavish"}, "BUDGET"),
        ({"mobility": "teleport"}, "MOBILITY"),
        ({"party": "crowd"}, "PARTY"),
        ({"touristiness": 1.5}, "TOURISTINESS"),
    ],
)
def test_bad_profile_fields_are_rejected_with_the_field_name(
    kwargs: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "interests": "local_food",
        "budget": "medium",
        "mobility": "walk",
        "touristiness": 0.0,
        "party": "solo",
        "vocabulary": VOCAB,
    }
    with pytest.raises(DemoInputError, match=message):
        build_profile(**{**base, **kwargs})  # type: ignore[arg-type]


def test_format_recommendations_prints_scores_signals_and_explanations() -> None:
    rec = {
        "rank": 1,
        "name": "Some Place",
        "category": "cafe",
        "utility": 0.5,
        "preference_score": 0.4,
        "context_compatibility": 0.9,
        "confidence": 0.7,
        "top_signals": [{"feature_group": "interest_match", "contribution": 0.31}],
        "explanation": ["Strong match with your stated interest in cafe"],
    }
    text = format_recommendations([rec])
    assert "Some Place" in text and "utility 0.500" in text
    assert "interest_match +0.31" in text
    assert "- Strong match with your stated interest in cafe" in text
