"""Category-conditioned templates for POI `name`, `description`, and `tags`.

Keeps text fields lexically varied within a category (feeds the later text-embedding
feature in `features/poi_features.py` — not built in this phase, but the text must not
be degenerate/identical within a category for that to be meaningful).
"""

from __future__ import annotations

import numpy as np

from poi_rank.datagen.taxonomy import TAGS

ADJECTIVES: list[str] = [
    "Cozy",
    "Charming",
    "Hidden",
    "Bustling",
    "Historic",
    "Modern",
    "Traditional",
    "Quiet",
    "Vibrant",
    "Authentic",
    "Elegant",
    "Rustic",
    "Scenic",
    "Lively",
    "Serene",
    "Iconic",
    "Local",
    "Beloved",
]

NAME_NOUNS: dict[str, list[str]] = {
    "restaurant": ["Kitchen", "Table", "Bistro", "Eatery", "Grill", "Noodle House", "Diner"],
    "cafe": ["Coffee House", "Cafe", "Roastery", "Tea Room", "Bakery"],
    "museum": ["Museum", "Gallery", "Exhibition Hall", "Collection"],
    "historic_site": ["Temple", "Palace", "Fortress", "Old Quarter", "Monument"],
    "nature_park": ["Park", "Garden", "Trail", "Riverside Walk", "Green"],
    "nightlife": ["Bar", "Lounge", "Club", "Speakeasy", "Rooftop"],
    "shopping": ["Market", "Boutique", "Arcade", "Bazaar", "Emporium"],
    "family_activity": ["Playland", "Adventure Park", "Discovery Center", "Fun Zone"],
    "wellness_spa": ["Spa", "Bathhouse", "Wellness Studio", "Retreat"],
    "religious_site": ["Shrine", "Cathedral", "Monastery", "Temple"],
    "viewpoint": ["Observation Deck", "Overlook", "Tower", "Skyline Point"],
    "entertainment": ["Theater", "Cinema", "Arcade", "Playhouse"],
}

# Neighborhood/district flavor words per destination — purely cosmetic, gives names a
# plausible local feel and adds lexical variety independent of category.
DISTRICT_WORDS: dict[str, list[str]] = {
    "seoul": ["Hongdae", "Itaewon", "Insadong", "Myeongdong", "Gangnam", "Bukchon", "Seongsu"],
    "kyoto": ["Gion", "Arashiyama", "Higashiyama", "Kawaramachi", "Fushimi", "Nishijin"],
    "barcelona": ["Gothic Quarter", "Gracia", "Eixample", "Born", "Barceloneta", "Poble Sec"],
}

DESCRIPTION_TEMPLATES: list[str] = [
    "A {adj1} {noun} known for its {tag1} atmosphere and {tag2} appeal.",
    "This {adj1} spot blends {tag1} charm with a {tag2} edge, popular among {crowd}.",
    "A {tag1}, {tag2} {noun} tucked into the {district} area.",
    "{adj1} and {adj2}, this {noun} is a favorite for a {tag1} experience.",
    "One of the more {tag1} {noun_lower}s in the district, prized for being {tag2}.",
]

CROWDS: list[str] = ["locals", "regulars", "first-time visitors", "return travelers", "neighbors"]


def generate_name(rng: np.random.Generator, destination: str, category: str) -> str:
    """Sample a lexically varied POI name from category/destination templates."""
    adj = rng.choice(ADJECTIVES)
    district = rng.choice(DISTRICT_WORDS[destination])
    noun = rng.choice(NAME_NOUNS[category])
    return f"{adj} {district} {noun}"


def generate_description(
    rng: np.random.Generator, destination: str, category: str, tags: list[str]
) -> str:
    """Sample a category/tag-conditioned description sentence."""
    template = str(rng.choice(DESCRIPTION_TEMPLATES))
    noun = str(rng.choice(NAME_NOUNS[category]))
    tag1 = tags[0] if tags else str(rng.choice(TAGS))
    tag2 = tags[1] if len(tags) > 1 else str(rng.choice(TAGS))
    return template.format(
        adj1=rng.choice(ADJECTIVES),
        adj2=rng.choice(ADJECTIVES),
        noun=noun,
        noun_lower=noun.lower(),
        tag1=tag1,
        tag2=tag2,
        district=rng.choice(DISTRICT_WORDS[destination]),
        crowd=rng.choice(CROWDS),
    )


def generate_tags(
    rng: np.random.Generator, category_tag_affinity: dict[str, float], n_tags: int
) -> list[str]:
    """Sample `n_tags` distinct tags, weighted toward the category's typical tags."""
    weights = np.array([category_tag_affinity.get(t, 0.15) for t in TAGS], dtype=np.float64)
    weights = weights / weights.sum()
    n_tags = min(n_tags, len(TAGS))
    chosen = rng.choice(len(TAGS), size=n_tags, replace=False, p=weights)
    return [TAGS[i] for i in chosen]
