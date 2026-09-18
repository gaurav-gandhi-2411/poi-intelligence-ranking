"""Shared category/tag vocabulary and the latent taste-vector space.

`TASTE_DIM = len(CATEGORIES) + len(TAGS)`. Both traveler true taste vectors and POI
`poi_semantic` vectors live in this one space: the first `len(CATEGORIES)` dims are a
one-hot-ish category axis (used directly for the `w_cat * taste_t[category_p]` utility
term), the remaining `len(TAGS)` dims are a tag-affinity axis. Cosine similarity
between the two vectors is the `w_taste` utility term. This is a deliberate design
choice (documented in docs/DATA_CARD.md) to keep the "semantic" and "category" utility
terms consistent with each other rather than sampling two unrelated latent spaces.
"""

from __future__ import annotations

CATEGORIES: list[str] = [
    "restaurant",
    "cafe",
    "museum",
    "historic_site",
    "nature_park",
    "nightlife",
    "shopping",
    "family_activity",
    "wellness_spa",
    "religious_site",
    "viewpoint",
    "entertainment",
]

SUBCATEGORIES: dict[str, list[str]] = {
    "restaurant": ["local_eatery", "fine_dining", "street_food", "fusion", "traditional"],
    "cafe": ["specialty_coffee", "tea_house", "dessert_cafe", "bakery"],
    "museum": ["art_museum", "history_museum", "science_museum", "gallery"],
    "historic_site": ["temple_site", "palace", "fortress", "old_town", "monument"],
    "nature_park": ["city_park", "garden", "hiking_trail", "botanical_garden", "riverside"],
    "nightlife": ["bar", "club", "live_music", "rooftop_bar", "speakeasy"],
    "shopping": ["market", "boutique", "mall", "artisan_shop", "flea_market"],
    "family_activity": [
        "amusement_park",
        "aquarium",
        "zoo",
        "playground",
        "interactive_museum",
    ],
    "wellness_spa": ["spa", "onsen", "yoga_studio", "massage"],
    "religious_site": ["temple_worship", "shrine", "cathedral", "monastery"],
    "viewpoint": ["observation_deck", "scenic_overlook", "tower", "rooftop_view"],
    "entertainment": ["theater", "cinema", "arcade", "escape_room", "live_show"],
}

# Relative generation frequency per category — restaurants/cafes are the long tail of
# any real POI catalog, wellness/religious sites are comparatively rare.
CATEGORY_PROBS: dict[str, float] = {
    "restaurant": 0.18,
    "cafe": 0.12,
    "museum": 0.08,
    "historic_site": 0.09,
    "nature_park": 0.08,
    "nightlife": 0.08,
    "shopping": 0.10,
    "family_activity": 0.06,
    "wellness_spa": 0.05,
    "religious_site": 0.05,
    "viewpoint": 0.05,
    "entertainment": 0.06,
}

TAGS: list[str] = [
    "local",
    "hidden-gem",
    "touristy",
    "authentic",
    "trendy",
    "historic",
    "family-friendly",
    "romantic",
    "budget-friendly",
    "luxury",
    "outdoor",
    "indoor",
    "lively",
    "shopping",
    "artsy",
    "foodie",
    "nature",
    "adventurous",
    "relaxing",
    "cultural",
]

# The stated-interest vocabulary is the union of categories and tags: both read as
# plausible traveler "interests" ("museum", "local food") and reusing this vocabulary
# avoids maintaining a second, disconnected interest taxonomy (see docs/DATA_CARD.md).
INTEREST_LABELS: list[str] = CATEGORIES + TAGS

TASTE_DIM: int = len(CATEGORIES) + len(TAGS)

CATEGORY_INDEX: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
TAG_INDEX: dict[str, int] = {t: len(CATEGORIES) + i for i, t in enumerate(TAGS)}
INTEREST_LABEL_INDEX: dict[str, int] = {label: i for i, label in enumerate(INTEREST_LABELS)}

# Messy, inconsistent variants of each category string injected into a subset of POI
# rows (docs/DATA_CARD.md dirtiness catalog). Cleaning is deliberately out of scope for
# this phase; a later `data/` phase is expected to canonicalize these back.
CATEGORY_STRING_VARIANTS: dict[str, list[str]] = {
    "restaurant": ["Restaurant", "restaurants", "Food & Dining"],
    "cafe": ["Cafe", "cafes", "Coffee Shop"],
    "museum": ["Museum", "museums", "Museum / Gallery"],
    "historic_site": ["Historic Site", "historic sites", "Historical Landmark"],
    "nature_park": ["Park", "parks & nature", "Nature / Park"],
    "nightlife": ["Nightlife", "bars & clubs", "Night Life"],
    "shopping": ["Shopping", "shops", "Retail / Shopping"],
    "family_activity": ["Family Activity", "family activities", "Kids & Family"],
    "wellness_spa": ["Wellness", "spa & wellness", "Spa / Wellness"],
    "religious_site": ["Religious Site", "religious sites", "Temple / Shrine"],
    "viewpoint": ["Viewpoint", "viewpoints", "Scenic View"],
    "entertainment": ["Entertainment", "entertainment venues", "Fun & Entertainment"],
}
