"""Category-conditioned templates for POI `name`, `description`, and `tags`.

Keeps text fields lexically varied within a category (feeds the later text-embedding
feature in `features/poi_features.py` — not built in this phase, but the text must not
be degenerate/identical within a category for that to be meaningful).

**DGP remediation, Block A, RC2c** (docs/DATA_CARD.md "DGP remediation, Block A"):
`generate_name`/`generate_description` now accept an optional `flavor` argument -- a
"dominant semantic flavor" tag derived from a POI's OWN `poi_semantic` latent vector
(`datagen/catalog.py::dominant_flavor_from_semantic`), not from its raw sampled `tags`
list. Confirmed bug this fixes (D10, `results/parts/dgp_diagnostics.json`): the
pre-remediation generator computed `poi_semantic` entirely AFTER text generation, from
`category`+`tags` alone, so the text fields had literally zero dependency edge on
`poi_semantic` (measured pairwise Spearman rho=0.21). Threading `flavor` through both
functions, and prioritizing it as the description's PRIMARY highlighted tag (rather
than an arbitrary `tags[0]`), gives two POIs with similar `poi_semantic` vectors
(including its own independent noise component) a genuinely higher chance of sharing
similar generated text, while still drawing from a broad general vocabulary the
remaining fraction of the time (preserving Phase 1's original lexical-variety
requirement -- text must not become degenerate/near-identical within a flavor).
"""

from __future__ import annotations

import numpy as np

from poi_rank.datagen.taxonomy import TAGS

# Small, hand-authored adjective pool per tag/"flavor" (RC2c) -- gives text generation
# a genuinely richer, flavor-differentiated vocabulary beyond the generic ADJECTIVES
# list below, so POIs sharing a dominant `poi_semantic` flavor read more similarly.
FLAVOR_ADJECTIVES: dict[str, list[str]] = {
    "local": ["Neighborhood", "Homegrown", "Unassuming", "Everyday"],
    "hidden-gem": ["Hidden", "Under-the-radar", "Secret", "Tucked-away"],
    "touristy": ["Iconic", "Must-see", "Landmark", "Crowd-pulling"],
    "authentic": ["Authentic", "Time-honored", "Genuine", "Old-school"],
    "trendy": ["Trendy", "Buzzy", "Fashionable", "On-trend"],
    "historic": ["Historic", "Storied", "Centuries-old", "Heritage"],
    "family-friendly": ["Family-friendly", "Kid-approved", "Easygoing", "Welcoming"],
    "romantic": ["Romantic", "Intimate", "Candlelit", "Charming"],
    "budget-friendly": ["Budget-friendly", "Wallet-friendly", "No-frills", "Affordable"],
    "luxury": ["Luxurious", "Upscale", "Refined", "Opulent"],
    "outdoor": ["Open-air", "Al-fresco", "Breezy", "Sun-drenched"],
    "indoor": ["Sheltered", "Climate-controlled", "Cozy indoor", "Snug"],
    "lively": ["Lively", "Buzzing", "Energetic", "Vibrant"],
    "shopping": ["Boutique-lined", "Retail-heavy", "Browse-worthy", "Curated"],
    "artsy": ["Artsy", "Gallery-like", "Creative", "Eclectic"],
    "foodie": ["Foodie-favorite", "Flavor-packed", "Delicious", "Savory"],
    "nature": ["Leafy", "Green", "Nature-filled", "Scenic-green"],
    "adventurous": ["Adventurous", "Thrilling", "Bold", "Daring"],
    "relaxing": ["Relaxing", "Serene", "Laid-back", "Peaceful"],
    "cultural": ["Cultural", "Traditional", "Heritage-rich", "Ceremonial"],
}

# Short closing phrase fragments per flavor (RC2c), appended to a subset of
# descriptions to give the flavor an additional, distinguishable lexical footprint
# beyond a single adjective/tag mention.
FLAVOR_PHRASES: dict[str, list[str]] = {
    "local": ["a real neighborhood favorite", "where locals actually go"],
    "hidden-gem": ["easy to miss if you're not looking", "a genuine hidden find"],
    "touristy": ["a bucket-list stop for most visitors", "always on the must-see list"],
    "authentic": ["true to its roots", "unchanged by passing trends"],
    "trendy": ["popular with the trend-conscious crowd", "frequently featured online"],
    "historic": ["steeped in local history", "a living piece of the past"],
    "family-friendly": ["easy to enjoy with kids in tow", "a solid pick for families"],
    "romantic": ["a favorite for couples", "made for a quiet evening together"],
    "budget-friendly": ["easy on the wallet", "great value for the price"],
    "luxury": ["a splurge-worthy choice", "for those wanting a touch of luxury"],
    "outdoor": ["best enjoyed in good weather", "an open-air experience"],
    "indoor": ["a reliable indoor option", "comfortable whatever the weather"],
    "lively": ["never short on energy", "buzzing most hours of the day"],
    "shopping": ["worth a browse even without buying", "a magnet for shoppers"],
    "artsy": ["a draw for the creatively inclined", "full of artistic touches"],
    "foodie": ["a name foodies pass around", "worth planning a meal around"],
    "nature": ["a welcome patch of green", "a breath of fresh air"],
    "adventurous": ["for those chasing a thrill", "not for the faint of heart"],
    "relaxing": ["a good spot to slow down", "made for unwinding"],
    "cultural": ["a window into local culture", "rich with cultural detail"],
}
FLAVOR_ADOPTION_RATE = 0.95

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

# Flavor-conditioned crowd descriptor (RC2c) -- another small, deterministic lexical
# channel tying generated text more strongly to `poi_semantic`'s dominant flavor.
FLAVOR_CROWDS: dict[str, list[str]] = {
    "local": ["locals", "neighbors"],
    "hidden-gem": ["in-the-know regulars", "those who ask around"],
    "touristy": ["first-time visitors", "tour groups"],
    "authentic": ["longtime regulars", "purists"],
    "trendy": ["the trend-conscious crowd", "younger visitors"],
    "historic": ["history buffs", "heritage tour groups"],
    "family-friendly": ["families", "parents with kids"],
    "romantic": ["couples", "date-night visitors"],
    "budget-friendly": ["budget travelers", "students"],
    "luxury": ["well-heeled visitors", "special-occasion diners"],
    "outdoor": ["outdoor enthusiasts", "day-trippers"],
    "indoor": ["regulars escaping the weather", "everyday visitors"],
    "lively": ["the after-work crowd", "nightlife regulars"],
    "shopping": ["shoppers", "browsers"],
    "artsy": ["art lovers", "the creative crowd"],
    "foodie": ["foodies", "food-tour groups"],
    "nature": ["nature lovers", "walkers"],
    "adventurous": ["thrill-seekers", "adventure travelers"],
    "relaxing": ["those looking to unwind", "solo visitors"],
    "cultural": ["culture seekers", "heritage travelers"],
}


def _pick_adjective(rng: np.random.Generator, flavor: str | None) -> str:
    """RC2c: with probability `FLAVOR_ADOPTION_RATE`, draw from `flavor`'s own
    adjective pool (when available); otherwise (or with no flavor) fall back to the
    general `ADJECTIVES` pool -- preserves lexical variety within a flavor."""
    pool = FLAVOR_ADJECTIVES.get(flavor) if flavor else None
    if pool and rng.random() < FLAVOR_ADOPTION_RATE:
        return str(rng.choice(pool))
    return str(rng.choice(ADJECTIVES))


def generate_name(
    rng: np.random.Generator, destination: str, category: str, flavor: str | None = None
) -> str:
    """Sample a lexically varied POI name from category/destination templates.

    `flavor` (RC2c, docs/DATA_CARD.md "DGP remediation, Block A") is the POI's
    `poi_semantic`-derived dominant flavor tag -- biases the adjective draw toward
    that flavor's own pool most of the time, never deterministically.
    """
    adj = _pick_adjective(rng, flavor)
    district = rng.choice(DISTRICT_WORDS[destination])
    noun = rng.choice(NAME_NOUNS[category])
    return f"{adj} {district} {noun}"


def generate_description(
    rng: np.random.Generator,
    destination: str,
    category: str,
    tags: list[str],
    flavors: list[str] | None = None,
) -> str:
    """Sample a category/tag-conditioned description sentence.

    RC2c: `tag1`/`tag2` (the description's two PRIMARY highlighted tags) are
    `flavors[0]`/`flavors[1]` when provided, rather than arbitrary members of
    `tags` -- this is what actually threads `poi_semantic`'s content into the
    generated text (see module docstring). Using the top-2 dominant flavors
    (rather than just one) reflects substantially more of `poi_semantic`'s real
    32-dim direction than a single argmax label could (docs/DATA_CARD.md). A short
    flavor-specific closing phrase is appended a fraction of the time for
    additional lexical signal.
    """
    template = str(rng.choice(DESCRIPTION_TEMPLATES))
    noun = str(rng.choice(NAME_NOUNS[category]))
    flavor = flavors[0] if flavors else None
    secondary_flavor = flavors[1] if flavors and len(flavors) > 1 else None
    tag1 = flavor if flavor else (tags[0] if tags else str(rng.choice(TAGS)))
    if secondary_flavor:
        tag2 = secondary_flavor
    else:
        remaining_tags = [t for t in tags if t != tag1]
        tag2 = remaining_tags[0] if remaining_tags else str(rng.choice(TAGS))
    crowd_pool = FLAVOR_CROWDS.get(flavor) if flavor else None
    crowd = (
        rng.choice(crowd_pool)
        if crowd_pool and rng.random() < FLAVOR_ADOPTION_RATE
        else rng.choice(CROWDS)
    )
    text = template.format(
        adj1=_pick_adjective(rng, flavor),
        adj2=_pick_adjective(rng, secondary_flavor),
        noun=noun,
        noun_lower=noun.lower(),
        tag1=tag1,
        tag2=tag2,
        district=rng.choice(DISTRICT_WORDS[destination]),
        crowd=crowd,
    )
    phrase_pool = FLAVOR_PHRASES.get(flavor) if flavor else None
    if phrase_pool and rng.random() < FLAVOR_ADOPTION_RATE:
        text = f"{text} It's {rng.choice(phrase_pool)}."
    if secondary_flavor:
        secondary_pool = FLAVOR_PHRASES.get(secondary_flavor)
        if secondary_pool and rng.random() < FLAVOR_ADOPTION_RATE:
            text = f"{text} Also known for being {rng.choice(secondary_pool)}."
    return text


def generate_tags(
    rng: np.random.Generator, category_tag_affinity: dict[str, float], n_tags: int
) -> list[str]:
    """Sample `n_tags` distinct tags, weighted toward the category's typical tags."""
    weights = np.array([category_tag_affinity.get(t, 0.15) for t in TAGS], dtype=np.float64)
    weights = weights / weights.sum()
    n_tags = min(n_tags, len(TAGS))
    chosen = rng.choice(len(TAGS), size=n_tags, replace=False, p=weights)
    return [TAGS[i] for i in chosen]
