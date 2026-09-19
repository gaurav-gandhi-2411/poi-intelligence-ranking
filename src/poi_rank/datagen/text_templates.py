"""Latent-dimension-conditioned templates for POI `name`, `description`, and `tags`.

Keeps text fields lexically varied within a category (feeds the later text-embedding
feature in `features/poi_features.py`) while making the text a genuine, noisy readout of
the POI's own latent `poi_semantic` vector.

**DGP remediation, Block A, RC2c -> A2** (docs/DATA_CARD.md "RC2c" and "A2"): RC2c threaded
a "dominant flavor" (argmax of `poi_semantic`'s tag axis) into the text and plateaued at
D10 rho=0.452, which spec-v3 section 2.1 traced to a thin flavor vocabulary (2-4 lexemes per
flavor, top-2 flavors only). A2 replaces that with:

1. A **phrase pool per latent dimension**: each of `poi_semantic`'s 32 dimensions (12
   categories + 20 tags, `taxonomy.py`) owns `phrases_per_dimension` (P, default 20, valid
   range 15-25) distinct, semantically-related paraphrases ("hidden atmosphere",
   "secret vibe", "under-the-radar feel", ...) -- see `build_phrase_pools`. Paraphrases
   within a dimension are SYNONYM-RICH by design (docs/DATA_CARD.md "A2" documents why
   and what that implies for the later TF-IDF-vs-MiniLM comparison, DR4).
2. **Loading-proportional phrase sampling** (`dimension_sampling_weights`,
   `sample_phrases`): 3-6 distinct phrases per POI, each dimension drawn with probability
   proportional to the POI's positive loading on it in `poi_semantic` (the noisy vector the
   utility actually uses, not the raw sampled tag list), mixed with a small uniform
   background rate so no phrase is deterministic.
3. **Surface realization independent of phrase choice** (`realize_description`): sentence
   frames, word order, connectives, and filler come from a SEPARATE random stream that
   never looks at which phrases were chosen, so lexical variety lives in the surface form
   and not in weakening the semantic link.

Each POI gets its own two seeded streams (`poi_text_rngs`), derived from the global seed,
destination, and POI index -- text generation consumes ZERO draws from the catalog's main
RNG, so changing any text-side knob (P, background rate, templates) leaves every other DGP
quantity bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from poi_rank.datagen.config import TextConfig
from poi_rank.datagen.taxonomy import CATEGORIES, TAGS, TASTE_DIM

FloatArray = npt.NDArray[np.float64]

# The 32 latent dimensions of `poi_semantic`, in vector order (categories first, then tags)
# -- must stay identical to `taxonomy.py`'s CATEGORY_INDEX/TAG_INDEX layout.
DIMENSION_NAMES: list[str] = [*CATEGORIES, *TAGS]

# Each dimension's synonym set: 5 lexemes that all say "this POI has trait X" in different
# words. A phrase is `{synonym} {head}` (below), so a pool of up to 5 x 5 = 25 distinct,
# semantically related paraphrases exists per dimension. Tag dimensions reuse and extend
# the Block A (RC2c) `FLAVOR_ADJECTIVES` hand-authored lists; category dimensions are new.
DIMENSION_SYNONYMS: dict[str, list[str]] = {
    # -- category dimensions ------------------------------------------------------------
    "restaurant": ["dining", "eatery", "culinary", "restaurant", "mealtime"],
    "cafe": ["coffee", "cafe", "espresso", "teahouse", "brewed"],
    "museum": ["museum", "exhibit", "curated-gallery", "collection", "showcase"],
    "historic_site": ["heritage-site", "ancient", "monument", "landmark-site", "old-town"],
    "nature_park": ["parkland", "garden", "woodland", "trailside", "greenspace"],
    "nightlife": ["after-dark", "nightlife", "late-night", "bar-scene", "evening-out"],
    "shopping": ["shopping", "retail", "marketplace", "storefront", "browsing"],
    "family_activity": ["kid-focused", "playful", "family-day", "activity-packed", "fun-filled"],
    "wellness_spa": ["spa", "wellness", "soothing", "bathhouse", "pampering"],
    "religious_site": ["sacred", "shrine", "spiritual", "devotional", "temple"],
    "viewpoint": ["panoramic", "lookout", "skyline", "overlook", "vista"],
    "entertainment": ["show", "entertainment", "theatrical", "screening", "amusement"],
    # -- tag dimensions -------------------------------------------------------------------
    "local": ["neighborhood", "homegrown", "unassuming", "everyday", "down-to-earth"],
    "hidden-gem": ["hidden", "under-the-radar", "secret", "tucked-away", "off-the-beaten-path"],
    "touristy": ["iconic", "must-see", "landmark", "crowd-pulling", "postcard"],
    "authentic": ["authentic", "time-honored", "genuine", "old-school", "true-to-form"],
    "trendy": ["trendy", "buzzy", "fashionable", "on-trend", "modish"],
    "historic": ["historic", "storied", "centuries-old", "heritage", "age-old"],
    "family-friendly": ["family-friendly", "kid-approved", "easygoing", "welcoming", "child-safe"],
    "romantic": ["romantic", "intimate", "candlelit", "charming", "dreamy"],
    "budget-friendly": ["budget-friendly", "wallet-friendly", "no-frills", "affordable", "bargain"],
    "luxury": ["luxurious", "upscale", "refined", "opulent", "lavish"],
    "outdoor": ["open-air", "al-fresco", "breezy", "sun-drenched", "sky-lit"],
    "indoor": ["sheltered", "climate-controlled", "cozy-indoor", "snug", "roofed"],
    "lively": ["lively", "buzzing", "energetic", "vibrant", "animated"],
    "shopping-tag": ["boutique-lined", "retail-heavy", "browse-worthy", "curated", "stall-filled"],
    "artsy": ["artsy", "gallery-like", "creative", "eclectic", "bohemian"],
    "foodie": ["foodie-favorite", "flavor-packed", "delicious", "savory", "mouthwatering"],
    "nature": ["leafy", "green", "nature-filled", "scenic-green", "verdant"],
    "adventurous": ["adventurous", "thrilling", "bold", "daring", "adrenaline-charged"],
    "relaxing": ["relaxing", "serene", "laid-back", "peaceful", "tranquil"],
    "cultural": ["cultural", "traditional", "heritage-rich", "ceremonial", "folkloric"],
}
# The tag "shopping" and the category "shopping" are two different latent dimensions that
# share a name; the tag dimension's synonym set lives under a distinct key to keep the dict
# unambiguous (`_synonym_key`).
_TAG_KEY_OVERRIDES: dict[int, str] = {len(CATEGORIES) + TAGS.index("shopping"): "shopping-tag"}

# Shared head nouns that turn a synonym into a short noun phrase ("hidden atmosphere").
PHRASE_HEADS: list[str] = ["atmosphere", "charm", "vibe", "feel", "character"]
N_SYNONYMS_PER_DIMENSION = 5
MAX_PHRASES_PER_DIMENSION = N_SYNONYMS_PER_DIMENSION * len(PHRASE_HEADS)  # 25

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

# --- surface realization (sampled independently of WHICH phrases were chosen) ---------------
# Frames are grouped by how many phrases they carry; the group structure, frame choice,
# connective, filler, and phrase order all come from the surface stream only.
SENTENCE_FRAMES: dict[int, list[str]] = {
    1: [
        "Known for its {p1}.",
        "Visitors mention the {p1}.",
        "You will notice the {p1} right away.",
        "Expect a {p1}.",
        "A place defined by its {p1}.",
    ],
    2: [
        "Blends {p1} with {p2}.",
        "Offers {p1} and {p2}.",
        "A mix of {p1} and {p2}.",
        "You get {p1}, plus {p2}.",
        "Pairs {p1} alongside {p2}.",
    ],
    3: [
        "{p1}, {p2} and {p3} set the tone.",
        "Combines {p1}, {p2}, and {p3}.",
        "Regulars point to the {p1}, the {p2}, and the {p3}.",
        "Think {p1}, {p2}, {p3}.",
    ],
}
CONNECTIVES: list[str] = ["Also,", "Beyond that,", "On top of that,", "In addition,", "Overall,"]
FILLER_SENTENCES: list[str] = [
    "Open most days.",
    "Worth a stop on any itinerary.",
    "Plan a short visit or a long one.",
    "Easy to find.",
]
# Openers name the POI's own category (a category-conditioned surface form, like the
# category-specific name nouns) and deliberately carry no district word: a rare open-class
# token in every description is pure lexical noise for any bag-of-words reader (A2 diagnostic,
# docs/DATA_CARD.md "A2" hypothesis table).
OPENERS: list[str] = [
    "This {category_word} sits nearby.",
    "A {category_word} worth a visit.",
    "One of the {category_word}s in the area.",
    "A {category_word} many visitors mention.",
]

_TEXT_STREAM_TAG = 7301  # constant mixed into every text-stream seed (namespace separation)
_PHRASE_STREAM = 0
_SURFACE_STREAM = 1


@dataclass(frozen=True)
class PoiText:
    """One POI's generated text fields plus which (dimension, variant) phrases it used."""

    name: str
    description: str
    phrase_ids: tuple[tuple[int, int], ...]  # (dimension index, variant index within pool)


def _synonym_key(dim_index: int) -> str:
    return _TAG_KEY_OVERRIDES.get(dim_index, DIMENSION_NAMES[dim_index])


def _anchor_word(dim_index: int) -> str:
    """Canonical trait word for a dimension (its taxonomy name, spaces for underscores)."""
    return DIMENSION_NAMES[dim_index].replace("_", " ")


def build_phrase_pools(n_per_dimension: int, anchor_phrases: bool = True) -> list[list[str]]:
    """Deterministic phrase pools, one per latent dimension (vector order).

    Pool for dimension d = `n_per_dimension` distinct phrases drawn from d's 5-word
    synonym set x the 5 shared heads. Variant k uses synonym `k % 5` and head `k // 5`, so
    a small pool (P=3) already spans 3 different synonyms and P=25 enumerates every
    combination. Phrases within a dimension are therefore paraphrases of one another
    (shared meaning, different words), not arbitrary unrelated tokens.

    `anchor_phrases=True` (shipped default) additionally keeps the dimension's canonical
    trait word in every phrase ("candlelit romantic charm"), the way real listings pair a
    descriptive paraphrase with the trait's plain name; `False` yields pure-synonym phrases
    ("candlelit charm") whose only cross-POI lexical link within a dimension is the
    synonym itself. Both variants are measured in the A2 sweep (docs/DATA_CARD.md "A2").
    """
    if not 1 <= n_per_dimension <= MAX_PHRASES_PER_DIMENSION:
        raise ValueError(
            f"phrases_per_dimension must be in [1, {MAX_PHRASES_PER_DIMENSION}], "
            f"got {n_per_dimension}"
        )
    pools: list[list[str]] = []
    for d in range(TASTE_DIM):
        synonyms = DIMENSION_SYNONYMS[_synonym_key(d)]
        anchor = _anchor_word(d)
        anchor_tokens = set(anchor.replace("-", " ").split())
        pool: list[str] = []
        for k in range(n_per_dimension):
            synonym = synonyms[k % N_SYNONYMS_PER_DIMENSION]
            head = PHRASE_HEADS[k // N_SYNONYMS_PER_DIMENSION]
            covered = anchor_tokens <= set(synonym.replace("-", " ").split())
            if anchor_phrases and not covered:
                pool.append(f"{synonym} {anchor} {head}")
            else:
                pool.append(f"{synonym} {head}")
        pools.append(pool)
    return pools


def dimension_sampling_weights(poi_semantic: FloatArray, background_rate: float) -> FloatArray:
    """Probability that a phrase draw lands on each latent dimension: with probability
    `1 - background_rate` proportional to the POI's POSITIVE loading on the dimension
    (negative loadings are pure noise and carry no trait), with probability
    `background_rate` uniform over all dimensions -- so every phrase in every pool has
    nonzero probability for every POI and none is deterministic."""
    loadings = np.clip(poi_semantic, 0.0, None)
    total = float(loadings.sum())
    uniform = np.full(len(poi_semantic), 1.0 / len(poi_semantic))
    if total <= 0.0:
        return uniform
    weights: FloatArray = (1.0 - background_rate) * loadings / total + background_rate * uniform
    return weights


def _systematic_counts(
    rng: np.random.Generator, weights: FloatArray, n: int
) -> npt.NDArray[np.intp]:
    """Systematic (Madow) unequal-probability sampling of `n` draws over dimensions: each
    dimension's expected count is exactly `n * weights[d]`, but the realized count is
    always floor or ceil of it -- the same marginal expectation as `n` independent
    multinomial draws with far lower variance. Dimension order is randomly permuted first
    so no dimension is systematically favoured by its position."""
    order = rng.permutation(len(weights))
    cumulative = np.cumsum(weights[order] * n)
    points = rng.random() + np.arange(n)
    slot = np.minimum(np.searchsorted(cumulative, points, side="left"), len(weights) - 1)
    counts: npt.NDArray[np.intp] = np.bincount(order[slot], minlength=len(weights))
    return counts


def sample_phrases(
    rng: np.random.Generator,
    poi_semantic: FloatArray,
    pools: list[list[str]],
    n_phrases: int,
    background_rate: float,
    sampling: str = "systematic",
) -> tuple[tuple[int, int], ...]:
    """Draw `n_phrases` (dimension, variant) pairs, never repeating a phrase.

    A dimension's expected number of mentions is `n_phrases * weights[d]`
    (`dimension_sampling_weights`, i.e. proportional to the POI's loading plus the uniform
    background). `sampling="multinomial"` draws every mention independently (pair
    probability `weights[d] / pool_size`, without replacement); `sampling="systematic"`
    (shipped default) realizes the same expectations with floor/ceil counts per dimension
    (`_systematic_counts`), then picks distinct variants uniformly within each dimension.
    Both variants are measured in the A2 sweep (docs/DATA_CARD.md "A2").
    """
    weights = dimension_sampling_weights(poi_semantic, background_rate)
    pool_size = len(pools[0])
    if sampling == "multinomial":
        p = np.repeat(weights / pool_size, pool_size)
        flat = rng.choice(len(weights) * pool_size, size=n_phrases, replace=False, p=p)
        return tuple((int(f) // pool_size, int(f) % pool_size) for f in flat)
    if sampling != "systematic":
        raise ValueError(f"unknown text sampling method: {sampling!r}")
    counts = _systematic_counts(rng, weights, n_phrases)
    picked: list[tuple[int, int]] = []
    for d in np.flatnonzero(counts):
        variants = rng.choice(pool_size, size=min(int(counts[d]), pool_size), replace=False)
        picked.extend((int(d), int(v)) for v in variants)
    return tuple(picked)


def poi_text_rngs(
    seed: int, destination_index: int, poi_index: int
) -> tuple[np.random.Generator, np.random.Generator]:
    """(phrase-selection stream, surface-realization stream) for one POI -- two
    independent, deterministic generators keyed by (seed, destination, POI index)."""
    return (
        np.random.default_rng(
            [seed, _TEXT_STREAM_TAG, destination_index, poi_index, _PHRASE_STREAM]
        ),
        np.random.default_rng(
            [seed, _TEXT_STREAM_TAG, destination_index, poi_index, _SURFACE_STREAM]
        ),
    )


def _capitalize(sentence: str) -> str:
    return sentence[0].upper() + sentence[1:] if sentence else sentence


def _uncapitalize(sentence: str) -> str:
    return sentence[0].lower() + sentence[1:] if sentence else sentence


def realize_description(
    rng: np.random.Generator, category: str, phrases: list[str], text_cfg: TextConfig
) -> str:
    """Surface form of a description carrying exactly `phrases` (their order, grouping
    into sentences, sentence frames, connectives, and filler are all sampled here).

    `rng` must be the POI's SURFACE stream: nothing in here branches on which
    phrases were chosen (only on how many), so surface variety cannot leak into or dilute
    the semantic link, and the same phrase set yields different text under different seeds.
    """
    fmt = {"category_word": category.replace("_", " ")}

    order = rng.permutation(len(phrases))
    shuffled = [phrases[int(i)] for i in order]

    k = text_cfg.surface_variants
    openers = OPENERS[:k]
    sentences: list[str] = [openers[int(rng.integers(len(openers)))].format(**fmt)]
    cursor = 0
    while cursor < len(shuffled):
        size = min(len(shuffled) - cursor, int(rng.integers(1, 4)))
        frames = SENTENCE_FRAMES[size][:k]
        frame = frames[int(rng.integers(len(frames)))]
        slots = {f"p{i + 1}": shuffled[cursor + i] for i in range(size)}
        sentence = _capitalize(frame.format(**slots))
        if rng.random() < text_cfg.connective_rate:
            connective = CONNECTIVES[int(rng.integers(len(CONNECTIVES)))]
            sentence = f"{connective} {_uncapitalize(sentence)}"
        sentences.append(sentence)
        cursor += size

    n_fillers = int(rng.integers(0, text_cfg.max_fillers + 1))
    for _ in range(n_fillers):
        filler = FILLER_SENTENCES[int(rng.integers(len(FILLER_SENTENCES)))].format(**fmt)
        sentences.insert(int(rng.integers(1, len(sentences) + 1)), filler)
    return " ".join(sentences)


def generate_name(
    rng: np.random.Generator, destination: str, category: str, adjective: str | None = None
) -> str:
    """Sample a lexically varied POI name from category/destination templates.

    `adjective` (when given) is a synonym of a latent dimension the POI actually loads on
    (`generate_poi_text` draws it loading-proportionally), so the name's one open-class
    adjective is signal rather than noise; without it a generic adjective is drawn."""
    adj = adjective.title() if adjective else ADJECTIVES[int(rng.integers(len(ADJECTIVES)))]
    district = DISTRICT_WORDS[destination][int(rng.integers(len(DISTRICT_WORDS[destination])))]
    noun = NAME_NOUNS[category][int(rng.integers(len(NAME_NOUNS[category])))]
    return f"{adj} {district} {noun}"


def generate_poi_text(
    phrase_rng: np.random.Generator,
    surface_rng: np.random.Generator,
    destination: str,
    category: str,
    poi_semantic: FloatArray,
    text_cfg: TextConfig,
    pools: list[list[str]],
) -> PoiText:
    """Name + description for one POI, conditioned on its own `poi_semantic`.

    Draws 3-6 phrases (`text_cfg.phrases_per_poi_min/max`) via loading-proportional
    sampling on `phrase_rng`, then realizes the surface text on `surface_rng`.
    """
    n_phrases = int(
        phrase_rng.integers(text_cfg.phrases_per_poi_min, text_cfg.phrases_per_poi_max + 1)
    )
    phrase_ids = sample_phrases(
        phrase_rng, poi_semantic, pools, n_phrases, text_cfg.background_rate, text_cfg.sampling
    )
    phrases = [pools[d][k] for d, k in phrase_ids]
    description = realize_description(surface_rng, category, phrases, text_cfg)
    name_dimension = int(
        phrase_rng.choice(
            TASTE_DIM, p=dimension_sampling_weights(poi_semantic, text_cfg.background_rate)
        )
    )
    name_synonyms = DIMENSION_SYNONYMS[_synonym_key(name_dimension)]
    adjective = name_synonyms[int(phrase_rng.integers(len(name_synonyms)))].replace("-", " ")
    name = generate_name(surface_rng, destination, category, adjective)
    return PoiText(name=name, description=description, phrase_ids=phrase_ids)


def generate_tags(
    rng: np.random.Generator, category_tag_affinity: dict[str, float], n_tags: int
) -> list[str]:
    """Sample `n_tags` distinct tags, weighted toward the category's typical tags."""
    weights = np.array([category_tag_affinity.get(t, 0.15) for t in TAGS], dtype=np.float64)
    weights = weights / weights.sum()
    n_tags = min(n_tags, len(TAGS))
    chosen = rng.choice(len(TAGS), size=n_tags, replace=False, p=weights)
    return [TAGS[i] for i in chosen]
