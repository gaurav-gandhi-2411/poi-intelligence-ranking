"""POI catalog generation: ~500 POIs per destination with realistic structure and
deliberately injected dirtiness (spec.md section 2.2).

`generate_all_catalogs` returns the full "true" (internal) catalog — including latent
oracle-only fields (`latent_quality`, `latent_localness`, `poi_semantic`) and the true
values of fields that get nulled on export (`price_level`, `expected_duration_min`,
`opening_hours`). `apply_catalog_dirtiness` derives the exported, messy catalog from
it. Downstream utility computation (`utility.py`) is given the true df, never the
export df — missingness in the export df represents an incomplete *listing*, not an
absent real-world attribute (documented in docs/DATA_CARD.md).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.taxonomy import (
    CATEGORIES,
    CATEGORY_INDEX,
    CATEGORY_PROBS,
    CATEGORY_STRING_VARIANTS,
    SUBCATEGORIES,
    TAG_INDEX,
    TAGS,
    TASTE_DIM,
)
from poi_rank.datagen.text_templates import (
    build_phrase_pools,
    generate_poi_text,
    generate_tags,
    poi_text_rngs,
)
from poi_rank.datagen.timeline import Timeline

DEST_CENTERS: dict[str, tuple[float, float]] = {
    "seoul": (37.5665, 126.9780),
    "kyoto": (35.0116, 135.7681),
    "barcelona": (41.3874, 2.1686),
}
DEST_CODES: dict[str, str] = {"seoul": "SEO", "kyoto": "KYO", "barcelona": "BCN"}

# Category-level generative priors. All hand-tuned to give a plausible, non-uniform
# catalog; exact values are not load-bearing for evaluation validity, only realism.
LOCALNESS_SHIFT: dict[str, float] = {
    "restaurant": 0.05,
    "cafe": 0.08,
    "museum": -0.05,
    "historic_site": -0.15,
    "nature_park": 0.05,
    "nightlife": 0.0,
    "shopping": -0.05,
    "family_activity": 0.0,
    "wellness_spa": 0.05,
    "religious_site": -0.05,
    "viewpoint": -0.15,
    "entertainment": -0.05,
}
POPULARITY_LOG_BONUS: dict[str, float] = {
    "restaurant": 0.0,
    "cafe": -0.1,
    "museum": 0.2,
    "historic_site": 0.4,
    "nature_park": 0.1,
    "nightlife": 0.1,
    "shopping": 0.2,
    "family_activity": 0.2,
    "wellness_spa": -0.1,
    "religious_site": 0.1,
    "viewpoint": 0.5,
    "entertainment": 0.1,
}
PRICE_LEVEL_PROBS: dict[str, list[float]] = {
    "restaurant": [0.25, 0.35, 0.25, 0.15],
    "cafe": [0.45, 0.40, 0.12, 0.03],
    "museum": [0.30, 0.40, 0.20, 0.10],
    "historic_site": [0.40, 0.35, 0.15, 0.10],
    "nature_park": [0.70, 0.20, 0.08, 0.02],
    "nightlife": [0.15, 0.35, 0.30, 0.20],
    "shopping": [0.20, 0.30, 0.30, 0.20],
    "family_activity": [0.25, 0.40, 0.25, 0.10],
    "wellness_spa": [0.10, 0.30, 0.35, 0.25],
    "religious_site": [0.80, 0.15, 0.04, 0.01],
    "viewpoint": [0.50, 0.30, 0.15, 0.05],
    "entertainment": [0.20, 0.40, 0.25, 0.15],
}
DURATION_BASE_MIN: dict[str, float] = {
    "restaurant": 75,
    "cafe": 45,
    "museum": 100,
    "historic_site": 70,
    "nature_park": 90,
    "nightlife": 120,
    "shopping": 60,
    "family_activity": 150,
    "wellness_spa": 90,
    "religious_site": 40,
    "viewpoint": 30,
    "entertainment": 110,
}
# (open_hour, close_hour) in 24h float; close < open means it wraps past midnight.
OPENING_HOURS_BASE: dict[str, tuple[float, float]] = {
    "restaurant": (11.0, 22.0),
    "cafe": (8.0, 19.0),
    "museum": (9.5, 17.5),
    "historic_site": (9.0, 18.0),
    "nature_park": (6.0, 20.0),
    "nightlife": (18.0, 2.0),
    "shopping": (10.0, 20.0),
    "family_activity": (9.0, 18.0),
    "wellness_spa": (10.0, 21.0),
    "religious_site": (7.0, 19.0),
    "viewpoint": (8.0, 21.0),
    "entertainment": (12.0, 23.0),
}
INDOOR_OUTDOOR_PROBS: dict[str, dict[str, float]] = {
    "restaurant": {"indoor": 0.85, "outdoor": 0.05, "mixed": 0.10},
    "cafe": {"indoor": 0.80, "outdoor": 0.05, "mixed": 0.15},
    "museum": {"indoor": 0.95, "outdoor": 0.0, "mixed": 0.05},
    "historic_site": {"indoor": 0.15, "outdoor": 0.45, "mixed": 0.40},
    "nature_park": {"indoor": 0.0, "outdoor": 0.85, "mixed": 0.15},
    "nightlife": {"indoor": 0.70, "outdoor": 0.05, "mixed": 0.25},
    "shopping": {"indoor": 0.70, "outdoor": 0.05, "mixed": 0.25},
    "family_activity": {"indoor": 0.30, "outdoor": 0.35, "mixed": 0.35},
    "wellness_spa": {"indoor": 0.90, "outdoor": 0.0, "mixed": 0.10},
    "religious_site": {"indoor": 0.55, "outdoor": 0.10, "mixed": 0.35},
    "viewpoint": {"indoor": 0.05, "outdoor": 0.75, "mixed": 0.20},
    "entertainment": {"indoor": 0.90, "outdoor": 0.0, "mixed": 0.10},
}
# (wheelchair_p, stroller_p, kid_friendly_p) per category.
ACCESSIBILITY_PROBS: dict[str, tuple[float, float, float]] = {
    "restaurant": (0.6, 0.6, 0.5),
    "cafe": (0.6, 0.65, 0.5),
    "museum": (0.75, 0.7, 0.6),
    "historic_site": (0.35, 0.3, 0.4),
    "nature_park": (0.4, 0.55, 0.7),
    "nightlife": (0.3, 0.1, 0.05),
    "shopping": (0.7, 0.65, 0.55),
    "family_activity": (0.6, 0.85, 0.95),
    "wellness_spa": (0.5, 0.15, 0.1),
    "religious_site": (0.4, 0.35, 0.4),
    "viewpoint": (0.35, 0.3, 0.45),
    "entertainment": (0.65, 0.4, 0.4),
}
RESERVATION_BASE_PROB: dict[str, float] = {
    "restaurant": 0.25,
    "cafe": 0.02,
    "museum": 0.10,
    "historic_site": 0.05,
    "nature_park": 0.02,
    "nightlife": 0.15,
    "shopping": 0.01,
    "family_activity": 0.10,
    "wellness_spa": 0.40,
    "religious_site": 0.02,
    "viewpoint": 0.05,
    "entertainment": 0.15,
}

CATEGORY_TAG_AFFINITY: dict[str, dict[str, float]] = {
    "restaurant": {"foodie": 0.9, "local": 0.6, "authentic": 0.6, "budget-friendly": 0.4},
    "cafe": {"foodie": 0.6, "trendy": 0.5, "relaxing": 0.5, "local": 0.4},
    "museum": {"cultural": 0.9, "artsy": 0.7, "historic": 0.4},
    "historic_site": {"historic": 0.9, "cultural": 0.7, "touristy": 0.5},
    "nature_park": {"nature": 0.9, "outdoor": 0.9, "relaxing": 0.6},
    "nightlife": {"lively": 0.9, "trendy": 0.6, "touristy": 0.3},
    "shopping": {"shopping": 0.9, "trendy": 0.5, "luxury": 0.3},
    "family_activity": {"family-friendly": 0.9, "outdoor": 0.4, "lively": 0.4},
    "wellness_spa": {"relaxing": 0.9, "luxury": 0.5},
    "religious_site": {"cultural": 0.7, "historic": 0.6, "touristy": 0.3},
    "viewpoint": {"touristy": 0.7, "scenic": 0.0, "outdoor": 0.6},
    "entertainment": {"lively": 0.6, "trendy": 0.5, "touristy": 0.3},
}


def _sample_category_column(rng: np.random.Generator, n: int) -> npt.NDArray[np.str_]:
    cats = list(CATEGORY_PROBS)
    probs = np.array([CATEGORY_PROBS[c] for c in cats])
    probs = probs / probs.sum()
    result: npt.NDArray[np.str_] = rng.choice(cats, size=n, p=probs)
    return result


def _poi_semantic_vector(
    rng: np.random.Generator, category: str, tags: list[str], noise_std: float = 0.06
) -> npt.NDArray[np.float64]:
    """Build a POI's latent semantic vector: category one-hot + tag weights + noise.

    Lives in the same TASTE_DIM space as traveler taste vectors (taxonomy.py), which
    is what makes `cos(taste_t, poi_semantic_p)` meaningful.

    RC2c (docs/DATA_CARD.md "DGP remediation, Block A"): `noise_std` lowered from
    0.25 to 0.06 -- measured directly (a noise-std sweep against the real TF-IDF
    text pipeline) to be the primary lever unlocking D10 headroom: at std=0.25 the
    independent per-dimension noise was comparable in magnitude to the
    category+tag signal itself (~2.0 vs ~2.8 total variance across 32 dims),
    capping ANY text-conditioning mechanism's achievable correlation regardless of
    template design. A small, non-zero noise floor is kept deliberately (never
    fully eliminated) so `poi_semantic` is not a purely deterministic function of
    `category`/`tags` alone.
    """
    vec = np.zeros(TASTE_DIM, dtype=np.float64)
    vec[CATEGORY_INDEX[category]] = 1.0
    for tag in tags:
        vec[TAG_INDEX[tag]] = 0.6
    vec += rng.normal(0.0, noise_std, size=TASTE_DIM)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def dominant_flavors_from_semantic(poi_semantic: npt.NDArray[np.float64], k: int = 3) -> list[str]:
    """RC2c (docs/DATA_CARD.md "DGP remediation, Block A"): the top-`k` TAGS by
    value in `poi_semantic`'s own tag axis -- i.e. the latent vector's OWN dominant
    "semantic flavors," including whatever independent noise they carry, not
    merely the raw sampled `tags` list. A single argmax label loses too much of a
    32-dim vector's actual direction to give text a strong enough correlation
    signal (measured, see docs/DATA_CARD.md); top-k reflects substantially more of
    `poi_semantic`'s real content. Used to condition text generation so it
    actually reflects `poi_semantic` (fixing D10's measured rho=0.21
    code-confirmed-zero dependency edge)."""
    tag_axis = poi_semantic[len(CATEGORIES) :]
    order = np.argsort(-tag_axis)[:k]
    return [TAGS[i] for i in order]


def _sample_opening_hours(
    rng: np.random.Generator, category: str
) -> dict[str, dict[str, str] | None]:
    """Category-conditioned weekly hours; some categories close specific days."""
    open_h, close_h = OPENING_HOURS_BASE[category]
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    hours: dict[str, dict[str, str] | None] = {}
    closed_day_p = 0.5 if category == "museum" else 0.05
    for day in days:
        if day == "mon" and rng.random() < closed_day_p and category == "museum":
            hours[day] = None
            continue
        if rng.random() < 0.05:
            hours[day] = None
            continue
        jitter = rng.normal(0, 0.4)
        o = (open_h + jitter) % 24
        c = (close_h + jitter) % 24
        hours[day] = {
            "open": f"{int(o):02d}:{int((o % 1) * 60):02d}",
            "close": f"{int(c):02d}:{int((c % 1) * 60):02d}",
        }
    return hours


def _sample_avg_crowd_by_hour(rng: np.random.Generator, category: str) -> list[float]:
    open_h, close_h = OPENING_HOURS_BASE[category]
    hours = np.arange(24)
    span = (close_h - open_h) % 24 or 24
    center = (open_h + span / 2) % 24
    width = max(span / 4, 2.0)
    delta = np.minimum(np.abs(hours - center), 24 - np.abs(hours - center))
    curve = np.exp(-0.5 * (delta / width) ** 2)
    curve += rng.normal(0, 0.05, size=24)
    curve = np.clip(curve, 0.0, None)
    max_val = curve.max()
    curve = curve / max_val if max_val > 0 else curve
    return [round(float(v), 3) for v in curve]


def _sample_seasonality(rng: np.random.Generator, indoor_outdoor: str) -> list[str]:
    if indoor_outdoor == "indoor":
        return []
    months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
    k = int(rng.integers(4, 8))
    chosen = rng.choice(months, size=min(k, len(months)), replace=False)
    return sorted(chosen.tolist(), key=months.index)


def generate_destination_pois(
    rng: np.random.Generator,
    destination: str,
    n_pois: int,
    cfg: DatagenConfig,
    timeline: Timeline,
) -> pd.DataFrame:
    """Generate the true (internal, pre-dirtiness) POI catalog for one destination.

    Includes near-duplicate cloning per `dirtiness.near_duplicate_rate`: `n_pois` is
    the TOTAL row count target, split into unique base POIs plus clones so the
    duplicate rate is hit by construction rather than as a post-hoc export step
    (duplicates are real, independently-exposable catalog rows, not a display-only
    artifact — see docs/DATA_CARD.md).
    """
    dup_rate = cfg.dirtiness.near_duplicate_rate
    n_unique = max(1, round(n_pois / (1 + dup_rate)))
    n_dup = n_pois - n_unique

    center_lat, center_lon = DEST_CENTERS[destination]
    code = DEST_CODES[destination]

    categories = _sample_category_column(rng, n_unique)

    # latent_localness computed BEFORE lat/lon (RC2b, docs/DATA_CARD.md "DGP
    # remediation, Block A") so geo generation can condition on it -- confirmed bug
    # this fixes (D7, results/parts/dgp_diagnostics.json): the pre-remediation
    # generator sampled lat/lon as pure Gaussian jitter around the destination
    # center, entirely independent of latent_localness (measured Spearman rho=0.012).
    latent_localness = np.clip(
        rng.beta(2.0, 2.0, size=n_unique)
        + np.array([LOCALNESS_SHIFT[c] for c in categories])
        + rng.normal(0, 0.05, size=n_unique),
        0.0,
        1.0,
    )
    latent_quality = np.clip(rng.beta(2.5, 2.2, size=n_unique), 0.0, 1.0)

    # RC2b: radial distance from the destination center is now biased by
    # latent_localness -- touristy (low-localness) POIs cluster TOWARD the center
    # (which `data/localness.py::compute_tourist_centroid` -- a review-count-weighted
    # centroid of the top-popularity decile -- naturally tracks, since popularity is
    # already negatively correlated with localness via `pop_mu` below), local
    # (high-localness) POIs spread FARTHER away, with realistic per-POI noise so the
    # relationship is a genuine correlation, not a deterministic mapping.
    # `GEO_LOCALNESS_RADIUS_GAIN` is a tuned strength constant (see
    # docs/DATA_CARD.md for the achieved Spearman rho after tuning).
    GEO_LOCALNESS_RADIUS_GAIN = 2.4
    radius_unit = np.abs(rng.normal(0, 1.0, size=n_unique)) + GEO_LOCALNESS_RADIUS_GAIN * (
        latent_localness - 0.5
    )
    radius_unit = np.clip(radius_unit, 0.05, None)
    angle = rng.uniform(0, 2 * np.pi, size=n_unique)
    # Anisotropic degree scale roughly matching the pre-remediation isotropic-in-km
    # footprint (lat sd ~0.06 deg, lon sd ~0.08 deg -- longitude degrees are shorter
    # in km at these latitudes).
    lat_jitter = radius_unit * np.cos(angle) * 0.045
    lon_jitter = radius_unit * np.sin(angle) * 0.060
    lats = center_lat + lat_jitter
    lons = center_lon + lon_jitter

    # Base/sigma tuned so the *natural* (pre-dirtiness-injection) sparse-POI rate is a
    # small few percent, leaving `sparse_review_count_rate` in datagen.yaml as the
    # dominant driver of the final sparse share rather than an accidental byproduct of
    # the lognormal tail (see tests/test_datagen_schema.py::TestDirtiness).
    pop_mu = 4.4 + np.array([POPULARITY_LOG_BONUS[c] for c in categories]) - 1.0 * latent_localness
    popularity_raw = rng.lognormal(mean=pop_mu, sigma=0.75)

    review_count_raw = (
        popularity_raw * np.exp(rng.normal(0, 0.4, size=n_unique)) * (0.7 + 0.6 * latent_quality)
    )
    review_count = np.clip(np.round(review_count_raw), 1, None).astype(int)
    sparse_mask = rng.random(n_unique) < cfg.dirtiness.sparse_review_count_rate
    review_count[sparse_mask] = rng.integers(1, 10, size=int(sparse_mask.sum()))

    log_pop = np.log1p(popularity_raw)
    z_pop = (log_pop - log_pop.mean()) / (log_pop.std() + 1e-9)
    foreign_review_ratio = np.clip(
        1.0
        / (
            1.0
            + np.exp(-(0.8 * z_pop - 1.0 * latent_localness + rng.normal(0, 0.3, size=n_unique)))
        ),
        0.0,
        1.0,
    )

    # RC2a: observation-noise on rating/review_count tightened (docs/DATA_CARD.md
    # "DGP remediation, Block A") -- rating_shrunk (the downstream observable
    # feature) is meant to carry meaningfully more real signal about latent_quality
    # than a near-pure-noise observation, without eliminating the genuine Bayes
    # ceiling this noise is intentionally there to preserve (spec.md's own design).
    # Constant lowered 1.5 -> 0.7 (see docs/DATA_CARD.md for the measured
    # before/after correlation between latent_quality and rating/review_count).
    rating_noise_std = 0.7 / np.sqrt(review_count + 1.0)
    rating = np.clip(
        1.0 + 4.0 * latent_quality + rng.normal(0, 1.0, size=n_unique) * rating_noise_std, 1.0, 5.0
    )
    rating = np.round(rating, 1)

    price_level = np.array(
        [rng.choice([1, 2, 3, 4], p=PRICE_LEVEL_PROBS[c]) for c in categories], dtype=int
    )
    expected_duration = np.clip(
        np.array([DURATION_BASE_MIN[c] for c in categories])
        * np.exp(rng.normal(0, 0.3, size=n_unique)),
        15,
        300,
    )

    indoor_outdoor = np.array(
        [
            rng.choice(
                list(INDOOR_OUTDOOR_PROBS[c]),
                p=list(INDOOR_OUTDOOR_PROBS[c].values()),
            )
            for c in categories
        ]
    )

    acc_probs = np.array([ACCESSIBILITY_PROBS[c] for c in categories])
    wheelchair = rng.random(n_unique) < acc_probs[:, 0]
    stroller = rng.random(n_unique) < acc_probs[:, 1]
    kid_friendly = rng.random(n_unique) < acc_probs[:, 2]

    reservation_base = np.array([RESERVATION_BASE_PROB[c] for c in categories])
    subcats = [rng.choice(SUBCATEGORIES[c]) for c in categories]
    fine_dining_bump = np.array([0.4 if sc == "fine_dining" else 0.0 for sc in subcats])
    reservation_required = rng.random(n_unique) < np.clip(reservation_base + fine_dining_bump, 0, 1)
    reservation_lead_days = np.where(reservation_required, rng.integers(1, 15, size=n_unique), 0)

    created_at_new_mask = rng.random(n_unique) < cfg.dirtiness.new_poi_rate
    created_established = timeline.start - pd.Timedelta(days=int(rng.integers(1, 3 * 365)))
    created_at = []
    for is_new in created_at_new_mask:
        if is_new:
            days_span = max((timeline.end - timeline.split).days, 1)
            created_at.append(timeline.split + pd.Timedelta(days=int(rng.integers(0, days_span))))
        else:
            days_span = max(
                (timeline.split - (timeline.start - pd.Timedelta(days=3 * 365))).days, 1
            )
            created_at.append(
                timeline.start
                - pd.Timedelta(days=3 * 365)
                + pd.Timedelta(days=int(rng.integers(0, days_span)))
            )
    del created_established

    dest_index = list(DEST_CENTERS).index(destination)
    phrase_pools = build_phrase_pools(cfg.text.phrases_per_dimension, cfg.text.anchor_phrases)

    rows: list[dict[str, Any]] = []
    for i in range(n_unique):
        cat = categories[i]
        tags = generate_tags(rng, CATEGORY_TAG_AFFINITY[cat], n_tags=int(rng.integers(3, 7)))
        # RC2c/A2: poi_semantic computed BEFORE the text fields, then threaded INTO
        # name/description generation -- fixes the confirmed zero-dependency-edge bug
        # (D10, docs/DATA_CARD.md "DGP remediation, Block A"): the pre-remediation
        # generator computed poi_semantic strictly AFTER text generation, from
        # category+tags alone, so text had no way to reflect it.
        poi_semantic = _poi_semantic_vector(rng, cat, tags)
        flavors = dominant_flavors_from_semantic(poi_semantic, k=3)
        # Guarantee the dominant flavors are themselves literal, observable tag
        # words (RC2c, unchanged by A2) -- real POI listings commonly carry tags
        # matching their most salient traits, and this makes the OBSERVABLE `tags`
        # column genuinely track `poi_semantic` (poi_semantic's independent noise can
        # make a DIFFERENT tag dimension dominant -- see
        # `dominant_flavors_from_semantic`).
        for flavor_tag in flavors:
            if flavor_tag not in tags:
                tags = [*tags, flavor_tag]
        # A2: text comes from per-POI seeded streams (`poi_text_rngs`), NOT the shared
        # `rng` -- text-side knobs (phrase pool size, background rate) therefore cannot
        # perturb any other DGP quantity.
        phrase_rng, surface_rng = poi_text_rngs(cfg.seed, dest_index, i)
        poi_text = generate_poi_text(
            phrase_rng, surface_rng, destination, cat, poi_semantic, cfg.text, phrase_pools
        )
        name = poi_text.name
        description = poi_text.description
        opening_hours = _sample_opening_hours(rng, cat)
        avg_crowd = _sample_avg_crowd_by_hour(rng, cat)
        seasonality = _sample_seasonality(rng, indoor_outdoor[i])
        rows.append(
            {
                "poi_id": f"P{code}{i + 1:04d}",
                "destination": destination,
                "name": name,
                "category": cat,
                "subcategory": subcats[i],
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "description": description,
                "tags": tags,
                "price_level": int(price_level[i]),
                "rating": float(rating[i]),
                "review_count": int(review_count[i]),
                "foreign_review_ratio": float(foreign_review_ratio[i]),
                "popularity_raw": float(popularity_raw[i]),
                "expected_duration_min": float(expected_duration[i]),
                "opening_hours": opening_hours,
                "reservation_required": bool(reservation_required[i]),
                "reservation_lead_days": int(reservation_lead_days[i]),
                "wheelchair": bool(wheelchair[i]),
                "stroller": bool(stroller[i]),
                "kid_friendly": bool(kid_friendly[i]),
                "indoor_outdoor": str(indoor_outdoor[i]),
                "seasonality": seasonality,
                "avg_crowd_by_hour": avg_crowd,
                "created_at": created_at[i],
                "latent_quality": float(latent_quality[i]),
                "latent_localness": float(latent_localness[i]),
                "poi_semantic": poi_semantic,
                "is_duplicate": False,
                "duplicate_of": None,
            }
        )

    base_df = pd.DataFrame(rows)

    dup_rows: list[dict[str, Any]] = []
    if n_dup > 0:
        dup_source_idx = rng.integers(0, n_unique, size=n_dup)
        suffixes = [" Annex", " - New Wing", " (Branch)", " 2", " - East Wing"]
        for j, src_idx in enumerate(dup_source_idx):
            src = base_df.iloc[int(src_idx)].to_dict()
            clone = dict(src)
            clone["poi_id"] = f"P{code}{n_unique + j + 1:04d}"
            clone["name"] = src["name"] + rng.choice(suffixes)
            # <50m jitter: ~0.00045 deg latitude ~= 50m.
            clone["lat"] = src["lat"] + rng.uniform(-0.00040, 0.00040)
            clone["lon"] = src["lon"] + rng.uniform(-0.00040, 0.00040)
            clone["review_count"] = max(1, int(src["review_count"] * rng.uniform(0.05, 0.35)))
            clone["popularity_raw"] = float(src["popularity_raw"] * rng.uniform(0.1, 0.4))
            clone["latent_quality"] = float(
                np.clip(src["latent_quality"] + rng.normal(0, 0.03), 0, 1)
            )
            clone["latent_localness"] = float(
                np.clip(src["latent_localness"] + rng.normal(0, 0.03), 0, 1)
            )
            clone["rating"] = float(np.clip(src["rating"] + rng.normal(0, 0.1), 1, 5))
            clone["created_at"] = src["created_at"] + pd.Timedelta(days=int(rng.integers(0, 120)))
            clone["is_duplicate"] = True
            clone["duplicate_of"] = src["poi_id"]
            dup_rows.append(clone)

    dup_df = pd.DataFrame(dup_rows) if dup_rows else pd.DataFrame(columns=base_df.columns)
    return pd.concat([base_df, dup_df], ignore_index=True)


def generate_all_catalogs(
    rng: np.random.Generator, cfg: DatagenConfig, timeline: Timeline
) -> pd.DataFrame:
    """Generate the true (internal) POI catalog across all destinations."""
    frames = [
        generate_destination_pois(rng, dest, cfg.scale.pois_per_destination, cfg, timeline)
        for dest in DEST_CENTERS
    ]
    return pd.concat(frames, ignore_index=True)


def apply_catalog_dirtiness(
    rng: np.random.Generator, true_df: pd.DataFrame, cfg: DatagenConfig
) -> pd.DataFrame:
    """Derive the exported, messy POI catalog from the true internal catalog.

    Nulls fields per the configured missingness rates, replaces `category` with an
    inconsistent variant string for a subset of rows, and JSON-serializes
    `opening_hours` (None when missing). Does not touch `review_count` (already a
    genuine long-tail characteristic of the true catalog, not a reporting gap) and
    does not touch latent columns (dropped entirely — those belong only in the
    oracle export).
    """
    n = len(true_df)
    df = true_df.copy()

    price_missing = rng.random(n) < cfg.dirtiness.missing_price_level_rate
    duration_missing = rng.random(n) < cfg.dirtiness.missing_expected_duration_rate
    hours_missing = rng.random(n) < cfg.dirtiness.missing_opening_hours_rate
    category_variant_mask = rng.random(n) < cfg.dirtiness.inconsistent_category_rate

    price_level = df["price_level"].astype("Int64")
    price_level[price_missing] = pd.NA

    expected_duration = df["expected_duration_min"].astype(float)
    expected_duration[duration_missing] = np.nan

    opening_hours_json = [
        None if missing else json.dumps(oh)
        for missing, oh in zip(hours_missing, df["opening_hours"], strict=True)
    ]

    category_export = df["category"].copy()
    for idx in np.where(category_variant_mask)[0]:
        cat = df.loc[idx, "category"]
        variants = CATEGORY_STRING_VARIANTS[cat]
        category_export.iloc[idx] = rng.choice(variants)

    accessibility = [
        {"wheelchair": bool(w), "stroller": bool(s), "kid_friendly": bool(k)}
        for w, s, k in zip(df["wheelchair"], df["stroller"], df["kid_friendly"], strict=True)
    ]

    export = pd.DataFrame(
        {
            "poi_id": df["poi_id"],
            "destination": df["destination"],
            "name": df["name"],
            "category": category_export,
            "subcategory": df["subcategory"],
            "lat": df["lat"],
            "lon": df["lon"],
            "description": df["description"],
            "tags": df["tags"],
            "price_level": price_level,
            "rating": df["rating"],
            "review_count": df["review_count"],
            "foreign_review_ratio": df["foreign_review_ratio"],
            "popularity_raw": df["popularity_raw"],
            "expected_duration_min": expected_duration,
            # JSON-encoded 7-day open/close map (or None when missing) — avoids
            # parquet nested-struct nullability edge cases; documented in DATA_CARD.md.
            "opening_hours": opening_hours_json,
            "reservation_required": df["reservation_required"],
            "reservation_lead_days": df["reservation_lead_days"],
            "accessibility": accessibility,
            "indoor_outdoor": df["indoor_outdoor"],
            "seasonality": df["seasonality"],
            "avg_crowd_by_hour": df["avg_crowd_by_hour"],
            "created_at": df["created_at"],
            "is_duplicate": df["is_duplicate"],
            "duplicate_of": df["duplicate_of"],
        }
    )
    return export
