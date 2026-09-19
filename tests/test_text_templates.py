"""`datagen/text_templates.py` (A2, docs/DATA_CARD.md "A2"): phrase pools, loading-
proportional sampling, surface-realization independence, determinism, config validation.
All hand-built inputs; the full-catalog effect on D10 is measured in
`tests/test_dgp_diagnostics.py` and `results/parts/d10_vocab_sweep.json`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poi_rank.datagen.config import DatagenConfig, TextConfig
from poi_rank.datagen.taxonomy import TASTE_DIM
from poi_rank.datagen.text_templates import (
    DIMENSION_NAMES,
    DIMENSION_SYNONYMS,
    MAX_PHRASES_PER_DIMENSION,
    _synonym_key,
    _systematic_counts,
    build_phrase_pools,
    dimension_sampling_weights,
    generate_poi_text,
    poi_text_rngs,
    realize_description,
    sample_phrases,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"


def _text_cfg(**overrides: object) -> TextConfig:
    base: dict[str, object] = {
        "phrases_per_dimension": 20,
        "phrases_per_poi_min": 10,
        "phrases_per_poi_max": 14,
        "background_rate": 0.02,
        "surface_variants": 2,
        "connective_rate": 0.25,
        "max_fillers": 1,
    }
    base.update(overrides)
    return TextConfig(**base)  # type: ignore[arg-type]


# ---- phrase pools -----------------------------------------------------------------------


@pytest.mark.parametrize("n_per_dimension", [3, 8, 15, 20, 25])
@pytest.mark.parametrize("anchor", [True, False])
def test_pool_size_and_within_dimension_distinctness(n_per_dimension: int, anchor: bool) -> None:
    pools = build_phrase_pools(n_per_dimension, anchor_phrases=anchor)
    assert len(pools) == TASTE_DIM == len(DIMENSION_NAMES)
    for pool in pools:
        assert len(pool) == n_per_dimension
        assert len(set(pool)) == n_per_dimension


def test_full_size_pools_are_globally_distinct_across_all_dimensions() -> None:
    pools = build_phrase_pools(MAX_PHRASES_PER_DIMENSION)
    flat = [phrase for pool in pools for phrase in pool]
    assert len(flat) == TASTE_DIM * MAX_PHRASES_PER_DIMENSION
    assert len(set(flat)) == len(flat)


def test_every_dimension_has_five_distinct_synonyms() -> None:
    for d in range(TASTE_DIM):
        synonyms = DIMENSION_SYNONYMS[_synonym_key(d)]
        assert len(synonyms) == 5
        assert len(set(synonyms)) == 5


def test_phrases_within_a_dimension_share_synonyms_not_arbitrary_words() -> None:
    """Paraphrase structure: a 25-phrase pool is built from just 5 synonyms (each reused
    with 5 heads), and with anchoring every phrase also carries the trait's plain word."""
    romantic = DIMENSION_NAMES.index("romantic")
    pool = build_phrase_pools(25)[romantic]
    assert {p.split()[0] for p in pool} == set(DIMENSION_SYNONYMS["romantic"])
    assert all("romantic" in p for p in pool)  # anchored (synonym "romantic" or explicit anchor)


def test_unanchored_pool_drops_the_plain_trait_word_for_non_covering_synonyms() -> None:
    hidden = DIMENSION_NAMES.index("hidden-gem")
    anchored = build_phrase_pools(5, anchor_phrases=True)[hidden]
    plain = build_phrase_pools(5, anchor_phrases=False)[hidden]
    assert all("hidden-gem" in p for p in anchored)
    assert all("hidden-gem" not in p for p in plain)


@pytest.mark.parametrize("bad", [0, MAX_PHRASES_PER_DIMENSION + 1])
def test_pool_size_out_of_range_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        build_phrase_pools(bad)


# ---- loading-proportional sampling ---------------------------------------------------------


def test_sampling_weights_hand_computed() -> None:
    semantic = np.zeros(TASTE_DIM)
    semantic[0], semantic[1], semantic[2], semantic[3] = 0.6, 0.3, 0.1, -0.5
    weights = dimension_sampling_weights(semantic, background_rate=0.1)
    # loadings clip to [0.6, 0.3, 0.1, 0, ...] sum 1.0; w = 0.9 * loading + 0.1 / 32
    uniform = 0.1 / TASTE_DIM
    assert weights[0] == pytest.approx(0.9 * 0.6 + uniform)
    assert weights[1] == pytest.approx(0.9 * 0.3 + uniform)
    assert weights[2] == pytest.approx(0.9 * 0.1 + uniform)
    assert weights[3] == pytest.approx(uniform)  # negative loading -> background only
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()  # no phrase is ever deterministic-excluded


def test_sampling_weights_all_nonpositive_falls_back_to_uniform() -> None:
    weights = dimension_sampling_weights(-np.ones(TASTE_DIM), background_rate=0.05)
    assert weights == pytest.approx(np.full(TASTE_DIM, 1.0 / TASTE_DIM))


@pytest.mark.parametrize("method", ["systematic", "multinomial"])
def test_mention_frequency_is_proportional_to_loading(method: str) -> None:
    pools = build_phrase_pools(20)
    semantic = np.zeros(TASTE_DIM)
    semantic[5], semantic[6], semantic[7] = 0.6, 0.3, 0.1
    n_phrases, trials = 6, 3000
    rng = np.random.default_rng(0)
    mentions = np.zeros(TASTE_DIM)
    for _ in range(trials):
        for d, _k in sample_phrases(rng, semantic, pools, n_phrases, 0.0, method):
            mentions[d] += 1
    freq = mentions / mentions.sum()
    assert freq[5] == pytest.approx(0.6, abs=0.03)
    assert freq[6] == pytest.approx(0.3, abs=0.03)
    assert freq[7] == pytest.approx(0.1, abs=0.03)
    assert freq[[i for i in range(TASTE_DIM) if i not in (5, 6, 7)]].sum() == 0.0


def test_background_rate_gives_every_dimension_some_mentions() -> None:
    pools = build_phrase_pools(20)
    semantic = np.zeros(TASTE_DIM)
    semantic[0] = 1.0
    rng = np.random.default_rng(1)
    seen = np.zeros(TASTE_DIM, dtype=bool)
    for _ in range(4000):
        for d, _k in sample_phrases(rng, semantic, pools, 12, 0.2):
            seen[d] = True
    assert seen.all()


def test_systematic_counts_are_floor_or_ceil_of_expectation() -> None:
    weights = np.array([0.5, 0.3, 0.2] + [0.0] * (TASTE_DIM - 3))
    rng = np.random.default_rng(2)
    for _ in range(200):
        counts = _systematic_counts(rng, weights, 7)
        assert counts.sum() == 7
        assert counts[0] in (3, 4)  # 7 * 0.5 = 3.5
        assert counts[1] in (2, 3)  # 7 * 0.3 = 2.1
        assert counts[2] in (1, 2)  # 7 * 0.2 = 1.4


def test_sampled_phrases_are_distinct_and_in_range() -> None:
    pools = build_phrase_pools(15)
    semantic = np.zeros(TASTE_DIM)
    semantic[0] = 1.0  # one dominant dimension -> repeated mentions of it
    rng = np.random.default_rng(3)
    for method in ("systematic", "multinomial"):
        picked = sample_phrases(rng, semantic, pools, 10, 0.0, method)
        assert len(set(picked)) == len(picked)
        assert all(0 <= d < TASTE_DIM and 0 <= k < 15 for d, k in picked)


def test_unknown_sampling_method_raises() -> None:
    with pytest.raises(ValueError):
        sample_phrases(
            np.random.default_rng(0), np.ones(TASTE_DIM), build_phrase_pools(5), 3, 0.1, "bogus"
        )


# ---- surface realization independence ----------------------------------------------------------


def test_same_phrase_set_gives_different_surface_text_across_seeds() -> None:
    phrases = ["candlelit romantic charm", "hidden hidden-gem vibe", "cozy-indoor indoor feel"]
    cfg = _text_cfg(surface_variants=4, connective_rate=0.5, max_fillers=1)
    texts = {
        realize_description(np.random.default_rng(seed), "cafe", phrases, cfg) for seed in range(40)
    }
    assert len(texts) > 10  # many distinct surface forms
    for text in texts:
        assert all(p in text.lower() for p in phrases)  # same phrase set every time


def test_surface_form_does_not_depend_on_which_phrases_were_chosen() -> None:
    """Same surface seed + same NUMBER of phrases -> identical skeleton regardless of the
    phrase identities (swap placeholders and the texts must match exactly)."""
    cfg = _text_cfg(surface_variants=4, connective_rate=0.5, max_fillers=2)
    a = ["AAA1 x", "AAA2 x", "AAA3 x", "AAA4 x", "AAA5 x"]
    b = ["BBB1 x", "BBB2 x", "BBB3 x", "BBB4 x", "BBB5 x"]
    for seed in range(25):
        text_a = realize_description(np.random.default_rng(seed), "museum", a, cfg)
        text_b = realize_description(np.random.default_rng(seed), "museum", b, cfg)
        assert text_a.lower().replace("aaa", "bbb") == text_b.lower()


def test_description_names_the_pois_own_category() -> None:
    text = realize_description(np.random.default_rng(0), "wellness_spa", ["x y"], _text_cfg())
    assert "wellness spa" in text


# ---- determinism + text depends on poi_semantic ---------------------------------------------


def test_generate_poi_text_is_deterministic_and_streams_are_keyed() -> None:
    pools = build_phrase_pools(20)
    semantic = np.random.default_rng(5).random(TASTE_DIM)
    cfg = _text_cfg()
    a = generate_poi_text(*poi_text_rngs(42, 0, 7), "seoul", "cafe", semantic, cfg, pools)
    b = generate_poi_text(*poi_text_rngs(42, 0, 7), "seoul", "cafe", semantic, cfg, pools)
    assert a == b
    c = generate_poi_text(*poi_text_rngs(42, 0, 8), "seoul", "cafe", semantic, cfg, pools)
    assert c != a  # different POI index -> different streams


def test_phrase_count_respects_configured_range() -> None:
    pools = build_phrase_pools(20)
    semantic = np.random.default_rng(6).random(TASTE_DIM)
    cfg = _text_cfg(phrases_per_poi_min=10, phrases_per_poi_max=14)
    counts = {
        len(
            generate_poi_text(
                *poi_text_rngs(42, 0, i), "kyoto", "museum", semantic, cfg, pools
            ).phrase_ids
        )
        for i in range(60)
    }
    assert counts <= set(range(10, 15))
    assert len(counts) > 1


def test_text_changes_when_only_poi_semantic_changes() -> None:
    pools = build_phrase_pools(20)
    cfg = _text_cfg()
    rng = np.random.default_rng(9)
    changed = 0
    n = 50
    for i in range(n):
        s1, s2 = rng.random(TASTE_DIM), rng.random(TASTE_DIM)
        a = generate_poi_text(*poi_text_rngs(42, 1, i), "kyoto", "cafe", s1, cfg, pools)
        b = generate_poi_text(*poi_text_rngs(42, 1, i), "kyoto", "cafe", s2, cfg, pools)
        changed += a.description != b.description
    assert changed / n > 0.9


def test_two_full_catalog_generations_are_identical() -> None:
    from poi_rank.datagen.catalog import generate_destination_pois
    from poi_rank.datagen.timeline import build_timeline

    cfg = DatagenConfig.from_yaml(CONFIG_PATH)
    timeline = build_timeline(cfg)
    a = generate_destination_pois(np.random.default_rng(1), "seoul", 80, cfg, timeline)
    b = generate_destination_pois(np.random.default_rng(1), "seoul", 80, cfg, timeline)
    assert a["description"].tolist() == b["description"].tolist()
    assert a["name"].tolist() == b["name"].tolist()


def test_text_is_not_degenerate_within_a_category() -> None:
    """Phase 1 non-degeneracy: descriptions within one category are overwhelmingly unique."""
    from poi_rank.datagen.catalog import generate_destination_pois
    from poi_rank.datagen.timeline import build_timeline

    cfg = DatagenConfig.from_yaml(CONFIG_PATH)
    df = generate_destination_pois(np.random.default_rng(4), "seoul", 500, cfg, build_timeline(cfg))
    df = df.loc[~df["is_duplicate"]]
    for _category, group in df.groupby("category"):
        assert group["description"].nunique() / len(group) > 0.95
    assert df["name"].nunique() / len(df) > 0.7


# ---- config validation ----------------------------------------------------------------------


def test_shipped_config_phrase_pool_size_is_in_spec_range() -> None:
    cfg = DatagenConfig.from_yaml(CONFIG_PATH)
    assert 15 <= cfg.text.phrases_per_dimension <= 25


def test_from_yaml_rejects_phrase_pool_size_outside_15_to_25(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "phrases_per_dimension: 20", "phrases_per_dimension: 8"
    )
    bad = tmp_path / "datagen.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="phrases_per_dimension"):
        DatagenConfig.from_yaml(bad)


def test_text_config_validation() -> None:
    with pytest.raises(ValueError):
        _text_cfg(phrases_per_dimension=0)
    with pytest.raises(ValueError):
        _text_cfg(phrases_per_poi_min=5, phrases_per_poi_max=4)
    with pytest.raises(ValueError):
        _text_cfg(background_rate=1.0)
    with pytest.raises(ValueError):
        _text_cfg(sampling="bogus")
    with pytest.raises(ValueError):
        _text_cfg(surface_variants=0)
    _text_cfg(phrases_per_dimension=3)  # sweep values below the shipped range are allowed
