"""`choice_index` must be a bit-identical, cheaper drop-in for `Generator.choice(n, p=p)`."""

from __future__ import annotations

import numpy as np

from poi_rank.datagen.sampling import choice_index


def test_choice_index_matches_generator_choice_and_stream_position() -> None:
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    src = np.random.default_rng(7)
    for _ in range(2000):
        n = int(src.integers(1, 40))
        p = src.random(n)
        p /= p.sum()
        assert choice_index(a, p) == int(b.choice(n, p=p))
    # Same number of draws consumed: the next raw draw agrees, so downstream RNG use is unchanged.
    assert a.random() == b.random()
