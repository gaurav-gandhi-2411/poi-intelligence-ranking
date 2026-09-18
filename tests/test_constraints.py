"""`eval/constraints.py` tests: hand-computed % top-10 with `compatibility >= 0.7`
(spec.md section 11.5)."""

from __future__ import annotations

import pytest

from poi_rank.eval.constraints import share_with_compatibility_above_target


def test_share_with_compatibility_above_target_hand_computed() -> None:
    values = [0.9, 0.5, 0.7, 0.65, 0.71]
    result = share_with_compatibility_above_target(values, target=0.7)
    assert result.n_recommended == 5
    assert result.n_above_target == 3  # 0.9, 0.7, 0.71
    assert result.share_above_target == pytest.approx(0.6)


def test_share_with_compatibility_above_target_empty_is_zero() -> None:
    result = share_with_compatibility_above_target([], target=0.7)
    assert result.n_recommended == 0
    assert result.share_above_target == 0.0
