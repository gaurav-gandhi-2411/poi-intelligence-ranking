"""Localness-index validation against the DGP's oracle latent value (spec.md section
4): the single most load-bearing correctness signal of Phase 2.

spec.md section 4 frames the rho > 0.6 target as "a measured number, printed by
evaluate.py" -- a success criterion reported in the scorecard (`docs/RESULTS.md`, with its
component-level diagnosis), NOT a build-blocking invariant. The former `xfail(strict=True)`
metric-target test was removed (spec-v2 S5): build-blocking tests are reserved for invariants
(hard constraints = 0, firewall, oracle isolation, determinism, no leakage). What stays here is
the invariant that the index is positively and significantly correlated with the latent value.
"""

from __future__ import annotations

from typing import Any

from poi_rank.eval.oracle import validate_localness_against_oracle


def test_localness_spearman_rho_is_at_least_positive_and_significant(
    prepared_data: dict[str, Any],
) -> None:
    """Even though the index misses the 0.6 target, it must still be a genuinely
    informative (positive, statistically significant) signal -- not noise."""
    prepared = prepared_data["prepared"]
    oracle_dir = prepared_data["output_dir"] / "_oracle"

    result = validate_localness_against_oracle(prepared, oracle_dir)

    assert result.spearman_rho > 0.3
    assert result.p_value < 0.01
