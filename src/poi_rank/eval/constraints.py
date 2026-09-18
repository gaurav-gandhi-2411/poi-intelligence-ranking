"""Constraint-compatibility summary metric (spec.md section 11.5): % of top-10
recommendations with `compatibility >= 0.7`. The build-blocking half of section
11.5 ("hard-constraint violation rate in top-10 must be 0.000") is ALREADY enforced
by `tests/test_hard_constraints.py` (Phase 6, genuinely passing, never `xfail`) and
by `scoring.output.assemble_output_payload`'s own defensive runtime assertion --
this module only adds the quick descriptive percentage spec.md also asks for, it
does not re-implement or duplicate the hard-constraint enforcement itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COMPATIBILITY_TARGET = 0.7


@dataclass(frozen=True)
class CompatibilityShareResult:
    n_recommended: int
    n_above_target: int
    share_above_target: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility_target": COMPATIBILITY_TARGET,
            "n_recommended": self.n_recommended,
            "n_above_target": self.n_above_target,
            "share_above_target": self.share_above_target,
        }


def share_with_compatibility_above_target(
    compatibility_values: list[float], target: float = COMPATIBILITY_TARGET
) -> CompatibilityShareResult:
    """`compatibility_values`: the `context_compatibility` field of every entry
    across every trip's top-10 recommendation list (flattened)."""
    n = len(compatibility_values)
    n_above = sum(1 for v in compatibility_values if v >= target)
    share = n_above / n if n > 0 else 0.0
    return CompatibilityShareResult(
        n_recommended=n, n_above_target=n_above, share_above_target=share
    )
