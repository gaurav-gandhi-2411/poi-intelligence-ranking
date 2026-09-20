"""Experiment H harness: augment the ranking frame with explicit cross features and score
configurations on the train-carved VALIDATION split only (same protocol and metric as
`ranker_sweep.py`: IPS-weighted NDCG@10, seeds 42/7/11/13). See
`docs/experiments/H-ranker-cross-features.md` for the pre-registered protocol and adoption bar.
"""

from __future__ import annotations

import numpy as np

SEEDS = (42, 7, 11, 13)
# ADOPTION_BAR = shipped 4-seed validation mean (0.3167, results/parts/ranker_sweep.json) + 0.010.
ADOPTION_BAR = 0.3267


def paired_seed_stats(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired comparison of two per-seed series (a - b): mean gap, sd, paired t and Wilcoxon."""
    from scipy import stats

    d = np.asarray(a) - np.asarray(b)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b)
        w_p = float(w.pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "mean_gap": float(d.mean()),
        "sd_gap": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "n_seeds": float(len(d)),
        "n_a_above_b": float((d > 0).sum()),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": w_p,
    }
