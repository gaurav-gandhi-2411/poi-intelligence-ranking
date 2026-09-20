"""Holdout effect of experiment H on its own (seed 42, same K, same corrected features).

The adopted configuration (20 xf_ cross features, num_leaves 15, linear label_gain) was selected on
validation only. This sandbox run re-runs the whole pipeline WITHOUT H (no xf_ columns via
`POI_RANK_DISABLE_XF=1`, and the pre-H model params num_leaves 31 / label_gain [0,1,3,7]) so the
holdout can be compared like-for-like with the committed pipeline. It changes no committed file
except `results/experiments/h/holdout_ablation_no_h_seed42.json`.

    uv run python scripts/h_holdout_ablation.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("C:/Users/gaura/h_ablation_scratch")


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "seedrep", ROOT / "scripts" / "seed_replication.py"
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["seedrep"] = m
    spec.loader.exec_module(m)
    cfgs = SCRATCH / "configs_src"
    if cfgs.exists():
        shutil.rmtree(cfgs)
    shutil.copytree(ROOT / "configs", cfgs)
    model = (cfgs / "model.yaml").read_text(encoding="utf-8")
    model, n1 = re.subn(r"(?m)^  num_leaves: 15", "  num_leaves: 31", model)
    model, n2 = re.subn(r"(?m)^  label_gain: \[0, 1, 2, 3\]", "  label_gain: [0, 1, 3, 7]", model)
    assert n1 == 1 and n2 == 1, (n1, n2)
    (cfgs / "model.yaml").write_text(model, encoding="utf-8")
    m.CONFIGS = cfgs
    os.environ["POI_RANK_DISABLE_XF"] = "1"
    res = m.run_seed(42, SCRATCH)
    head = m.headline(res)
    out = {
        "description": "seed 42, K as committed, corrected features, WITHOUT experiment H "
        "(no xf_ columns, num_leaves 31, label_gain [0,1,3,7]); holdout headline",
        "headline": head,
        "ndcg10_primary_ci": [
            res["systems"]["lambdamart_ips"]["metrics"]["ndcg@10"]["ci_low"],
            res["systems"]["lambdamart_ips"]["metrics"]["ndcg@10"]["ci_high"],
        ],
        "wilcoxon_primary_vs_content_cosine_p": res["wilcoxon"]["lambdamart_ips_vs_content_cosine"][
            "p_value"
        ],
    }
    path = ROOT / "results" / "experiments" / "h" / "holdout_ablation_no_h_seed42.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
