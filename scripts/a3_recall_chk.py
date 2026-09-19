"""A3 step 0: per-stratum candidate recall vs chance + oracle-top-K recall ceiling (diagnostic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from poi_rank.candidates.recall_metrics import _canonical_holdout, recall_with_chance_lift

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "synthetic"
pois = pd.read_parquet(D / "pois_prepared.parquet")
c = pd.read_parquet(D / "candidates.parquet")
h = pd.read_parquet(D / "interactions_holdout_random.parquet")
r = recall_with_chance_lift(pois, c, h)
for k, v in r.items():
    print(k, {a: round(b, 3) for a, b in v.items()})

u = pd.read_parquet(D / "_oracle" / "holdout_utility_true.parquet")
u = u[u.poi_id.isin(set(pois.poi_id))]
hh = _canonical_holdout(pois, h)
pos = hh[hh.label >= 1].groupby("trip_id").poi_id.apply(set)
csz = c.groupby("trip_id").size()
rec = []
for t, g in u.groupby("trip_id"):
    if t not in pos.index:
        continue
    top = set(g.nlargest(int(csz[t]), "utility_true").poi_id)
    rec.append(len(pos[t] & top) / len(pos[t]))
ceil = float(np.mean(rec))
print("oracle-topK recall", ceil, len(rec))
out = {"recall_by_stratum": r, "oracle_topK_recall_ceiling_diagnostic": ceil}
(ROOT / "results" / "parts" / "a3_step0_recall.json").write_text(json.dumps(out, indent=2))
