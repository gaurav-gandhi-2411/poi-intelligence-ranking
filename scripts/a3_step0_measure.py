"""A3 step 0 (spec-v3 + GG's 2026-09-19 rulings): measure D9 (within-trip), D11 and the two
ablations on the EXISTING (A2-data) features BEFORE writing any new representation code.
Reporting only; writes results/parts/a3_step0.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from poi_rank.eval.representation import (
    d9_and_ablations,
    load_reference_inputs,
    ridge_recoverability,
)
from poi_rank.features.config import FeatureBuildConfig
from poi_rank.features.traveler_features import assemble_traveler_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"

t0 = time.perf_counter()
cfg = FeatureBuildConfig.from_yaml(ROOT / "configs" / "features.yaml")
inputs = load_reference_inputs(DATA)
pois, semantic = inputs["pois"], inputs["semantic"]
pf = pd.read_parquet(DATA / "poi_features.parquet").set_index("poi_id").loc[pois["poi_id"]]
emb = pf[[f"text_emb_{i:02d}" for i in range(64)]].to_numpy(dtype=np.float64)

rec = ridge_recoverability(emb, semantic)
pred = rec.pop("_pred")
print("D11 text-only", rec, f"{time.perf_counter()-t0:.0f}s", flush=True)

trips = pd.read_parquet(DATA / "trips.parquet")
travelers = pd.read_parquet(DATA / "travelers.parquet")
train = pd.read_parquet(DATA / "interactions_train.parquet")
pre = pd.read_parquet(DATA / "interactions_pretrip.parquet")
est_sem = assemble_traveler_features(
    travelers, trips, train, pois, semantic.astype(np.float32), cfg, pre
)
print("built est-taste over true semantic", f"{time.perf_counter()-t0:.0f}s", flush=True)
tf = pd.read_parquet(DATA / "traveler_features.parquet")
holdout = set(trips.loc[trips["is_holdout"], "trip_id"].astype(str))
t2t = dict(zip(trips["trip_id"].astype(str), trips["traveler_id"].astype(str), strict=True))
d9 = d9_and_ablations(DATA, tf, emb, est_sem, holdout, t2t, inputs, pred)
print(json.dumps(d9, indent=1))
out = {"d11_text_only": rec, "d9_within_trip": d9, "n_holdout_trips": len(holdout)}
(ROOT / "results" / "parts" / "a3_step0.json").write_text(json.dumps(out, indent=2))
