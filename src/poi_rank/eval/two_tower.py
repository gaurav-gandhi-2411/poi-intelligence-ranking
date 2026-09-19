"""Small two-tower neural ranker for Decision-Register DR2 (learning curve vs LambdaMART).

Two towers (traveler-side MLP, POI-side MLP) into a shared 64-d space, score = dot product +
POI bias, trained pointwise (BCE on label >= 1) with the SAME clipped IPS weights the LightGBM
ranker uses. By construction it sees only SEPARABLE features: traveler-side and POI-side columns
go through their own tower; the engineered traveler x POI interaction columns (`interact_*`)
cannot enter a dot-product model, which is part of what DR2 measures.

Eval-side only (never imported by the shipped pipeline); CPU torch, seeded, single-process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from poi_rank.models.baselines import categorical_feature_columns, numeric_feature_columns

FloatArray = npt.NDArray[np.float32]

_ITEM_PREFIXES = ("text_emb_", "behav_", "num_", "geo_")
_PAIR_PREFIX = "interact_"


def split_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(traveler-side numeric columns, POI-side numeric columns); `interact_*` excluded."""
    numeric = numeric_feature_columns(frame)
    item = [c for c in numeric if c.startswith(_ITEM_PREFIXES)]
    pair = [c for c in numeric if c.startswith(_PAIR_PREFIX)]
    user = [c for c in numeric if c not in set(item) | set(pair)]
    return user, item


@dataclass
class Standardizer:
    mean: FloatArray
    std: FloatArray

    @classmethod
    def fit(cls, x: FloatArray) -> Standardizer:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        return cls(mean=np.nan_to_num(mean), std=np.where(np.nan_to_num(std) > 1e-6, std, 1.0))

    def apply(self, x: FloatArray) -> FloatArray:
        out = (x - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0).astype(np.float32)


class TwoTower(nn.Module):
    def __init__(self, d_user: int, d_item: int, hidden: int = 128, out: int = 64) -> None:
        super().__init__()
        self.user = nn.Sequential(
            nn.Linear(d_user, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, out)
        )
        self.item = nn.Sequential(
            nn.Linear(d_item, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, out)
        )
        self.item_bias = nn.Linear(out, 1)

    def forward(self, xu: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        u, v = self.user(xu), self.item(xi)
        out: torch.Tensor = (u * v).sum(-1) + self.item_bias(v).squeeze(-1)
        return out


@dataclass
class TwoTowerData:
    """Per-trip traveler matrix + per-POI matrix + (trip_idx, poi_idx) index arrays."""

    user_matrix: FloatArray
    item_matrix: FloatArray
    trip_idx: npt.NDArray[np.int64]
    poi_idx: npt.NDArray[np.int64]


class TwoTowerEncoder:
    """Fits the standardisers / one-hot layout on the TRAIN frame; encodes any frame."""

    def __init__(self, train_frame: pd.DataFrame) -> None:
        self.user_cols, self.item_cols = split_columns(train_frame)
        self.cat_cols = categorical_feature_columns(train_frame)
        first_trip = train_frame.drop_duplicates("trip_id")
        first_poi = train_frame.drop_duplicates("poi_id")
        self.user_std = Standardizer.fit(first_trip[self.user_cols].to_numpy(np.float32))
        self.item_std = Standardizer.fit(first_poi[self.item_cols].to_numpy(np.float32))
        self.dummy_cols = list(
            pd.get_dummies(first_poi[self.cat_cols].astype(str), prefix=self.cat_cols).columns
        )

    def encode(self, frame: pd.DataFrame) -> TwoTowerData:
        trips = frame.drop_duplicates("trip_id")
        pois = frame.drop_duplicates("poi_id")
        dummies = (
            pd.get_dummies(pois[self.cat_cols].astype(str), prefix=self.cat_cols)
            .reindex(columns=self.dummy_cols, fill_value=0)
            .to_numpy(np.float32)
        )
        item = np.concatenate(
            [self.item_std.apply(pois[self.item_cols].to_numpy(np.float32)), dummies], axis=1
        )
        return TwoTowerData(
            user_matrix=self.user_std.apply(trips[self.user_cols].to_numpy(np.float32)),
            item_matrix=item,
            trip_idx=pd.Index(trips["trip_id"]).get_indexer(frame["trip_id"]).astype(np.int64),
            poi_idx=pd.Index(pois["poi_id"]).get_indexer(frame["poi_id"]).astype(np.int64),
        )


def _predict(model: TwoTower, data: TwoTowerData) -> npt.NDArray[np.float64]:
    model.eval()
    with torch.no_grad():
        xu = torch.from_numpy(data.user_matrix)
        xi = torch.from_numpy(data.item_matrix)
        u, v = model.user(xu), model.item(xi)
        bias = model.item_bias(v).squeeze(-1)
        ti, pi = torch.from_numpy(data.trip_idx), torch.from_numpy(data.poi_idx)
        scores = (u[ti] * v[pi]).sum(-1) + bias[pi]
    result: npt.NDArray[np.float64] = scores.numpy().astype(np.float64)
    return result


def train_two_tower(
    encoder: TwoTowerEncoder,
    fit_frame: pd.DataFrame,
    fit_weight: npt.NDArray[np.float64],
    val_frame: pd.DataFrame,
    val_ndcg: Any,
    seed: int,
    epochs: int = 10,
    batch_size: int = 4096,
    patience: int = 3,
) -> tuple[TwoTower, TwoTowerEncoder, dict[str, float]]:
    """Train on `fit_frame` (pointwise BCE, IPS weights), early-stop on `val_ndcg(scores)` over
    `val_frame` (train-carved; the holdout is never consulted). Returns the best-epoch model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    data = encoder.encode(fit_frame)
    val_data = encoder.encode(val_frame)
    y = torch.from_numpy((fit_frame["label"].to_numpy() >= 1).astype(np.float32))
    w = torch.from_numpy(fit_weight.astype(np.float32))
    xu_all, xi_all = torch.from_numpy(data.user_matrix), torch.from_numpy(data.item_matrix)
    ti_all, pi_all = torch.from_numpy(data.trip_idx), torch.from_numpy(data.poi_idx)

    model = TwoTower(xu_all.shape[1], xi_all.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    gen = torch.Generator().manual_seed(seed)
    n = len(y)
    best, best_state, bad = -1.0, None, 0
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            logits = model(xu_all[ti_all[b]], xi_all[pi_all[b]])
            loss = (loss_fn(logits, y[b]) * w[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = float(val_ndcg(_predict(model, val_data)))
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, encoder, {"best_val_ndcg10": best}


def score_frame(model: TwoTower, encoder: TwoTowerEncoder, frame: pd.DataFrame) -> pd.Series:
    scores = _predict(model, encoder.encode(frame))
    return pd.Series(scores, index=frame.index, name="score_two_tower")
