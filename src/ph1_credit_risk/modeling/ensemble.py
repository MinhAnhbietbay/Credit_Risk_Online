from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import CV_FOLDS, RANDOM_SEED

__all__ = ["Ensemble", "rank_average", "stacking_oof", "compare_methods", "select_ensemble", "logit"]

EPS = 1e-6

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))

def _matrix(preds: Mapping[str, np.ndarray], members: list[str]) -> np.ndarray:
    missing = [m for m in members if m not in preds]
    if missing:
        raise KeyError(f"Thiếu dự đoán của: {missing}")
    return np.column_stack([np.asarray(preds[m], dtype=np.float64) for m in members])

def rank_average(preds: Mapping[str, np.ndarray], weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Trung bình có trọng số của thứ hạng chuẩn hóa về [0, 1]."""
    members = list(preds)
    if weights is None:
        weights = {m: 1.0 for m in members}
    if set(weights) != set(members):
        raise ValueError(f"weights phải có đúng các key {members}, nhận {list(weights)}")
    M = _matrix(preds, members)
    n = M.shape[0]
    ranks = np.column_stack([(rankdata(M[:, j]) - 1) / max(n - 1, 1) for j in range(M.shape[1])])
    w = np.array([weights[m] for m in members], dtype=np.float64)
    return ranks @ w / w.sum()

def _new_stacker() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)

def stacking_oof(
    preds: Mapping[str, np.ndarray], y: np.ndarray, n_splits: int = CV_FOLDS
) -> tuple[np.ndarray, LogisticRegression]:
    """OOF của meta-model (CV lồng, không rò rỉ) + meta-model cuối fit trên toàn bộ OOF."""
    members = list(preds)
    Z = logit(_matrix(preds, members))
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof = np.full(len(y), -1.0)
    for tr, va in skf.split(Z, y):
        oof[va] = _new_stacker().fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
    assert not np.any(oof == -1.0)
    return oof, _new_stacker().fit(Z, y)

@dataclass
class Ensemble:
    """Model cuối: cách trộn + danh sách member (+ trọng số hoặc meta-model)."""

    method: str                      # "rank_average" | "stacking" | "single"
    members: list[str]
    weights: dict[str, float] = field(default_factory=dict)
    stacker: Any = None
    oof_auc: float = float("nan")

    def combine(self, preds: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.method == "single":
            return np.asarray(_matrix(preds, self.members)[:, 0])
        if self.method == "rank_average":
            return rank_average(_require(preds, self.members), self.weights)
        if self.method == "stacking":
            return self.stacker.predict_proba(logit(_matrix(preds, self.members)))[:, 1]
        raise ValueError(self.method)

def _require(preds: Mapping[str, np.ndarray], members: list[str]) -> dict[str, np.ndarray]:
    _matrix(preds, members)  # ném KeyError nếu thiếu
    return {m: preds[m] for m in members}

def compare_methods(
    preds: Mapping[str, np.ndarray], y: np.ndarray, weights: Mapping[str, float] | None = None, n_splits: int = CV_FOLDS
) -> pd.DataFrame:
    """Bảng AUC OOF: từng model đơn, rank_average, stacking — sắp xếp giảm dần."""
    y = np.asarray(y)
    rows = [{"method": m, "auc": float(roc_auc_score(y, p)), "kind": "single"} for m, p in preds.items()]
    rows.append({"method": "rank_average", "auc": float(roc_auc_score(y, rank_average(preds, weights))), "kind": "ensemble"})
    stack_oof, _ = stacking_oof(preds, y, n_splits)
    rows.append({"method": "stacking", "auc": float(roc_auc_score(y, stack_oof)), "kind": "ensemble"})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)

def select_ensemble(
    preds: Mapping[str, np.ndarray], y: np.ndarray, weights: Mapping[str, float] | None = None, n_splits: int = CV_FOLDS
) -> Ensemble:
    """Chọn cách trộn có AUC OOF cao nhất; nếu không vượt model đơn tốt nhất thì giữ model đơn."""
    y = np.asarray(y)
    members = list(preds)
    table = compare_methods(preds, y, weights, n_splits)
    best_single = table[table["kind"] == "single"].iloc[0]
    best = table.iloc[0]
    if best["kind"] == "single" or best["auc"] < best_single["auc"]:
        return Ensemble(method="single", members=[best_single["method"]], oof_auc=float(best_single["auc"]))
    if best["method"] == "rank_average":
        w = dict(weights) if weights is not None else {m: 1.0 for m in members}
        return Ensemble(method="rank_average", members=members, weights=w, oof_auc=float(best["auc"]))
    _, stacker = stacking_oof(preds, y, n_splits)
    return Ensemble(method="stacking", members=members, stacker=stacker, oof_auc=float(best["auc"]))
