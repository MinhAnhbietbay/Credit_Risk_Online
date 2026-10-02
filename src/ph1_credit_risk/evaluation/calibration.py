# src/ph1_credit_risk/evaluation/calibration.py
"""Hiệu chỉnh xác suất. 

ECE dùng bin **bằng số lượng** (phân vị) vì điểm dồn gần 0; bin chia đều [0,1] như `metrics.calibration_data` để trống gần hết các bin trên 0.3.  Calibrator fit out-of-fold qua
`cross_calibrate` để số "sau hiệu chỉnh" không lạc quan."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED

__all__ = ["ece", "calibration_summary", "quantile_reliability", "make_calibrator", "cross_calibrate"]

_EPS = 1e-6

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)

def quantile_reliability(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    bins = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    return df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), count=("y", "size")).reset_index(drop=True)

def ece(y, p, n_bins: int = 10) -> float:
    t = quantile_reliability(y, p, n_bins)
    return float((t["count"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["count"].sum())

def calibration_summary(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "ece": ece(y, p), "citl": float(p.mean() - y.mean())}

class _Platt:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6).fit(_logit(np.asarray(p)), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(np.asarray(p)))[:, 1]

class _Isotonic:
    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p), y)
        return self

    def predict(self, p):
        return self.iso.predict(np.asarray(p))

def make_calibrator(method: str):
    if method == "platt":
        return _Platt()
    if method == "isotonic":
        return _Isotonic()
    raise ValueError(f"method phải là 'platt' hoặc 'isotonic', nhận {method!r}")

def cross_calibrate(method: str, p, y, n_splits: int = 5, seed: int = RANDOM_SEED) -> np.ndarray:
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(int)
    out = np.empty_like(p)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(p, y):
        out[va] = make_calibrator(method).fit(p[tr], y[tr]).predict(p[va])
    return out