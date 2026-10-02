from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_SEED
from src.ph1_credit_risk.evaluation.metrics import brier_score, ks_statistic

__all__ = ["DEFAULT_METRICS", "stratified_resample", "bootstrap_ci", "paired_bootstrap_diff"]

Metric = Callable[[np.ndarray, np.ndarray], float]
DEFAULT_METRICS: dict[str, Metric] = {
    "auc": lambda y, p: float(roc_auc_score(y, p)),
    "ks": ks_statistic,
    "brier": brier_score,
}

def stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])

def bootstrap_ci(y, p, metrics: Mapping[str, Metric], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        idx = stratified_resample(y, rng)
        for m, fn in metrics.items():
            draws[m][b] = fn(y[idx], p[idx])
    return pd.DataFrame([{"metric": m, "point": fn(y, p),
                          "lo": float(np.quantile(draws[m], alpha / 2)), "hi": float(np.quantile(draws[m], 1 - alpha / 2)),
                          "se": float(np.std(draws[m], ddof=1))} for m, fn in metrics.items()])

def paired_bootstrap_diff(y, p_a, p_b, metric: Metric, n_boot: int = 1000, alpha: float = 0.05,
                          seed: int = RANDOM_SEED) -> dict[str, float]:
    """metric(a) − metric(b) trên cùng bộ dòng rút. `share_le_zero` = tỉ lệ lần rút mà a không hơn b."""
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = stratified_resample(y, rng)
        d[b] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    return {"diff": float(metric(y, p_a) - metric(y, p_b)), "lo": float(np.quantile(d, alpha / 2)),
            "hi": float(np.quantile(d, 1 - alpha / 2)), "share_le_zero": float((d <= 0).mean())}