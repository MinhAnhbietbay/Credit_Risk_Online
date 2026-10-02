# src/ph1_credit_risk/evaluation/fairness.py
"""Số đo công bằng theo nhóm. Chỉ đo, không sửa model. Nhóm có n < min_n: ghi n, để trống chỉ số."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_SEED
from src.ph1_credit_risk.evaluation.bootstrap import stratified_resample

__all__ = ["AGE_BANDS", "age_band", "align_attributes", "group_metrics", "disparity_summary", "flag_rate_ratio_ci"]

AGE_BANDS = [(-np.inf, 25, "<25"), (25, 35, "25-34"), (35, 45, "35-44"), (45, 55, "45-54"), (55, np.inf, "55+")]
_BAND_ORDER = {lab: i for i, (*_, lab) in enumerate(AGE_BANDS)}
_METRICS = ["default_rate", "mean_pd", "calib_gap", "flag_rate", "tpr", "fpr", "precision", "auc"]

def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.25
    edges = [AGE_BANDS[0][0]] + [hi for _, hi, _ in AGE_BANDS]
    return pd.cut(years, edges, right=False, labels=[lab for *_, lab in AGE_BANDS]).astype(str)

def align_attributes(ids: np.ndarray, attrs: pd.DataFrame) -> pd.DataFrame:
    a = attrs.set_index("SK_ID_CURR")
    missing = set(ids.tolist()) - set(a.index)
    if missing:
        raise KeyError(f"{len(missing)} SK_ID_CURR không có thuộc tính, vd {sorted(missing)[:5]}")
    return a.loc[ids].reset_index()

def _one(y, p, threshold) -> dict:
    flag = p >= threshold
    pos, neg = y == 1, y == 0
    return {"default_rate": y.mean(), "mean_pd": p.mean(), "calib_gap": p.mean() - y.mean(), "flag_rate": flag.mean(),
            "tpr": flag[pos].mean() if pos.any() else np.nan, "fpr": flag[neg].mean() if neg.any() else np.nan,
            "precision": y[flag].mean() if flag.any() else np.nan,
            "auc": roc_auc_score(y, p) if pos.any() and neg.any() else np.nan}

def group_metrics(y, p, group, threshold: float, min_n: int = 30) -> pd.DataFrame:
    y, p, group = np.asarray(y).astype(int), np.asarray(p, dtype=float), np.asarray(group)
    rows = []
    for g in pd.unique(group):
        m = group == g
        row = {"group": g, "n": int(m.sum())}
        row.update(_one(y[m], p[m], threshold) if m.sum() >= min_n else {k: np.nan for k in _METRICS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "group",
        key=lambda col: col.map(lambda x: (0, _BAND_ORDER[x]) if x in _BAND_ORDER else (1, str(x)))
    ).reset_index(drop=True)

def disparity_summary(tbl: pd.DataFrame) -> dict[str, float]:
    t = tbl.dropna(subset=["flag_rate"])
    return {"flag_rate_ratio": float(t["flag_rate"].min() / t["flag_rate"].max()),
            "tpr_gap": float(t["tpr"].max() - t["tpr"].min()), "fpr_gap": float(t["fpr"].max() - t["fpr"].min()),
            "auc_gap": float(t["auc"].max() - t["auc"].min()),
            "calib_gap_range": float(t["calib_gap"].max() - t["calib_gap"].min())}

def flag_rate_ratio_ci(y, p, group, threshold: float, n_boot: int = 1000, seed: int = RANDOM_SEED,
                       alpha: float = 0.05, min_n: int = 30) -> tuple[float, float]:
    y, p, group = np.asarray(y).astype(int), np.asarray(p, dtype=float), np.asarray(group)
    rng = np.random.default_rng(seed)
    r = np.empty(n_boot)
    for b in range(n_boot):
        idx = stratified_resample(y, rng)
        r[b] = disparity_summary(group_metrics(y[idx], p[idx], group[idx], threshold, min_n))["flag_rate_ratio"]
    return float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))
