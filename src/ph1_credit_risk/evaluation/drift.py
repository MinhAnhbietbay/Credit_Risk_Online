"""PSI theo feature. 
Bin theo phân vị của tập gốc, NaN là một bin riêng; feature phân loại: mỗi giá trị một bin. 
Mốc 0.1 / 0.25 là quy ước ngành (tham chiếu), không phải ngưỡng có csdl."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["psi", "psi_level", "psi_table"]


def _shares(counts: pd.Series, total: int, eps: float) -> np.ndarray:
    return np.clip(counts.to_numpy(dtype=float) / max(total, 1), eps, None)

def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10, categorical: bool = False, eps: float = 1e-4) -> float:
    if categorical:
        e, a = expected.astype(str), actual.astype(str)   # NaN -> "nan", tự thành một nhóm
        cats = sorted(set(e) | set(a))
        ec = e.value_counts().reindex(cats, fill_value=0)
        ac = a.value_counts().reindex(cats, fill_value=0)
    else:
        e, a = expected.astype(float), actual.astype(float)
        edges = np.unique(np.nanquantile(e, np.linspace(0, 1, n_bins + 1))) if e.notna().any() else np.array([0.0])
        inner = edges[1:-1]
        def count(s):
            b = pd.Series(np.where(s.isna(), -1, np.searchsorted(inner, s.to_numpy(), side="right")))
            return b.value_counts().reindex(range(-1, len(inner) + 1), fill_value=0)
        ec, ac = count(e), count(a)
    pe, pa = _shares(ec, len(expected), eps), _shares(ac, len(actual), eps)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

def psi_level(v: float) -> str:
    return "ổn định" if v < 0.1 else ("theo dõi" if v < 0.25 else "lệch lớn")

def psi_table(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], categorical: Sequence[str] = ()) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(ref[c], cur[c], categorical=c in categorical)} for c in cols]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    t["level"] = t["psi"].map(psi_level)
    return t
