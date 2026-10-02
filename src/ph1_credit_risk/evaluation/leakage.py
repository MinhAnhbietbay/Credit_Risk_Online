"""Kiểm rò rỉ ở cấp feature 

Mọi cột DAYS_* / MONTHS_BALANCE ở bảng phụ tính tương đối so với ngày nộp đơn hiện tại (âm = quá khứ).
Cột ghi **sự kiện đã xảy ra** mà có giá trị dương = thông tin từ sau ngày nộp đơn = rò rỉ. 
Cột ghi **lịch dự kiến** có giá trị dương là hợp lệ (lịch đã ký từ trước). Phân loại theo HomeCredit_columns_description.csv.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__all__ = ["DATE_COLUMNS", "DERIVED_FROM", "SENTINEL", "temporal_audit", "features_from_column", "univariate_auc"]

SENTINEL = 365243  # mã "không xác định" của Home Credit

ACTUAL, PLANNED = "sự kiện thực", "lịch dự kiến"
DATE_COLUMNS: dict[str, dict[str, str]] = {
    "previous_application": {
        "DAYS_DECISION": ACTUAL, "DAYS_FIRST_DRAWING": ACTUAL, "DAYS_LAST_DUE": ACTUAL, "DAYS_TERMINATION": ACTUAL,
        "DAYS_FIRST_DUE": PLANNED, "DAYS_LAST_DUE_1ST_VERSION": PLANNED,
    },
    "bureau": {
        "DAYS_CREDIT": ACTUAL, "DAYS_ENDDATE_FACT": ACTUAL, "DAYS_CREDIT_UPDATE": ACTUAL,
        "DAYS_CREDIT_ENDDATE": PLANNED,
    },
    "installments_payments": {"DAYS_ENTRY_PAYMENT": ACTUAL, "DAYS_INSTALMENT": PLANNED},
    "bureau_balance": {"MONTHS_BALANCE": ACTUAL},
    "POS_CASH_balance": {"MONTHS_BALANCE": ACTUAL},
    "credit_card_balance": {"MONTHS_BALANCE": ACTUAL},
}

# Ánh xạ cột ngày thô -> các tên feature dẫn xuất từ cột ngày (ví dụ: DPD, DBD, LATE_RATIO từ chênh lệch ngày trả và ngày đến hạn)
DERIVED_FROM: dict[tuple[str, str], list[str]] = {
    ("installments_payments", "DAYS_ENTRY_PAYMENT"): ["DPD", "DBD", "LATE_RATIO"],
    ("installments_payments", "DAYS_INSTALMENT"): ["DPD", "DBD", "LATE_RATIO"],
}

# Tiền tố feature theo bảng (khớp builder: PREV_, PREV_APPROVED_, BUREAU_ACTIVE_, ...)
TABLE_PREFIX = {
    "previous_application": r"PREV_(?:APPROVED_|REFUSED_)?",
    "bureau": r"BUREAU_(?:ACTIVE_|CLOSED_)?",
    "installments_payments": r"INST_",
    "bureau_balance": r"BUREAU_(?:ACTIVE_|CLOSED_)?BB_",
    "POS_CASH_balance": r"POS_",
    "credit_card_balance": r"CC_",
}
AGG_SUFFIX = r"_(?:MEAN|MAX|MIN|SUM|STD|SIZE|COUNT)"

def temporal_audit(table: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, kind in DATE_COLUMNS.get(table, {}).items():
        if col not in df.columns:
            continue
        raw = df[col]
        s = raw.mask(raw == SENTINEL)
        n_valid = int(s.notna().sum())
        n_future = int((s > 0).sum())
        share_future = float(n_future / n_valid) if n_valid > 0 else 0.0
        verdict = "ok"
        if share_future > 0:
            verdict = "RÒ RỈ" if kind == ACTUAL else "ok (lịch dự kiến)"
        rows.append({"table": table, "column": col, "kind": kind, "n": int(len(raw)),
                     "share_sentinel": float((raw == SENTINEL).mean()), "share_future": share_future,
                     "n_future": n_future, "max": float(s.max()), "verdict": verdict})
    return pd.DataFrame(rows, columns=["table", "column", "kind", "n", "share_sentinel", "share_future", "n_future", "max", "verdict"])

def features_from_column(table: str, column: str, features: list[str]) -> list[str]:
    prefix = TABLE_PREFIX[table]
    patterns = [rf"{prefix}{re.escape(column)}{AGG_SUFFIX}"]
    for d in DERIVED_FROM.get((table, column), []):
        patterns.append(rf"{prefix}{re.escape(d)}(?:{AGG_SUFFIX})?")
    pat = re.compile(rf"(?:{'|'.join(patterns)})")
    seen = set()
    result = []
    for f in features:
        if f not in seen and pat.fullmatch(f):
            seen.add(f)
            result.append(f)
    return result

def univariate_auc(X: pd.DataFrame, y: np.ndarray, skip: Sequence[str] = ()) -> pd.DataFrame:
    """AUC của từng feature khi đứng một mình, chỉ trên dòng không NaN. `strength` = max(AUC, 1-AUC)."""
    y = np.asarray(y)
    rows = []
    for c in X.columns:
        if c in skip:
            continue
        s = X[c].to_numpy(dtype=float)
        mask = ~np.isnan(s)
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y[mask], s[mask]))
        rows.append({"feature": c, "auc": auc, "strength": max(auc, 1 - auc), "coverage": float(mask.mean())})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)