from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.ph1_credit_risk.data import loader
from src.ph1_credit_risk.features._common import count_categories, numeric_aggs, numeric_columns

__all__ = [
    "agg_previous", "prev_id_map", "build_previous_features", "clean_days",
    "PREV_AGG_PATH", "PREV_ID_MAP_PATH", "DAYS_SENTINEL_COLS", "MODE_COLS",
]

PREV_AGG_PATH = PROCESSED_DIR / "agg_previous.parquet"
PREV_ID_MAP_PATH = PROCESSED_DIR / "prev_id_map.parquet"

# Các cột DAYS_* dùng 365243 làm mã "không xác định"
DAYS_SENTINEL = 365243
DAYS_SENTINEL_COLS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]

STATUS_VALUES = ["Approved", "Refused", "Canceled", "Unused offer"]
MODE_COLS = ["NAME_CONTRACT_TYPE", "NAME_YIELD_GROUP", "PRODUCT_COMBINATION"]

_KEYS = ("SK_ID_PREV", "SK_ID_CURR")

def clean_days(prev: pd.DataFrame) -> pd.DataFrame:
    """Thay mã 365243 bằng NaN ở các cột DAYS_* (trả bản sao)."""
    out = prev.copy()
    for col in DAYS_SENTINEL_COLS:
        if col in out.columns:
            out[col] = out[col].mask(out[col] == DAYS_SENTINEL, np.nan)
    return out

def _mode_by_group(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    """Giá trị xuất hiện nhiều nhất của `col` trong mỗi nhóm `key` (vector hóa).

    Hòa nhau thì lấy giá trị đứng trước theo thứ tự sort — ổn định giữa các lần chạy.
    """
    counts = df.groupby([key, col], sort=True, observed=True).size().reset_index(name="n")
    best = counts.sort_values([key, "n"], ascending=[True, False]).drop_duplicates(key)
    return best.set_index(key)[col]

def agg_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Gộp về 1 dòng / SK_ID_CURR."""
    df = clean_days(prev)

    # Feature dẫn xuất ở mức từng đơn: xin bao nhiêu / được duyệt bao nhiêu
    credit = df["AMT_CREDIT"].replace(0, np.nan)
    df["APP_CREDIT_RATIO"] = df["AMT_APPLICATION"] / credit

    num_cols = numeric_columns(df, exclude=_KEYS)
    out = numeric_aggs(df, "SK_ID_CURR", num_cols, "PREV_")

    status = df["NAME_CONTRACT_STATUS"].astype("str")
    out = out.join(count_categories(status, df["SK_ID_CURR"], STATUS_VALUES, "PREV_STATUS_"))
    out["PREV_COUNT"] = df.groupby("SK_ID_CURR", sort=True).size()
    out["PREV_REFUSED_RATIO"] = out["PREV_STATUS_REFUSED_COUNT"] / out["PREV_COUNT"]
    out["PREV_APPROVED_RATIO"] = out["PREV_STATUS_APPROVED_COUNT"] / out["PREV_COUNT"]

    # Nhóm riêng theo kết quả đơn — hồ sơ từng bị từ chối là tín hiệu mạnh
    status_arr = status.to_numpy()
    for label, prefix in (("Approved", "PREV_APPROVED_"), ("Refused", "PREV_REFUSED_")):
        subset = df[status_arr == label]
        cols = [c for c in num_cols if subset[c].notna().any()]
        if len(subset) and cols:
            out = out.join(numeric_aggs(subset, "SK_ID_CURR", cols, prefix))

    for col in MODE_COLS:
        out[f"PREV_{col}_MODE"] = _mode_by_group(df, "SK_ID_CURR", col).astype("str").astype("category")

    out.index.name = "SK_ID_CURR"
    return out

def prev_id_map(prev: pd.DataFrame) -> pd.DataFrame:
    """Bảng 2 cột (SK_ID_PREV, SK_ID_CURR), mỗi SK_ID_PREV đúng 1 dòng."""
    return (
        prev[["SK_ID_PREV", "SK_ID_CURR"]]
        .drop_duplicates(subset="SK_ID_PREV")
        .reset_index(drop=True)
    )

def build_previous_features() -> pd.DataFrame:
    """Đọc từ đĩa, ghi `agg_previous.parquet` + `prev_id_map.parquet`."""
    t0 = time.time()
    prev = loader.load_table("previous_application")

    id_map = prev_id_map(prev)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    id_map.to_parquet(PREV_ID_MAP_PATH)
    print(f"[previous] id_map: {id_map.shape} -> {PREV_ID_MAP_PATH}")

    out = loader.downcast(agg_previous(prev))
    del prev
    gc.collect()
    out.to_parquet(PREV_AGG_PATH)
    print(f"[previous] agg_previous: {out.shape} -> {PREV_AGG_PATH} ({time.time() - t0:.1f}s)")
    return out
