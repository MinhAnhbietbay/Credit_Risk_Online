from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.ph1_credit_risk.data import loader
from src.ph1_credit_risk.features._common import count_categories, numeric_aggs, numeric_columns

__all__ = ["agg_bureau_balance", "agg_bureau", "build_bureau_features", "BUREAU_AGG_PATH"]

BUREAU_AGG_PATH = PROCESSED_DIR / "agg_bureau.parquet"

# Thứ tự cố định để cột output ổn định giữa các lần chạy
_STATUS_VALUES = ["0", "1", "2", "3", "4", "5", "C", "X"]
_DPD_STATUSES = ["1", "2", "3", "4", "5"]

_CREDIT_ACTIVE_VALUES = ["Active", "Closed", "Sold", "Bad debt"]

# 1: bureau_balance -> SK_ID_BUREAU
def _bb_partial(bb: pd.DataFrame) -> pd.DataFrame:
    """Bộ đếm thô cho một chunk: mọi cột đều cộng/min/max dồn được giữa các chunk."""
    status = bb["STATUS"].astype("str")
    counts = pd.get_dummies(status).reindex(columns=_STATUS_VALUES, fill_value=False)
    counts.columns = [f"STATUS_{s}_COUNT" for s in _STATUS_VALUES]
    work = pd.concat(
        [bb[["SK_ID_BUREAU", "MONTHS_BALANCE"]].reset_index(drop=True),
         counts.astype("int32").reset_index(drop=True)],
        axis=1,
    )

    grouped = work.groupby("SK_ID_BUREAU", sort=True)
    out = grouped.agg(
        MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
        MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
        MONTHS_BALANCE_SIZE=("MONTHS_BALANCE", "size"),
    )
    return out.join(grouped[list(counts.columns)].sum())

def _combine_bb_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Cộng dồn bộ đếm thô từ nhiều chunk (một SK_ID_BUREAU có thể nằm ở 2 chunk)."""
    stacked = pd.concat(partials)
    how = {c: "sum" for c in stacked.columns}
    how["MONTHS_BALANCE_MIN"] = "min"
    how["MONTHS_BALANCE_MAX"] = "max"
    return stacked.groupby(level="SK_ID_BUREAU", sort=True).agg(how)

def _bb_finalize(partial: pd.DataFrame) -> pd.DataFrame:
    """Từ bộ đếm thô -> feature cuối (tỉ lệ, DPD), gắn tiền tố BB_."""
    out = partial.copy()
    size = out["MONTHS_BALANCE_SIZE"]

    dpd_cols = [f"STATUS_{s}_COUNT" for s in _DPD_STATUSES]
    out["DPD_RATIO"] = out[dpd_cols].sum(axis=1) / size

    # Mức trễ hạn cao nhất từng chạm: 0 nếu chưa từng trễ
    levels = np.zeros(len(out), dtype="int8")
    for s in _DPD_STATUSES:
        levels = np.where(out[f"STATUS_{s}_COUNT"].to_numpy() > 0, int(s), levels)
    out["MAX_DPD_LEVEL"] = levels

    for s in _STATUS_VALUES:
        out[f"STATUS_{s}_RATIO"] = out[f"STATUS_{s}_COUNT"] / size

    out.columns = [f"BB_{c}" for c in out.columns]
    return out

def agg_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    return _bb_finalize(_bb_partial(bb))

# 2: bureau (+ bb_agg) -> SK_ID_CURR
def agg_bureau(bureau: pd.DataFrame, bb_agg: pd.DataFrame | None) -> pd.DataFrame:
    """Gộp `bureau` (đã ghép feature bb) về 1 dòng / SK_ID_CURR, tiền tố `BUREAU_`."""
    df = bureau
    if bb_agg is not None:
        df = df.merge(bb_agg, how="left", left_on="SK_ID_BUREAU", right_index=True)

    num_cols = numeric_columns(df, exclude=("SK_ID_CURR", "SK_ID_BUREAU"))
    out = numeric_aggs(df, "SK_ID_CURR", num_cols, "BUREAU_")

    # Đếm theo trạng thái khoản vay + tỉ lệ đang hoạt động
    active = df["CREDIT_ACTIVE"].astype("str")
    out = out.join(
        count_categories(active, df["SK_ID_CURR"], _CREDIT_ACTIVE_VALUES, "BUREAU_CREDIT_ACTIVE_")
    )

    out["BUREAU_COUNT"] = df.groupby("SK_ID_CURR", sort=True).size()
    out["BUREAU_ACTIVE_RATIO"] = out["BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT"] / out["BUREAU_COUNT"]
    out["BUREAU_CREDIT_TYPE_NUNIQUE"] = df.groupby("SK_ID_CURR", sort=True)["CREDIT_TYPE"].nunique()

    # Nhóm chỉ tính trên khoản đang hoạt động
    active_only = df[active.to_numpy() == "Active"]
    if len(active_only):
        out = out.join(numeric_aggs(active_only, "SK_ID_CURR", num_cols, "BUREAU_ACTIVE_"))

    out.index.name = "SK_ID_CURR"
    return out

# Chạy trên data thật
def build_bureau_features(chunksize: int | None = 5_000_000) -> pd.DataFrame:
    t0 = time.time()

    if chunksize is None:
        bb_agg = agg_bureau_balance(loader.load_table("bureau_balance"))
    else:
        partials = []
        for chunk in loader.iter_chunks("bureau_balance", chunksize=chunksize):
            partials.append(_bb_partial(chunk))
            del chunk
            gc.collect()
        bb_agg = _bb_finalize(_combine_bb_partials(partials))
        del partials
        gc.collect()
    print(f"[bureau] bb_agg: {bb_agg.shape} ({time.time() - t0:.1f}s)")

    bureau = loader.load_table("bureau")
    out = agg_bureau(bureau, bb_agg)
    del bureau, bb_agg
    gc.collect()

    out = loader.downcast(out)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(BUREAU_AGG_PATH)
    print(f"[bureau] agg_bureau: {out.shape} -> {BUREAU_AGG_PATH} ({time.time() - t0:.1f}s)")
    return out
