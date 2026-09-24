from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.ph1_credit_risk.data import loader
from src.ph1_credit_risk.features._common import (
    attach_customer, count_categories, numeric_aggs, numeric_columns, write_parquet,
)
from src.ph1_credit_risk.features.previous import PREV_ID_MAP_PATH

__all__ = ["agg_credit_card", "build_credit_card_features", "CC_AGG_PATH"]

CC_AGG_PATH = PROCESSED_DIR / "agg_credit_card.parquet"

STATUS_VALUES = ["Active", "Completed", "Signed", "Demand", "Sent proposal"]
_KEYS = ("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE")

def agg_credit_card(cc: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    df = attach_customer(cc, id_map)

    # Tỉ lệ dùng hạn mức ở mức từng tháng; hạn mức 0 -> NaN
    limit = df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    df["UTILIZATION"] = df["AMT_BALANCE"] / limit

    num_cols = numeric_columns(df, exclude=_KEYS)
    out = numeric_aggs(df, "SK_ID_CURR", num_cols, "CC_")

    grouped = df.groupby("SK_ID_CURR", sort=True)
    out["CC_COUNT"] = grouped.size()
    out["CC_MONTHS_BALANCE_MIN"] = grouped["MONTHS_BALANCE"].min()
    out["CC_MONTHS_WITH_BALANCE"] = (df["AMT_BALANCE"] > 0).groupby(df["SK_ID_CURR"].to_numpy()).sum()
    out["CC_PREV_NUNIQUE"] = grouped["SK_ID_PREV"].nunique()

    status = df["NAME_CONTRACT_STATUS"].astype("str")
    out = out.join(count_categories(status, df["SK_ID_CURR"], STATUS_VALUES, "CC_STATUS_"))
    out.index.name = "SK_ID_CURR"
    return out

def build_credit_card_features() -> pd.DataFrame:
    t0 = time.time()
    id_map = pd.read_parquet(PREV_ID_MAP_PATH)
    cc = loader.load_table("credit_card_balance")
    out = agg_credit_card(cc, id_map)
    del cc
    gc.collect()
    return write_parquet(out, CC_AGG_PATH, "credit_card", t0)
