from __future__ import annotations

import gc
import time

import pandas as pd

from src.config import PROCESSED_DIR
from src.ph1_credit_risk.data import loader
from src.ph1_credit_risk.features._common import attach_customer, count_categories, flatten_columns, write_parquet
from src.ph1_credit_risk.features.previous import PREV_ID_MAP_PATH

__all__ = ["agg_pos_cash", "build_pos_cash_features", "POS_AGG_PATH"]

POS_AGG_PATH = PROCESSED_DIR / "agg_pos_cash.parquet"

STATUS_VALUES = ["Active", "Completed", "Signed", "Approved", "Returned to the store", "Demand"]

_AGGS = {
    "MONTHS_BALANCE": ["min", "max", "size"],
    "CNT_INSTALMENT": ["mean", "max"],
    "CNT_INSTALMENT_FUTURE": ["mean", "max"],
    "SK_DPD": ["mean", "max", "sum"],
    "SK_DPD_DEF": ["mean", "max", "sum"],
}

def agg_pos_cash(pos: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    df = attach_customer(pos, id_map)
    out = flatten_columns(df.groupby("SK_ID_CURR", sort=True).agg(_AGGS), "POS_")

    status = df["NAME_CONTRACT_STATUS"].astype("str")
    out = out.join(count_categories(status, df["SK_ID_CURR"], STATUS_VALUES, "POS_STATUS_"))
    out["POS_COUNT"] = out["POS_MONTHS_BALANCE_SIZE"]
    out["POS_PREV_NUNIQUE"] = df.groupby("SK_ID_CURR", sort=True)["SK_ID_PREV"].nunique()
    out.index.name = "SK_ID_CURR"
    return out

def build_pos_cash_features() -> pd.DataFrame:
    t0 = time.time()
    id_map = pd.read_parquet(PREV_ID_MAP_PATH)
    pos = loader.load_table("pos_cash_balance")
    out = agg_pos_cash(pos, id_map)
    del pos
    gc.collect()
    return write_parquet(out, POS_AGG_PATH, "pos_cash", t0)
