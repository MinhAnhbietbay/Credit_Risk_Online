from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.ph1_credit_risk.data import loader
from src.ph1_credit_risk.features._common import attach_customer, flatten_columns, write_parquet
from src.ph1_credit_risk.features.previous import PREV_ID_MAP_PATH

__all__ = ["add_row_features", "agg_installments", "build_installments_features", "INST_AGG_PATH"]

INST_AGG_PATH = PROCESSED_DIR / "agg_installments.parquet"

_ROW_AGGS = ["mean", "max", "sum", "std"]
_AGGS = {
    "PAYMENT_PERC": _ROW_AGGS,
    "PAYMENT_DIFF": _ROW_AGGS,
    "DPD": _ROW_AGGS,
    "DBD": _ROW_AGGS,
    "AMT_INSTALMENT": ["mean", "max", "sum"],
    "AMT_PAYMENT": ["mean", "max", "sum"],
    "DAYS_ENTRY_PAYMENT": ["max", "min"],
    "NUM_INSTALMENT_VERSION": ["nunique"],
    "NUM_INSTALMENT_NUMBER": ["max"],
}

def add_row_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Thêm PAYMENT_PERC / PAYMENT_DIFF / DPD / DBD cho từng kỳ trả (trả bản sao)."""
    out = inst.copy()
    instalment = out["AMT_INSTALMENT"].replace(0, np.nan)  # chia 0 -> NaN, không inf
    out["PAYMENT_PERC"] = out["AMT_PAYMENT"] / instalment
    out["PAYMENT_DIFF"] = out["AMT_INSTALMENT"] - out["AMT_PAYMENT"]
    late = out["DAYS_ENTRY_PAYMENT"] - out["DAYS_INSTALMENT"]
    out["DPD"] = late.clip(lower=0)
    out["DBD"] = (-late).clip(lower=0)
    return out

def agg_installments(inst: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    df = add_row_features(attach_customer(inst, id_map))
    out = flatten_columns(df.groupby("SK_ID_CURR", sort=True).agg(_AGGS), "INST_")
    out["INST_COUNT"] = df.groupby("SK_ID_CURR", sort=True).size()
    out["INST_LATE_RATIO"] = (df["DPD"] > 0).groupby(df["SK_ID_CURR"].to_numpy()).mean()
    out["INST_UNDERPAID_RATIO"] = (df["PAYMENT_DIFF"] > 0).groupby(df["SK_ID_CURR"].to_numpy()).mean()
    out.index.name = "SK_ID_CURR"
    return out

def build_installments_features() -> pd.DataFrame:
    t0 = time.time()
    id_map = pd.read_parquet(PREV_ID_MAP_PATH)
    inst = loader.load_table("installments_payments")
    out = agg_installments(inst, id_map)
    del inst
    gc.collect()
    return write_parquet(out, INST_AGG_PATH, "installments", t0)
