from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_application_features", "DAYS_EMPLOYED_SENTINEL", "EXT_SOURCE_COLS"]

DAYS_EMPLOYED_SENTINEL = 365243
EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.mask(b == 0, np.nan)

def add_application_features(app: pd.DataFrame) -> pd.DataFrame:
    out = app.copy()

    anom = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    out["DAYS_EMPLOYED_ANOM"] = anom.astype("int8")
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].mask(anom, np.nan)

    out["CREDIT_INCOME_RATIO"] = _safe_div(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["ANNUITY_INCOME_RATIO"] = _safe_div(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_TERM"] = _safe_div(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["DAYS_EMPLOYED_RATIO"] = _safe_div(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])
    out["INCOME_PER_PERSON"] = _safe_div(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])

    ext = out[EXT_SOURCE_COLS]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_STD"] = ext.std(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_MAX"] = ext.max(axis=1)

    return out
