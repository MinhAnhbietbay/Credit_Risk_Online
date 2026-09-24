from __future__ import annotations

from pathlib import Path

from src.config import HOME_CREDIT_DIR

__all__ = ["TABLES", "KEYS", "DTYPES", "HOME_CREDIT_DIR", "table_path", "infer_dtype", "dtype_for"]

TABLES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

KEYS: dict[str, list[str]] = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR", "SK_ID_PREV"],
    "pos_cash_balance": ["SK_ID_CURR", "SK_ID_PREV"],
    "installments_payments": ["SK_ID_CURR", "SK_ID_PREV"],
    "credit_card_balance": ["SK_ID_CURR", "SK_ID_PREV"],
}

DTYPES: dict[str, dict[str, str]] = {
    "application_train": {"SK_ID_CURR": "int32", "TARGET": "int8"},
    "application_test": {"SK_ID_CURR": "int32"},
    "bureau": {"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32"},
    # MONTHS_BALANCE ở các bảng lịch sử là số nguyên đầy đủ, không NaN
    "bureau_balance": {"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16"},
    "previous_application": {"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"},
    "pos_cash_balance": {"SK_ID_CURR": "int32", "SK_ID_PREV": "int32", "MONTHS_BALANCE": "int16"},
    "installments_payments": {"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"},
    "credit_card_balance": {"SK_ID_CURR": "int32", "SK_ID_PREV": "int32", "MONTHS_BALANCE": "int16"},
}

_FLOAT_PREFIXES = (
    "AMT_", "DAYS_", "CNT_", "RATE_", "NUM_", "SK_DPD", "EXT_SOURCE_",
    "HOUR_", "OWN_CAR_AGE", "REGION_", "OBS_", "DEF_", "SELLERPLACE_",
    "NFLAG_", "APARTMENTS_", "BASEMENTAREA_", "YEARS_", "COMMONAREA_",
    "ELEVATORS_", "ENTRANCES_", "FLOORS", "LANDAREA_", "LIVINGAPARTMENTS_",
    "LIVINGAREA_", "NONLIVINGAPARTMENTS_", "NONLIVINGAREA_", "TOTALAREA_",
)

_CATEGORY_PREFIXES = ("NAME_", "CODE_", "WEEKDAY_", "FLAG_OWN_", "FLAG_LAST_APPL")

_CATEGORY_EXACT = frozenset(
    {
        "STATUS", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
        "PRODUCT_COMBINATION", "CHANNEL_TYPE",
    }
)

def table_path(name: str) -> Path:
    if name not in TABLES:
        raise KeyError(f"Bảng không tồn tại: {name!r}. Hợp lệ: {sorted(TABLES)}")
    return HOME_CREDIT_DIR / TABLES[name]

def infer_dtype(column: str) -> str | None:
    if column.startswith("SK_ID_"):
        return "int32"
    if column in _CATEGORY_EXACT or column.startswith(_CATEGORY_PREFIXES):
        return "category"
    if column.startswith(_FLOAT_PREFIXES) or column == "MONTHS_BALANCE":
        return "float32"
    return None

def dtype_for(name: str, columns: list[str]) -> dict[str, str]:
    overrides = DTYPES[name]
    out: dict[str, str] = {}
    for col in columns:
        dtype = overrides.get(col, infer_dtype(col))
        if dtype is not None:
            out[col] = dtype
    return out
