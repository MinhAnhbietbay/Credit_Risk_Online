from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.online.feature_set import feature_set_version, load_categorical, load_selected_features
from src.ph1_credit_risk.explain.glossary import decode_category, describe

__all__ = [
    "selected_features", "categorical_features", "current_feature_set_version",
    "feature_medians", "form_features", "shap_ranking", "model_comparison",
    "describe_feature", "format_value",
]

MEDIANS_PATH = PROCESSED_DIR / "feature_medians.json"
FORM_PATH = PROCESSED_DIR / "web_form_features.json"
SHAP_GLOBAL_PATH = REPORTS_DIR / "shap_global.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

@lru_cache(maxsize=1)
def selected_features() -> tuple[str, ...]:
    return tuple(load_selected_features())

@lru_cache(maxsize=1)
def categorical_features() -> tuple[str, ...]:
    return tuple(load_categorical(selected_features()))

@lru_cache(maxsize=1)
def current_feature_set_version() -> str:
    return feature_set_version(selected_features())

@lru_cache(maxsize=1)
def feature_medians() -> dict:
    if not MEDIANS_PATH.exists():
        raise FileNotFoundError(
            f"Thiếu {MEDIANS_PATH}. Chạy `python scripts/ph1_web_assets.py` trước.")
    return json.loads(MEDIANS_PATH.read_text())

@lru_cache(maxsize=1)
def form_features() -> tuple[dict, ...]:
    """10–15 feature quan trọng nhất theo SHAP, kèm khoảng nhập / danh sách lựa chọn."""
    if not FORM_PATH.exists():
        raise FileNotFoundError(
            f"Thiếu {FORM_PATH}. Chạy `python scripts/ph1_web_assets.py` trước.")
    return tuple(json.loads(FORM_PATH.read_text()))

def shap_ranking() -> pd.DataFrame:
    if not SHAP_GLOBAL_PATH.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mo_ta"])
    return pd.read_csv(SHAP_GLOBAL_PATH)

def model_comparison() -> pd.DataFrame:
    """Bảng so sánh 6 model. Rỗng nếu chưa chạy `ph1_report.py`."""
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)

def describe_feature(name: str) -> str:
    return describe(name)

def format_value(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "thiếu dữ liệu"
    label = decode_category(feature, value)
    if label is not None:
        return label
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"
