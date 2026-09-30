from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.api.services import catalog
from src.config import MODELS_DIR
from src.ph1_credit_risk.explain import shap_explain as se

__all__ = ["NOTE", "EXPLAIN_MODEL_FILE", "load_explainer_model", "explain", "factors"]

EXPLAIN_MODEL_FILE = "ph1_catboost_fold0.joblib"

NOTE = (
    "SHAP dưới đây mô tả CatBoost — model cây thành phần tốt nhất — chứ không phải ensemble "
    "stacking đang dùng để chấm PD. Giá trị SHAP ở thang log-odds. Xác suất CatBoost là điểm xếp "
    "hạng chưa hiệu chỉnh (mức nền ~35% do scale_pos_weight), không phải PD."
)

@lru_cache(maxsize=1)
def load_explainer_model():
    path = MODELS_DIR / EXPLAIN_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}. Chạy `python scripts/ph1_train.py` trước.")
    return joblib.load(path)


def explain(X: pd.DataFrame, k: int = 5) -> dict:
    return se.explain_one(load_explainer_model(), X, k=k)

def factors(one: dict) -> list[dict]:
    return [{
        "feature": f["feature"],
        "value": catalog.format_value(f["feature"], f["value"]),
        "direction": f["direction"],
        "shap": round(float(f["shap"]), 4),
        "mo_ta": f["mo_ta"],
    } for f in one["factors"]]
