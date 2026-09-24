"""SHAP
SHAP mô tả model thành phần, không phải model cuối. 
Model cuối của PH1 là stacking 5 model, SHAP dạng tree không áp thẳng lên nó được. 
**model cây thành phần tốt nhất là CatBoost** (OOF AUC 0.7839 so với 0.7861 của ensemble)

**SHAP out-of-fold:** CatBoost có 5 bản fold. Mỗi dòng được giải thích bằng **đúng model fold không
train trên dòng đó** (`oof_shap_values`)

Đơn vị: SHAP của CatBoost nằm trong thang **log-odds (margin)**, cộng được:

    sum(shap của dòng) + base_value == margin(dòng)      (khớp đến ~1e-15, có test)
    xác suất = sigmoid(margin)

Nên "đẩy rủi ro lên/xuống" trong `top_factors` là đẩy trên thang log-odds, không phải cộng trừ điểm
phần trăm
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import shap

from src.ph1_credit_risk.explain.glossary import decode_category, describe

__all__ = [
    "shap_values", "oof_shap_values", "margin_to_proba", "global_importance",
    "top_factors", "explain_one", "SHAP_MODEL_NOTE",
]

SHAP_MODEL_NOTE = (
    "CatBoost fit với scale_pos_weight nên xác suất nó trả về là ĐIỂM RỦI RO CHƯA HIỆU CHỈNH "
    "(mức nền ~35% chứ không phải tỉ lệ vỡ nợ thật ~8%); PD dùng cho quyết định lấy từ ensemble."
)

def shap_values(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:  # một số phiên bản trả (n, p, 2) cho bài toán nhị phân
        values = values[..., 1]
    return values, float(np.ravel(explainer.expected_value)[-1])

def oof_shap_values(
    models: Sequence[Any],
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[float]]:

    folds = list(folds)
    values = np.full((len(X), X.shape[1]), np.nan, dtype=np.float64)
    bases: list[float] = []
    for model, (_, va) in zip(models, folds):
        v, b = shap_values(model, X.iloc[va])
        values[va] = v
        bases.append(b)
    if np.isnan(values).any():
        raise ValueError("Các fold không phủ kín mọi dòng của X — SHAP out-of-fold sẽ có dòng trống.")
    return values, bases

def margin_to_proba(margin: np.ndarray | float) -> np.ndarray | float:
    """sigmoid: đổi log-odds sang xác suất vỡ nợ."""
    return 1.0 / (1.0 + np.exp(-np.asarray(margin, dtype=np.float64)))


def global_importance(values: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mo_ta": [describe(f) for f in feature_names],
    })
    return imp.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def _format_value(feature: str, value: Any) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "thiếu dữ liệu"
    label = decode_category(feature, value)
    if label is not None:
        return label
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3g}"
    return str(value)

def top_factors(
    shap_row: np.ndarray,
    row: pd.Series,
    feature_names: Sequence[str],
    k: int = 5,
) -> list[dict[str, Any]]:

    shap_row = np.asarray(shap_row, dtype=np.float64)
    order = np.argsort(-np.abs(shap_row))[:k]
    out = []
    for i in order:
        name = feature_names[i]
        pushes_up = shap_row[i] > 0
        direction = "tăng" if pushes_up else "giảm"
        value = _format_value(name, row[name])
        out.append({
            "feature": name,
            "value": row[name],
            "shap": float(shap_row[i]),
            "direction": direction,
            "mo_ta": describe(name),
            "sentence": (
                f"{name} = {value} ({describe(name)}) làm rủi ro {direction} "
                f"(SHAP {shap_row[i]:+.3f} log-odds)"
            ),
        })
    return out

def explain_one(model: Any, row: pd.DataFrame, k: int = 5) -> dict[str, Any]:
    
    if len(row) != 1:
        raise ValueError(f"explain_one cần đúng 1 dòng, nhận {len(row)}")
    values, base = shap_values(model, row)
    shap_row = values[0]
    margin = float(shap_row.sum() + base)
    return {
        "base_value": base,
        "base_probability": float(margin_to_proba(base)),
        "margin": margin,
        "probability": float(margin_to_proba(margin)),
        "factors": top_factors(shap_row, row.iloc[0], list(row.columns), k=k),
        "shap_values": shap_row,
        "note": SHAP_MODEL_NOTE,
    }
