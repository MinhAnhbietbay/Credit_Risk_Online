from __future__ import annotations

import hashlib
import json
from typing import Sequence

from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH
from src.ph1_credit_risk.features.selection import SELECTED_FEATURES_PATH

__all__ = ["feature_set_version", "load_selected_features", "load_categorical"]

def feature_set_version(features: Sequence[str]) -> str:
    """Hash 12 ký tự, **không phụ thuộc thứ tự** (sắp xếp trước khi băm)."""
    payload = json.dumps(sorted(features), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]

def load_selected_features() -> list[str]:
    return json.loads(SELECTED_FEATURES_PATH.read_text())

def load_categorical(features: Sequence[str] | None = None) -> list[str]:
    cats = json.loads(CATEGORICAL_COLUMNS_PATH.read_text())
    if features is None:
        return cats
    keep = set(features)
    return [c for c in cats if c in keep]


def to_model_frame(df, features: Sequence[str], categorical: Sequence[str] | None = None):
    import pandas as pd

    cats = set(categorical or ())
    out = pd.DataFrame(index=df.index)
    for c in features:
        s = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(
            [float("nan")] * len(df), index=df.index)
        # categorical: giữ kiểu nguyên cho phép NaN (Int64), LightGBM/CatBoost hiểu được
        out[c] = s.astype("Int64") if c in cats else s.astype("float64")
    return out
