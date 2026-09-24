from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED

__all__ = [
    "ModelSpec", "REGISTRY", "get_model", "list_models",
    "pos_weight", "build_and_fit", "predict_proba",
    "EARLY_STOPPING_ROUNDS", "EVAL_FRACTION",
]

EARLY_STOPPING_ROUNDS = 50
EVAL_FRACTION = 0.1 

@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[..., Any]  # build(categorical=(), scale_pos_weight=None) -> estimator chưa fit
    handles_nan: bool
    handles_categorical: bool
    needs_scaling: bool
    early_stopping: bool = False

# Tiền xử lý cho model sklearn (LR / RF)
class _NumericColumns:
    def __init__(self, categorical: Sequence[str]):
        self.categorical = frozenset(categorical)

    def __call__(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in self.categorical]

def _preprocessor(categorical: Sequence[str], scale: bool) -> ColumnTransformer:
    """Cột số: impute median + cờ missing (D3) [+ scaler]; cột categorical: one-hot (D4)."""
    cat = list(categorical)
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    transformers = [("num", Pipeline(num_steps), _NumericColumns(cat))]
    if cat:
        # mã -1 (missing) và mã chưa thấy ở train -> vector 0 (handle_unknown="ignore")
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), cat))
    return ColumnTransformer(transformers, sparse_threshold=0.0)

def _build_logistic_regression(categorical: Sequence[str] = (), scale_pos_weight: float | None = None) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(categorical, scale=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_SEED)),
    ])

def _build_random_forest(categorical: Sequence[str] = (), scale_pos_weight: float | None = None) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(categorical, scale=False)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                       class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
    ])

# Boosting: giữ NaN, scale_pos_weight, early stopping
def _build_xgboost(categorical: Sequence[str] = (), scale_pos_weight: float | None = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=scale_pos_weight or 1.0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric="auc",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )

def _build_lightgbm(categorical: Sequence[str] = (), scale_pos_weight: float | None = None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight or 1.0,
        # chỉ theo dõi AUC: nếu để thêm binary_logloss mặc định, logloss trên eval set
        # (không weight) tăng ngay từ đầu khi train có scale_pos_weight -> early stop ở iter 1
        metric="auc",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )

def _build_catboost(categorical: Sequence[str] = (), scale_pos_weight: float | None = None) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        cat_features=list(categorical), scale_pos_weight=scale_pos_weight or 1.0,
        eval_metric="AUC", random_seed=RANDOM_SEED, thread_count=-1,
        verbose=0, allow_writing_files=False,
    )

REGISTRY: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec("logistic_regression", _build_logistic_regression,
                                     handles_nan=False, handles_categorical=False, needs_scaling=True),
    "random_forest": ModelSpec("random_forest", _build_random_forest,
                               handles_nan=False, handles_categorical=False, needs_scaling=False),
    "xgboost": ModelSpec("xgboost", _build_xgboost,
                         handles_nan=True, handles_categorical=False, needs_scaling=False, early_stopping=True),
    "lightgbm": ModelSpec("lightgbm", _build_lightgbm,
                          handles_nan=True, handles_categorical=True, needs_scaling=False, early_stopping=True),
    "catboost": ModelSpec("catboost", _build_catboost,
                          handles_nan=True, handles_categorical=True, needs_scaling=False, early_stopping=True),
}

def get_model(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Model '{name}' không có trong registry. Có: {list_models()}")
    return REGISTRY[name]

def list_models() -> list[str]:
    return sorted(REGISTRY)

def pos_weight(y: pd.Series | np.ndarray) -> float:
    """scale_pos_weight = n_neg / n_pos (spec §6, thay cho SMOTE)."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    return float((len(y) - n_pos) / max(n_pos, 1))

def build_and_fit(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: Sequence[str] = (),
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Any:
    spec = get_model(name)
    cat = [c for c in categorical if c in X.columns]
    est = spec.build(categorical=cat, scale_pos_weight=pos_weight(y))

    if not spec.early_stopping:
        return est.fit(X, y)

    if X_val is None:
        X, X_val, y, y_val = train_test_split(
            X, y, test_size=EVAL_FRACTION, stratify=y, random_state=RANDOM_SEED
        )
    if name == "lightgbm":
        est.fit(X, y, eval_set=[(X_val, y_val)], categorical_feature=cat or "auto",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    elif name == "xgboost":
        est.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "catboost":
        est.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0)
    return est

def predict_proba(est: Any, X: pd.DataFrame) -> np.ndarray:
    """Xác suất lớp 1, mảng 1 chiều."""
    return np.asarray(est.predict_proba(X))[:, 1]

def fit_timed(name: str, X: pd.DataFrame, y: pd.Series, categorical: Sequence[str] = (), **kw) -> tuple[Any, float]:
    t0 = time.time()
    est = build_and_fit(name, X, y, categorical, **kw)
    return est, time.time() - t0
