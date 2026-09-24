"""Cắt feature table từ ~760 cột xuống 60-100.
(1) loại cột ít giá trị — thiếu nhiều, hằng số, tương quan cao;
(2) xếp hạng phần còn lại bằng LightGBM gain importance trung bình qua k-fold.
Chỉ chạy trên phần CV (80%) của `features_train` — xếp hạng có nhìn nhãn nên không
được chạm holdout
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH, features_path
from src.ph1_credit_risk.modeling.split import cv_portion

__all__ = [
    "drop_low_value_columns", "rank_by_lgbm_gain", "select_features",
    "FEATURES_TRAIN_PATH", "SELECTED_FEATURES_PATH", "IMPORTANCE_PATH", "REPORT_PATH", "PROTECTED",
    "LGBM_RANK_PARAMS", "EXCLUDED",
]

FEATURES_TRAIN_PATH = features_path("train")
SELECTED_FEATURES_PATH = PROCESSED_DIR / "selected_features.json"
IMPORTANCE_PATH = PROCESSED_DIR / "feature_importance.csv"  # xếp hạng đầy đủ sau lọc, cho evidence
REPORT_PATH = REPORTS_DIR / "feature_selection.md"

PROTECTED = ("SK_ID_CURR", "TARGET")

EXCLUDED = ("CODE_GENDER",)

# Model xếp hạng: nhỏ, nhanh, chỉ cần đủ để so gain giữa các cột
LGBM_RANK_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
)


def drop_low_value_columns(
    df: pd.DataFrame, missing_threshold: float = 0.85, corr_threshold: float = 0.95
) -> tuple[pd.DataFrame, list[dict]]:
    
    dropped: list[dict] = []
    cols = [c for c in df.columns if c not in PROTECTED]

    for c in cols:
        if c in EXCLUDED:
            dropped.append({"column": c, "reason": "excluded", "detail": "thuộc tính được bảo vệ, xem EXCLUDED"})
    cols = [c for c in cols if c not in EXCLUDED]

    missing = df[cols].isna().mean()
    for c in cols:
        if missing[c] > missing_threshold:
            dropped.append({"column": c, "reason": "missing", "detail": f"{missing[c]:.3f}"})
    cols = [c for c in cols if missing[c] <= missing_threshold]

    nunique = df[cols].nunique(dropna=True)
    for c in cols:
        if nunique[c] <= 1:
            dropped.append({"column": c, "reason": "constant", "detail": f"nunique={nunique[c]}"})
    cols = [c for c in cols if nunique[c] > 1]

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num_cols].astype("float32").corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    removed: set[str] = set()
    for i, ci in enumerate(num_cols):
        if ci in removed:
            continue
        for j in range(i + 1, len(num_cols)):
            cj = num_cols[j]
            if cj in removed or not corr[i, j] > corr_threshold:
                continue
            loser, keeper = (ci, cj) if missing[ci] > missing[cj] else (cj, ci)
            removed.add(loser)
            dropped.append({"column": loser, "reason": "correlated", "detail": f"{keeper} r={corr[i, j]:.3f}"})
            if loser == ci:
                break
    cols = [c for c in cols if c not in removed]

    keep = [c for c in df.columns if c in PROTECTED or c in set(cols)]
    return df[keep], dropped

def rank_by_lgbm_gain(
    X: pd.DataFrame, y: pd.Series, categorical: list[str], n_splits: int = 3
) -> pd.Series:
    """Gain importance trung bình qua `n_splits` fold, chuẩn hóa tổng = 1, giảm dần."""
    cat = [c for c in categorical if c in X.columns]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    total = np.zeros(X.shape[1])
    for tr, _ in skf.split(X, y):
        model = lgb.LGBMClassifier(**LGBM_RANK_PARAMS, importance_type="gain")
        model.fit(X.iloc[tr], y.iloc[tr], categorical_feature=cat or "auto")
        total += model.feature_importances_
    imp = pd.Series(total / n_splits, index=X.columns)
    return (imp / imp.sum()).sort_values(ascending=False)

def _write_report(imp: pd.Series, dropped: list[dict], chosen: list[str], n_before: int, elapsed: float) -> None:
    by_reason = pd.Series([d["reason"] for d in dropped]).value_counts() if dropped else pd.Series(dtype=int)
    lines = [
        "# Feature selection (PH1 Task 6)", "",
        f"- Cột đầu vào (trừ khóa/nhãn): {n_before}",
        f"- Sau lọc cột ít giá trị: {len(imp)}",
        f"- Chọn cuối: {len(chosen)}  (thời gian {elapsed:.0f}s)", "",
        "## Số cột bị loại theo lý do", "",
        "| Lý do | Số cột |", "|---|---|",
        *[f"| {r} | {n} |" for r, n in by_reason.items()],
        "", f"## Top {len(chosen)} theo LightGBM gain (chuẩn hóa)", "",
        "| # | Feature | Gain |", "|---|---|---|",
        *[f"| {i + 1} | {c} | {imp[c]:.4f} |" for i, c in enumerate(chosen)],
        "", "## Cột bị loại (chi tiết)", "",
        "| Cột | Lý do | Chi tiết |", "|---|---|---|",
        *[f"| {d['column']} | {d['reason']} | {d['detail']} |" for d in dropped],
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))

def select_features(n_features: int = 80, n_splits: int = 3) -> list[str]:
    """Lọc + xếp hạng trên phần CV của `features_train`, ghi `selected_features.json` và báo cáo."""
    t0 = time.time()
    df = cv_portion(pd.read_parquet(FEATURES_TRAIN_PATH))
    categorical = json.loads(CATEGORICAL_COLUMNS_PATH.read_text())
    n_before = df.shape[1] - len(PROTECTED)

    df, dropped = drop_low_value_columns(df)
    print(f"[selection] {n_before} -> {df.shape[1] - len(PROTECTED)} cột sau lọc ({time.time() - t0:.0f}s)")

    X = df.drop(columns=list(PROTECTED))
    imp = rank_by_lgbm_gain(X, df["TARGET"], categorical, n_splits=n_splits)
    chosen = list(imp.index[:n_features])
    print(f"[selection] chọn {len(chosen)} cột ({time.time() - t0:.0f}s)")

    SELECTED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_FEATURES_PATH.write_text(json.dumps(chosen, indent=1))
    imp.rename("gain").to_csv(IMPORTANCE_PATH, index_label="feature")
    _write_report(imp, dropped, chosen, n_before, time.time() - t0)
    return chosen
