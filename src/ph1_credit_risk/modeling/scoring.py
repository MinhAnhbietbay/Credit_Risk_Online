"""Chấm điểm một bảng feature bằng model cuối: mỗi member = trung bình các model fold, rồi `Ensemble.combine`.
Dùng chung cho holdout và application_test."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.ph1_credit_risk.modeling import registry as rg

__all__ = ["predict_ensemble"]

def predict_ensemble(ensemble, X: pd.DataFrame, models_dir: Path = MODELS_DIR) -> dict[str, np.ndarray]:
    preds, n_folds = {}, {}
    for m in ensemble.members:
        folds = sorted(Path(models_dir).glob(f"ph1_{m}_fold*.joblib"))
        if not folds:
            raise FileNotFoundError(f"không có model fold cho {m} trong {models_dir}")
        n_folds[m] = len(folds)
        preds[m] = np.mean([rg.predict_proba(joblib.load(f), X) for f in folds], axis=0)
        # Trộn thiếu fold hoặc lệch chiều vẫn ra một mảng xác suất trông bình thường: chặn tại đây.
        assert preds[m].shape == (len(X),), f"{m}: ra {preds[m].shape}, cần ({len(X)},)"
    if len(set(n_folds.values())) != 1:
        raise ValueError(f"số model fold không đều giữa các member: {n_folds}")
    preds["ensemble"] = ensemble.combine(preds)
    print(f"[score] {len(X):,} dòng × {len(ensemble.members)} member, mỗi member trung bình "
          f"{next(iter(n_folds.values()))} model fold checked", flush=True)
    return preds