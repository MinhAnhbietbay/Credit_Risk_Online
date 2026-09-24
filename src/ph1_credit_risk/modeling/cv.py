from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import CV_FOLDS, MODELS_DIR, OUTPUT_DIR, RANDOM_SEED
from src.ph1_credit_risk.modeling import registry as rg

__all__ = ["CVResult", "run_cv", "save_cv_result", "OOF_INIT", "OOF_DIR"]

OOF_DIR = OUTPUT_DIR / "oof"
OOF_INIT = -1.0 

@dataclass
class CVResult:
    model_name: str
    oof_pred: np.ndarray            # dài bằng len(X), theo vị trí dòng
    fold_scores: list[float]        # AUC từng fold
    mean_auc: float
    std_auc: float
    fit_seconds: float
    fitted_models: list = field(default_factory=list)              # 1 model / fold
    fold_indices: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    fold_seconds: list[float] = field(default_factory=list)

def run_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: Sequence[str] = (),
    n_splits: int = CV_FOLDS,
    verbose: bool = False,
) -> CVResult:
    y_arr = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof = np.full(len(X), OOF_INIT, dtype=np.float64)
    scores, models, indices, secs = [], [], [], []
    t_all = time.time()
    for k, (tr, va) in enumerate(skf.split(X, y_arr)):
        t0 = time.time()
        # build model mới mỗi fold; boosting tự cắt eval set từ train-fold, không nhìn valid-fold
        est = rg.build_and_fit(model_name, X.iloc[tr], pd.Series(y_arr[tr]), categorical)
        oof[va] = rg.predict_proba(est, X.iloc[va])
        auc = float(roc_auc_score(y_arr[va], oof[va]))
        scores.append(auc); models.append(est); indices.append((tr, va)); secs.append(time.time() - t0)
        if verbose:
            print(f"  [{model_name}] fold {k}: AUC {auc:.4f}  {secs[-1]:.0f}s", flush=True)
    assert not np.any(oof == OOF_INIT), "OOF chưa phủ kín mọi dòng"
    return CVResult(
        model_name=model_name, oof_pred=oof, fold_scores=scores,
        mean_auc=float(np.mean(scores)), std_auc=float(np.std(scores)),
        fit_seconds=time.time() - t_all, fitted_models=models, fold_indices=indices, fold_seconds=secs,
    )

def save_cv_result(res: CVResult, models_dir: Path = MODELS_DIR, oof_dir: Path = OOF_DIR) -> dict[str, Any]:
    """`outputs/models/ph1_<model>_fold<k>.joblib` + `outputs/oof/<model>_oof.npy`."""
    models_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{res.model_name}_oof.npy"
    np.save(oof_path, res.oof_pred)
    model_paths = []
    for k, m in enumerate(res.fitted_models):
        p = models_dir / f"ph1_{res.model_name}_fold{k}.joblib"
        joblib.dump(m, p)
        model_paths.append(p)
    return {"oof": oof_path, "models": model_paths}
