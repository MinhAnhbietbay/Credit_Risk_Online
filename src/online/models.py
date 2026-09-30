"""Nạp model để chấm điểm, và đăng ký champion offline vào sổ `model_versions`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import CV_FOLDS, MODELS_DIR, REPORTS_DIR
from src.online.feature_set import feature_set_version, load_selected_features
from src.online.schema import ROLE_CHAMPION, ModelVersion
from src.ph1_credit_risk.modeling import registry as rg

__all__ = ["EnsembleScorer", "SingleModelScorer", "load_scorer", "champion_threshold",
           "register_offline_champion", "CHAMPION_NAME"]

CHAMPION_NAME = "ph1_ensemble_stacking"
HOLDOUT_METRICS_PATH = REPORTS_DIR / "holdout_metrics.json"

class EnsembleScorer:

    def __init__(self, members: dict[str, list], ensemble: Any):
        self.members = members
        self.ensemble = ensemble

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> "EnsembleScorer":
        members = {name: [joblib.load(models_dir / f"ph1_{name}_fold{k}.joblib")
                          for k in range(CV_FOLDS)]
                   for name in rg.list_models()}
        return cls(members, joblib.load(models_dir / "ph1_ensemble.joblib"))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = {name: np.mean([rg.predict_proba(m, X) for m in models], axis=0)
                 for name, models in self.members.items()}
        p = np.asarray(self.ensemble.combine(preds))
        return np.column_stack([1 - p, p])

class SingleModelScorer:

    def __init__(self, estimator: Any):
        self.estimator = estimator

    @classmethod
    def load(cls, path: str | Path) -> "SingleModelScorer":
        return cls(joblib.load(Path(path)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))


def load_scorer(mv: ModelVersion | None) -> EnsembleScorer | SingleModelScorer:
    if mv is None or mv.model_name == CHAMPION_NAME:
        return EnsembleScorer.load()
    if not mv.artifact_path:
        raise FileNotFoundError(f"Bản model {mv.model_name} v{mv.version} không có artifact_path")
    return SingleModelScorer.load(mv.artifact_path)


def champion_threshold() -> float:
    return float(json.loads(HOLDOUT_METRICS_PATH.read_text())["threshold"])


def register_offline_champion(session, activate: bool = True) -> ModelVersion:
    from src.online import repository as repo

    holdout = json.loads(HOLDOUT_METRICS_PATH.read_text())
    return repo.register_model(
        session, CHAMPION_NAME,
        artifact_path=str(MODELS_DIR / "ph1_ensemble.joblib"),
        feature_set_version=feature_set_version(load_selected_features()),
        n_train_samples=None,
        train_metrics={"auc": holdout.get("oof_auc")},
        valid_metrics={"auc": holdout.get("auc"), "ks": holdout.get("ks"),
                       "brier": holdout.get("brier"), "precision": holdout.get("precision"),
                       "recall": holdout.get("recall"), "n": holdout.get("n")},
        threshold=holdout.get("threshold"),
        role=ROLE_CHAMPION,
        note="Ensemble stacking offline (Task 9); số valid là HOLDOUT chấm một lần ở Task 10",
        activate=activate,
    )