from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from src.config import MODELS_DIR, RANDOM_SEED
from src.ph1_credit_risk.evaluation import metrics as mt
from src.online import repository as repo
from src.online.feature_set import (
    feature_set_version, load_categorical, load_selected_features, to_model_frame,
)
from src.online.schema import ROLE_CHALLENGER
from src.ph1_credit_risk.modeling import registry as rg

__all__ = ["RetrainResult", "MIN_TRAINING_ROWS", "MIN_POSITIVES", "VALID_FRACTION",
           "can_retrain", "retrain_from_database", "CHALLENGER_NAME"]

CHALLENGER_NAME = "ph1_lightgbm_online"
CHALLENGER_DIR = MODELS_DIR / "online"

MIN_TRAINING_ROWS = 2_000   # dưới mức này AUC dao động quá mạnh, không đáng train
MIN_POSITIVES = 100         # cần đủ ca vỡ nợ, không chỉ đủ dòng
VALID_FRACTION = 0.2

@dataclass
class RetrainResult:
    model_name: str
    version: int
    n_train: int
    n_valid: int
    n_positives: int
    auc_train: float
    auc_valid: float
    ks_valid: float
    fit_seconds: float
    artifact_path: str
    feature_set_version: str
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)

def can_retrain(session: Session) -> tuple[bool, str]:
    fsv = feature_set_version(load_selected_features())
    c = repo.labeled_training_counts(session, fsv)
    n, n_pos = c["n"], c["n_positives"]
    if n < MIN_TRAINING_ROWS:
        return False, (f"Mới có {n:,} hồ sơ đã biết kết quả, cần tối thiểu "
                       f"{MIN_TRAINING_ROWS:,}. Tiết lộ thêm nhãn rồi thử lại.")
    if n_pos < MIN_POSITIVES:
        return False, f"Mới có {n_pos} ca vỡ nợ, cần tối thiểu {MIN_POSITIVES}."
    return True, f"Sẵn sàng: {n:,} hồ sơ đã biết kết quả, trong đó {n_pos:,} ca vỡ nợ."


def retrain_from_database(session: Session, models_dir: Path = CHALLENGER_DIR,
                          activate: bool = False) -> RetrainResult:
    import time

    features = load_selected_features()
    fsv = feature_set_version(features)
    categorical = load_categorical(features)

    df = repo.labeled_training_frame(session, fsv, features)
    ok, reason = can_retrain(session)
    if not ok:
        raise ValueError(reason)

    X = to_model_frame(df, features, categorical)
    y = pd.to_numeric(df["TARGET"], errors="coerce").astype(int)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=VALID_FRACTION, stratify=y, random_state=RANDOM_SEED)

    t0 = time.time()
    est = rg.build_and_fit("lightgbm", X_tr, y_tr, categorical=categorical)
    fit_seconds = time.time() - t0

    p_tr = rg.predict_proba(est, X_tr)
    p_va = rg.predict_proba(est, X_va)
    auc_train = float(roc_auc_score(y_tr, p_tr))
    auc_valid = float(roc_auc_score(y_va, p_va))
    ks_valid = float(mt.ks_statistic(y_va.to_numpy(), p_va))

    models_dir.mkdir(parents=True, exist_ok=True)
    version = repo.next_version(session, CHALLENGER_NAME)
    path = models_dir / f"{CHALLENGER_NAME}_v{version}.joblib"
    joblib.dump(est, path)

    threshold = float(mt.threshold_for_flag_rate(p_va, _champion_flag_rate()))
    mv = repo.register_model(
        session, CHALLENGER_NAME,
        artifact_path=str(path), feature_set_version=fsv,
        n_train_samples=int(len(X_tr)),
        train_metrics={"auc": auc_train},
        valid_metrics={"auc": auc_valid, "ks": ks_valid, "n": int(len(X_va))},
        threshold=threshold, role=ROLE_CHALLENGER,
        note=(f"LightGBM train từ DB lúc {datetime.now(UTC):%Y-%m-%d %H:%M} — "
              "CHƯA qua holdout, số valid chỉ để theo dõi, không so được với bảng Task 10"),
        activate=activate,
    )

    return RetrainResult(
        model_name=CHALLENGER_NAME, version=mv.version,
        n_train=len(X_tr), n_valid=len(X_va), n_positives=int(y.sum()),
        auc_train=auc_train, auc_valid=auc_valid, ks_valid=ks_valid,
        fit_seconds=fit_seconds, artifact_path=str(path),
        feature_set_version=fsv, threshold=threshold,
    )

def _champion_flag_rate() -> float:
    import json

    from src.config import REPORTS_DIR
    basis = json.loads((REPORTS_DIR / "threshold_basis.json").read_text())
    return float(basis["flag_rate"])
