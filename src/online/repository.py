from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

import pandas as pd
from sqlalchemy import Integer, func, select, update
from sqlalchemy.orm import Session

from src.online.schema import (
    INPUT_MANUAL, ROLE_CHALLENGER, Applicant, ApplicantFeatures, ModelVersion, Prediction,
)

__all__ = [
    "counts", "list_applicants", "get_applicant", "get_features", "features_frame",
    "labeled_training_frame", "labeled_training_counts", "register_model", "activate_model", "active_model",
    "list_models", "next_version", "save_prediction", "predictions_frame",
    "pending_predictions", "set_actual_label", "label_progress", "risk_band_stats",
    "default_rate_by_buckets",
]

# Tổng quan
def counts(session: Session) -> dict[str, int]:
    def n(model, *where):
        return int(session.execute(select(func.count()).select_from(model).where(*where)).scalar() or 0)

    return {
        "applicants": n(Applicant),
        "labeled": n(Applicant, Applicant.target.is_not(None)),
        "unlabeled": n(Applicant, Applicant.target.is_(None)),
        "holdout": n(Applicant, Applicant.is_holdout.is_(True)),
        "features": n(ApplicantFeatures),
        "predictions": n(Prediction),
        "model_versions": n(ModelVersion),
    }


# Hồ sơ

def list_applicants(session: Session, limit: int = 300, unlabeled_only: bool = True,
                    exclude_holdout: bool = True) -> list[dict[str, Any]]:
    stmt = select(Applicant)
    if unlabeled_only:
        stmt = stmt.where(Applicant.target.is_(None))
    if exclude_holdout:
        stmt = stmt.where(Applicant.is_holdout.is_(False))
    rows = session.execute(stmt.order_by(Applicant.sk_id_curr).limit(limit)).scalars().all()
    return [{
        "sk_id_curr": r.sk_id_curr,
        "amt_credit": r.amt_credit,
        "amt_annuity": r.amt_annuity,
        "amt_income_total": r.amt_income_total,
        "days_birth": r.days_birth,
        "days_employed": r.days_employed,
        "ext_source_mean": r.ext_source_mean,
        "target": r.target,
    } for r in rows]

def get_applicant(session: Session, sk_id_curr: int) -> Applicant | None:
    return session.get(Applicant, int(sk_id_curr))

def get_features(session: Session, sk_id_curr: int, feature_set_version: str) -> dict | None:
    """80 feature của một hồ sơ, đúng phiên bản bộ feature yêu cầu."""
    row = session.execute(
        select(ApplicantFeatures).where(
            ApplicantFeatures.sk_id_curr == int(sk_id_curr),
            ApplicantFeatures.feature_set_version == feature_set_version,
        )
    ).scalars().first()
    return dict(row.features) if row else None

def features_frame(session: Session, feature_set_version: str, columns: Sequence[str],
                   limit: int | None = None, labeled_only: bool = False,
                   exclude_holdout: bool = True) -> pd.DataFrame:
    stmt = (select(ApplicantFeatures.sk_id_curr, ApplicantFeatures.features, Applicant.target)
            .join(Applicant, Applicant.sk_id_curr == ApplicantFeatures.sk_id_curr)
            .where(ApplicantFeatures.feature_set_version == feature_set_version))
    if labeled_only:
        stmt = stmt.where(Applicant.target.is_not(None), Applicant.label_revealed_at.is_not(None))
    if exclude_holdout:
        stmt = stmt.where(Applicant.is_holdout.is_(False))
    if limit:
        stmt = stmt.limit(limit)

    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["SK_ID_CURR", *columns, "TARGET"])

    df = pd.DataFrame([{"SK_ID_CURR": sk, **feats, "TARGET": tgt} for sk, feats, tgt in rows])
    for c in columns:                     # cột thiếu trong JSON -> NaN, không để KeyError
        if c not in df.columns:
            df[c] = pd.NA
    return df[["SK_ID_CURR", *columns, "TARGET"]]

def labeled_training_frame(session: Session, feature_set_version: str,
                           columns: Sequence[str]) -> pd.DataFrame:
    return features_frame(session, feature_set_version, columns,
                          labeled_only=True, exclude_holdout=True)

def labeled_training_counts(session: Session, feature_set_version: str) -> dict[str, int]:
    n, n_pos = session.execute(
        select(func.count(), func.sum(func.cast(Applicant.target == 1, Integer)))
        .select_from(ApplicantFeatures)
        .join(Applicant, Applicant.sk_id_curr == ApplicantFeatures.sk_id_curr)
        .where(
            ApplicantFeatures.feature_set_version == feature_set_version,
            Applicant.target.is_not(None), Applicant.label_revealed_at.is_not(None),
            Applicant.is_holdout.is_(False),
        )
    ).one()
    return {"n": int(n or 0), "n_positives": int(n_pos or 0)}

# Sổ đăng ký model

def next_version(session: Session, model_name: str) -> int:
    cur = session.execute(
        select(func.max(ModelVersion.version)).where(ModelVersion.model_name == model_name)
    ).scalar()
    return int(cur or 0) + 1

def register_model(session: Session, model_name: str, *, artifact_path: str | None = None,
                   feature_set_version: str | None = None, n_train_samples: int | None = None,
                   train_metrics: dict | None = None, valid_metrics: dict | None = None,
                   threshold: float | None = None, role: str = ROLE_CHALLENGER,
                   note: str | None = None, activate: bool = False) -> ModelVersion:
    mv = ModelVersion(
        model_name=model_name, version=next_version(session, model_name), role=role,
        artifact_path=artifact_path, feature_set_version=feature_set_version,
        n_train_samples=n_train_samples, train_metrics=train_metrics, valid_metrics=valid_metrics,
        threshold=threshold, note=note, is_active=False, trained_at=datetime.now(UTC),
    )
    session.add(mv)
    session.commit()
    if activate:
        activate_model(session, mv.id)
    return mv

def activate_model(session: Session, model_version_id: int) -> ModelVersion:
    session.execute(update(ModelVersion).values(is_active=False))
    session.execute(update(ModelVersion)
                    .where(ModelVersion.id == int(model_version_id)).values(is_active=True))
    session.commit()
    return session.get(ModelVersion, int(model_version_id))

def active_model(session: Session) -> ModelVersion | None:
    return session.execute(
        select(ModelVersion).where(ModelVersion.is_active.is_(True))
    ).scalars().first()

def list_models(session: Session) -> pd.DataFrame:
    rows = session.execute(select(ModelVersion).order_by(ModelVersion.id)).scalars().all()
    return pd.DataFrame([{
        "id": r.id, "model": r.model_name, "version": r.version, "vai_tro": r.role,
        "dang_dung": "✓" if r.is_active else "",
        "auc_train": (r.train_metrics or {}).get("auc"),
        "auc_valid": (r.valid_metrics or {}).get("auc"),
        "ks_valid": (r.valid_metrics or {}).get("ks"),
        "n_train": r.n_train_samples, "nguong": r.threshold,
        "bo_feature": r.feature_set_version,
        "train_luc": r.trained_at.strftime("%Y-%m-%d %H:%M") if r.trained_at else "",
        "ghi_chu": r.note or "",
    } for r in rows])


# Dự đoán

def save_prediction(session: Session, *, proba: float, threshold: float,
                    sk_id_curr: int | None = None, model_version_id: int | None = None,
                    input_type: str = INPUT_MANUAL, features: dict | None = None) -> Prediction:
    p = Prediction(
        sk_id_curr=sk_id_curr, model_version_id=model_version_id,
        predicted_proba=float(proba), predicted_label=int(proba >= threshold),
        threshold=float(threshold), input_type=input_type, features=features,
    )
    session.add(p)
    session.commit()
    return p

def predictions_frame(session: Session, limit: int = 500) -> pd.DataFrame:
    rows = session.execute(
        select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit)
    ).scalars().all()
    return pd.DataFrame([{
        "id": r.id, "sk_id_curr": r.sk_id_curr, "pd": r.predicted_proba,
        "du_doan": r.predicted_label, "thuc_te": r.actual_label, "nguong": r.threshold,
        "nguon": r.input_type, "model_version_id": r.model_version_id,
        "luc": r.created_at,
    } for r in rows])

def pending_predictions(session: Session, limit: int = 200) -> pd.DataFrame:
    rows = session.execute(
        select(Prediction).where(Prediction.actual_label.is_(None),
                                 Prediction.sk_id_curr.is_not(None))
        .order_by(Prediction.created_at.desc()).limit(limit)
    ).scalars().all()
    return pd.DataFrame([{
        "id": r.id, "sk_id_curr": r.sk_id_curr, "pd": round(r.predicted_proba, 4),
        "du_doan": r.predicted_label, "luc": r.created_at,
    } for r in rows])

def set_actual_label(session: Session, prediction_id: int, actual: int) -> bool:
    p = session.get(Prediction, int(prediction_id))
    if p is None:
        return False
    p.actual_label = int(actual)
    if p.sk_id_curr is not None:
        a = session.get(Applicant, p.sk_id_curr)
        if a is not None and a.target is None:
            a.target, a.label_revealed_at = int(actual), datetime.now(UTC)
    session.commit()
    return True

# Thống kê cho tab nghiệp vụ 

def label_progress(session: Session) -> dict[str, int]:
    c = counts(session)
    return {"labeled": c["labeled"], "unlabeled": c["unlabeled"], "total": c["applicants"]}

def risk_band_stats(session: Session, feature_set_version: str,
                    bands: Sequence[float] = (0.05, 0.15, 0.30)) -> pd.DataFrame:
    """Phân bố dự đoán theo nhóm rủi ro, kèm tỉ lệ vỡ nợ thật ở nhóm đã có nhãn."""
    df = predictions_frame(session, limit=100_000)
    if df.empty:
        return pd.DataFrame(columns=["nhom", "so_ho_so", "pd_tb", "n_co_nhan", "ti_le_vo_no_that"])

    edges = [0.0, *bands, 1.01]
    labels = [f"< {bands[0]:.0%}"] + \
             [f"{lo:.0%}–{hi:.0%}" for lo, hi in zip(bands, bands[1:])] + \
             [f"≥ {bands[-1]:.0%}"]
    df["nhom"] = pd.cut(df["pd"], bins=edges, labels=labels, right=False)
    g = df.groupby("nhom", observed=True)
    return pd.DataFrame({
        "nhom": g.size().index,
        "so_ho_so": g.size().to_numpy(),
        "pd_tb": g["pd"].mean().to_numpy(),
        "n_co_nhan": g["thuc_te"].count().to_numpy(),
        "ti_le_vo_no_that": g["thuc_te"].mean().to_numpy(),
    })

def _bucket_frame(df: pd.DataFrame, column: str, n_bins: int) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["nhom", "ti_le_vo_no", "so_ho_so"])
    if df.empty or df[column].isna().all():
        return empty
    s = pd.to_numeric(df[column], errors="coerce")
    try:
        bins = pd.qcut(s, n_bins, duplicates="drop")
    except ValueError:
        return empty
    g = df.assign(_bin=bins).groupby("_bin", observed=True)["TARGET"]
    return pd.DataFrame({
        "nhom": [str(i) for i in g.mean().index],
        "ti_le_vo_no": g.mean().to_numpy(),
        "so_ho_so": g.size().to_numpy(),
    })

def default_rate_by_buckets(session: Session, feature_set_version: str, columns: Sequence[str],
                            n_bins: int = 6) -> dict[str, pd.DataFrame]:
    df = features_frame(session, feature_set_version, list(columns),
                        labeled_only=True, exclude_holdout=True)
    return {column: _bucket_frame(df, column, n_bins) for column in columns}