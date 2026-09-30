from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

JSONType = JSON().with_variant(JSONB, "postgresql")

__all__ = ["Base", "Applicant", "ApplicantFeatures", "ModelVersion", "Prediction",
           "ROLE_CHAMPION", "ROLE_CHALLENGER", "INPUT_EXISTING", "INPUT_MANUAL"]

ROLE_CHAMPION = "champion"      # model offline đã qua holdout 
ROLE_CHALLENGER = "challenger"  # model train lại từ dashboard, chưa qua holdout

INPUT_EXISTING = "ho_so_co_san"  # chấm một hồ sơ đã có trong DB
INPUT_MANUAL = "nhap_tay"        # chấm hồ sơ nhập tay trên form


class Applicant(Base):
    """Một hồ sơ xin vay. `sk_id_curr` giữ nguyên id của Home Credit để đối chiếu ngược."""

    __tablename__ = "applicants"

    sk_id_curr = Column(Integer, primary_key=True, autoincrement=False)

    amt_credit = Column(Float)          # số tiền vay được duyệt
    amt_annuity = Column(Float)         # tiền trả mỗi kỳ
    amt_income_total = Column(Float)    # thu nhập khai báo
    days_birth = Column(Integer)        # số ngày tới ngày sinh (âm)
    days_employed = Column(Integer)     # số ngày kể từ khi đi làm (âm)
    ext_source_mean = Column(Float)     # điểm tín dụng nguồn ngoài, trung bình 3 nguồn

    target = Column(Integer, nullable=True)
    label_revealed_at = Column(DateTime, nullable=True)

    is_holdout = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    features = relationship("ApplicantFeatures", back_populates="applicant",
                            cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_applicants_target", "target"),
        Index("ix_applicants_holdout", "is_holdout"),
    )


class ApplicantFeatures(Base):

    __tablename__ = "applicant_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sk_id_curr = Column(Integer, ForeignKey("applicants.sk_id_curr", ondelete="CASCADE"),
                        nullable=False)
    feature_set_version = Column(String(32), nullable=False)  # hash bộ feature, xem `feature_set.py`
    features = Column(JSONType, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    applicant = relationship("Applicant", back_populates="features")

    __table_args__ = (
        UniqueConstraint("sk_id_curr", "feature_set_version", name="uq_features_per_version"),
        Index("ix_features_version", "feature_set_version"),
    )

class ModelVersion(Base):

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False)
    version = Column(Integer, nullable=False)
    role = Column(String(16), nullable=False, default=ROLE_CHALLENGER)
    is_active = Column(Boolean, default=False, nullable=False)

    artifact_path = Column(String(512))          # đường dẫn joblib; ensemble thì trỏ tới thư mục
    feature_set_version = Column(String(32))
    n_train_samples = Column(Integer)
    train_metrics = Column(JSONType)             # {"auc": ..., "ks": ...}
    valid_metrics = Column(JSONType)
    threshold = Column(Float)                   
    note = Column(String(512))
    trained_at = Column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
        Index("ix_model_active", "is_active"),
    )

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sk_id_curr = Column(Integer, ForeignKey("applicants.sk_id_curr"), nullable=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)

    predicted_proba = Column(Float, nullable=False)
    predicted_label = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)
    actual_label = Column(Integer, nullable=True)     # điền khi biết kết quả thật
    input_type = Column(String(32), nullable=False)
    features = Column(JSONType)                        # ảnh chụp feature lúc chấm, để truy vết

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("ix_pred_created", "created_at"),
        Index("ix_pred_actual", "actual_label"),
    )