"""Nạp hồ sơ + 80 feature từ `features_train.parquet` vào DB.

Thiết kế cho vòng đời online: khi seed, một phần hồ sơ **giấu nhãn** (`target = NULL`) để tab
Retrain có việc mà làm — nhãn được tiết lộ dần bằng `simulator.py`, giống cách dữ liệu thật về
theo thời gian.

Hồ sơ thuộc holdout offline vẫn được nạp nhưng đánh dấu `is_holdout = True` và
**không bao giờ** vào tập retrain.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.config import RANDOM_SEED
from src.online.engine import get_session, init_db
from src.online.feature_set import feature_set_version, load_selected_features
from src.online.schema import Applicant, ApplicantFeatures
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED
from src.ph1_credit_risk.modeling.split import load_holdout_ids

__all__ = ["DISPLAY_COLUMNS", "seed_database", "read_source_frame", "clear_database",
           "exclude_holdout"]

# Cột phi chuẩn hoá lên bảng `applicants` để duyệt/lọc nhanh, không phải join JSONB
DISPLAY_COLUMNS = {
    "amt_credit": "AMT_CREDIT",
    "amt_annuity": "AMT_ANNUITY",
    "days_birth": "DAYS_BIRTH",
    "days_employed": "DAYS_EMPLOYED",
    "ext_source_mean": "EXT_SOURCE_MEAN",
}
# AMT_INCOME_TOTAL không nằm trong 80 feature đã chọn -> đọc thêm từ parquet cho phần hiển thị
INCOME_COLUMN = "AMT_INCOME_TOTAL"

BATCH = 2_000

def exclude_holdout(df: pd.DataFrame, holdout_ids: Iterable[int]) -> pd.DataFrame:
    holdout = {int(i) for i in holdout_ids}
    return df[~df["SK_ID_CURR"].isin(holdout)].reset_index(drop=True)


def read_source_frame(features: Sequence[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys([*PROTECTED, *features, INCOME_COLUMN]))
    return exclude_holdout(pd.read_parquet(FEATURES_TRAIN_PATH, columns=cols), load_holdout_ids())


def _clean(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value)

def _rows(df: pd.DataFrame, features: Sequence[str], holdout: set[int],
          labeled_mask: np.ndarray, fsv: str) -> Iterator[tuple[dict, dict]]:
    now = datetime.now(UTC)
    for (_, r), revealed in zip(df.iterrows(), labeled_mask):
        sk = int(r["SK_ID_CURR"])
        applicant = {
            "sk_id_curr": sk,
            "amt_income_total": _clean(r.get(INCOME_COLUMN)),
            **{db_col: _clean(r.get(src)) for db_col, src in DISPLAY_COLUMNS.items()},
            "target": int(r["TARGET"]) if revealed else None,
            "label_revealed_at": now if revealed else None,
            "is_holdout": sk in holdout,
            "created_at": now,
        }
        feats = {"sk_id_curr": sk, "feature_set_version": fsv, "created_at": now,
                 "features": {c: _clean(r.get(c)) for c in features}}
        yield applicant, feats


def clear_database(session: Session) -> None:
    """Xoá hồ sơ + feature (giữ nguyên sổ model và dự đoán)."""
    session.query(ApplicantFeatures).delete()
    session.query(Applicant).delete()
    session.commit()

def seed_database(labeled_fraction: float = 0.6, url: str | None = None,
                  replace: bool = True, verbose: bool = True) -> dict:
    features = load_selected_features()
    fsv = feature_set_version(features)
    init_db(url) if url else init_db()
    session = get_session(url) if url else get_session()

    try:
        if replace:
            clear_database(session)

        df = read_source_frame(features)
        holdout = set(load_holdout_ids())

        n_leak = int(df["SK_ID_CURR"].isin(holdout).sum())
        if n_leak:
            raise AssertionError(f"{n_leak} hồ sơ holdout lọt vào tập seed — holdout không bao giờ "
                                 "được nạp vào DB online (chỉ dùng chấm offline một lần, Task 10)")

        rng = np.random.default_rng(RANDOM_SEED)
        revealed = rng.random(len(df)) < labeled_fraction

        n_app = n_feat = 0
        app_buf, feat_buf = [], []
        for applicant, feats in _rows(df, features, holdout, revealed, fsv):
            app_buf.append(applicant)
            feat_buf.append(feats)
            if len(app_buf) >= BATCH:
                session.bulk_insert_mappings(Applicant, app_buf)
                session.bulk_insert_mappings(ApplicantFeatures, feat_buf)
                session.commit()
                n_app += len(app_buf); n_feat += len(feat_buf)
                app_buf, feat_buf = [], []
                if verbose:
                    print(f"  [seed] {n_app:,} hồ sơ", flush=True)
        if app_buf:
            session.bulk_insert_mappings(Applicant, app_buf)
            session.bulk_insert_mappings(ApplicantFeatures, feat_buf)
            session.commit()
            n_app += len(app_buf); n_feat += len(feat_buf)

        return {
            "n_applicants": n_app, "n_features": n_feat,
            "n_labeled": int(revealed.sum()), "n_unlabeled": int((~revealed).sum()),
            "n_holdout": int(df["SK_ID_CURR"].isin(holdout).sum()),
            "feature_set_version": fsv, "n_feature_columns": len(features),
        }
    finally:
        session.close()
