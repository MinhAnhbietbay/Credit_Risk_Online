"""Tab Insight nghiệp vụ"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas.catalog import Counts
from src.api.schemas.insights import DefaultRatesResponse, FeatureInsight, ModelComparisonResponse
from src.api.serialize import figure, records
from src.api.services import catalog, charts, db
from src.online import repository as repo

router = APIRouter(prefix="/insights", tags=["insights"])

INSIGHT_FEATURES = [
    "EXT_SOURCE_MEAN", "INST_LATE_RATIO", "CREDIT_TERM",
    "DAYS_EMPLOYED", "PREV_REFUSED_RATIO", "ANNUITY_INCOME_RATIO",
]

SHAP_NOTE = ("SHAP mô tả CatBoost (model cây thành phần tốt nhất), không phải ensemble stacking "
             "đang dùng để chấm PD.")


def _is_monotonic(rates: pd.Series) -> bool:
    return bool(rates.is_monotonic_increasing or rates.is_monotonic_decreasing)


def insight_sentence(feature: str, frame: pd.DataFrame) -> str:
    """Câu nhận xét sinh từ số đo, không viết cứng."""
    rates = frame["ti_le_vo_no"]
    lo, hi = float(rates.iloc[0]), float(rates.iloc[-1])
    n = int(frame["so_ho_so"].sum())
    verb, prep = ("tăng", "lên") if hi > lo else ("giảm", "xuống")
    small = min(hi, lo)
    ratio = f" (gấp {max(hi, lo) / small:.1f} lần)" if small > 0 else ""
    text = (f"{feature}: {catalog.describe_feature(feature)}. Đi từ nhóm thấp nhất sang cao nhất, "
            f"tỉ lệ vỡ nợ {verb} từ {lo:.2%} {prep} {hi:.2%}{ratio}, đo trên {n:,} hồ sơ đã biết "
            "kết quả trong hệ thống.")
    if not _is_monotonic(rates):
        text += " Quan hệ không đơn điệu (nhóm cao nhất không phải nhóm rủi ro nhất)."
    return text


@router.get("/default-rates", response_model=DefaultRatesResponse)
def default_rates(features: str | None = Query(None, description="Tên feature, phân cách dấu phẩy"),
                  session: Session = Depends(get_db)) -> DefaultRatesResponse:
    selected = set(catalog.selected_features())
    available = [f for f in INSIGHT_FEATURES if f in selected]
    picked = available[:4] if features is None else [f for f in features.split(",") if f]
    unknown = [f for f in picked if f not in available]
    if unknown:
        raise HTTPException(422, f"Yếu tố không hỗ trợ: {', '.join(unknown)}. "
                                 f"Chọn trong: {', '.join(available)}.")

    counts = repo.counts(session)
    if counts["labeled"] == 0:
        return DefaultRatesResponse(counts=Counts(**counts), available_features=available,
                                    picked=picked, features=[])

    fsv = catalog.current_feature_set_version()
    frames = db.default_rate_by_buckets(session, fsv, tuple(picked), 6)
    out = []
    for feature in picked:
        frame = frames[feature]
        if frame.empty:
            out.append(FeatureInsight(feature=feature, mo_ta=catalog.describe_feature(feature),
                                      buckets=[], chart=None, sentence=None, monotonic=None))
            continue
        out.append(FeatureInsight(
            feature=feature, mo_ta=catalog.describe_feature(feature), buckets=records(frame),
            chart=figure(charts.bucket_chart(feature, frame)),
            sentence=insight_sentence(feature, frame),
            monotonic=_is_monotonic(frame["ti_le_vo_no"])))
    return DefaultRatesResponse(counts=Counts(**counts), available_features=available,
                                picked=picked, features=out)


@router.get("/model-comparison", response_model=ModelComparisonResponse)
def model_comparison() -> ModelComparisonResponse:
    return ModelComparisonResponse(
        model_comparison=records(catalog.model_comparison()),
        shap_ranking=records(catalog.shap_ranking().head(15)),
        shap_note=SHAP_NOTE,
    )
