"""Tab Phân khúc rủi ro"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas.segmentation import RiskSegmentResponse
from src.api.serialize import figure, records
from src.api.services import catalog, charts
from src.online import repository as repo

router = APIRouter(tags=["segmentation"])


@router.get("/risk-segments", response_model=RiskSegmentResponse)
def risk_segments(session: Session = Depends(get_db)) -> RiskSegmentResponse:
    frame = repo.risk_band_stats(session, catalog.current_feature_set_version())
    if frame.empty:
        return RiskSegmentResponse(bands=[], total=0, n_labeled=0, pd_mean=None,
                                   band_chart=None, calibration_chart=None)
    total = int(frame["so_ho_so"].sum())
    n_labeled = int(frame["n_co_nhan"].fillna(0).sum())
    return RiskSegmentResponse(
        bands=records(frame), total=total, n_labeled=n_labeled,
        pd_mean=float((frame["pd_tb"] * frame["so_ho_so"]).sum() / total),
        band_chart=figure(charts.band_chart(frame)),
        calibration_chart=figure(charts.calibration_chart(frame)) if n_labeled > 0 else None,
    )
