"""Tab Dự đoán vs thực tế"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sklearn.metrics import roc_auc_score
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas.predictions import PredictionsResponse, PredictionsSummary
from src.api.serialize import figure, records
from src.api.services import charts
from src.online import repository as repo

router = APIRouter(tags=["predictions"])


@router.get("/predictions", response_model=PredictionsResponse)
def predictions(limit: int = Query(1_000, ge=1, le=100_000),
                session: Session = Depends(get_db)) -> PredictionsResponse:
    df = repo.predictions_frame(session, limit=limit)
    if df.empty:
        return PredictionsResponse(
            rows=[], summary=PredictionsSummary(total=0, n_labeled=0, pd_mean=None, flag_rate=None),
            auc=None, auc_note=None, confusion=None, distribution=None)

    labeled = df.dropna(subset=["thuc_te"])
    auc, auc_note = None, None
    if labeled["thuc_te"].nunique() > 1:
        auc = float(roc_auc_score(labeled["thuc_te"].astype(int), labeled["pd"]))
    elif not labeled.empty:
        auc_note = "Phần đã có kết quả mới chỉ có một loại nhãn nên chưa tính được AUC."

    has_labels = not labeled.empty
    return PredictionsResponse(
        rows=records(df),
        summary=PredictionsSummary(total=len(df), n_labeled=len(labeled),
                                   pd_mean=float(df["pd"].mean()),
                                   flag_rate=float(df["du_doan"].mean())),
        auc=auc, auc_note=auc_note,
        confusion=figure(charts.confusion(df)) if has_labels else None,
        distribution=figure(charts.distribution(df)) if has_labels else None,
    )
