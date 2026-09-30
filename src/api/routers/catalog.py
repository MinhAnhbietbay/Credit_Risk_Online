"""GET /api/status - nội dung thanh bên: DB sống không, model đang dùng, số đếm, bộ feature."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas.catalog import Counts, StatusResponse
from src.api.services import catalog, db
from src.online import repository as repo

router = APIRouter(tags=["catalog"])


@router.get("/status", response_model=StatusResponse)
def status(session: Session = Depends(get_db)) -> StatusResponse:
    fsv, n_features = catalog.current_feature_set_version(), len(catalog.selected_features())
    ok, message = db.connection_status(session)
    if not ok:
        return StatusResponse(db_ready=False, message=message, active_model=None, counts=None,
                              feature_set_version=fsv, n_features=n_features)
    return StatusResponse(
        db_ready=True, message=message, active_model=db.active_model_info(session),
        counts=Counts(**repo.counts(session)), feature_set_version=fsv, n_features=n_features)
