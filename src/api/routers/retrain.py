"""Tab ReTrain"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.cache import clear_ttl_caches
from src.api.deps import get_db
from src.api.schemas.retrain import (
    LabelRequest, LabelResponse, ModelRef, Progress, PromoteRequest, RetrainResultResponse,
    RetrainStatusResponse, RevealRequest, RevealResponse, RunRequest,
)
from src.api.serialize import records
from src.online import models
from src.online import repository as repo
from src.online import retrain as rt
from src.online import simulator as sim
from src.online.schema import ModelVersion

router = APIRouter(prefix="/retrain", tags=["retrain"])

OVERFIT_GAP = 0.15   # chép từ tab Streamlit: chênh AUC train–valid lớn hơn mức này thì cảnh báo


def _ref(mv: ModelVersion) -> ModelRef:
    return ModelRef(id=mv.id, model_name=mv.model_name, version=mv.version, role=mv.role)


@router.get("/status", response_model=RetrainStatusResponse)
def status(session: Session = Depends(get_db)) -> RetrainStatusResponse:
    ok, reason = rt.can_retrain(session)
    return RetrainStatusResponse(
        progress=Progress(**repo.label_progress(session)),
        available_to_reveal=sim.available_to_reveal(session),
        can_retrain=ok, reason=reason,
        min_training_rows=rt.MIN_TRAINING_ROWS, min_positives=rt.MIN_POSITIVES,
        pending=records(repo.pending_predictions(session, limit=100)),
        models=records(repo.list_models(session)),
    )


@router.post("/reveal-labels", response_model=RevealResponse)
def reveal_labels(req: RevealRequest, session: Session = Depends(get_db)) -> RevealResponse:
    result = sim.reveal_labels(session, n=req.n)
    clear_ttl_caches()
    return RevealResponse(**result)


@router.post("/labels", response_model=LabelResponse)
def set_label(req: LabelRequest, session: Session = Depends(get_db)) -> LabelResponse:
    if not repo.set_actual_label(session, req.prediction_id, req.actual):
        raise HTTPException(404, f"Không tìm thấy bản ghi dự đoán #{req.prediction_id}.")
    clear_ttl_caches()
    return LabelResponse(prediction_id=req.prediction_id, actual=req.actual)


@router.post("/run", response_model=RetrainResultResponse)
def run(req: RunRequest, session: Session = Depends(get_db)) -> RetrainResultResponse:
    try:
        result = rt.retrain_from_database(session, activate=req.activate)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    clear_ttl_caches()
    return RetrainResultResponse(**result.as_dict(),
                                 overfit_warning=result.auc_train - result.auc_valid > OVERFIT_GAP)


@router.post("/promote", response_model=ModelRef)
def promote(req: PromoteRequest, session: Session = Depends(get_db)) -> ModelRef:
    # Kiểm tồn tại TRƯỚC: activate_model tắt mọi bản rồi mới bật bản được chọn,
    # id sai sẽ để hệ thống không còn model nào đang dùng.
    if session.get(ModelVersion, req.model_version_id) is None:
        raise HTTPException(404, f"Không có bản model #{req.model_version_id}.")
    mv = repo.activate_model(session, req.model_version_id)
    clear_ttl_caches()
    return _ref(mv)


@router.post("/register-champion", response_model=ModelRef)
def register_champion(session: Session = Depends(get_db)) -> ModelRef:
    if repo.counts(session)["model_versions"] > 0:
        raise HTTPException(400, "Sổ đăng ký đã có model - chọn một bản để kích hoạt thay vì "
                                 "đăng ký lại.")
    mv = models.register_offline_champion(session)
    clear_ttl_caches()
    return _ref(mv)
