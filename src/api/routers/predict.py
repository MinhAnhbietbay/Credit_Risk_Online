"""Tab Dự đoán"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.cache import clear_ttl_caches
from src.api.deps import get_db
from src.api.schemas.predict import (
    ApplicantDetailResponse, ApplicantOption, FilledFeature, FormConfigResponse, PredictRequest,
    PredictResponse, ShapResult, TopFeature,
)
from src.api.serialize import figure
from src.api.services import catalog, charts, db, explaining, scoring
from src.api.services.theme import decision_text, risk_level
from src.online import repository as repo
from src.online.schema import INPUT_EXISTING, INPUT_MANUAL

router = APIRouter(prefix="/predict", tags=["predict"])

NO_ACTIVE_MODEL = ("Chưa có model nào được kích hoạt. Sang tab Train lại để đăng ký/kích hoạt.")


def _applicant_label(a: dict) -> str:
    tuoi = int(-(a["days_birth"] or 0) / 365.25)
    return (f"{a['sk_id_curr']}  ·  vay {a['amt_credit'] or 0:,.0f}  ·  "
            f"trả kỳ {a['amt_annuity'] or 0:,.0f}  ·  {tuoi} tuổi")


def _features_or_404(session: Session, sk_id_curr: int) -> dict:
    fsv = catalog.current_feature_set_version()
    feats = repo.get_features(session, sk_id_curr, fsv)
    if feats is None:
        raise HTTPException(404, f"Hồ sơ {sk_id_curr} không có bộ feature phiên bản `{fsv}`.")
    return feats


@router.get("/form-config", response_model=FormConfigResponse)
def form_config(session: Session = Depends(get_db)) -> FormConfigResponse:
    rows = repo.list_applicants(session, limit=300, unlabeled_only=True)
    return FormConfigResponse(
        fields=list(catalog.form_features()),
        n_features_total=len(catalog.selected_features()),
        applicants=[ApplicantOption(sk_id_curr=a["sk_id_curr"], label=_applicant_label(a))
                    for a in rows],
        active_model=db.active_model_info(session),
    )


@router.get("/applicants/{sk_id_curr}", response_model=ApplicantDetailResponse)
def applicant_detail(sk_id_curr: int,
                     session: Session = Depends(get_db)) -> ApplicantDetailResponse:
    feats = _features_or_404(session, sk_id_curr)
    return ApplicantDetailResponse(sk_id_curr=sk_id_curr, top_features=[
        TopFeature(feature=f["feature"], value=catalog.format_value(f["feature"], feats.get(f["feature"])),
                   mo_ta=f["mo_ta"])
        for f in catalog.form_features()])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest, session: Session = Depends(get_db)) -> PredictResponse:
    model_info = db.active_model_info(session)
    if model_info is None:
        raise HTTPException(400, NO_ACTIVE_MODEL)

    if req.sk_id_curr is not None:
        X, filled = scoring.build_feature_row(_features_or_404(session, req.sk_id_curr))
        input_type = INPUT_EXISTING
    else:
        errors = scoring.validate_manual_values(req.values)
        if errors:
            raise HTTPException(422, "; ".join(errors))
        X, filled = scoring.build_feature_row(req.values)
        input_type = INPUT_MANUAL

    try:
        result = scoring.score_row(X, model_info, filled)
    except scoring.FeatureSetMismatch as exc:
        raise HTTPException(400, str(exc)) from exc

    prediction_id = scoring.save(session, result, X, req.sk_id_curr, input_type)
    clear_ttl_caches()

    shap, shap_error = None, None
    try:
        one = explaining.explain(X, k=5)
        shap = ShapResult(note=explaining.NOTE, probability=one["probability"],
                          base_probability=one["base_probability"],
                          factors=explaining.factors(one))
    except FileNotFoundError as exc:
        shap_error = str(exc)

    level, color = risk_level(result.pd_value, result.threshold)
    return PredictResponse(
        pd=result.pd_value, threshold=result.threshold, flagged=result.flagged,
        risk_level=level, risk_color=color,
        decision=decision_text(result.pd_value, result.threshold),
        model_label=result.model_label, prediction_id=prediction_id,
        filled_from_median=[FilledFeature(feature=f, mo_ta=catalog.describe_feature(f))
                            for f in result.filled_from_median],
        n_features_total=len(catalog.selected_features()),
        gauge=figure(charts.risk_gauge(result.pd_value, result.threshold)),
        shap=shap, shap_error=shap_error,
    )
