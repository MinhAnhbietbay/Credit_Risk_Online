from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator

from src.api.schemas.catalog import ActiveModel


class ApplicantOption(BaseModel):
    sk_id_curr: int
    label: str

class FormConfigResponse(BaseModel):
    fields: list[dict[str, Any]]
    n_features_total: int
    applicants: list[ApplicantOption]
    active_model: ActiveModel | None

class TopFeature(BaseModel):
    feature: str
    value: str
    mo_ta: str

class ApplicantDetailResponse(BaseModel):
    sk_id_curr: int
    top_features: list[TopFeature]

class PredictRequest(BaseModel):
    sk_id_curr: int | None = None
    values: dict[str, float] | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "PredictRequest":
        if (self.sk_id_curr is None) == (self.values is None):
            raise ValueError("Gửi đúng một trong hai: `sk_id_curr` (hồ sơ có sẵn) "
                             "hoặc `values` (nhập tay)")
        return self

class FilledFeature(BaseModel):
    feature: str
    mo_ta: str

class ShapFactor(BaseModel):
    feature: str
    value: str
    direction: str
    shap: float
    mo_ta: str

class ShapResult(BaseModel):
    note: str
    probability: float
    base_probability: float
    factors: list[ShapFactor]

class PredictResponse(BaseModel):
    pd: float
    threshold: float
    flagged: bool
    risk_level: str
    risk_color: str
    decision: str
    model_label: str
    prediction_id: int
    filled_from_median: list[FilledFeature]
    n_features_total: int
    gauge: dict[str, Any]
    shap: ShapResult | None
    shap_error: str | None
