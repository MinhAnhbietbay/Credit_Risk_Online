from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.api.schemas.catalog import Counts

class FeatureInsight(BaseModel):
    feature: str
    mo_ta: str
    buckets: list[dict[str, Any]]
    chart: dict[str, Any] | None
    sentence: str | None
    monotonic: bool | None

class DefaultRatesResponse(BaseModel):
    counts: Counts
    available_features: list[str]
    picked: list[str]
    features: list[FeatureInsight]

class ModelComparisonResponse(BaseModel):
    model_comparison: list[dict[str, Any]]
    shap_ranking: list[dict[str, Any]]
    shap_note: str
