from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

class Progress(BaseModel):
    total: int
    labeled: int
    unlabeled: int

class RetrainStatusResponse(BaseModel):
    progress: Progress
    available_to_reveal: int
    can_retrain: bool
    reason: str
    min_training_rows: int
    min_positives: int
    pending: list[dict[str, Any]]
    models: list[dict[str, Any]]

class RevealRequest(BaseModel):
    n: int = Field(1_000, ge=100, le=5_000)

class RevealResponse(BaseModel):
    n_revealed: int
    n_remaining: int
    n_default: int

class LabelRequest(BaseModel):
    prediction_id: int
    actual: Literal[0, 1]

class LabelResponse(BaseModel):
    prediction_id: int
    actual: int

class RunRequest(BaseModel):
    activate: bool = False

class RetrainResultResponse(BaseModel):
    model_name: str
    version: int
    n_train: int
    n_valid: int
    n_positives: int
    auc_train: float
    auc_valid: float
    ks_valid: float
    fit_seconds: float
    artifact_path: str
    feature_set_version: str
    threshold: float
    overfit_warning: bool

class PromoteRequest(BaseModel):
    model_version_id: int

class ModelRef(BaseModel):
    id: int
    model_name: str
    version: int
    role: str
