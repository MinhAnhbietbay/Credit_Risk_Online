from __future__ import annotations

from pydantic import BaseModel

class ActiveModel(BaseModel):
    id: int
    model_name: str
    version: int
    role: str
    threshold: float | None
    feature_set_version: str | None
    auc: float | None
    note: str

class Counts(BaseModel):
    applicants: int
    labeled: int
    predictions: int
    model_versions: int

class StatusResponse(BaseModel):
    db_ready: bool
    message: str
    active_model: ActiveModel | None
    counts: Counts | None
    feature_set_version: str
    n_features: int
