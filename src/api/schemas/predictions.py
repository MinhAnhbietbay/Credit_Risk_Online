from __future__ import annotations

from typing import Any

from pydantic import BaseModel

class PredictionsSummary(BaseModel):
    total: int
    n_labeled: int
    pd_mean: float | None
    flag_rate: float | None

class PredictionsResponse(BaseModel):
    rows: list[dict[str, Any]]
    summary: PredictionsSummary
    auc: float | None
    auc_note: str | None
    confusion: dict[str, Any] | None
    distribution: dict[str, Any] | None
