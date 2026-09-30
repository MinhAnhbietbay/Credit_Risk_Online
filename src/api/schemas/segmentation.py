from __future__ import annotations

from typing import Any

from pydantic import BaseModel

class RiskSegmentResponse(BaseModel):
    bands: list[dict[str, Any]]
    total: int
    n_labeled: int
    pd_mean: float | None
    band_chart: dict[str, Any] | None
    calibration_chart: dict[str, Any] | None
