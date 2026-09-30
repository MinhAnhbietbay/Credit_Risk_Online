from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

class ScenarioRow(BaseModel):
    name: str
    channel: str
    description: str
    shock: str
    source: str

class ScenariosResponse(BaseModel):
    warning: str
    non_monotonic_note: str
    scenarios: list[ScenarioRow]

class StressRunRequest(BaseModel):
    n: int = Field(3_000, ge=500, le=10_000)

class StressRow(BaseModel):
    scenario: str
    channel: str
    shock: str
    pd_mean: float
    flag_rate: float
    flag_rate_delta: float

class StressRunResponse(BaseModel):
    n_sample: int
    threshold: float
    rows: list[StressRow]
    band_chart: dict[str, Any]
