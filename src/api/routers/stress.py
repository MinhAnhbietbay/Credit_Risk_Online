"""Tab Stress test"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas.stress import (
    ScenarioRow, ScenariosResponse, StressRow, StressRunRequest, StressRunResponse,
)
from src.api.serialize import figure
from src.api.services import catalog, charts, db, scoring, stressing
from src.ph1_credit_risk.stress import macro_scenarios as ms

router = APIRouter(prefix="/stress", tags=["stress"])


def _shock(channel: str, annuity_multiplier: float, percentile_shift: float) -> str:
    if channel == ms.CHANNEL_COST:
        return f"×{annuity_multiplier:.4f} annuity"
    return f"+{percentile_shift * 100:.0f} điểm phân vị"


def _source(s) -> str:
    if s.quantile is not None and s.reference_quantile is not None:
        return f"p{s.quantile * 100:.0f} vs p{s.reference_quantile * 100:.0f} đo trên dữ liệu"
    return "mốc, không đổi"


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    return ScenariosResponse(
        warning=stressing.WARNING, non_monotonic_note=stressing.NON_MONOTONIC_NOTE,
        scenarios=[ScenarioRow(name=s.name, channel=s.channel, description=s.description,
                               shock=_shock(s.channel, s.annuity_multiplier, s.percentile_shift),
                               source=_source(s))
                   for s in stressing.load_scenarios()])


@router.post("/run", response_model=StressRunResponse)
def run(req: StressRunRequest, session: Session = Depends(get_db)) -> StressRunResponse:
    model_info = db.active_model_info(session)
    if model_info is None:
        raise HTTPException(400, "Chưa có model nào được kích hoạt.")
    try:
        scoring.check_feature_set(model_info)
    except scoring.FeatureSetMismatch as exc:
        raise HTTPException(400, str(exc)) from exc

    X = stressing.portfolio_sample(session, catalog.current_feature_set_version(), req.n)
    if X.empty:
        raise HTTPException(400, "Không lấy được hồ sơ nào từ DB - chạy "
                                 "`python scripts/ph1_db_seed.py`.")

    threshold = scoring.active_threshold(model_info)
    scorer = scoring.load_scorer_for(model_info["model_name"], model_info["version"],
                                     model_info["artifact_path"])
    table = stressing.run_impact(scorer, X, stressing.load_scenarios(), threshold)
    rows = [StressRow(scenario=r.scenario, channel=r.channel,
                      shock=_shock(r.channel, r.annuity_multiplier, r.percentile_shift),
                      pd_mean=r.pd_mean, flag_rate=r.flag_rate, flag_rate_delta=r.flag_rate_delta)
            for r in table.itertuples()]
    return StressRunResponse(n_sample=len(X), threshold=threshold, rows=rows,
                             band_chart=figure(charts.stress_band_chart(table)))
