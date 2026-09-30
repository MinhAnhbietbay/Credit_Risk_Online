from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy.orm import Session

from src.api.cache import ttl_cache
from src.api.services import catalog
from src.online import repository as repo
from src.online.feature_set import to_model_frame
from src.ph1_credit_risk.stress import macro_scenarios as ms

__all__ = ["WARNING", "NON_MONOTONIC_NOTE", "load_scenarios", "portfolio_sample",
           "run_impact"]

WARNING = ms.WARNING

NON_MONOTONIC_NOTE = (
    "Kênh chi phí chỉ đáng tin ở cú sốc nhẹ. Quan hệ giữa gánh nặng trả "
    "nợ và vỡ nợ trong dữ liệu không đơn điệu - thập phân vị CREDIT_TERM cao nhất lại là nhóm ít "
    "vỡ nợ nhất (4.93%), vì đó là vay tiêu dùng POS món nhỏ kỳ hạn ngắn. Đẩy cả danh mục vào vùng "
    "đó làm model chấm rủi ro thấp đi. Kênh hành vi đơn điệu trên mọi cột nên tin được ở cả hai mức."
)

@lru_cache(maxsize=1)
def load_scenarios() -> tuple:
    return tuple(ms.load_scenarios())

@ttl_cache(seconds=300)
def portfolio_sample(session: Session, feature_set_version: str, n: int = 3_000) -> pd.DataFrame:
    features = catalog.selected_features()
    df = repo.features_frame(session, feature_set_version, features, limit=int(n))
    if df.empty:
        return df
    return to_model_frame(df, features, catalog.categorical_features())

def run_impact(scorer, X: pd.DataFrame, scenarios, threshold: float) -> pd.DataFrame:
    return ms.portfolio_impact(scorer, X, list(scenarios), threshold=threshold)