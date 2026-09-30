from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.api.cache import ttl_cache
from src.online import repository as repo

__all__ = ["connection_status", "active_model_info", "default_rate_by_buckets"]

def connection_status(session: Session) -> tuple[bool, str]:
    try:
        session.execute(text("select 1"))
    except OperationalError as exc:
        return False, f"{type(exc).__name__}: {exc}"[:300]
    return True, "ok"

def active_model_info(session: Session) -> dict | None:
    mv = repo.active_model(session)
    if mv is None:
        return None
    valid = mv.valid_metrics or {}
    return {
        "id": mv.id, "model_name": mv.model_name, "version": mv.version, "role": mv.role,
        "threshold": mv.threshold, "artifact_path": mv.artifact_path,
        "feature_set_version": mv.feature_set_version,
        "valid_metrics": valid, "train_metrics": mv.train_metrics or {},
        "n_train_samples": mv.n_train_samples, "note": mv.note or "",
        "auc": valid.get("auc"),
    }

@ttl_cache(seconds=120)
def default_rate_by_buckets(session: Session, feature_set_version: str, columns: tuple[str, ...],
                            n_bins: int = 6) -> dict[str, pd.DataFrame]:
    return repo.default_rate_by_buckets(session, feature_set_version, columns, n_bins=n_bins)
