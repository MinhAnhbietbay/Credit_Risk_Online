from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sqlalchemy.orm import Session

from src.api.services import catalog
from src.online import repository as repo
from src.online.feature_set import to_model_frame
from src.online.models import champion_threshold, load_scorer
from src.online.schema import ModelVersion

__all__ = ["FeatureSetMismatch", "ScoreResult", "load_scorer_for", "check_feature_set",
           "active_threshold", "build_feature_row", "validate_manual_values", "score_row", "save"]

class FeatureSetMismatch(RuntimeError):
    """Model được train trên bộ feature khác bộ feature đang có trong dữ liệu."""

@dataclass
class ScoreResult:
    pd_value: float
    threshold: float
    flagged: bool
    model_label: str
    model_version_id: int
    filled_from_median: list[str]

@lru_cache(maxsize=4)
def load_scorer_for(model_name: str, version: int, artifact_path: str | None):
    """Nạp model một lần cho mỗi bản (khoá = tên + version + đường dẫn), dùng chung mọi request."""
    return load_scorer(ModelVersion(model_name=model_name, version=version,
                                    artifact_path=artifact_path))

def check_feature_set(model_info: dict) -> None:
    current = catalog.current_feature_set_version()
    declared = model_info.get("feature_set_version")
    if declared and declared != current:
        raise FeatureSetMismatch(
            f"Model đang dùng train trên bộ feature `{declared}`, còn dữ liệu hiện tại là "
            f"`{current}`. Train lại model hoặc kích hoạt bản khớp trước khi chấm điểm.")

def active_threshold(model_info: dict) -> float:
    """Ngưỡng của bản model đang dùng; không có thì rơi về ngưỡng champion."""
    if model_info.get("threshold"):
        return float(model_info["threshold"])
    return champion_threshold()

def build_feature_row(user_values: dict) -> tuple[pd.DataFrame, list[str]]:
    """(DataFrame một dòng đủ cột đúng thứ tự, danh sách feature đã điền trung vị)."""
    features = catalog.selected_features()
    medians = catalog.feature_medians()
    row, filled = {}, []
    for name in features:
        if user_values.get(name) is not None:
            row[name] = user_values[name]
        else:
            row[name] = medians.get(name)
            filled.append(name)
    df = pd.DataFrame([row], columns=list(features))
    return to_model_frame(df, features, catalog.categorical_features()), filled

def validate_manual_values(values: dict[str, float]) -> list[str]:
    fields = {f["feature"]: f for f in catalog.form_features()}
    errors = []
    for name, v in values.items():
        field = fields.get(name)
        if field is None:
            errors.append(f"{name}: không phải feature trên form")
        elif not math.isfinite(v):
            errors.append(f"{name}: giá trị phải là số hữu hạn")
        elif field["categorical"]:
            if v not in {o["code"] for o in field["options"]}:
                errors.append(f"{name}: mã {v:g} không có trong danh sách lựa chọn")
        elif not field["min"] <= v <= field["max"]:
            errors.append(f"{name}: {v:g} nằm ngoài khoảng [{field['min']:g}, {field['max']:g}]")
    return errors

def score_row(X: pd.DataFrame, model_info: dict, filled: list[str]) -> ScoreResult:
    check_feature_set(model_info)
    scorer = load_scorer_for(model_info["model_name"], model_info["version"],
                             model_info["artifact_path"])
    proba = float(scorer.predict_proba(X)[:, 1][0])
    threshold = active_threshold(model_info)
    return ScoreResult(
        pd_value=proba, threshold=threshold, flagged=proba >= threshold,
        model_label=f"{model_info['model_name']} v{model_info['version']} ({model_info['role']})",
        model_version_id=model_info["id"], filled_from_median=list(filled))

def save(session: Session, result: ScoreResult, X: pd.DataFrame, sk_id_curr: int | None,
         input_type: str) -> int:
    """Lưu lần chấm vào DB, trả id bản ghi. Giá trị numpy -> Python để cột JSON ghi được."""
    features = {c: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for c, v in X.iloc[0].items()}
    p = repo.save_prediction(
        session, proba=result.pd_value, threshold=result.threshold, sk_id_curr=sk_id_curr,
        model_version_id=result.model_version_id, input_type=input_type, features=features)
    return p.id
