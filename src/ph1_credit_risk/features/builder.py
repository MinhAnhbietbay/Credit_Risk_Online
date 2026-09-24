from __future__ import annotations

import gc
import json
import time
from typing import Literal

import pandas as pd

from src.config import PROCESSED_DIR
from src.ph1_credit_risk.data import loader
from src.ph1_credit_risk.features.application import add_application_features
from src.ph1_credit_risk.features.bureau import BUREAU_AGG_PATH
from src.ph1_credit_risk.features.credit_card import CC_AGG_PATH
from src.ph1_credit_risk.features.installments import INST_AGG_PATH
from src.ph1_credit_risk.features.pos_cash import POS_AGG_PATH
from src.ph1_credit_risk.features.previous import PREV_AGG_PATH

__all__ = [
    "join_features", "encode_categoricals", "build_feature_table", "features_path",
    "AGG_PATHS", "LABEL_MAPS_PATH", "CATEGORICAL_COLUMNS_PATH", "UNKNOWN_CODE",
]

# Thứ tự join cố định -> thứ tự cột output ổn định giữa các lần chạy
AGG_PATHS = [BUREAU_AGG_PATH, PREV_AGG_PATH, POS_AGG_PATH, INST_AGG_PATH, CC_AGG_PATH]
LABEL_MAPS_PATH = PROCESSED_DIR / "label_maps.json"
CATEGORICAL_COLUMNS_PATH = PROCESSED_DIR / "categorical_columns.json"

UNKNOWN_CODE = -1

LabelMaps = dict[str, dict[str, int]]

def features_path(split: str):
    return PROCESSED_DIR / f"features_{split}.parquet"

def join_features(app: pd.DataFrame, aggs: list[pd.DataFrame]) -> pd.DataFrame:
    """Left join các bảng aggregate (index = SK_ID_CURR) vào bảng chính"""
    out = app
    seen = set(out.columns)
    for agg in aggs:
        dup = seen.intersection(agg.columns)
        if dup:
            raise ValueError(f"Cột trùng khi join: {sorted(dup)}")
        seen.update(agg.columns)
        out = out.merge(agg, how="left", left_on="SK_ID_CURR", right_index=True)
    return out


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object or df[c].dtype == "str"
    ]


def encode_categoricals(
    df: pd.DataFrame, label_maps: LabelMaps | None = None
) -> tuple[pd.DataFrame, LabelMaps]:
    
    out = df.copy()
    cols = _categorical_columns(out)
    if label_maps is None:
        label_maps = {
            c: {v: i for i, v in enumerate(sorted(out[c].dropna().astype("str").unique()))}
            for c in cols
        }
    elif set(cols) != set(label_maps):
        raise ValueError(
            f"Cột categorical không khớp bộ mã đã fit: thiếu {sorted(set(label_maps) - set(cols))}, "
            f"thừa {sorted(set(cols) - set(label_maps))}"
        )
    for c in cols:
        codes = out[c].astype("str").map(label_maps[c])
        out[c] = codes.fillna(UNKNOWN_CODE).astype("int16")
    return out, label_maps


def _load_application(split: str) -> pd.DataFrame:
    return loader.load_table(f"application_{split}")

def build_feature_table(split: Literal["train", "test"]) -> pd.DataFrame:
    t0 = time.time()
    app = add_application_features(_load_application(split))
    print(f"[builder:{split}] application: {app.shape} ({time.time() - t0:.1f}s)")

    out = app
    for path in AGG_PATHS:
        agg = pd.read_parquet(path)
        out = join_features(out, [agg])
        del agg
        gc.collect()
    print(f"[builder:{split}] joined: {out.shape} ({time.time() - t0:.1f}s)")

    if split == "train":
        out, label_maps = encode_categoricals(out)
        LABEL_MAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LABEL_MAPS_PATH.write_text(json.dumps(label_maps, indent=1, ensure_ascii=False))
        CATEGORICAL_COLUMNS_PATH.write_text(json.dumps(list(label_maps), indent=1))
    else:
        if not LABEL_MAPS_PATH.exists():
            raise FileNotFoundError(f"Thiếu {LABEL_MAPS_PATH}: chạy build_feature_table('train') trước")
        out, _ = encode_categoricals(out, json.loads(LABEL_MAPS_PATH.read_text()))

    out = loader.downcast(out)
    path = features_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"[builder:{split}] {out.shape} -> {path} ({time.time() - t0:.1f}s)")
    return out
