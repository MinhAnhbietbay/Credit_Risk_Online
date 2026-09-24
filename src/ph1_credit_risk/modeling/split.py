from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_SEED, TEST_SIZE

__all__ = ["holdout_split", "cv_portion", "make_holdout_split", "save_holdout_ids", "load_holdout_ids", "HOLDOUT_IDS_PATH"]

HOLDOUT_IDS_PATH = PROCESSED_DIR / "holdout_ids.json"

def holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Trả (index phần CV, index holdout) — stratified theo TARGET, seed cố định."""
    cv_idx, ho_idx = train_test_split(
        df.index.to_numpy(), test_size=TEST_SIZE, stratify=df["TARGET"], random_state=RANDOM_SEED
    )
    return np.sort(cv_idx), np.sort(ho_idx)

def cv_portion(df: pd.DataFrame) -> pd.DataFrame:
    """Phần 80% dùng cho mọi bước fit/so sánh."""
    return df.loc[holdout_split(df)[0]]

def make_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(phần CV 80%, holdout 20%) dạng DataFrame — cùng một phép chia với `holdout_split`."""
    cv_idx, ho_idx = holdout_split(df)
    return df.loc[cv_idx], df.loc[ho_idx]
def save_holdout_ids(holdout: pd.DataFrame, path: Path = HOLDOUT_IDS_PATH) -> Path:
    """Ghi SK_ID_CURR của holdout để mọi task sau chấm đúng một tập."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(int(i) for i in holdout["SK_ID_CURR"])))
    return path

def load_holdout_ids(path: Path = HOLDOUT_IDS_PATH) -> list[int]:
    return sorted(json.loads(Path(path).read_text()))
