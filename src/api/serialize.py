from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

__all__ = ["records", "figure"]

def records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))

def figure(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())
