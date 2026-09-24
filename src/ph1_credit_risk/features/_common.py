from __future__ import annotations

import pandas as pd

DEFAULT_NUM_AGGS = ["mean", "sum", "max", "min"]

def _clean(name: str) -> str:
    return name.upper().replace(" ", "_").replace("-", "_").replace("/", "_")

def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df.columns = [_clean(f"{prefix}{col}_{agg}") for col, agg in df.columns]
    return df

def numeric_aggs(
    df: pd.DataFrame,
    key: str,
    cols: list[str],
    prefix: str,
    aggs: list[str] | None = None,
) -> pd.DataFrame:
    grouped = df.groupby(key, sort=True)[cols].agg(aggs or DEFAULT_NUM_AGGS)
    return flatten_columns(grouped, prefix)

def numeric_columns(df: pd.DataFrame, exclude: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def count_categories(
    series: pd.Series,
    key: pd.Series,
    values: list[str],
    prefix: str,
    with_ratio: bool = False,
) -> pd.DataFrame:

    dummies = pd.get_dummies(series.astype("str")).reindex(columns=values, fill_value=False)
    dummies.columns = [_clean(f"{prefix}{v}_COUNT") for v in values]
    counts = dummies.astype("int32").groupby(key.to_numpy(), sort=True).sum()
    if with_ratio:
        total = counts.sum(axis=1)
        for v in values:
            counts[_clean(f"{prefix}{v}_RATIO")] = counts[_clean(f"{prefix}{v}_COUNT")] / total
    counts.index.name = key.name
    return counts

def attach_customer(df: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    base = df.drop(columns=["SK_ID_CURR"], errors="ignore")
    return base.merge(id_map[["SK_ID_PREV", "SK_ID_CURR"]], on="SK_ID_PREV", how="inner")


def write_parquet(df: pd.DataFrame, path, label: str, t0: float) -> pd.DataFrame:
    import time

    from src.ph1_credit_risk.data.loader import downcast

    out = downcast(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"[{label}] {out.shape} -> {path} ({time.time() - t0:.1f}s)")
    return out
