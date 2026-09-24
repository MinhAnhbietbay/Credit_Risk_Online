from __future__ import annotations

from typing import Iterator

import pandas as pd

from src.ph1_credit_risk.data import schema

__all__ = ["load_table", "iter_chunks", "downcast", "table_columns"]


def table_columns(name: str) -> list[str]:
    """Danh sách cột của bảng, đọc mỗi dòng header."""
    return list(pd.read_csv(schema.table_path(name), nrows=0).columns)


def _read_kwargs(name: str, columns: list[str] | None, chunked: bool = False) -> dict:
    cols = columns if columns is not None else table_columns(name)
    dtypes = schema.dtype_for(name, cols)
    if chunked:
        dtypes = {c: ("str" if d == "category" else d) for c, d in dtypes.items()}
    kwargs: dict = {"dtype": dtypes}
    if columns is not None:
        kwargs["usecols"] = columns
    return kwargs


def load_table(
    name: str,
    columns: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(schema.table_path(name), nrows=nrows, **_read_kwargs(name, columns))
    if columns is not None:
        df = df[columns]
    return df


def iter_chunks(name: str, chunksize: int = 2_000_000) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(
        schema.table_path(name), chunksize=chunksize, **_read_kwargs(name, None, chunked=True)
    )
    for chunk in reader:
        yield chunk.reset_index(drop=True)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        kind = out[col].dtype.kind
        if kind == "f":
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif kind in "iu":

            out[col] = pd.to_numeric(out[col], downcast="integer")
    return out
