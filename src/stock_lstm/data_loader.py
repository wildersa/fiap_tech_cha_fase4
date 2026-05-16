from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


EXPECTED_MAP = {
    "date": "Date",
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # yfinance pode retornar MultiIndex em algumas versões/configurações.
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        if any(col in {"Open", "High", "Low", "Close", "Adj Close", "Volume"} for col in level0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = ["_".join(str(x) for x in col if str(x) != "").strip() for col in df.columns]

    new_cols = {}
    for col in df.columns:
        key = str(col).strip().replace("_", " ").lower()
        new_cols[col] = EXPECTED_MAP.get(key, str(col).strip())

    df = df.rename(columns=new_cols)
    return df


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.loc[df.index.notna()].copy()
    df.index.name = "Date"
    df = df.sort_index()
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = ensure_datetime_index(df)
    return df


def load_yfinance(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol}.")

    df = normalize_columns(df)
    df = ensure_datetime_index(df)
    return df
