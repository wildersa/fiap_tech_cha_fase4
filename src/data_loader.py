"""
Carregamento e preparacao da serie de fechamento.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        if any(col in {"Close", "Adj Close"} for col in level0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = ["_".join(str(x) for x in col if str(x) != "").strip() for col in df.columns]

    expected = {
        "date": "Date",
        "datetime": "Date",
        "close": "Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
    }
    return df.rename(columns={col: expected.get(str(col).strip().replace("_", " ").lower(), str(col).strip()) for col in df.columns})


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[df.index.notna()].copy()
    df.index.name = "Date"
    return df.sort_index()


def load_csv(path: str | Path) -> pd.DataFrame:
    return ensure_datetime_index(normalize_columns(pd.read_csv(path)))


def load_yfinance(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
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
    return ensure_datetime_index(normalize_columns(df))


def build_close_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime_index(normalize_columns(df))
    if "Close" not in df.columns:
        raise ValueError("Coluna obrigatoria ausente: Close")

    close = pd.to_numeric(df["Close"], errors="coerce")
    close_frame = pd.DataFrame({"Close": close}).replace([np.inf, -np.inf], np.nan).dropna()
    close_frame = close_frame.loc[close_frame["Close"] > 0].sort_index()
    if len(close_frame) < 3:
        raise ValueError("Historico insuficiente de fechamentos.")
    return close_frame


def build_return_frame(df: pd.DataFrame) -> pd.DataFrame:
    close_frame = build_close_frame(df)
    close_frame["Log_Return"] = np.log(close_frame["Close"] / close_frame["Close"].shift(1))
    return close_frame.replace([np.inf, -np.inf], np.nan).dropna()
