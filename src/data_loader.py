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
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
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
    df = ensure_datetime_index(normalize_columns(pd.read_csv(path)))
    # Capitalize the columns after normalize_columns just in case
    df.columns = [col.capitalize() for col in df.columns]
    return df


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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features multivariadas para a LSTM.
    Inclui medias moveis, log returns, volatilidade, momentum e normalizacao de volume.
    """
    data = df.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Features de preco/tendencia
    data["SMA_7"] = data["Close"].rolling(7).mean()
    data["SMA_21"] = data["Close"].rolling(21).mean()

    # Features estacionarias/relativas
    data["Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Volatility_21"] = data["Log_Return"].rolling(21).std()
    data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1.0
    data["Range_Pct"] = (data["High"] - data["Low"]) / data["Close"]
    data["Volume_Z"] = (
        (data["Volume"] - data["Volume"].rolling(21).mean())
        / data["Volume"].rolling(21).std()
    )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data["Volume"] > 0].copy()
    return data
