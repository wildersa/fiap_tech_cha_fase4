from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import normalize_columns, ensure_datetime_index


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

PRICE_COLS = ["Open", "High", "Low", "Close"]

STATIONARY_COLS = [
    "Volume_Log",
    "Volume_Change",
    "Return_1",
    "Return_5",
    "Volatility_5",
    "Volatility_10",
    "Range_Pct",
    "Candle_Body_Pct",
    "MA_Ratio_5",
    "MA_Ratio_10",
    "Momentum_5",
]

ALL_FEATURE_COLS = PRICE_COLS + STATIONARY_COLS


def prepare_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_datetime_index(df)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    df = df[REQUIRED_COLS].copy()

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[df["Volume"] > 0].copy()
    df = df.sort_index()
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_ohlcv_frame(df)

    close = df["Close"]

    df["Volume_Log"] = np.log1p(df["Volume"])
    df["Volume_Change"] = df["Volume_Log"].diff()

    df["Return_1"] = np.log(close / close.shift(1))
    df["Return_5"] = np.log(close / close.shift(5))

    df["Volatility_5"] = df["Return_1"].rolling(5).std()
    df["Volatility_10"] = df["Return_1"].rolling(10).std()

    df["Range_Pct"] = (df["High"] - df["Low"]) / close
    df["Candle_Body_Pct"] = (df["Close"] - df["Open"]) / df["Open"]

    df["MA_Ratio_5"] = close / close.rolling(5).mean() - 1
    df["MA_Ratio_10"] = close / close.rolling(10).mean() - 1

    df["Momentum_5"] = close / close.shift(5) - 1

    df = df[ALL_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    return df


@dataclass
class WindowPreprocessor:
    window_size: int = 30
    price_cols: Sequence[str] = field(default_factory=lambda: PRICE_COLS.copy())
    stationary_cols: Sequence[str] = field(default_factory=lambda: STATIONARY_COLS.copy())
    feature_cols: Sequence[str] = field(default_factory=lambda: ALL_FEATURE_COLS.copy())
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    def fit(self, df_feat: pd.DataFrame, train_end_row: int) -> "WindowPreprocessor":
        train_values = df_feat.iloc[:train_end_row][list(self.stationary_cols)].values
        self.scaler.fit(train_values)
        self.fitted = True
        return self

    def create_windows(self, df_feat: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("WindowPreprocessor precisa ser fitado antes de criar janelas.")

        data = df_feat[list(self.feature_cols)].values.astype(np.float32)
        closes = df_feat["Close"].values.astype(np.float32)
        dates = df_feat.index.to_numpy()

        price_idx = [list(self.feature_cols).index(c) for c in self.price_cols]
        stat_idx = [list(self.feature_cols).index(c) for c in self.stationary_cols]

        X, y = [], []
        anchors, last_closes, target_dates, target_rows = [], [], [], []

        for i in range(self.window_size, len(data)):
            # Janela termina em i-1. Alvo é i.
            window = data[i - self.window_size:i].copy()

            anchor_price = closes[i - self.window_size]
            last_close = closes[i - 1]
            target_close = closes[i]

            if anchor_price <= 0:
                continue

            window[:, price_idx] = window[:, price_idx] / anchor_price
            window[:, stat_idx] = self.scaler.transform(window[:, stat_idx])

            target_ratio = target_close / anchor_price

            X.append(window)
            y.append(target_ratio)
            anchors.append(anchor_price)
            last_closes.append(last_close)
            target_dates.append(dates[i])
            target_rows.append(i)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32).reshape(-1, 1),
            np.asarray(anchors, dtype=np.float32),
            np.asarray(last_closes, dtype=np.float32),
            np.asarray(target_dates),
            np.asarray(target_rows),
        )

    def build_latest_window(self, df_feat: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("WindowPreprocessor precisa estar fitado.")

        if len(df_feat) < self.window_size:
            raise ValueError(
                f"Dados insuficientes. Necessário ao menos {self.window_size} linhas após feature engineering."
            )

        data = df_feat[list(self.feature_cols)].values.astype(np.float32)
        closes = df_feat["Close"].values.astype(np.float32)

        window = data[-self.window_size:].copy()
        anchor_price = closes[-self.window_size]
        last_close = closes[-1]

        price_idx = [list(self.feature_cols).index(c) for c in self.price_cols]
        stat_idx = [list(self.feature_cols).index(c) for c in self.stationary_cols]

        window[:, price_idx] = window[:, price_idx] / anchor_price
        window[:, stat_idx] = self.scaler.transform(window[:, stat_idx])

        X = window.reshape(1, self.window_size, len(self.feature_cols)).astype(np.float32)
        return X, float(anchor_price), float(last_close)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "WindowPreprocessor":
        return joblib.load(path)
