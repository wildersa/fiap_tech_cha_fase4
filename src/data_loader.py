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


def add_features(df: pd.DataFrame, required_features: list[str] | None = None) -> pd.DataFrame:
    """
    Adiciona features multivariadas para a LSTM.
    Inclui medias moveis, log returns, volatilidade, momentum, normalizacao de volume,
    RSI, MACD, Bollinger Bands Width, ATR, lags de retorno, retornos acumulados moveis,
    dia da semana e log do volume.
    Se required_features for fornecido, apenas calcula o que for necessário.
    """
    data = df.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    
    req_set = set(required_features) if required_features is not None else None

    def needs(col: str) -> bool:
        return req_set is None or col in req_set

    if missing:
        # Se required_features for nulo, tenta calcular tudo, então quebra se faltar base
        if req_set is None:
            raise ValueError(f"Colunas obrigatorias ausentes: {missing}")
        else:
            # Se mode for single e as colunas OHLCV não estiverem todas presentes, não quebra
            # a menos que uma das features requeridas dependa dessas colunas
            pass
    else:
        for col in required:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Sempre precisamos de Close numérico se formos fazer qualquer conta
    if "Close" in data.columns:
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    # Features de preco/tendencia
    if needs("SMA_7") and "Close" in data.columns: data["SMA_7"] = data["Close"].rolling(7).mean()
    if needs("SMA_21") and "Close" in data.columns: data["SMA_21"] = data["Close"].rolling(21).mean()

    # Log_Return é muito usado (target, lags, volatilidade), verificamos dependências diretas e indiretas
    needs_log_return = needs("Log_Return") or needs("Volatility_21") or needs("Log_Return_Lag1") or needs("Log_Return_Lag2") or needs("Log_Return_Lag3") or needs("Log_Return_Lag5")
    
    if needs("Return") and "Close" in data.columns: data["Return"] = data["Close"].pct_change()
    if needs_log_return and "Close" in data.columns: data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    
    if needs("Volatility_21") and "Log_Return" in data.columns: data["Volatility_21"] = data["Log_Return"].rolling(21).std()
    if needs("Momentum_5") and "Close" in data.columns: data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1.0
    if needs("Range_Pct") and "High" in data.columns and "Low" in data.columns and "Close" in data.columns: 
        data["Range_Pct"] = (data["High"] - data["Low"]) / data["Close"]
        
    if needs("Volume_Z") and "Volume" in data.columns:
        data["Volume_Z"] = (data["Volume"] - data["Volume"].rolling(21).mean()) / data["Volume"].rolling(21).std()

    # 1. RSI_14
    if needs("RSI_14") and "Close" in data.columns:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        data["RSI_14"] = 100 - (100 / (1 + rs))

    # 2. MACD
    if (needs("MACD") or needs("MACD_Signal") or needs("MACD_Hist")) and "Close" in data.columns:
        ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = ema_12 - ema_26
        if needs("MACD_Signal") or needs("MACD_Hist"):
            data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        if needs("MACD_Hist"):
            data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # 3. Bollinger Band Width
    if needs("BB_Width") and "Close" in data.columns:
        sma_20 = data["Close"].rolling(window=20).mean()
        std_20 = data["Close"].rolling(window=20).std()
        data["BB_Width"] = (2 * 2 * std_20) / (sma_20 + 1e-9)

    # 4. ATR_14
    if needs("ATR_14") and "High" in data.columns and "Low" in data.columns and "Close" in data.columns:
        high_low = data["High"] - data["Low"]
        high_close_prev = (data["High"] - data["Close"].shift(1)).abs()
        low_close_prev = (data["Low"] - data["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data["ATR_14"] = tr.ewm(com=13, adjust=False).mean()

    # 5. Lags de retorno
    if needs("Log_Return_Lag1") and "Log_Return" in data.columns: data["Log_Return_Lag1"] = data["Log_Return"].shift(1)
    if needs("Log_Return_Lag2") and "Log_Return" in data.columns: data["Log_Return_Lag2"] = data["Log_Return"].shift(2)
    if needs("Log_Return_Lag3") and "Log_Return" in data.columns: data["Log_Return_Lag3"] = data["Log_Return"].shift(3)
    if needs("Log_Return_Lag5") and "Log_Return" in data.columns: data["Log_Return_Lag5"] = data["Log_Return"].shift(5)

    # 6. Rolling returns
    if needs("Rolling_Return_5") and "Close" in data.columns: data["Rolling_Return_5"] = data["Close"] / data["Close"].shift(5) - 1.0
    if needs("Rolling_Return_20") and "Close" in data.columns: data["Rolling_Return_20"] = data["Close"] / data["Close"].shift(20) - 1.0

    # 7. Day of Week
    if needs("Day_Of_Week") and isinstance(data.index, pd.DatetimeIndex):
        data["Day_Of_Week"] = data.index.dayofweek.astype(float)

    # 8. Log Volume
    if needs("Log_Volume") and "Volume" in data.columns:
        data["Log_Volume"] = np.log(data["Volume"] + 1e-9)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Manter a verificação do Volume apenas se Volume for necessário ou for o padrão antigo?
    # Se "Volume" for obrigatório ou presente, podemos remover Volume = 0.
    if "Volume" in data.columns:
        data = data.loc[data["Volume"] > 0].copy()
        
    return data
