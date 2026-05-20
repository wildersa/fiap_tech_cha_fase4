import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.data_loader import normalize_columns, ensure_datetime_index, add_features, load_yfinance, load_csv

def test_normalize_columns():
    df = pd.DataFrame({
        "close": [1, 2],
        "adj_close": [0.9, 1.8],
        "Open": [0.8, 1.6],
        "volume": [100, 200]
    })
    normalized = normalize_columns(df)
    assert "Close" in normalized.columns
    assert "Adj Close" in normalized.columns
    assert "Open" in normalized.columns
    assert "Volume" in normalized.columns

def test_normalize_columns_multiindex():
    columns = pd.MultiIndex.from_tuples([("Close", "PETR4.SA"), ("Volume", "PETR4.SA")])
    df = pd.DataFrame([[1, 100], [2, 200]], columns=columns)
    normalized = normalize_columns(df)
    assert list(normalized.columns) == ["Close", "Volume"]

def test_ensure_datetime_index():
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Close": [10, 11]
    })
    indexed = ensure_datetime_index(df)
    assert isinstance(indexed.index, pd.DatetimeIndex)
    assert indexed.index.name == "Date"

def test_add_features():
    dates = pd.date_range(start="2023-01-01", periods=30)
    df = pd.DataFrame({
        "Open": np.random.rand(30) + 10,
        "High": np.random.rand(30) + 11,
        "Low": np.random.rand(30) + 9,
        "Close": np.random.rand(30) + 10.5,
        "Volume": np.random.randint(100, 1000, 30)
    }, index=dates)

    featured = add_features(df)

    expected_cols = [
        "SMA_7", "SMA_21", "Return", "Log_Return", "Volatility_21", "Momentum_5", "Range_Pct", "Volume_Z",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Width", "ATR_14",
        "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5",
        "Rolling_Return_5", "Rolling_Return_20", "Day_Of_Week", "Log_Volume"
    ]
    for col in expected_cols:
        assert col in featured.columns

    # Sem required_features, mantem o comportamento legacy: calcula tudo e
    # remove NaNs globalmente para comparabilidade com runs antigas.
    assert not featured.isnull().values.any()

def test_add_features_only_calculates_required_features():
    dates = pd.date_range(start="2023-01-01", periods=30)
    df = pd.DataFrame({
        "Open": np.random.rand(30) + 10,
        "High": np.random.rand(30) + 11,
        "Low": np.random.rand(30) + 9,
        "Close": np.random.rand(30) + 10.5,
        "Volume": np.random.randint(100, 1000, 30)
    }, index=dates)
    df.iloc[0, df.columns.get_loc("Volume")] = 0

    featured = add_features(df, required_features=["Log_Return", "Close"])

    assert "Log_Return" in featured.columns
    assert "RSI_14" not in featured.columns
    assert "MACD" not in featured.columns
    assert len(featured) == len(df) - 1

def test_add_features_missing_columns():
    df = pd.DataFrame({"Close": [1, 2]})
    with pytest.raises(ValueError, match="Colunas obrigatorias ausentes"):
        add_features(df)

@patch("yfinance.download")
def test_load_yfinance(mock_download):
    mock_df = pd.DataFrame({
        "Open": [10], "High": [11], "Low": [9], "Close": [10.5], "Volume": [1000]
    }, index=pd.DatetimeIndex(["2023-01-01"], name="Date"))
    mock_download.return_value = mock_df

    loaded = load_yfinance("PETR4.SA", "2023-01-01")
    assert not loaded.empty
    assert "Close" in loaded.columns
    mock_download.assert_called_once()

@patch("yfinance.download")
def test_load_yfinance_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="Nenhum dado retornado"):
        load_yfinance("INVALID", "2023-01-01")

def test_load_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "close": [10, 11],
        "open": [9.5, 10.5],
        "high": [10.5, 11.5],
        "low": [9, 10],
        "volume": [1000, 1100]
    })
    df.to_csv(csv_file, index=False)

    loaded = load_csv(csv_file)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert "Close" in loaded.columns
    assert "Open" in loaded.columns
