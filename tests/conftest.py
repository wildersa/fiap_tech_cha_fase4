import pytest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from unittest.mock import MagicMock
import torch
from pathlib import Path

@pytest.fixture
def synthetic_df():
    """Gera um DataFrame sintético com colunas OHLCV."""
    dates = pd.date_range(start="2020-01-01", periods=100)
    data = {
        "Open": np.linspace(10, 20, 100) + np.random.randn(100),
        "High": np.linspace(11, 21, 100) + np.random.randn(100),
        "Low": np.linspace(9, 19, 100) + np.random.randn(100),
        "Close": np.linspace(10.5, 20.5, 100) + np.random.randn(100),
        "Volume": np.random.randint(1000, 5000, 100),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df

@pytest.fixture
def temp_mlflow_dir():
    """Cria um diretório temporário para o MLflow."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture(autouse=True)
def mock_mlflow_env(temp_mlflow_dir, monkeypatch):
    """Configura o MLflow para usar o diretório temporário em todos os testes."""
    uri = Path(temp_mlflow_dir).as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    import mlflow
    mlflow.set_tracking_uri(uri)

@pytest.fixture
def temp_model_dir():
    """Cria um diretório temporário para salvar modelos durante os testes."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)

@pytest.fixture
def mock_onnx_session():
    """Mock para onnxruntime.InferenceSession."""
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    # Retorna um valor fixo para a predição (log-retorno escalonado)
    mock_session.run.return_value = [np.array([[[0.1]]], dtype=np.float32)]
    return mock_session
