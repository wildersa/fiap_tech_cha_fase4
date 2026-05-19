import pytest
from fastapi.testclient import TestClient
import numpy as np
import joblib
from unittest.mock import patch, MagicMock
from src.api import app, MODEL_DIR
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

client = TestClient(app)

@pytest.fixture
def mock_preprocessor():
    scaler = StandardScaler()
    scaler.fit(np.array([[0.1], [0.2], [0.3]]))
    return {
        "window_size": 10,
        "feature_scaler": scaler,
        "target_scaler": scaler,
        "target_mode": "log_returns",
        "feature_mode": "single"
    }

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_telemetry():
    response = client.get("/telemetry")
    assert response.status_code == 200
    assert "uptime_seconds" in response.json()

def test_model_card_empty():
    # Test model-card when no files exist
    with patch("src.api.MODEL_DIR", Path("/non/existent")):
        response = client.get("/model-card")
        assert response.status_code == 200
        assert response.json()["evaluation_details"]["artifacts_available"]["model_onnx"] is False

def test_model_card_types():
    with patch("src.api.MODEL_DIR", Path("/non/existent/single")), patch("src.api.MODEL_DIR_MULTI", Path("/non/existent/multi")):
        response_single = client.get("/model-card?type=single")
        assert response_single.status_code == 200
        assert response_single.json()["model_overview"]["model_name"] == "StockLSTM (Univariado)"

        response_multi = client.get("/model-card?type=multi")
        assert response_multi.status_code == 200
        assert response_multi.json()["model_overview"]["model_name"] == "StockLSTM (Multivariado)"

        response_default = client.get("/model-card")
        assert response_default.status_code == 200
        assert response_default.json()["model_overview"]["model_name"] == "StockLSTM (Univariado)"

def test_model_image_not_found():
    with patch("src.api.MODEL_DIR", Path("/non/existent/single")), patch("src.api.MODEL_DIR_MULTI", Path("/non/existent/multi")):
        response_single = client.get("/model-image?type=single")
        assert response_single.status_code == 404

        response_multi = client.get("/model-image?type=multi")
        assert response_multi.status_code == 404

@patch("src.api.load_preprocessor")
@patch("src.api.load_predictor")
def test_predict_success(mock_load_pred, mock_load_prep, mock_preprocessor):
    mock_load_prep.return_value = mock_preprocessor

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    # The error was likely here, session.run returns a list of outputs
    # and each output is a numpy array.
    # src/api.py:169: predicted_scaled = float(ort_outs[0][0][0])
    mock_session.run.return_value = [np.array([[0.1]], dtype=np.float32)]

    mock_load_pred.return_value = mock_session

    # Needs window_size + 1 = 11 closes for log_returns
    payload = {"symbol": "PETR4.SA", "closes": [10.0 + float(i) for i in range(11)]}
    response = client.post("/predict", json=payload)

    if response.status_code != 200:
        print(response.json())

    assert response.status_code == 200
    data = response.json()
    assert "predicted_close" in data
    assert data["symbol"] == "PETR4.SA"

def test_predict_invalid_input():
    # Empty closes
    response = client.post("/predict", json={"symbol": "PETR4.SA", "closes": []})
    assert response.status_code == 400

    # Negative closes
    response = client.post("/predict", json={"symbol": "PETR4.SA", "closes": [-1.0, 10.0]})
    assert response.status_code == 400

@patch("src.api.run_training_pipeline")
def test_train_endpoint(mock_run_train):
    mock_run_train.return_value = {
        "metrics": {"mae": 0.1},
        "output_dir": "some/dir"
    }

    payload = {
        "symbol": "PETR4.SA",
        "max_epochs": 1,
        "batch_size": 4
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@patch("src.api.run_training_pipeline")
def test_train_endpoint_custom_params(mock_run_train):
    mock_run_train.return_value = {
        "metrics": {"mae": 0.1},
        "output_dir": "some/dir"
    }

    payload = {
        "symbol": "PETR4.SA",
        "feature_mode": "custom",
        "selected_features": ["Log_Return", "RSI_14"],
        "feature_preset": "custom",
        "max_epochs": 1,
        "batch_size": 4
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@patch("src.api.MlflowClient")
def test_runs_endpoint(mock_mlflow_client_class):
    mock_client = MagicMock()
    mock_mlflow_client_class.return_value = mock_client

    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    mock_client.get_experiment_by_name.return_value = mock_exp

    mock_run = MagicMock()
    mock_run.info.run_id = "run123"
    mock_run.info.status = "FINISHED"
    mock_run.info.start_time = 123456789
    mock_run.info.end_time = 123456790
    mock_run.data.metrics = {"mae": 0.1}
    mock_run.data.params = {}
    mock_run.data.tags = {}

    mock_client.search_runs.return_value = [mock_run]

    response = client.get("/runs")
    assert response.status_code == 200
    assert len(response.json()["runs"]) == 1
    assert response.json()["runs"][0]["run_id"] == "run123"

def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard"


@patch("src.api.MlflowClient")
def test_delete_run_endpoint(mock_mlflow_client_class):
    mock_client = MagicMock()
    mock_mlflow_client_class.return_value = mock_client
    
    response = client.delete("/runs/run123")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_client.delete_run.assert_called_once_with("run123")


def test_train_endpoint_invalid_features():
    payload = {
        "symbol": "PETR4.SA",
        "feature_mode": "custom",
        "selected_features": ["invalid_feat", "on"],
        "max_epochs": 1,
        "batch_size": 4
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 500
    assert "Features invalidas/desconhecidas no registro" in response.json()["detail"]


@patch("src.api.load_preprocessor")
@patch("src.api.load_predictor")
def test_predict_ohlcv_success(mock_load_pred, mock_load_prep, mock_preprocessor):
    mock_load_prep.return_value = mock_preprocessor

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[0.1]], dtype=np.float32)]
    mock_load_pred.return_value = mock_session

    # Cria 15 linhas de OHLCV de teste
    rows = []
    for i in range(15):
        rows.append({
            "date": f"2024-01-{i+2:02d}",
            "open": 30.0 + float(i),
            "high": 31.0 + float(i),
            "low": 29.0 + float(i),
            "close": 30.5 + float(i),
            "volume": 1000000.0
        })
    payload = {"symbol": "PETR4.SA", "rows": rows}
    response = client.post("/predict/ohlcv", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_close" in data
    assert data["symbol"] == "PETR4.SA"


def test_predict_ohlcv_invalid_input():
    # Envio vazio
    response = client.post("/predict/ohlcv", json={"symbol": "PETR4.SA", "rows": []})
    assert response.status_code == 400
