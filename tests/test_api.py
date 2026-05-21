import pytest
from fastapi.testclient import TestClient
import numpy as np
import joblib
from unittest.mock import patch, MagicMock
from src.api import app, MODEL_DIR
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
from src.api import _run_selection_record, _select_best_run, refresh_models_for_runtime

client = TestClient(app)


def _mock_finished_run(run_id, metrics, params=None):
    run = MagicMock()
    run.info.run_id = run_id
    run.info.status = "FINISHED"
    run.data.metrics = metrics
    run.data.params = params or {"window_size": "20", "target_mode": "log_returns", "feature_mode": "single"}
    return run

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


def test_best_run_selection_prioritizes_positive_baseline_gain():
    record = _run_selection_record(_mock_finished_run(
        "formula_gain",
        {"test_lstm_mape_pct": 1.20, "test_baseline_mape_pct": 1.50, "directional_accuracy_pct": 54.0},
    ))

    assert record["baseline_gain_pct"] == pytest.approx(20.0)


def test_best_run_selection_uses_tie_margin_and_tiebreakers():
    max_gain_worse_direction = _run_selection_record(_mock_finished_run(
        "max_gain",
        {"test_lstm_mape_pct": 1.30, "baseline_gain_pct": 2.0, "directional_accuracy_pct": 54.0},
        {"window_size": "60", "target_mode": "raw_close", "feature_mode": "single"},
    ))
    tie_higher_direction = _run_selection_record(_mock_finished_run(
        "tie_direction",
        {"test_lstm_mape_pct": 1.40, "baseline_gain_pct": 1.9, "directional_accuracy_pct": 57.0},
        {"window_size": "60", "target_mode": "raw_close", "feature_mode": "single"},
    ))
    tie_same_direction_lower_mape = _run_selection_record(_mock_finished_run(
        "tie_mape",
        {"test_lstm_mape_pct": 1.20, "baseline_gain_pct": 1.85, "directional_accuracy_pct": 57.0},
        {"window_size": "40", "target_mode": "raw_close", "feature_mode": "single"},
    ))
    tie_same_direction_same_mape_smaller_payload = _run_selection_record(_mock_finished_run(
        "tie_payload",
        {"test_lstm_mape_pct": 1.20, "baseline_gain_pct": 1.80, "directional_accuracy_pct": 57.0},
        {"window_size": "20", "target_mode": "raw_close", "feature_mode": "single"},
    ))

    best = _select_best_run([
        max_gain_worse_direction,
        tie_higher_direction,
        tie_same_direction_lower_mape,
        tie_same_direction_same_mape_smaller_payload,
    ])

    assert best["run"].info.run_id == "tie_payload"


def test_best_run_selection_respects_gain_band_before_tiebreakers():
    outside_gain_band_better_direction = _run_selection_record(_mock_finished_run(
        "outside_band",
        {"test_lstm_mape_pct": 1.10, "baseline_gain_pct": 1.69, "directional_accuracy_pct": 99.0},
        {"window_size": "20", "target_mode": "raw_close", "feature_mode": "single"},
    ))
    max_gain_worse_direction = _run_selection_record(_mock_finished_run(
        "max_gain",
        {"test_lstm_mape_pct": 1.30, "baseline_gain_pct": 2.0, "directional_accuracy_pct": 50.0},
        {"window_size": "20", "target_mode": "raw_close", "feature_mode": "single"},
    ))

    best = _select_best_run([outside_gain_band_better_direction, max_gain_worse_direction])

    assert best["run"].info.run_id == "max_gain"


def test_best_run_selection_treats_exact_margin_as_technical_tie():
    max_gain_worse_direction = _run_selection_record(_mock_finished_run(
        "max_gain",
        {"test_lstm_mape_pct": 1.30, "baseline_gain_pct": 1.8, "directional_accuracy_pct": 50.0},
        {"window_size": "20", "target_mode": "raw_close", "feature_mode": "single"},
    ))
    exact_margin_better_direction = _run_selection_record(_mock_finished_run(
        "exact_margin",
        {"test_lstm_mape_pct": 1.20, "baseline_gain_pct": 1.5, "directional_accuracy_pct": 60.0},
        {"window_size": "20", "target_mode": "raw_close", "feature_mode": "single"},
    ))

    best = _select_best_run([max_gain_worse_direction, exact_margin_better_direction])

    assert best["run"].info.run_id == "exact_margin"


def test_best_run_selection_rejects_logged_gain_that_does_not_beat_baseline_mape():
    contradictory_gain = _run_selection_record(_mock_finished_run(
        "bad_gain",
        {"test_lstm_mape_pct": 1.30, "test_baseline_mape_pct": 1.20, "baseline_gain_pct": 3.0, "directional_accuracy_pct": 99.0},
    ))

    assert _select_best_run([contradictory_gain]) is None


def test_best_run_selection_accepts_baseline_gain_alias():
    alias_gain = _run_selection_record(_mock_finished_run(
        "alias_gain",
        {"test_lstm_mape_pct": 1.20, "baseline_gain_pct": 3.0, "directional_accuracy_pct": 54.0},
    ))
    logged_gain = _run_selection_record(_mock_finished_run(
        "logged_gain",
        {"test_lstm_mape_pct": 1.18, "gain_mape_pct": 1.5, "directional_accuracy_pct": 58.0},
    ))

    best = _select_best_run([logged_gain, alias_gain])

    assert best["run"].info.run_id == "alias_gain"


def test_custom_feature_runs_are_auto_promotion_eligible():
    record = _run_selection_record(_mock_finished_run(
        "custom_gain",
        {"test_lstm_mape_pct": 1.18, "test_baseline_mape_pct": 1.21, "directional_accuracy_pct": 60.0},
        {"window_size": "20", "target_mode": "log_returns", "feature_mode": "custom"},
    ))

    best = _select_best_run([record])

    assert best["run"].info.run_id == "custom_gain"


def test_best_run_selection_returns_none_without_positive_baseline_gain():
    zero_gain = _run_selection_record(_mock_finished_run(
        "zero_gain",
        {"test_lstm_mape_pct": 1.07, "baseline_gain_pct": 0.0, "directional_accuracy_pct": 53.2},
    ))
    negative_gain = _run_selection_record(_mock_finished_run(
        "negative_gain",
        {"test_lstm_mape_pct": 1.21, "gain_mape_pct": -0.2, "directional_accuracy_pct": 50.8},
    ))

    best = _select_best_run([negative_gain, zero_gain])

    assert best is None


@patch("src.api.sync_best_model_from_mlflow")
def test_refresh_models_syncs_only_in_dev_train_mode(mock_sync):
    with patch("src.api.ENABLE_TRAINING_API", True):
        refresh_models_for_runtime(force_best=True)
    mock_sync.assert_called_once_with(force_best=True)

    mock_sync.reset_mock()
    with patch("src.api.ENABLE_TRAINING_API", False):
        refresh_models_for_runtime(force_best=True)
    mock_sync.assert_not_called()

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


@patch("src.api.refresh_models_for_runtime")
def test_model_champion_returns_single_and_multi_champions(mock_refresh, tmp_path):
    single_dir = tmp_path / "single"
    multi_dir = tmp_path / "multi"
    single_dir.mkdir()
    multi_dir.mkdir()

    (single_dir / "metrics.json").write_text(
        """
        {
          "lstm_test": {"mape_pct": 1.0},
          "baseline_test": {"mape_pct": 1.2},
          "relative_gain_vs_baseline_pct": {"mape_pct": 16.6667},
          "directional_accuracy_test_lstm_pct": 55.0
        }
        """,
        encoding="utf-8",
    )
    (single_dir / "metadata.json").write_text(
        """
        {
          "run_id": "single-run",
          "feature_mode": "single",
          "target_mode": "log_returns",
          "window_size": 20
        }
        """,
        encoding="utf-8",
    )
    (multi_dir / "metrics.json").write_text(
        """
        {
          "lstm_test": {"mape_pct": 0.9},
          "baseline_test": {"mape_pct": 1.2},
          "relative_gain_vs_baseline_pct": {"mape_pct": 25.0},
          "directional_accuracy_test_lstm_pct": 57.0
        }
        """,
        encoding="utf-8",
    )
    (multi_dir / "metadata.json").write_text(
        """
        {
          "run_id": "multi-run",
          "feature_mode": "custom",
          "target_mode": "log_returns",
          "window_size": 20
        }
        """,
        encoding="utf-8",
    )

    with patch("src.api.MODEL_DIR", single_dir), patch("src.api.MODEL_DIR_MULTI", multi_dir):
        response = client.get("/model-champion")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model_type"] == "multi"
    assert payload["run_id"] == "multi-run"
    assert payload["champions"]["single"]["has_champion"] is True
    assert payload["champions"]["single"]["run_id"] == "single-run"
    assert payload["champions"]["multi"]["has_champion"] is True
    assert payload["champions"]["multi"]["run_id"] == "multi-run"

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


@patch("src.api.load_preprocessor_multi")
@patch("src.api.load_predictor_multi")
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
