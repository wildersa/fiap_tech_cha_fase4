import pytest
import pandas as pd
import numpy as np
import torch
import mlflow
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.train import (
    TrainConfig,
    create_windowed_sequences,
    make_loader,
    regression_metrics,
    directional_accuracy,
    run_training_pipeline,
    predict_numpy,
    NpEncoder,
    write_json,
    resolve_feature_columns
)

def test_create_windowed_sequences(synthetic_df):
    window_size = 10
    feature_cols = ["Close"]
    target_col = "Close"
    train_end_row = 70

    X, y, last_closes, target_dates, target_rows, feature_scaler, target_scaler = create_windowed_sequences(
        synthetic_df, window_size, feature_cols, target_col, train_end_row
    )

    expected_samples = len(synthetic_df) - window_size
    assert X.shape == (expected_samples, window_size, 1)
    assert y.shape == (expected_samples, 1)
    assert len(last_closes) == expected_samples
    assert len(target_dates) == expected_samples
    assert len(target_rows) == expected_samples

def test_make_loader():
    X = np.random.randn(100, 10, 1).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    batch_size = 16

    loader = make_loader(X, y, batch_size, shuffle=True)

    assert len(loader.dataset) == 100
    batch_X, batch_y = next(iter(loader))
    assert batch_X.shape[0] == batch_size
    assert batch_y.shape[0] == batch_size

def test_regression_metrics():
    y_true = [10.0, 11.0, 12.0]
    y_pred = [10.5, 10.5, 12.5]
    metrics = regression_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape_pct" in metrics
    assert metrics["mae"] == pytest.approx(0.5)

def test_directional_accuracy():
    y_true = [11.0, 9.0, 12.0]
    y_pred = [12.0, 8.0, 11.0] # Directions: Up, Down, Up (but predicted Down for last)
    last_close = [10.0, 10.0, 10.0]

    acc = directional_accuracy(y_true, y_pred, last_close)
    # y_true - last: [1, -1, 2] -> signs [1, -1, 1]
    # y_pred - last: [2, -2, 1] -> signs [1, -1, 1]
    # All match!
    assert acc == 100.0

    y_pred_wrong = [9.0, 11.0, 9.0] # Directions: Down, Up, Down -> signs [-1, 1, -1]
    acc_wrong = directional_accuracy(y_true, y_pred_wrong, last_close)
    assert acc_wrong == 0.0

@patch("src.train.load_yfinance")
def test_run_training_pipeline_minimal(mock_load_yf, synthetic_df, temp_model_dir):
    mock_load_yf.return_value = synthetic_df

    cfg = TrainConfig(
        symbol="TEST",
        window_size=5,
        max_epochs=1,
        batch_size=4,
        output_dir=str(temp_model_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        hidden_size=8,
        num_layers=1,
        device="cpu"
    )

    # Ensure no other run is active
    mlflow.end_run()
    results = run_training_pipeline(cfg)

    assert "metrics" in results
    assert "history" in results
    assert Path(temp_model_dir).exists()
    assert (Path(temp_model_dir) / "model.onnx").exists()
    assert (Path(temp_model_dir) / "preprocessor.joblib").exists()
    assert (Path(temp_model_dir) / "metadata.json").exists()

def test_predict_numpy():
    from src.model import StockLSTM
    model = StockLSTM(1, 8, 1, 0.0)
    X = np.random.randn(10, 5, 1).astype(np.float32)
    preds = predict_numpy(model, X, torch.device("cpu"))
    assert preds.shape == (10,)

def test_json_safe():
    import json
    data = {
        "np_int": np.int64(1),
        "np_float": np.float64(1.5),
        "np_array": np.array([1, 2, 3]),
        "list": [1, 2],
        "nested": {"a": 1}
    }
    encoded_str = json.dumps(data, cls=NpEncoder)
    loaded = json.loads(encoded_str)
    assert loaded["np_int"] == 1
    assert loaded["np_float"] == 1.5
    assert loaded["np_array"] == [1, 2, 3]

def test_write_json(tmp_path):
    path = tmp_path / "test.json"
    data = {"a": 1}
    write_json(path, data)
    assert path.exists()

def test_train_config_default():
    cfg = TrainConfig()
    assert cfg.symbol == "PETR4.SA"

def test_run_training_pipeline_csv(tmp_path, synthetic_df, temp_model_dir):
    csv_path = tmp_path / "data.csv"
    synthetic_df.to_csv(csv_path)

    cfg = TrainConfig(
        window_size=5,
        max_epochs=1,
        batch_size=4,
        output_dir=str(temp_model_dir),
        device="cpu"
    )

    mlflow.end_run()
    results = run_training_pipeline(cfg, csv_path=str(csv_path))
    assert results["metrics"]["lstm_test"]["mae"] >= 0

def test_run_training_pipeline_invalid_modes():
    cfg = TrainConfig(target_mode="invalid")
    mlflow.end_run()
    with pytest.raises(ValueError, match="Target mode desconhecido"):
        run_training_pipeline(cfg)

    cfg = TrainConfig(feature_mode="invalid")
    mlflow.end_run()
    with pytest.raises(ValueError, match="Feature mode desconhecido"):
        run_training_pipeline(cfg)


def test_create_windowed_sequences_flexible_scaling(synthetic_df):
    window_size = 10
    feature_cols = ["Close"]
    target_col = "Close"
    train_end_row = 70

    # Test MinMax
    X_minmax, y_minmax, _, _, _, _, _ = create_windowed_sequences(
        synthetic_df, window_size, feature_cols, target_col, train_end_row,
        feature_scaler_type="minmax", target_scaler_type="minmax"
    )
    # Check bounds on the first training window: MinMax scale must be between [0, 1]
    assert np.all(X_minmax[0] >= -1e-6) and np.all(X_minmax[0] <= 1.000001)

    # Test Robust
    X_robust, y_robust, _, _, _, _, _ = create_windowed_sequences(
        synthetic_df, window_size, feature_cols, target_col, train_end_row,
        feature_scaler_type="robust", target_scaler_type="robust"
    )
    assert X_robust.shape == X_minmax.shape


@patch("src.train.load_yfinance")
def test_run_training_pipeline_custom_features(mock_load_yf, synthetic_df, temp_model_dir):
    mock_load_yf.return_value = synthetic_df

    cfg = TrainConfig(
        symbol="TEST",
        window_size=5,
        max_epochs=1,
        batch_size=4,
        output_dir=str(temp_model_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        hidden_size=8,
        num_layers=1,
        device="cpu",
        feature_mode="custom",
        selected_features=["Log_Return", "RSI_14", "MACD"]
    )

    mlflow.end_run()
    results = run_training_pipeline(cfg)
    assert "metrics" in results
    assert results["metrics"]["lstm_test"]["mae"] >= 0


@patch("src.train.load_yfinance")
def test_run_training_pipeline_feature_preset(mock_load_yf, synthetic_df, temp_model_dir):
    mock_load_yf.return_value = synthetic_df

    cfg = TrainConfig(
        symbol="TEST",
        window_size=5,
        max_epochs=1,
        batch_size=4,
        output_dir=str(temp_model_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        hidden_size=8,
        num_layers=1,
        device="cpu",
        feature_mode="technical_features",
        feature_preset="returns_basic"
    )

    mlflow.end_run()
    results = run_training_pipeline(cfg)
    assert "metrics" in results


def test_resolve_feature_columns_single():
    cfg = TrainConfig(feature_mode="single")
    cols = resolve_feature_columns(cfg, "Log_Return")
    assert cols == ["Log_Return"]


def test_resolve_feature_columns_technical_preset():
    cfg = TrainConfig(feature_mode="technical_features", feature_preset="returns_basic")
    cols = resolve_feature_columns(cfg, "Log_Return")
    assert cols == ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5"]


def test_resolve_feature_columns_custom_empty():
    cfg = TrainConfig(feature_mode="custom", selected_features=[])
    with pytest.raises(ValueError, match="O modo 'custom' exige que 'selected_features' nao esteja vazio"):
        resolve_feature_columns(cfg, "Log_Return")


def test_resolve_feature_columns_invalid_feature():
    cfg = TrainConfig(feature_mode="custom", selected_features=["InvalidFeatureName"])
    with pytest.raises(ValueError, match="Features invalidas/desconhecidas no registro"):
        resolve_feature_columns(cfg, "Log_Return")


@patch("shutil.rmtree")
@patch("src.train.load_yfinance")
def test_metadata_and_preprocessor_preservation(mock_load_yf, mock_rmtree, synthetic_df):
    import json
    import joblib
    import shutil
    
    temp_challenger = Path("models/.temp_challenger")
    if temp_challenger.exists():
        try:
            shutil.rmtree(temp_challenger)
        except Exception:
            pass

    mock_load_yf.return_value = synthetic_df
    cfg = TrainConfig(
        symbol="TEST",
        window_size=5,
        max_epochs=1,
        batch_size=4,
        output_dir="models/some_dir",
        train_ratio=0.6,
        val_ratio=0.2,
        hidden_size=8,
        num_layers=1,
        device="cpu",
        feature_mode="technical_features",
        feature_preset="returns_basic"
    )
    mlflow.end_run()
    try:
        results = run_training_pipeline(cfg)
        
        # Check metadata.json in temp_challenger
        metadata_path = temp_challenger / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        
        expected_features = ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5"]
        assert metadata["selected_features"] == expected_features
        assert metadata["feature_cols"] == expected_features
        assert metadata["feature_count"] == len(expected_features)

        # Check preprocessor.joblib preserves exact order of feature_cols
        preproc_path = temp_challenger / "preprocessor.joblib"
        assert preproc_path.exists()
        preprocessor = joblib.load(preproc_path)
        assert preprocessor["feature_cols"] == expected_features
        assert preprocessor["selected_features"] == expected_features
    finally:
        if temp_challenger.exists():
            try:
                shutil.rmtree(temp_challenger)
            except Exception:
                pass


@patch("src.train.load_yfinance")
def test_non_single_model_no_promotion(mock_load_yf, synthetic_df, temp_model_dir):
    import shutil
    mock_load_yf.return_value = synthetic_df
    
    final_output_dir = Path(temp_model_dir) / "final_prod"
    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
        
    cfg = TrainConfig(
        symbol="TEST",
        window_size=5,
        max_epochs=1,
        batch_size=4,
        output_dir=str(final_output_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        hidden_size=8,
        num_layers=1,
        device="cpu",
        feature_mode="ohlcv" # Not single, so should be rejected for auto-promotion
    )
    mlflow.end_run()
    results = run_training_pipeline(cfg)
    
    # Because it is not "single", it is not promoted and the final output dir is not created/saved
    model_file = final_output_dir / "model.onnx"
    assert not model_file.exists()
