"""
Pipeline completo de treinamento para o Tech Challenge Fase 4.

O modelo usa apenas o preco de fechamento. Internamente, o pipeline transforma
os fechamentos em log-retornos, treina a LSTM para prever o proximo log-retorno
e converte a previsao de volta para preco:

    predicted_close = last_close * exp(predicted_log_return)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import os
from dotenv import load_dotenv

load_dotenv()

from data_loader import build_close_frame, build_return_frame, load_csv, load_yfinance
from model import StockLSTM


@dataclass
class TrainConfig:
    symbol: str = "PETR4.SA"
    start_date: str = "2018-01-01"
    end_date: str | None = None
    window_size: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 20
    output_dir: str = os.getenv("MODEL_DIR", "models/lstm_petr4")
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_preprocessor(return_frame: pd.DataFrame, train_end_row: int, window_size: int) -> dict:
    scaler = StandardScaler()
    scaler.fit(return_frame.iloc[:train_end_row][["Log_Return"]].values)
    return {
        "window_size": window_size,
        "input_columns": ["Log_Return"],
        "target": "next_log_return",
        "scaler": scaler,
    }


def create_windows(return_frame: pd.DataFrame, preprocessor: dict):
    window_size = int(preprocessor["window_size"])
    scaler = preprocessor["scaler"]
    returns = return_frame["Log_Return"].values.astype(np.float32)
    closes = return_frame["Close"].values.astype(np.float32)
    dates = return_frame.index.to_numpy()

    X, y, last_closes, target_closes, target_dates, target_rows = [], [], [], [], [], []
    for i in range(window_size, len(return_frame)):
        window_returns = returns[i - window_size:i].reshape(-1, 1)
        scaled_window = scaler.transform(window_returns)
        target_return = returns[i]

        X.append(scaled_window)
        y.append(target_return)
        last_closes.append(closes[i - 1])
        target_closes.append(closes[i])
        target_dates.append(dates[i])
        target_rows.append(i)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
        np.asarray(last_closes, dtype=np.float32),
        np.asarray(target_closes, dtype=np.float32),
        np.asarray(target_dates),
        np.asarray(target_rows),
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_numpy(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch.to(device)).cpu().numpy())
    return np.vstack(preds).reshape(-1)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def directional_accuracy(y_true, y_pred, last_close) -> float:
    return float(np.mean((np.asarray(y_true) > np.asarray(last_close)) == (np.asarray(y_pred) > np.asarray(last_close))) * 100)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def plot_performance(output_dir: Path, close_frame: pd.DataFrame, train_end_row: int, val_end_row: int, dates_test, y_true, y_pred, metrics: dict, symbol: str) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(close_frame.iloc[:train_end_row].index, close_frame.iloc[:train_end_row]["Close"], color="#1F2933", label="Treino", linewidth=1.4)
    plt.plot(close_frame.iloc[train_end_row:val_end_row].index, close_frame.iloc[train_end_row:val_end_row]["Close"], color="#8A94A6", label="Validacao", linewidth=1.4)
    plt.plot(pd.to_datetime(dates_test), y_true, color="#1455D9", label="Teste real", linewidth=1.4)
    plt.plot(pd.to_datetime(dates_test), y_pred, color="#D62F2F", label="Previsao LSTM", linewidth=1.4)
    plt.title(f"Performance do modelo LSTM - {symbol}")
    plt.xlabel("Data")
    plt.ylabel("Preco de fechamento")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    text = (
        f"MAE: {metrics['lstm']['mae']:.4f}\n"
        f"RMSE: {metrics['lstm']['rmse']:.4f}\n"
        f"MAPE: {metrics['lstm']['mape_pct']:.2f}%"
    )
    plt.gca().text(0.99, 0.03, text, transform=plt.gca().transAxes, ha="right", va="bottom", bbox={"facecolor": "white", "alpha": 0.9})
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance.png", dpi=180)
    plt.close()


def run_training_pipeline(cfg: TrainConfig, csv_path: str | None = None) -> dict:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if csv_path is None:
        csv_path = os.getenv("DATA_CSV_PATH") or None
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    if not mlflow_uri.startswith(("file://", "http://", "https://")):
        mlflow_path = Path(mlflow_uri)
        if not mlflow_path.is_absolute():
            mlflow_path = Path(__file__).resolve().parent.parent / mlflow_path
        mlflow_uri = mlflow_path.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    
    mlflow.set_experiment("stock_lstm_training")
    mlflow.start_run(run_name=f"{cfg.symbol}_log_returns")
    mlflow.log_params(asdict(cfg))
    mlflow.log_param("target_mode", "log_returns")

    df_raw = load_csv(csv_path) if csv_path else load_yfinance(cfg.symbol, cfg.start_date, cfg.end_date)
    data_source = str(csv_path) if csv_path else "yfinance"
    mlflow.log_param("data_source", data_source)
    close_frame = build_close_frame(df_raw)
    return_frame = build_return_frame(df_raw)

    n_rows = len(return_frame)
    train_end_row = int(n_rows * cfg.train_ratio)
    val_end_row = int(n_rows * (cfg.train_ratio + cfg.val_ratio))
    preprocessor = fit_preprocessor(return_frame, train_end_row, cfg.window_size)

    X_all, y_all, last_close_all, target_close_all, dates_all, rows_all = create_windows(return_frame, preprocessor)
    train_mask = rows_all < train_end_row
    val_mask = (rows_all >= train_end_row) & (rows_all < val_end_row)
    test_mask = rows_all >= val_end_row

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    last_close_test = last_close_all[test_mask]
    y_true_close = target_close_all[test_mask]
    dates_test = dates_all[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Split gerou conjunto vazio. Aumente o historico ou reduza window_size.")

    print("Dataset")
    print(f"- fonte: {data_source}")
    print(f"- periodo: {close_frame.index.min().date()} ate {close_frame.index.max().date()}")
    print(f"- fechamentos: {len(close_frame)}")
    print(f"- log-retornos: {len(return_frame)}")
    print("Janelas")
    print(f"- X_train: {X_train.shape}")
    print(f"- X_val:   {X_val.shape}")
    print(f"- X_test:  {X_test.shape}")
    mlflow.log_metrics(
        {
            "rows_close": len(close_frame),
            "rows_log_return": len(return_frame),
            "samples_train": len(X_train),
            "samples_val": len(X_val),
            "samples_test": len(X_test),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(
        input_size=X_train.shape[2],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Device: {device}")
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb.to(device)), yb.to(device)).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train={train_loss:.8f} | val={val_loss:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stopping na epoca {epoch}.")
                break

    model.load_state_dict(best_state)
    predicted_log_return = predict_numpy(model, X_test, device)
    y_pred_close = last_close_test * np.exp(predicted_log_return)
    baseline_close = last_close_test

    lstm_metrics = regression_metrics(y_true_close, y_pred_close)
    baseline_metrics = regression_metrics(y_true_close, baseline_close)
    metrics = {
        "lstm": lstm_metrics,
        "baseline_last_close": baseline_metrics,
        "lstm_directional_accuracy_pct": directional_accuracy(y_true_close, y_pred_close, last_close_test),
        "relative_gain_vs_baseline_pct": {
            "mae": (baseline_metrics["mae"] - lstm_metrics["mae"]) / baseline_metrics["mae"] * 100,
            "rmse": (baseline_metrics["rmse"] - lstm_metrics["rmse"]) / baseline_metrics["rmse"] * 100,
            "mape": (baseline_metrics["mape_pct"] - lstm_metrics["mape_pct"]) / baseline_metrics["mape_pct"] * 100,
        },
    }

    print()
    print("Metricas no teste")
    print(f"- LSTM MAE:  {lstm_metrics['mae']:.4f} | Baseline MAE:  {baseline_metrics['mae']:.4f}")
    print(f"- LSTM RMSE: {lstm_metrics['rmse']:.4f} | Baseline RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"- LSTM MAPE: {lstm_metrics['mape_pct']:.2f}% | Baseline MAPE: {baseline_metrics['mape_pct']:.2f}%")
    mlflow.log_metrics(
        {
            "lstm_mae": lstm_metrics["mae"],
            "lstm_rmse": lstm_metrics["rmse"],
            "lstm_mape_pct": lstm_metrics["mape_pct"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_mape_pct": baseline_metrics["mape_pct"],
            "directional_accuracy_pct": metrics["lstm_directional_accuracy_pct"],
            "gain_mae_pct": metrics["relative_gain_vs_baseline_pct"]["mae"],
            "gain_rmse_pct": metrics["relative_gain_vs_baseline_pct"]["rmse"],
            "gain_mape_pct": metrics["relative_gain_vs_baseline_pct"]["mape"],
        }
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": int(X_train.shape[2]),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    metadata = {
        "symbol": cfg.symbol,
        "data_source": data_source,
        "window_size": cfg.window_size,
        "prediction_horizon": "next_trading_day",
        "input_description": "Janela de log-retornos calculados a partir dos fechamentos anteriores",
        "target_description": "Proximo log-retorno; o preco previsto e last_close * exp(predicted_log_return)",
        "feature_period_start": close_frame.index.min(),
        "feature_period_end": close_frame.index.max(),
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "history.json", history)

    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(dates_test),
            "last_close": last_close_test,
            "actual_close": y_true_close,
            "predicted_close_lstm": y_pred_close,
            "predicted_log_return": predicted_log_return,
            "baseline_last_close": baseline_close,
            "absolute_error_lstm": np.abs(y_true_close - y_pred_close),
            "absolute_error_baseline": np.abs(y_true_close - baseline_close),
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    plot_performance(output_dir, close_frame, train_end_row, val_end_row, dates_test, y_true_close, y_pred_close, metrics, cfg.symbol)
    mlflow.log_artifacts(str(output_dir))

    print()
    print(f"Artefatos salvos em: {output_dir}")
    print("MLflow run:", mlflow.active_run().info.run_id)
    mlflow.end_run()
    return {"metrics": metrics, "history": history, "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina uma LSTM para prever fechamento de acoes.")
    parser.add_argument("--csv", type=str, default=None, help="Caminho opcional para CSV local.")
    parser.add_argument("--symbol", type=str, default="PETR4.SA")
    parser.add_argument("--start-date", type=str, default="2018-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/lstm_petr4")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    
    args = parse_args()
    cfg = TrainConfig(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        window_size=args.window_size,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    run_training_pipeline(cfg, args.csv)


if __name__ == "__main__":
    main()
