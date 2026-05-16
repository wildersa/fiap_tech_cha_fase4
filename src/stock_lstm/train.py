from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .data_loader import load_csv, load_yfinance
from .model import build_model
from .preprocessing import build_feature_frame, WindowPreprocessor
from .utils import (
    set_seed,
    regression_metrics,
    directional_accuracy,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina LSTM para previsão de fechamento.")
    parser.add_argument("--csv", type=str, default=None, help="Caminho do CSV local.")
    parser.add_argument("--symbol", type=str, default="PETR4.SA")
    parser.add_argument("--start-date", type=str, default="2018-01-01")
    parser.add_argument("--output-dir", type=str, default="models/lstm_petr4")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def predict_numpy(model, X, device):
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=256, shuffle=False)

    with torch.no_grad():
        for xb in loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            preds.append(pred)

    return np.vstack(preds).reshape(-1)


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        symbol=args.symbol,
        start_date=args.start_date,
        window_size=args.window_size,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        df_raw = load_csv(args.csv)
        data_source = str(args.csv)
    else:
        df_raw = load_yfinance(cfg.symbol, cfg.start_date)
        data_source = "yfinance"

    df_feat = build_feature_frame(df_raw)

    n_rows = len(df_feat)
    train_end_row = int(n_rows * cfg.train_ratio)
    val_end_row = int(n_rows * (cfg.train_ratio + cfg.val_ratio))

    preprocessor = WindowPreprocessor(window_size=cfg.window_size)
    preprocessor.fit(df_feat, train_end_row=train_end_row)

    X_all, y_all, anchors_all, last_close_all, dates_all, rows_all = preprocessor.create_windows(df_feat)

    train_mask = rows_all < train_end_row
    val_mask = (rows_all >= train_end_row) & (rows_all < val_end_row)
    test_mask = rows_all >= val_end_row

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    anchors_test = anchors_all[test_mask]
    last_close_test = last_close_all[test_mask]
    dates_test = dates_all[test_mask]

    print("Dataset")
    print(f"- fonte: {data_source}")
    print(f"- período features: {df_feat.index.min().date()} até {df_feat.index.max().date()}")
    print(f"- linhas após features: {len(df_feat)}")
    print()
    print("Janelas")
    print(f"- X_train: {X_train.shape}")
    print(f"- X_val:   {X_val.shape}")
    print(f"- X_test:  {X_test.shape}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Split gerou conjunto vazio. Aumente o histórico ou reduza WINDOW_SIZE.")

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(
        input_size=X_train.shape[2],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            print(f"Early stopping na época {epoch}.")
            break

    model.load_state_dict(best_state)

    pred_ratio_test = predict_numpy(model, X_test, device)

    y_true_close = y_test.reshape(-1) * anchors_test
    y_pred_close = pred_ratio_test * anchors_test
    baseline_close = last_close_test

    lstm_metrics = regression_metrics(y_true_close, y_pred_close)
    baseline_metrics = regression_metrics(y_true_close, baseline_close)
    lstm_directional_accuracy = directional_accuracy(y_true_close, y_pred_close, last_close_test)

    metrics = {
        "lstm": lstm_metrics,
        "baseline_last_close": baseline_metrics,
        "lstm_directional_accuracy_pct": lstm_directional_accuracy,
        "relative_gain_vs_baseline_pct": {
            "mae": (baseline_metrics["mae"] - lstm_metrics["mae"]) / baseline_metrics["mae"] * 100,
            "rmse": (baseline_metrics["rmse"] - lstm_metrics["rmse"]) / baseline_metrics["rmse"] * 100,
            "mape": (baseline_metrics["mape_pct"] - lstm_metrics["mape_pct"]) / baseline_metrics["mape_pct"] * 100,
        },
    }

    print()
    print("Métricas no teste")
    print("LSTM:", metrics["lstm"])
    print("Baseline:", metrics["baseline_last_close"])
    print("Ganho relativo:", metrics["relative_gain_vs_baseline_pct"])

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": int(X_train.shape[2]),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "config": cfg.to_dict(),
    }

    torch.save(checkpoint, output_dir / "model.pt")
    preprocessor.save(str(output_dir / "preprocessor.pkl"))

    metadata = {
        "symbol": cfg.symbol,
        "data_source": data_source,
        "window_size": cfg.window_size,
        "prediction_horizon": "next_trading_day",
        "feature_cols": list(preprocessor.feature_cols),
        "price_cols": list(preprocessor.price_cols),
        "stationary_cols": list(preprocessor.stationary_cols),
        "feature_period_start": df_feat.index.min(),
        "feature_period_end": df_feat.index.max(),
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "target_description": "Close do próximo pregão como ratio em relação ao anchor_price da janela",
    }

    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "history.json", history)

    predictions = pd.DataFrame({
        "target_date": pd.to_datetime(dates_test),
        "actual_close": y_true_close,
        "predicted_close_lstm": y_pred_close,
        "baseline_last_close": baseline_close,
        "absolute_error_lstm": np.abs(y_true_close - y_pred_close),
        "absolute_error_baseline": np.abs(y_true_close - baseline_close),
    })
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print()
    print("Artefatos salvos em:", output_dir)


if __name__ == "__main__":
    main()
