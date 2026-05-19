"""
Pipeline completo de treinamento para o Tech Challenge Fase 4.

Versao dinamica (Roteiro de Testes): Suporta multiples feature_modes e target_modes.
Normalizacao via StandardScaler separados (Features e Target).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from dotenv import load_dotenv

load_dotenv()

from src.data_loader import add_features, load_csv, load_yfinance
from src.model import StockLSTM


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
    grad_clip: float | None = 1.0
    output_dir: str = os.getenv("MODEL_DIR", "models/lstm_petr4")
    seed: int = 42
    target_mode: str = "log_returns"
    feature_mode: str = "single"
    feature_scaler_type: str = "standard"
    target_scaler_type: str = "standard"
    device: str = "auto"
    parent_run_id: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_windowed_sequences(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: list[str],
    target_col: str,
    train_end_row: int,
    feature_scaler_type: str = "standard",
    target_scaler_type: str = "standard"
):
    X_values = df[feature_cols].values.astype(np.float32)
    y_values = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    closes = df["Close"].values.astype(np.float32)
    dates = df.index.to_numpy()

    def get_scaler(scaler_type: str):
        if scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    feature_scaler = get_scaler(feature_scaler_type)
    feature_scaler.fit(X_values[:train_end_row])
    scaled_X = feature_scaler.transform(X_values).astype(np.float32)
    
    target_scaler = get_scaler(target_scaler_type)
    target_scaler.fit(y_values[:train_end_row])
    scaled_y = target_scaler.transform(y_values).astype(np.float32)

    X, y, last_closes, target_closes, target_dates, target_rows = [], [], [], [], [], []

    for i in range(window_size, len(df)):
        X.append(scaled_X[i - window_size : i])
        y.append(scaled_y[i, 0])
        last_closes.append(closes[i - 1])
        target_closes.append(closes[i])
        target_dates.append(dates[i])
        target_rows.append(i)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
        np.asarray(last_closes, dtype=np.float32),
        np.asarray(target_dates),
        np.asarray(target_rows),
        feature_scaler,
        target_scaler
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_numpy(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
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
    true_dir = np.sign(np.asarray(y_true) - np.asarray(last_close))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(last_close))
    return float(np.mean(true_dir == pred_dir) * 100)


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


def plot_performance(
    output_dir: Path,
    df_feat: pd.DataFrame,
    train_end: int,
    val_end: int,
    dates_test,
    y_true,
    y_pred,
    metrics: dict,
    symbol: str,
    feature_mode: str,
    target_mode: str
) -> None:
    plt.figure(figsize=(14, 6))
    
    plt.plot(
        df_feat.iloc[:train_end].index, 
        df_feat.iloc[:train_end]["Close"], 
        color="#1F2933", label="Treino", linewidth=1.4
    )
    plt.plot(
        df_feat.iloc[train_end:val_end].index, 
        df_feat.iloc[train_end:val_end]["Close"], 
        color="#8A94A6", label="Validacao", linewidth=1.4
    )
    
    plt.plot(pd.to_datetime(dates_test), y_true, color="#1455D9", label="Teste real", linewidth=1.4)
    plt.plot(pd.to_datetime(dates_test), y_pred, color="#D62F2F", label="Previsao LSTM", linewidth=1.4)
    
    plt.title(f"Performance do modelo LSTM [{feature_mode} | {target_mode}] - {symbol}")
    plt.xlabel("Data")
    plt.ylabel("Preco de fechamento")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    
    text = (
        f"MAE: {metrics['lstm_test']['mae']:.4f}\n"
        f"RMSE: {metrics['lstm_test']['rmse']:.4f}\n"
        f"MAPE: {metrics['lstm_test']['mape_pct']:.2f}%"
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
    if "://" not in mlflow_uri:
        mlflow_path = Path(mlflow_uri)
        if not mlflow_path.is_absolute():
            mlflow_path = Path(__file__).resolve().parent.parent / mlflow_path
        mlflow_uri = mlflow_path.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment("stock_lstm_hypersearch")
    
    with mlflow.start_run(run_name=f"{cfg.symbol}_{cfg.feature_mode}_{cfg.target_mode}", nested=True) as run:
        if cfg.parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", cfg.parent_run_id)
            mlflow.set_tag("derived_from", cfg.parent_run_id)
        mlflow.log_params(asdict(cfg))

        df_raw = load_csv(csv_path) if csv_path else load_yfinance(cfg.symbol, cfg.start_date, cfg.end_date)
        data_source = str(csv_path) if csv_path else "yfinance"
        mlflow.log_param("data_source", data_source)

        df_feat = add_features(df_raw)
        
        if cfg.target_mode == "log_returns":
            target_col = "Log_Return"
        elif cfg.target_mode == "raw_close":
            target_col = "Close"
        elif cfg.target_mode == "returns":
            target_col = "Return"
        else:
            raise ValueError(f"Target mode desconhecido: {cfg.target_mode}")
            
        if cfg.feature_mode == "single":
            feature_cols = [target_col]
        elif cfg.feature_mode == "ohlcv":
            feature_cols = ["Open", "High", "Low", "Close", "Volume"]
        elif cfg.feature_mode == "ohlcv_returns":
            feature_cols = ["Open", "High", "Low", "Close", "Volume", "Log_Return"]
        elif cfg.feature_mode == "technical_features":
            feature_cols = [
                "Log_Return", "SMA_7", "SMA_21", "Volatility_21", "Momentum_5", "Range_Pct", "Volume_Z",
                "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Width", "ATR_14",
                "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5",
                "Rolling_Return_5", "Rolling_Return_20", "Day_Of_Week", "Log_Volume"
            ]
        else:
            raise ValueError(f"Feature mode desconhecido: {cfg.feature_mode}")

        mlflow.log_param("feature_cols", ",".join(feature_cols))

        n_rows = len(df_feat)
        train_end = int(n_rows * cfg.train_ratio)
        val_end = int(n_rows * (cfg.train_ratio + cfg.val_ratio))

        X_all, y_all, last_close_all, dates_all, rows_all, feature_scaler, target_scaler = create_windowed_sequences(
            df_feat,
            cfg.window_size,
            feature_cols,
            target_col,
            train_end,
            feature_scaler_type=cfg.feature_scaler_type,
            target_scaler_type=cfg.target_scaler_type
        )

        train_mask = rows_all < train_end
        val_mask = (rows_all >= train_end) & (rows_all < val_end)
        test_mask = rows_all >= val_end

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        last_close_train = last_close_all[train_mask]
        last_close_val = last_close_all[val_mask]
        last_close_test = last_close_all[test_mask]

        dates_test = dates_all[test_mask]

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("Split gerou conjunto vazio. Aumente o historico ou reduza window_size.")

        print(f"Dataset Dinamico: {cfg.feature_mode} | target: {cfg.target_mode}")
        print(f"- fonte: {data_source}")
        print(f"- registros: {len(df_feat)}")
        print("Janelas")
        print(f"- X_train: {X_train.shape}")
        print(f"- X_val:   {X_val.shape}")
        print(f"- X_test:  {X_test.shape}")
        
        mlflow.log_metrics(
            {
                "rows_feat": len(df_feat),
                "samples_train": len(X_train),
                "samples_val": len(X_val),
                "samples_test": len(X_test),
            }
        )

        if cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.device)
            
        model = StockLSTM(
            input_size=X_train.shape[2],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        
        train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

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
                loss = criterion(model(xb), yb)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_losses.append(criterion(model(xb.to(device)), yb.to(device)).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if epoch == 1 or epoch % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:03d} | train={train_loss:.8f} | val={val_loss:.8f} | lr={lr:.6g}")

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
        
        # Avaliacao
        pred_train_scaled = predict_numpy(model, X_train, device)
        pred_val_scaled = predict_numpy(model, X_val, device)
        pred_test_scaled = predict_numpy(model, X_test, device)

        y_train_raw = target_scaler.inverse_transform(y_train).reshape(-1)
        pred_train_raw = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).reshape(-1)
        
        y_val_raw = target_scaler.inverse_transform(y_val).reshape(-1)
        pred_val_raw = target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).reshape(-1)
        
        y_test_raw = target_scaler.inverse_transform(y_test).reshape(-1)
        pred_test_raw = target_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).reshape(-1)

        if cfg.target_mode == "log_returns":
            true_train_close = last_close_train * np.exp(y_train_raw)
            pred_train_close = last_close_train * np.exp(pred_train_raw)

            true_val_close = last_close_val * np.exp(y_val_raw)
            pred_val_close = last_close_val * np.exp(pred_val_raw)

            true_test_close = last_close_test * np.exp(y_test_raw)
            pred_test_close = last_close_test * np.exp(pred_test_raw)
        elif cfg.target_mode == "returns":
            true_train_close = last_close_train * (1.0 + y_train_raw)
            pred_train_close = last_close_train * (1.0 + pred_train_raw)

            true_val_close = last_close_val * (1.0 + y_val_raw)
            pred_val_close = last_close_val * (1.0 + pred_val_raw)

            true_test_close = last_close_test * (1.0 + y_test_raw)
            pred_test_close = last_close_test * (1.0 + pred_test_raw)
        else:
            true_train_close = y_train_raw
            pred_train_close = pred_train_raw

            true_val_close = y_val_raw
            pred_val_close = pred_val_raw

            true_test_close = y_test_raw
            pred_test_close = pred_test_raw

        baseline_test_close = last_close_test

        metrics_train = regression_metrics(true_train_close, pred_train_close)
        metrics_val = regression_metrics(true_val_close, pred_val_close)
        metrics_test = regression_metrics(true_test_close, pred_test_close)
        metrics_baseline = regression_metrics(true_test_close, baseline_test_close)

        lstm_direction = directional_accuracy(true_test_close, pred_test_close, last_close_test)

        metrics = {
            "lstm_train": metrics_train,
            "lstm_val": metrics_val,
            "lstm_test": metrics_test,
            "baseline_test": metrics_baseline,
            "directional_accuracy_test_lstm_pct": lstm_direction,
            "relative_gain_vs_baseline_pct": {
                k: (metrics_baseline[k] - metrics_test[k]) / metrics_baseline[k] * 100
                for k in ["mae", "rmse", "mape_pct"]
            },
        }

        print()
        print("Metricas no teste")
        print(f"- LSTM MAE:  {metrics_test['mae']:.4f} | Baseline MAE:  {metrics_baseline['mae']:.4f}")
        print(f"- LSTM RMSE: {metrics_test['rmse']:.4f} | Baseline RMSE: {metrics_baseline['rmse']:.4f}")
        print(f"- LSTM MAPE: {metrics_test['mape_pct']:.2f}% | Baseline MAPE: {metrics_baseline['mape_pct']:.2f}%")
        print(f"- LSTM Direcional: {lstm_direction:.2f}%")
        
        mlflow.log_metrics(
            {
                "val_lstm_mae": metrics_val["mae"],
                "val_lstm_rmse": metrics_val["rmse"],
                "val_lstm_mape_pct": metrics_val["mape_pct"],
                "test_lstm_mae": metrics_test["mae"],
                "test_lstm_rmse": metrics_test["rmse"],
                "test_lstm_mape_pct": metrics_test["mape_pct"],
                "test_baseline_mae": metrics_baseline["mae"],
                "test_baseline_rmse": metrics_baseline["rmse"],
                "test_baseline_mape_pct": metrics_baseline["mape_pct"],
                "directional_accuracy_pct": lstm_direction,
                "gain_mae_pct": metrics["relative_gain_vs_baseline_pct"]["mae"],
                "gain_rmse_pct": metrics["relative_gain_vs_baseline_pct"]["rmse"],
                "gain_mape_pct": metrics["relative_gain_vs_baseline_pct"]["mape_pct"],
            }
        )

        # MLOps: Champion/Challenger Promotion Evaluation
        is_better = True
        
        # 1. Impede promoção automática de modelos multivariados/experimentais para não quebrar a API univariada
        if cfg.feature_mode != "single":
            print(f"[Promocao Rejeitada] O novo modelo possui feature_mode='{cfg.feature_mode}'. Apenas modelos univariados (single) sao elegiveis para promocao automatica na producao.")
            is_better = False
        
        champion_mape = None
        champion_metadata_path = output_dir / "metadata.json"
        champion_metrics_path = output_dir / "metrics.json"

        if is_better and champion_metadata_path.exists() and champion_metrics_path.exists():
            try:
                champ_metrics = json.loads(champion_metrics_path.read_text(encoding="utf-8"))
                # Prioriza o MAPE de validação para comparação de promoção
                champion_mape = (
                    champ_metrics.get("lstm_val", {}).get("mape_pct") or
                    champ_metrics.get("val_lstm_mape_pct") or
                    champ_metrics.get("lstm_test", {}).get("mape_pct") or
                    champ_metrics.get("test_lstm_mape_pct")
                )
                if champion_mape is not None:
                    val_mape = metrics_val["mape_pct"]
                    print(f"Modelo atual na producao (Champion) possui Validation MAPE: {champion_mape:.2f}%")
                    if val_mape < champion_mape:
                        print(f"[Promocao Aprovada] O novo modelo superou o atual no conjunto de validacao ({val_mape:.2f}% < {champion_mape:.2f}%).")
                    else:
                        print(f"[Promocao Rejeitada] O novo modelo nao superou o atual no conjunto de validacao ({val_mape:.2f}% >= {champion_mape:.2f}%).")
                        is_better = False
            except Exception as e:
                print(f"Falha ao avaliar Champion atual: {e}. Sobrescrevendo...")

        if not is_better:
            save_dir = Path("models/.temp_challenger")
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Salvando artefatos temporarios do Challenger em {save_dir} (nao serao promovidos para {output_dir})")
        else:
            save_dir = output_dir

        # Exportando pesos PyTorch (Legado/Compatibilidade)
        torch.save(model.state_dict(), save_dir / "model.pt")

        # Exportando pesos de forma segura usando Safetensors (imune a RCEs baseados em pickle)
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), save_dir / "model.safetensors")
            print("Pesos do modelo exportados com sucesso em formato seguro (model.safetensors).")
        except Exception as e:
            print(f"Aviso: Nao foi possivel salvar os pesos em formato safetensors: {e}")

        # MLOps: Compilacao para formato estatico ONNX (Production Ready)
        dummy_input = torch.randn(1, cfg.window_size, X_train.shape[2]).to(device)
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(save_dir / "model.onnx"),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        preprocess = {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "window_size": cfg.window_size,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "feature_mode": cfg.feature_mode,
            "target_mode": cfg.target_mode
        }
        joblib.dump(preprocess, save_dir / "preprocessor.joblib")

        metadata = {
            "run_id": run.info.run_id,
            "symbol": cfg.symbol,
            "data_source": data_source,
            "window_size": cfg.window_size,
            "prediction_horizon": "next_trading_day",
            "target_mode": cfg.target_mode,
            "feature_mode": cfg.feature_mode,
            "feature_scaler_type": cfg.feature_scaler_type,
            "target_scaler_type": cfg.target_scaler_type,
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape,
        }
        
        write_json(save_dir / "metadata.json", metadata)
        write_json(save_dir / "metrics.json", metrics)
        write_json(save_dir / "history.json", history)

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(dates_test),
                "last_close": last_close_test,
                "actual_close": true_test_close,
                "predicted_close_lstm": pred_test_close,
                "baseline_last_close": baseline_test_close,
                "absolute_error_lstm": np.abs(true_test_close - pred_test_close),
                "absolute_error_baseline": np.abs(true_test_close - baseline_test_close),
            }
        )
        predictions.to_csv(save_dir / "test_predictions.csv", index=False)
        
        plot_performance(
            save_dir, df_feat, train_end, val_end, dates_test, 
            true_test_close, pred_test_close, metrics, cfg.symbol, 
            cfg.feature_mode, cfg.target_mode
        )
        
        mlflow.log_artifacts(str(save_dir))
        
        if not is_better:
            try:
                shutil.rmtree(save_dir)
            except Exception as e:
                pass

        print()
        print(f"Artefatos de execucao registrados com sucesso.")
        print("MLflow run:", run.info.run_id)
        
    return {"metrics": metrics, "history": history, "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina uma LSTM multivariada/univariada configuravel.")
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
    parser.add_argument("--target-mode", type=str, default="log_returns", choices=["log_returns", "raw_close", "returns"])
    parser.add_argument("--feature-mode", type=str, default="single", choices=["single", "ohlcv", "ohlcv_returns", "technical_features"])
    parser.add_argument("--feature-scaler-type", type=str, default="standard", choices=["standard", "minmax", "robust"])
    parser.add_argument("--target-scaler-type", type=str, default="standard", choices=["standard", "minmax", "robust"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
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
        target_mode=args.target_mode,
        feature_mode=args.feature_mode,
        feature_scaler_type=args.feature_scaler_type,
        target_scaler_type=args.target_scaler_type,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    run_training_pipeline(cfg, args.csv)


if __name__ == "__main__":
    main()
