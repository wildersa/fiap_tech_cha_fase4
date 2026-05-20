"""
Orquestrador do Pipeline de Treinamento e Seleção de Modelos.

Este módulo implementa o fluxo completo de MLOps para o StockLSTM:
1. Coleta de dados (Yahoo Finance ou CSV).
2. Engenharia de features e split temporal (Train/Val/Test).
3. Ciclo de treinamento com Early Stopping e monitoramento MLflow.
4. Avaliação contra Baseline (Naive Persistence).
5. Promoção Automática baseada no algoritmo Champion vs Challenger.
6. Exportação de artefatos em múltiplos formatos (PyTorch, Safetensors, ONNX).
"""

from __future__ import annotations

import os
import shutil
import copy
import json
import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import src.train
from src.model import StockLSTM
from src.train.config import TrainConfig, resolve_target_column, resolve_feature_columns
from src.train.champion_selection import (
    MIN_BASELINE_GAIN,
    build_selection_record,
    should_promote_candidate,
    is_auto_promotion_eligible,
    AUTO_PROMOTION_FEATURE_MODES,
)
from src.train.data_prep import set_seed, create_windowed_sequences, make_loader
from src.train.trainer import regression_metrics, directional_accuracy, predict_numpy
from src.train.artifacts import write_json, plot_performance
from dotenv import load_dotenv
load_dotenv()


def _empty_date(value: str | None) -> bool:
    """Verifica se uma string de data está vazia ou é nula."""
    return value is None or str(value).strip().lower() in {"", "none", "null", "nan"}


def _requested_end_date(value: str | None) -> str | None:
    """Normaliza a data final solicitada pelo usuário."""
    if _empty_date(value):
        return None
    return str(pd.Timestamp(value).date())


def _yfinance_end_date_exclusive(value: str | None) -> str | None:
    """Ajusta a data final para exclusividade exigida pela API do yfinance."""
    requested = _requested_end_date(value)
    if requested is None:
        return None
    return str((pd.Timestamp(requested) + pd.Timedelta(days=1)).date())


def _dataset_date(value) -> str:
    """Converte objeto timestamp para string YYYY-MM-DD."""
    return str(pd.Timestamp(value).date())


def _split_date(values, index: int) -> str:
    """Extrai data de um índice específico de uma série temporal."""
    return _dataset_date(values[index])


def run_training_pipeline(cfg: TrainConfig, csv_path: str | None = None) -> dict:
    """
    Executa o fluxo completo de treinamento.
    
    Args:
        cfg: Objeto de configuração com hiperparâmetros e parâmetros de dados.
        csv_path: Caminho opcional para arquivo de dados local.

    Returns:
        Dicionário com métricas finais e diretório de saída dos artefatos.
    """
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csv_path = os.getenv("DATA_CSV_PATH") or None

    # Configura rastreamento do MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if "://" not in mlflow_uri:
        mlflow_path = Path(mlflow_uri)
        if not mlflow_path.is_absolute():
            mlflow_path = Path(__file__).resolve().parent.parent.parent / mlflow_path
        mlflow_uri = mlflow_path.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment("stock_lstm_hypersearch")
    
    with mlflow.start_run(run_name=f"{cfg.symbol}_{cfg.feature_mode}_{cfg.target_mode}", nested=True) as run:
        if cfg.parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", cfg.parent_run_id)
            mlflow.set_tag("derived_from", cfg.parent_run_id)

        mlflow.log_params(asdict(cfg))
        end_date_requested = _requested_end_date(cfg.end_date)
        yfinance_end_date_exclusive = _yfinance_end_date_exclusive(cfg.end_date)
        preprocessing_mode = "selected_feature_dropna"
        mlflow.log_params(
            {
                "preprocessing_mode": preprocessing_mode,
                "end_date_requested": end_date_requested,
                "yfinance_end_date_exclusive": yfinance_end_date_exclusive,
            }
        )

        # 1. Carregamento de dados
        df_raw = src.train.load_csv(csv_path) if csv_path else src.train.load_yfinance(cfg.symbol, cfg.start_date, yfinance_end_date_exclusive)
        data_source = str(csv_path) if csv_path else "yfinance"
        mlflow.log_param("data_source", data_source)
        dataset_start_real = _dataset_date(df_raw.index.min())
        dataset_end_real = _dataset_date(df_raw.index.max())
        mlflow.log_params(
            {
                "dataset_start_real": dataset_start_real,
                "dataset_end_real": dataset_end_real,
                "dataset_rows_raw": len(df_raw),
            }
        )

        # 2. Resolução de features e target
        target_col = resolve_target_column(cfg.target_mode)
        feature_cols = resolve_feature_columns(cfg, target_col)

        # Close é sempre necessário para reconstruir preço e baseline.
        required_cols = list(set(feature_cols + [target_col, "Close"]))
        df_feat = src.train.add_features(df_raw, required_cols)
        
        missing_required = [f for f in required_cols if f not in df_feat.columns]
        if missing_required:
            raise ValueError(f"Colunas obrigatorias ausentes no DataFrame gerado: {missing_required}")

        mlflow.log_param("dataset_rows_feat", len(df_feat))
        mlflow.log_param("feature_cols", ",".join(feature_cols))

        # 3. Engenharia de sequências e Janelamento (Lookback)
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

        # Split temporal (sem shuffle entre os conjuntos)
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
        dates_train = dates_all[train_mask]
        dates_val = dates_all[val_mask]

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("Split gerou conjunto vazio. Aumente o historico ou reduza window_size.")

        split_date_params = {
            "train_start_date": _split_date(dates_train, 0),
            "train_end_date": _split_date(dates_train, -1),
            "val_start_date": _split_date(dates_val, 0),
            "val_end_date": _split_date(dates_val, -1),
            "test_start_date": _split_date(dates_test, 0),
            "test_end_date": _split_date(dates_test, -1),
        }
        mlflow.log_params(split_date_params)

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

        # 4. Inicialização do Modelo e Otimizadores
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

        # 5. Ciclo de Épocas (Treinamento e Validação)
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

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    print(f"Early stopping na epoca {epoch}.")
                    break

        # Recupera o melhor estado encontrado (ponto de menor loss de validação)
        model.load_state_dict(best_state)
        
        # 6. Avaliação Final (Conjunto de Teste)
        pred_train_scaled = predict_numpy(model, X_train, device)
        pred_val_scaled = predict_numpy(model, X_val, device)
        pred_test_scaled = predict_numpy(model, X_test, device)

        y_train_raw = target_scaler.inverse_transform(y_train).reshape(-1)
        pred_train_raw = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).reshape(-1)
        
        y_val_raw = target_scaler.inverse_transform(y_val).reshape(-1)
        pred_val_raw = target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).reshape(-1)
        
        y_test_raw = target_scaler.inverse_transform(y_test).reshape(-1)
        pred_test_raw = target_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).reshape(-1)

        # Conversão de retornos para preços absolutos para cálculo de métricas de negócio
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

        # Baseline Naive (Previsão de hoje = preço de ontem)
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

        # 7. Avaliação Champion/Challenger (MLOps)
        champion_metadata_path = output_dir / "metadata.json"
        champion_metrics_path = output_dir / "metrics.json"
        champion_record = None
        candidate_record = build_selection_record(metrics, asdict(cfg))

        if champion_metadata_path.exists() and champion_metrics_path.exists():
            try:
                champion_metrics = json.loads(champion_metrics_path.read_text(encoding="utf-8"))
                champion_metadata = json.loads(champion_metadata_path.read_text(encoding="utf-8"))
                champion_record = build_selection_record(champion_metrics, champion_metadata)
            except Exception as e:
                print(f"Falha ao avaliar Champion atual: {e}. Sobrescrevendo...")

        is_better = True

        if not is_auto_promotion_eligible(cfg.feature_mode):
            eligible_modes = ", ".join(sorted(AUTO_PROMOTION_FEATURE_MODES))
            print(f"[Promocao Rejeitada] O novo modelo possui feature_mode='{cfg.feature_mode}'. Apenas modelos com feature_mode em {{{eligible_modes}}} sao elegiveis para promocao automatica na producao.")
            is_better = False
        elif candidate_record is None:
            print("[Promocao Rejeitada] O novo modelo nao gerou metricas validas para selecao.")
            is_better = False
        else:
            is_better = should_promote_candidate(candidate_record, champion_record)
            gain_text = f"{candidate_record['baseline_gain_pct']:.2f}%" if candidate_record["baseline_gain_pct"] is not None else "sem ganho positivo"
            if is_better:
                print(
                    f"[Promocao Aprovada] O novo modelo venceu o ranking baseline/direcional "
                    f"(ganho={gain_text}; direcional={candidate_record['directional_accuracy']:.2f}%; "
                    f"MAPE={candidate_record['mape_lstm']:.4f}%; linhas={candidate_record['inference_required_rows']}; "
                    f"janela={candidate_record['window_size']})."
                )
            elif candidate_record["baseline_gain_pct"] is None or candidate_record["baseline_gain_pct"] <= MIN_BASELINE_GAIN:
                print("[Promocao Rejeitada] O novo modelo nao superou o baseline persistente.")
            else:
                print("[Promocao Rejeitada] O novo modelo nao venceu o Champion atual pelo ranking baseline/direcional.")

        if not is_better:
            save_dir = Path("models/.temp_challenger")
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Salvando artefatos temporarios do Challenger em {save_dir} (nao serao promovidos para {output_dir})")
        else:
            save_dir = output_dir

        # 8. Exportação de Artefatos
        # Pesos PyTorch
        torch.save(model.state_dict(), save_dir / "model.pt")

        # Safetensors (Formato seguro contra RCE)
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), save_dir / "model.safetensors")
            print("Pesos do modelo exportados com sucesso em formato seguro (model.safetensors).")
        except Exception as e:
            print(f"Aviso: Nao foi possivel salvar os pesos em formato safetensors: {e}")

        # Compilacao para formato estático ONNX (Aceleração em Inferência)
        dummy_input = torch.randn(1, cfg.window_size, X_train.shape[2]).to(device)
        model.eval()
        batch_dim = torch.export.Dim("batch_size", min=1, max=2048)
        dynamic_shapes = {
            "x": {0: batch_dim}
        }

        torch.onnx.export(
            model,
            dummy_input,
            str(save_dir / "model.onnx"),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=dynamic_shapes,
        )

        resolved_selected_features = feature_cols if cfg.feature_mode in {"technical_features", "custom"} else None

        # Salva preprocessador (Normalizadores e Metadados de Janela)
        preprocess = {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "window_size": cfg.window_size,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "feature_mode": cfg.feature_mode,
            "target_mode": cfg.target_mode,
            "feature_scaler_type": cfg.feature_scaler_type,
            "target_scaler_type": cfg.target_scaler_type,
            "selected_features": resolved_selected_features,
            "feature_preset": cfg.feature_preset,
            "feature_schema_version": "v1",
            "feature_registry_version": "2026-05-19"
        }
        joblib.dump(preprocess, save_dir / "preprocessor.joblib")

        # Salva metadados de linhagem e governança
        metadata = {
            "run_id": run.info.run_id,
            "symbol": cfg.symbol,
            "data_source": data_source,
            "preprocessing_mode": preprocessing_mode,
            "end_date_requested": end_date_requested,
            "yfinance_end_date_exclusive": yfinance_end_date_exclusive,
            "dataset_start_real": dataset_start_real,
            "dataset_end_real": dataset_end_real,
            **split_date_params,
            "window_size": cfg.window_size,
            "prediction_horizon": "next_trading_day",
            "target_mode": cfg.target_mode,
            "feature_mode": cfg.feature_mode,
            "feature_scaler_type": cfg.feature_scaler_type,
            "target_scaler_type": cfg.target_scaler_type,
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape,
            "selected_features": resolved_selected_features,
            "feature_cols": feature_cols,
            "feature_preset": cfg.feature_preset,
            "feature_schema_version": "v1",
            "feature_registry_version": "2026-05-19",
            "feature_count": len(feature_cols),
        }
        
        write_json(save_dir / "metadata.json", metadata)
        write_json(save_dir / "metrics.json", metrics)
        write_json(save_dir / "history.json", history)

        # 9. Log de predições do conjunto de teste para auditoria
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
        
        # Gera gráfico de performance
        plot_performance(
            save_dir, df_feat, train_end, val_end, dates_test, 
            true_test_close, pred_test_close, metrics, cfg.symbol, 
            cfg.feature_mode, cfg.target_mode
        )
        
        # Envia todos os artefatos locais para o MLflow
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
    parser.add_argument(
        "--feature-mode", 
        type=str, 
        default="single", 
        choices=["single", "ohlcv", "ohlcv_returns", "technical_features", "custom"]
    )
    parser.add_argument("--feature-scaler-type", type=str, default="standard", choices=["standard", "minmax", "robust"])
    parser.add_argument("--target-scaler-type", type=str, default="standard", choices=["standard", "minmax", "robust"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--selected-features", type=str, default=None, help="Lista de features separadas por virgula para o modo custom.")
    parser.add_argument(
        "--feature-preset", 
        type=str, 
        default=None, 
        choices=["returns_basic", "returns_trend", "returns_volatility", "technical_complete"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    selected = None
    if args.selected_features:
        selected = [f.strip() for f in args.selected_features.split(",") if f.strip()]

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
        selected_features=selected,
        feature_preset=args.feature_preset,
    )
    run_training_pipeline(cfg, args.csv)
