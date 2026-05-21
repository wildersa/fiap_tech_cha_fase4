from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train.config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI dedicado para executar o treinamento StockLSTM com saida operacional."
    )
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
        choices=["single", "ohlcv", "ohlcv_returns", "technical_features", "custom"],
    )
    parser.add_argument("--feature-scaler-type", type=str, default="standard", choices=["standard", "minmax", "robust"])
    parser.add_argument("--target-scaler-type", type=str, default="standard", choices=["standard", "minmax", "robust"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--selected-features", type=str, default=None, help="Features separadas por virgula para modo custom.")
    parser.add_argument(
        "--feature-preset",
        type=str,
        default=None,
        choices=["returns_basic", "returns_trend", "returns_volatility", "technical_complete"],
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    selected = None
    if args.selected_features:
        selected = [item.strip() for item in args.selected_features.split(",") if item.strip()]

    return TrainConfig(
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


def metric(metrics: dict[str, Any], section: str, key: str) -> float | None:
    value = metrics.get(section, {})
    if not isinstance(value, dict):
        return None
    raw = value.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def print_config(cfg: TrainConfig, csv_path: str | None) -> None:
    print("=== StockLSTM Training CLI ===", flush=True)
    print("Iniciando treinamento...", flush=True)
    print(f"- symbol: {cfg.symbol}", flush=True)
    print(f"- periodo: {cfg.start_date} ate {cfg.end_date or 'hoje/disponivel'}", flush=True)
    print(f"- fonte: {csv_path or 'yfinance'}", flush=True)
    print(f"- feature_mode: {cfg.feature_mode}", flush=True)
    print(f"- target_mode: {cfg.target_mode}", flush=True)
    print(f"- window_size: {cfg.window_size}", flush=True)
    print(f"- max_epochs: {cfg.max_epochs}", flush=True)
    print(f"- output_dir: {cfg.output_dir}", flush=True)
    print("", flush=True)


def print_result(result: dict[str, Any], elapsed_sec: float) -> None:
    metrics = result.get("metrics", {})
    lstm_mape = metric(metrics, "lstm_test", "mape_pct")
    baseline_mape = metric(metrics, "baseline_test", "mape_pct")
    gain_mape = metric(metrics, "relative_gain_vs_baseline_pct", "mape_pct")
    direction = metrics.get("directional_accuracy_test_lstm_pct")

    summary = {
        "status": "success",
        "elapsed_sec": round(elapsed_sec, 2),
        "output_dir": result.get("output_dir"),
        "metrics": {
            "test_lstm_mape_pct": lstm_mape,
            "test_baseline_mape_pct": baseline_mape,
            "gain_mape_pct": gain_mape,
            "directional_accuracy_pct": direction,
        },
    }

    print("", flush=True)
    print("=== Resultado do Treinamento ===", flush=True)
    print(f"- status: {summary['status']}", flush=True)
    print(f"- tempo: {summary['elapsed_sec']}s", flush=True)
    print(f"- output_dir: {summary['output_dir']}", flush=True)
    if lstm_mape is not None:
        print(f"- MAPE LSTM teste: {lstm_mape:.4f}%", flush=True)
    if baseline_mape is not None:
        print(f"- MAPE baseline teste: {baseline_mape:.4f}%", flush=True)
    if gain_mape is not None:
        print(f"- ganho MAPE vs baseline: {gain_mape:+.4f}%", flush=True)
    if direction is not None:
        print(f"- acuracia direcional: {float(direction):.2f}%", flush=True)

    print("", flush=True)
    print("Resposta JSON:", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    print_config(cfg, args.csv)

    started_at = time.time()
    try:
        from src.train.pipeline import run_training_pipeline

        result = run_training_pipeline(cfg, args.csv)
    except Exception as exc:
        elapsed_sec = time.time() - started_at
        error_payload = {
            "status": "error",
            "elapsed_sec": round(elapsed_sec, 2),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "config": asdict(cfg),
        }
        print("", flush=True)
        print("=== Falha no Treinamento ===", flush=True)
        print(json.dumps(error_payload, indent=2, ensure_ascii=False), flush=True)
        return 1

    print_result(result, time.time() - started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
