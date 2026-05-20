import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.train.trainer import NpEncoder


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, cls=NpEncoder, indent=2, ensure_ascii=False), encoding="utf-8")


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
