"""
Ajuste de hiperparametros para o modelo StockLSTM com registro no MLflow.
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import itertools
import os
import random
from pathlib import Path
import mlflow
from src.train import TrainConfig, run_training_pipeline
from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tuning de hiperparametros para LSTM.")
    parser.add_argument("--symbol", type=str, default="PETR4.SA")
    parser.add_argument("--csv", type=str, default=None, help="Caminho para o CSV de dados local.")
    parser.add_argument("--max-epochs", type=int, default=30, help="Numero maximo de epochs por trial (curto para busca rapida).")
    parser.add_argument("--n-trials", type=int, default=5, help="Numero maximo de combinacoes aleatorias a testar.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configura MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if "://" not in mlflow_uri:
        mlflow_path = Path(mlflow_uri)
        if not mlflow_path.is_absolute():
            mlflow_path = Path(__file__).resolve().parent.parent / mlflow_path
        mlflow_uri = mlflow_path.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("stock_lstm_hypersearch")

    # Define o espaco de busca
    param_grid = {
        "hidden_size": [32, 64],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "window_size": [20, 30, 60],
        "batch_size": [32, 64],
        "feature_scaler_type": ["standard", "robust"],
        "target_scaler_type": ["standard", "minmax"]
    }

    # Gera todas as combinacoes
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Embaralha e limita ao numero de trials para busca aleatoria
    random.seed(args.seed)
    random.shuffle(all_combinations)
    trials_to_run = all_combinations[:args.n_trials]

    print(f"Iniciando busca de hiperparametros para {args.symbol}...")
    print(f"Total de combinacoes no espaco de busca: {len(all_combinations)}")
    print(f"Combinacoes selecionadas para execucao (n_trials): {len(trials_to_run)}")

    best_mape = float("inf")
    best_params = None

    # Cria run pai
    with mlflow.start_run(run_name=f"Tuning_{args.symbol}") as parent_run:
        mlflow.set_tag("type", "hyperparameter_tuning")
        mlflow.set_tag("symbol", args.symbol)

        for i, params in enumerate(trials_to_run, 1):
            print(f"\n--- Trial {i}/{len(trials_to_run)} ---")
            print("Parametros:", params)

            # Instancia configuracao de treino herdando o parent_run_id para aninhamento
            cfg = TrainConfig(
                symbol=args.symbol,
                max_epochs=args.max_epochs,
                parent_run_id=parent_run.info.run_id,
                seed=args.seed,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
                window_size=params["window_size"],
                batch_size=params["batch_size"],
                feature_scaler_type=params["feature_scaler_type"],
                target_scaler_type=params["target_scaler_type"],
                feature_mode="technical_features",
                target_mode="log_returns"
            )

            try:
                res = run_training_pipeline(cfg, args.csv)
                val_mape = res["metrics"]["lstm_val"]["mape_pct"]
                print(f"Trial {i} finalizado com sucesso. Validation MAPE: {val_mape:.2f}%")

                if val_mape < best_mape:
                    best_mape = val_mape
                    best_params = params
                    print(f"Novo melhor modelo encontrado! MAPE: {best_mape:.2f}%")
            except Exception as e:
                print(f"Erro no Trial {i} com parametros {params}: {e}")
                continue

        if best_params:
            print("\n=========================================")
            print("Busca Finalizada!")
            print(f"Melhor Validation MAPE: {best_mape:.2f}%")
            print("Melhores Parametros:", best_params)
            print("=========================================")
            
            # Loga os melhores parametros no run pai
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_mape_pct", best_mape)
        else:
            print("\nNenhum trial foi executado com sucesso.")


if __name__ == "__main__":
    main()
