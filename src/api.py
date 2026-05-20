"""
API de inferencia do modelo LSTM.

Contrato de inferencia:

    {
      "symbol": "PETR4.SA",
      "closes": [60 fechamentos anteriores]
    }

A API calcula os log-retornos, aplica o mesmo scaler do treino, executa a LSTM
e converte o log-retorno previsto para preco de fechamento.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import os
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from src.data_loader import normalize_columns, ensure_datetime_index, add_features
import psutil
import onnxruntime as ort
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, REGISTRY
from pydantic import BaseModel, Field

# Métricas oficiais do Prometheus para monitoramento de recursos e performance
PROM_CPU_USAGE = Gauge('api_cpu_usage_percent', 'Uso de CPU do processo em porcentagem')
PROM_MEMORY_USAGE = Gauge('api_memory_usage_bytes', 'Uso de memoria do processo em bytes')
PROM_SYSTEM_MEMORY = Gauge('api_system_memory_percent', 'Uso de memoria do sistema em porcentagem')
PROM_LAST_LATENCY = Gauge('api_last_latency_ms', 'Tempo de resposta da ultima requisicao em ms')
PROM_TOTAL_ERRORS = Counter('api_errors_total', 'Contador acumulado de erros na API')

ENABLE_TRAINING_API = os.getenv("ENABLE_TRAINING_API", "true").lower() == "true"
if ENABLE_TRAINING_API:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.train import TrainConfig, run_training_pipeline
        from src.model import StockLSTM
    except ImportError:
        ENABLE_TRAINING_API = False


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/lstm_petr4"))
MODEL_DIR_MULTI = Path(os.getenv("MODEL_DIR_MULTI", "models/lstm_petr4_multi"))
BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_TEMPLATE = BASE_DIR / "dashboard.html"
START_TIME = time.time()
APP_METRICS = {
    "total_requests": 0,
    "total_latency_sec": 0.0,
    "last_latency_sec": 0.0,
    "total_errors": 0,
    "prediction_requests": 0,
    "last_prediction": None,
    "last_training_time_sec": None,
    "total_inference_time_sec": 0.0,
    "inference_requests": 0,
    "last_inference_time_sec": 0.0,
}
TELEMETRY_HISTORY = deque(maxlen=100)


class PredictRequest(BaseModel):
    symbol: str = Field(
        "PETR4.SA",
        description="Código do ativo financeiro a ser previsto.",
        examples=["PETR4.SA"]
    )
    closes: List[float] = Field(
        ...,
        description="Lista cronológica de preços de fechamento anteriores do ativo. A quantidade mínima exigida é igual ao window_size + 1 (ex: se window_size=60, envie pelo menos 61 fechamentos).",
        examples=[[30.1, 30.2, 30.5, 30.4, 30.7, 30.9, 31.0, 31.2, 31.5, 31.4, 31.8, 31.9, 32.1, 32.0, 32.4, 32.5, 32.7, 32.6, 32.9, 33.1, 33.0, 33.4, 33.5, 33.7, 33.6, 33.9, 34.1, 34.0, 34.4, 34.5, 34.7, 34.6, 34.9, 35.1, 35.0, 35.4, 35.5, 35.7, 35.6, 35.9, 36.1, 36.0, 36.4, 36.5, 36.7, 36.6, 36.9, 37.1, 37.0, 37.4, 37.5, 37.7, 37.6, 37.9, 38.1, 38.0, 38.4, 38.5, 38.7, 38.6, 38.9]]
    )


class OhlcvRow(BaseModel):
    date: str = Field(..., description="Data no formato YYYY-MM-DD", examples=["2024-01-02"])
    open: float = Field(..., description="Preço de Abertura", examples=[35.1])
    high: float = Field(..., description="Preço Máximo", examples=[36.0])
    low: float = Field(..., description="Preço Mínimo", examples=[34.8])
    close: float = Field(..., description="Preço de Fechamento", examples=[35.7])
    volume: float = Field(..., description="Volume de negociação", examples=[12345600])


class PredictOhlcvRequest(BaseModel):
    symbol: str = Field(
        "PETR4.SA",
        description="Código do ativo financeiro a ser previsto.",
        examples=["PETR4.SA"]
    )
    rows: List[OhlcvRow] = Field(
        ...,
        description="Lista cronológica recente de dados OHLCV (enviar dados suficientes para lookback window e cálculo de indicadores).",
        examples=[
            [
                {"date": "2024-01-02", "open": 35.1, "high": 36.0, "low": 34.8, "close": 35.7, "volume": 12345600}
            ]
        ]
    )


class TrainRequest(BaseModel):
    symbol: str = Field(
        "PETR4.SA",
        description="Código do ativo para busca no Yahoo Finance.",
        examples=["PETR4.SA"]
    )
    start_date: str = Field(
        "2018-01-01",
        description="Data de início da coleta de dados históricos no formato YYYY-MM-DD.",
        examples=["2018-01-01"]
    )
    end_date: str | None = Field(
        None,
        description="Data final da coleta no formato YYYY-MM-DD. Se nulo, coleta até a data atual.",
        examples=["2024-12-31"]
    )
    window_size: int = Field(
        60,
        description="Tamanho da janela de lookback (dias de histórico para alimentar a LSTM).",
        examples=[60]
    )
    train_ratio: float = Field(
        0.70,
        description="Proporção dos dados temporais para treinamento (0.0 a 1.0).",
        examples=[0.70]
    )
    val_ratio: float = Field(
        0.15,
        description="Proporção dos dados temporais para validação (0.0 a 1.0).",
        examples=[0.15]
    )
    hidden_size: int = Field(
        64,
        description="Número de neurônios na camada oculta da LSTM.",
        examples=[64]
    )
    num_layers: int = Field(
        1,
        description="Número de camadas empilhadas na LSTM.",
        examples=[1]
    )
    dropout: float = Field(
        0.20,
        description="Fator de dropout para regularização (apenas aplicável se num_layers > 1).",
        examples=[0.20]
    )
    learning_rate: float = Field(
        1e-3,
        description="Taxa de aprendizado inicial do otimizador AdamW.",
        examples=[0.001]
    )
    weight_decay: float = Field(
        1e-4,
        description="Fator de decaimento de peso (regularização L2) no AdamW.",
        examples=[0.0001]
    )
    batch_size: int = Field(
        32,
        description="Tamanho do lote de treinamento.",
        examples=[32]
    )
    max_epochs: int = Field(
        100,
        description="Quantidade máxima de épocas de treino.",
        examples=[100]
    )
    patience: int = Field(
        20,
        description="Paciência do Early Stopping baseada na perda de validação.",
        examples=[20]
    )
    target_mode: str = Field(
        "log_returns",
        description="Modo do alvo de treino: 'log_returns' (retornos logarítmicos) ou 'raw_close' (preço de fechamento bruto).",
        examples=["log_returns"]
    )
    feature_mode: str = Field(
        "single",
        description="Tipo de features: 'single', 'ohlcv', 'ohlcv_returns', 'technical_features' ou 'custom'.",
        examples=["single"]
    )
    feature_scaler_type: str = Field(
        "standard",
        description="Tipo do normalizador das features: 'standard', 'minmax' ou 'robust'.",
        examples=["standard"]
    )
    target_scaler_type: str = Field(
        "standard",
        description="Tipo do normalizador do alvo (target): 'standard', 'minmax' ou 'robust'.",
        examples=["standard"]
    )
    grad_clip: float | None = Field(
        1.0,
        description="Valor limite para corte de gradiente (gradient clipping). Evita explosão de gradientes.",
        examples=[1.0]
    )
    device: str = Field(
        "auto",
        description="Dispositivo de execução: 'auto' (detecta automaticamente), 'cpu' ou 'cuda'.",
        examples=["auto"]
    )
    parent_run_id: str | None = Field(
        None,
        description="ID de uma run do MLflow para associar a linhagem (tags de pai/filho).",
        examples=["52c6f10c0e1847c2b530514fe96b96db"]
    )
    selected_features: List[str] | None = Field(
        None,
        description="Lista de features selecionadas (obrigatório se feature_mode='custom').",
        examples=[["Log_Return", "RSI_14", "MACD"]]
    )
    feature_preset: str | None = Field(
        None,
        description="Preset de features a ser utilizado (returns_basic, returns_trend, returns_volatility, technical_complete).",
        examples=["returns_trend"]
    )


app = FastAPI(
    title="Tech Challenge LSTM API",
    description="API para prever o proximo fechamento a partir de fechamentos historicos.",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# Garantir que a API aponte para o MLflow correto localmente
if ENABLE_TRAINING_API:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if "://" not in mlflow_uri:
        mlflow_path = Path(mlflow_uri)
        if not mlflow_path.is_absolute():
            mlflow_path = Path(__file__).resolve().parent.parent / mlflow_path
        mlflow_uri = mlflow_path.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)


def sync_best_model_from_mlflow() -> None:
    """
    MLOps Automatic Promotion:
    Sincroniza automaticamente o melhor modelo de tipo 'single' (univariado) para MODEL_DIR,
    e o melhor modelo de tipo multivariado (qualquer outro feature_mode) para MODEL_DIR_MULTI.
    """
    if not ENABLE_TRAINING_API:
        return

    try:
        from mlflow.tracking import MlflowClient
        import shutil
        client = MlflowClient()
        experiments = client.search_experiments()
        
        # Encontrar os melhores modelos no MLflow
        best_run_single = None
        best_mape_single = float("inf")
        best_run_multi = None
        best_mape_multi = float("inf")

        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for r in runs:
                feature_mode = r.data.params.get("feature_mode", "single")
                mape = (
                    r.data.metrics.get("val_lstm_mape_pct") or 
                    r.data.metrics.get("lstm_mape_pct") or
                    r.data.metrics.get("test_lstm_mape_pct")
                )
                if mape is not None and mape > 0:
                    if feature_mode == "single":
                        if mape < best_mape_single:
                            best_mape_single = mape
                            best_run_single = r
                    else:
                        if mape < best_mape_multi:
                            best_mape_multi = mape
                            best_run_multi = r

        # Função auxiliar para promover modelo para um diretório específico
        def promote_to_dir(best_run, best_mape, target_dir, label):
            if best_run is None:
                return False
            
            current_mape = float("inf")
            metrics_path = target_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    current_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    current_mape = (
                        current_metrics.get("lstm_val", {}).get("mape_pct") or
                        current_metrics.get("val_lstm_mape_pct") or
                        current_metrics.get("lstm_test", {}).get("mape_pct") or
                        current_metrics.get("test_lstm_mape_pct") or
                        current_metrics.get("lstm_mape_pct") or
                        float("inf")
                    )
                except Exception:
                    pass

            if best_mape < (current_mape - 1e-6):
                print(f"[MLOps] Novo Campeao {label} detectado no MLflow (Run {best_run.info.run_id}) com Validation MAPE {best_mape:.4f}% (anterior em disco: {current_mape:.4f}%)")
                target_dir.mkdir(parents=True, exist_ok=True)
                local_path = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id)
                src_path = Path(local_path)
                
                copied_any = False
                for filename in ["model.onnx", "model.pt", "model.safetensors", "preprocessor.joblib", "metadata.json", "metrics.json", "model_performance.png"]:
                    src_file = src_path / filename
                    if src_file.exists():
                        shutil.copy(str(src_file), str(target_dir / filename))
                        copied_any = True
                
                if copied_any:
                    print(f"[MLOps] Modelo {label} do MLflow promovido com sucesso para {target_dir}.")
                    return True
                else:
                    print(f"[MLOps] Aviso: Nenhum artefato {label} copiado de {src_path}")
            else:
                print(f"[MLOps] O modelo {label} em disco ({current_mape:.4f}%) ja e o melhor ou empata com o melhor do MLflow ({best_mape:.4f}%)")
            return False

        # Promove o modelo univariado
        updated_single = promote_to_dir(best_run_single, best_mape_single, MODEL_DIR, "Univariado")
        
        # Promove o modelo multivariado
        updated_multi = promote_to_dir(best_run_multi, best_mape_multi, MODEL_DIR_MULTI, "Multivariado")

        if updated_single or updated_multi:
            load_preprocessor.cache_clear()
            load_predictor.cache_clear()
            load_preprocessor_multi.cache_clear()
            load_predictor_multi.cache_clear()
            
    except Exception as e:
        print(f"[MLOps] Erro ao sincronizar melhor modelo do MLflow: {e}")


@app.on_event("startup")
def startup_event():
    sync_best_model_from_mlflow()


@lru_cache(maxsize=1)
def load_preprocessor() -> dict:
    path = MODEL_DIR / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessador nao encontrado: {path}. Execute src/train.py primeiro.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_predictor():
    model_path = MODEL_DIR / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo ONNX nao encontrado: {model_path}. Execute o treinamento primeiro para gerar o arquivo ONNX.")

    session = ort.InferenceSession(str(model_path))
    return session


@lru_cache(maxsize=1)
def load_preprocessor_multi() -> dict:
    path = MODEL_DIR_MULTI / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessador multivariado nao encontrado em {path}. Execute o treinamento com feature_mode multivariado primeiro.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_predictor_multi():
    model_path = MODEL_DIR_MULTI / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo ONNX multivariado nao encontrado em {model_path}. Execute o treinamento com feature_mode multivariado primeiro.")

    session = ort.InferenceSession(str(model_path))
    return session


def prepare_closes(closes: List[float]) -> np.ndarray:
    arr = np.asarray(closes, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("closes precisa ser uma lista simples de numeros.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("closes contem valores invalidos.")
    if np.any(arr <= 0):
        raise ValueError("Todos os fechamentos precisam ser maiores que zero.")
    return arr


def build_latest_window(closes: np.ndarray, preprocessor: dict) -> np.ndarray:
    window_size = int(preprocessor["window_size"])
    target_mode = preprocessor.get("target_mode", "log_returns")
    feature_mode = preprocessor.get("feature_mode", "single")
    
    if feature_mode != "single":
        raise ValueError(
            f"Feature mode '{feature_mode}' não é suportado para inferência em tempo real na API simplificada /predict. "
            "Por favor, use o endpoint multivariado '/predict/ohlcv'."
        )

    if target_mode in {"log_returns", "returns"}:
        required_closes = window_size + 1
        if len(closes) < required_closes:
            raise ValueError(f"Envie pelo menos {required_closes} fechamentos para gerar {window_size} retornos.")
        if target_mode == "log_returns":
            feat_series = np.log(closes[1:] / closes[:-1]).reshape(-1, 1)
        else: # returns
            feat_series = (closes[1:] / closes[:-1] - 1.0).reshape(-1, 1)
        latest_feat = feat_series[-window_size:]
    else: # raw_close
        required_closes = window_size
        if len(closes) < required_closes:
            raise ValueError(f"Envie pelo menos {required_closes} fechamentos.")
        latest_feat = closes[-window_size:].reshape(-1, 1)
        
    scaler_key = "feature_scaler" if "feature_scaler" in preprocessor else "scaler"
    scaled_window = preprocessor[scaler_key].transform(latest_feat)
    return scaled_window.reshape(1, window_size, 1).astype(np.float32)


def predict_next(closes_payload: List[float]) -> dict:
    preprocessor = load_preprocessor()
    session = load_predictor()
    closes = prepare_closes(closes_payload)
    X = build_latest_window(closes, preprocessor)
    last_close = float(closes[-1])

    ort_inputs = {session.get_inputs()[0].name: X}
    ort_outs = session.run(None, ort_inputs)
    predicted_scaled = float(ort_outs[0][0][0])
    
    scaler_key = "target_scaler" if "target_scaler" in preprocessor else "scaler"
    predicted_raw = float(preprocessor[scaler_key].inverse_transform([[predicted_scaled]])[0][0])

    target_mode = preprocessor.get("target_mode", "log_returns")
    if target_mode == "log_returns":
        predicted_close = last_close * float(np.exp(predicted_raw))
        predicted_log_return = predicted_raw
        predicted_return_pct = float((np.exp(predicted_raw) - 1) * 100)
    elif target_mode == "returns":
        predicted_close = last_close * float(1.0 + predicted_raw)
        predicted_log_return = float(np.log(1.0 + predicted_raw)) if (1.0 + predicted_raw) > 0 else 0.0
        predicted_return_pct = float(predicted_raw * 100)
    else: # raw_close
        predicted_close = predicted_raw
        predicted_log_return = float(np.log(predicted_close / last_close)) if (predicted_close > 0 and last_close > 0) else 0.0
        predicted_return_pct = float((predicted_close / last_close - 1) * 100) if last_close else 0.0

    change_abs = predicted_close - last_close
    change_pct = (change_abs / last_close) * 100 if last_close else 0.0
    return {
        "prediction_horizon": "next_trading_day",
        "last_close": last_close,
        "predicted_close": float(predicted_close),
        "predicted_log_return": predicted_log_return,
        "predicted_return_pct": float(predicted_return_pct),
        "predicted_change_abs": float(change_abs),
        "predicted_change_pct": float(change_pct),
        "predicted_direction": "alta" if change_abs >= 0 else "queda",
    }


def predict_next_ohlcv(rows_payload: List[dict]) -> dict:
    preprocessor = load_preprocessor_multi()
    session = load_predictor_multi()

    # 1. Converter para DataFrame
    df = pd.DataFrame(rows_payload)

    # 2. Normalizar nomes das colunas
    df = normalize_columns(df)

    # 3. Garantir index datetime e ordenação
    df = ensure_datetime_index(df)

    # 4. Validar colunas necessárias
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no payload OHLCV: {missing}")

    # 5. Adicionar/calcular features baseado no preprocessor do modelo treinado
    feature_mode = preprocessor.get("feature_mode", "single")
    if feature_mode == "single":
        target_col = preprocessor.get("target_col", "Log_Return")
        if target_col == "Log_Return":
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        elif target_col == "Return":
            df["Return"] = df["Close"].pct_change()
        # Remove as linhas com NaN resultantes do shift/pct_change
        df = df.dropna(subset=[target_col])
    else:
        # Modo multivariado: calcular todas as features
        df = add_features(df)

    # 6. Validar que as colunas de feature esperadas foram geradas
    feature_cols = preprocessor.get("feature_cols", ["Log_Return"])
    missing_feats = [f for f in feature_cols if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Não foi possível calcular todas as features necessárias: {missing_feats}. Envie mais linhas de histórico.")

    # 7. Validar window_size mínimo após limpeza/cálculos
    window_size = int(preprocessor["window_size"])
    if len(df) < window_size:
        raise ValueError(
            f"Após o cálculo de indicadores, restaram apenas {len(df)} linhas válidas "
            f"(mínimo necessário para window_size de {window_size}: {window_size} linhas). "
            "Por favor, envie um histórico de dados mais longo para estabilização das médias/indicadores."
        )

    # 8. Extrair a janela de lookback
    latest_df = df[feature_cols].tail(window_size)
    X_values = latest_df.values.astype(np.float32)

    # 9. Escalar features usando o normalizador salvo
    scaler_key = "feature_scaler" if "feature_scaler" in preprocessor else "scaler"
    scaled_window = preprocessor[scaler_key].transform(X_values)

    # 10. Ajustar dimensão para LSTM: (batch_size=1, window_size, input_size=len(feature_cols))
    X = scaled_window.reshape(1, window_size, len(feature_cols)).astype(np.float32)

    # 11. Executar predição na sessão ONNX
    ort_inputs = {session.get_inputs()[0].name: X}
    ort_outs = session.run(None, ort_inputs)
    predicted_scaled = float(ort_outs[0][0][0])

    # 12. Desnormalizar o target
    target_scaler_key = "target_scaler" if "target_scaler" in preprocessor else "scaler"
    predicted_raw = float(preprocessor[target_scaler_key].inverse_transform([[predicted_scaled]])[0][0])

    # 13. Reverter escala de retorno/log-retorno se aplicável para obter o preço final
    last_close = float(df["Close"].iloc[-1])
    target_mode = preprocessor.get("target_mode", "log_returns")

    if target_mode == "log_returns":
        predicted_close = last_close * float(np.exp(predicted_raw))
        predicted_log_return = predicted_raw
        predicted_return_pct = float((np.exp(predicted_raw) - 1) * 100)
    elif target_mode == "returns":
        predicted_close = last_close * float(1.0 + predicted_raw)
        predicted_log_return = float(np.log(1.0 + predicted_raw)) if (1.0 + predicted_raw) > 0 else 0.0
        predicted_return_pct = float(predicted_raw * 100)
    else: # raw_close
        predicted_close = predicted_raw
        predicted_log_return = float(np.log(predicted_close / last_close)) if (predicted_close > 0 and last_close > 0) else 0.0
        predicted_return_pct = float((predicted_close / last_close - 1) * 100) if last_close else 0.0

    change_abs = predicted_close - last_close
    change_pct = (change_abs / last_close) * 100 if last_close else 0.0

    return {
        "prediction_horizon": "next_trading_day",
        "last_close": last_close,
        "predicted_close": float(predicted_close),
        "predicted_log_return": predicted_log_return,
        "predicted_return_pct": float(predicted_return_pct),
        "predicted_change_abs": float(change_abs),
        "predicted_change_pct": float(change_pct),
        "predicted_direction": "alta" if change_abs >= 0 else "queda",
    }


@app.middleware("http")
async def collect_app_metrics(request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.time()
    response = None
    has_error = False
    status_code = 200

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        has_error = True
        status_code = 500
        raise
    finally:
        latency = time.time() - start
        APP_METRICS["total_requests"] += 1
        APP_METRICS["total_latency_sec"] += latency
        APP_METRICS["last_latency_sec"] = latency
        if has_error or status_code >= 400:
            APP_METRICS["total_errors"] += 1
            PROM_TOTAL_ERRORS.inc()
            
        latency_ms = round(latency * 1000, 2)
        cpu = psutil.cpu_percent()
        mem_bytes = psutil.Process().memory_info().rss
        mem_mb = round(mem_bytes / (1024 * 1024), 2)
        sys_mem = psutil.virtual_memory().percent

        # Atualiza as métricas oficiais do Prometheus registradas no processo
        PROM_CPU_USAGE.set(cpu)
        PROM_MEMORY_USAGE.set(mem_bytes)
        PROM_SYSTEM_MEMORY.set(sys_mem)
        PROM_LAST_LATENCY.set(latency_ms)

        snapshot = {
            "timestamp": time.strftime("%H:%M:%S"),
            "latency_ms": round(APP_METRICS["last_inference_time_sec"] * 1000, 2),
            "cpu_percent": cpu,
            "memory_mb": mem_mb
        }
        TELEMETRY_HISTORY.append(snapshot)
        if response:
            response.headers["X-Process-Time"] = f"{latency:.6f}"


def model_card_payload(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    metrics_path = model_dir / "metrics.json"
    metadata = {}
    metrics = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    preprocessor = None
    if (model_dir / "preprocessor.joblib").exists():
        try:
            if model_dir == MODEL_DIR_MULTI:
                preprocessor = load_preprocessor_multi()
            else:
                preprocessor = load_preprocessor()
        except Exception:
            preprocessor = None

    # Carrega textos qualitativos de model_card_template.json se existir
    template_path = Path(__file__).resolve().parent / "model_card_template.json"
    is_multi = (model_dir == MODEL_DIR_MULTI)
    text_defaults = {
        "model_name": "StockLSTM (Multivariado)" if is_multi else "StockLSTM (Univariado)",
        "model_description": "Modelo Deep Learning baseado em LSTM configurável (Multivariado) projetado para previsão de séries temporais de ativos financeiros usando múltiplos indicadores de mercado." if is_multi else "Modelo Deep Learning baseado em LSTM configurável (Univariado) projetado para previsão de séries temporais de fechamento de ativos financeiros.",
        "intended_uses": "Auxílio à tomada de decisão em estratégias de trading de curto prazo (D+1) para a ação PETR4. Não recomendado para uso autônomo de alta frequência (HFT) sem supervisão humana.",
        "training_observations": "Modelo treinado com otimizador AdamW, decaimento de peso (weight decay) e parada antecipada (early stopping) baseada na perda do conjunto de validação. O processamento separa as escalas de feature/target usando Anchor Price.",
        "evaluation_dataset": "Split temporal Out-of-Time de 15% da base de dados histórica.",
        "ethical_considerations": "Este modelo foi criado exclusivamente para fins acadêmicos (Tech Challenge FIAP) e não constitui conselho financeiro ou indicação de compra/venda.",
        "caveats_and_recommendations": "O modelo assume estabilidade relativa do mercado. Eventos de cisne negro (black swan), crises geopolíticas extremas ou alterações corporativas bruscas invalidam as previsões temporais devido ao conceito de Data Drift."
    }
    if template_path.exists():
        try:
            text_defaults.update(json.loads(template_path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[ModelCard] Erro ao ler template json: {e}")

    # Garante a diferenciação pós-template para o nome do modelo
    if is_multi:
        if not text_defaults["model_name"].endswith(" (Multivariado)"):
            text_defaults["model_name"] = text_defaults["model_name"] + " (Multivariado)"
    else:
        if not text_defaults["model_name"].endswith(" (Univariado)"):
            text_defaults["model_name"] = text_defaults["model_name"] + " (Univariado)"

    # Monta a estrutura inspirada no AWS SageMaker Model Cards
    return {
        "model_overview": {
            "model_name": text_defaults["model_name"],
            "model_description": text_defaults["model_description"],
            "model_version": "1.0.0",
            "model_status": "Approved" if (model_dir / "model.onnx").exists() else "Draft",
            "risk_rating": "Medium",
            "intended_uses": text_defaults["intended_uses"]
        },
        "training_details": {
            "symbol": metadata.get("symbol", "PETR4.SA"),
            "data_source": metadata.get("data_source", "yfinance"),
            "window_size": metadata.get("window_size") or (preprocessor or {}).get("window_size") or 60,
            "feature_mode": metadata.get("feature_mode") or (preprocessor or {}).get("feature_mode") or ("ohlcv" if is_multi else "single"),
            "target_mode": metadata.get("target_mode") or (preprocessor or {}).get("target_mode") or "log_returns",
            "feature_scaler_type": metadata.get("feature_scaler_type") or (preprocessor or {}).get("feature_scaler_type") or "standard",
            "target_scaler_type": metadata.get("target_scaler_type") or (preprocessor or {}).get("target_scaler_type") or "standard",
            "mlflow_run_id": metadata.get("run_id", "Treinamento Local / Sem Run ID"),
            "feature_preset": metadata.get("feature_preset") or (preprocessor or {}).get("feature_preset"),
            "selected_features": metadata.get("selected_features") or (preprocessor or {}).get("selected_features"),
            "feature_count": metadata.get("feature_count") or (len(preprocessor["feature_cols"]) if preprocessor and "feature_cols" in preprocessor else 1),
            "feature_cols": metadata.get("feature_cols") or (preprocessor or {}).get("feature_cols") or [],
            "training_observations": text_defaults["training_observations"]
        },
        "evaluation_details": {
            "evaluation_dataset": text_defaults["evaluation_dataset"],
            "metrics": metrics,
            "artifacts_available": {
                "model_onnx": (model_dir / "model.onnx").exists(),
                "model_safetensors": (model_dir / "model.safetensors").exists(),
                "preprocessor_joblib": (model_dir / "preprocessor.joblib").exists(),
                "metadata_json": metadata_path.exists(),
            }
        },
        "additional_information": {
            "ethical_considerations": text_defaults["ethical_considerations"],
            "caveats_and_recommendations": text_defaults["caveats_and_recommendations"]
        }
    }


def telemetry_payload() -> dict:
    total_requests = APP_METRICS["total_requests"]
    avg_latency_ms = (APP_METRICS["total_latency_sec"] / total_requests * 1000) if total_requests else 0.0

    cpu = REGISTRY.get_sample_value('api_cpu_usage_percent') or 0.0
    mem_bytes = REGISTRY.get_sample_value('api_memory_usage_bytes') or 0.0
    sys_mem = REGISTRY.get_sample_value('api_system_memory_percent') or 0.0
    last_lat = REGISTRY.get_sample_value('api_last_latency_ms') or 0.0
    errs = REGISTRY.get_sample_value('api_errors_total_total') or REGISTRY.get_sample_value('api_errors_total') or 0.0

    avg_inference_ms = (APP_METRICS["total_inference_time_sec"] / APP_METRICS["inference_requests"] * 1000) if APP_METRICS["inference_requests"] else 0.0
    last_inference_ms = APP_METRICS["last_inference_time_sec"] * 1000

    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "api": {
            "total_requests": total_requests,
            "prediction_requests": APP_METRICS["prediction_requests"],
            "total_errors": int(errs),
            "average_response_time_ms": round(avg_latency_ms, 2),
            "last_response_time_ms": round(last_lat, 2),
        },
        "inference": {
            "average_time_ms": round(avg_inference_ms, 2),
            "last_time_ms": round(last_inference_ms, 2),
        },
        "training": {
            "enabled": ENABLE_TRAINING_API,
            "last_time_sec": round(APP_METRICS["last_training_time_sec"], 2) if APP_METRICS.get("last_training_time_sec") is not None else None,
        },
        "resources": {
            "cpu_percent": cpu,
            "process_memory_mb": round(mem_bytes / (1024 * 1024), 2),
            "system_memory_percent": sys_mem,
        },
        "model": {
            "model_dir": str(MODEL_DIR),
            "loaded": (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "preprocessor.joblib").exists(),
        },
        "last_prediction": APP_METRICS["last_prediction"],
        "history": list(TELEMETRY_HISTORY),
    }


@app.get(
    "/health",
    summary="Verificação de Saúde (Liveness/Readiness)",
    description="Retorna se a API está online e indica o caminho absoluto do diretório ativo de modelos no disco.",
    response_description="Status de saúde da API e diretório configurado de modelo.",
    tags=["Monitoramento & Diagnóstico"]
)
def health():
    return {"status": "ok", "model_dir": str(MODEL_DIR)}


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get(
    "/model-card",
    summary="Ficha Técnica (AWS SageMaker Model Card)",
    description="Retorna a ficha técnica detalhada do modelo em formato JSON, estruturada no padrão do AWS SageMaker Model Cards (seções Model Overview, Training Details, Evaluation Details e Additional Information). Suporta os tipos 'single' ou 'multi'.",
    response_description="Payload formatado no padrão AWS SageMaker Model Cards contendo governança, parâmetros de treino, métricas de validação e limites de responsabilidade.",
    tags=["Monitoramento & Diagnóstico"]
)
def model_card(type: str = "single"):
    # Garante sincronização das últimas runs antes de responder
    sync_best_model_from_mlflow()
    
    if type == "multi":
        return model_card_payload(MODEL_DIR_MULTI)
    return model_card_payload(MODEL_DIR)


@app.get(
    "/model-image",
    summary="Gráfico de Performance do Modelo",
    description="Retorna o gráfico de avaliação offline em formato PNG (gerado durante a fase de validação pós-treino do modelo de produção ativo).",
    response_description="Arquivo de imagem PNG do gráfico de performance e curva de perdas.",
    tags=["Monitoramento & Diagnóstico"]
)
def get_model_image(type: str = "single"):
    target_dir = MODEL_DIR_MULTI if type == "multi" else MODEL_DIR
    image_path = target_dir / "model_performance.png"
    if image_path.exists():
        return FileResponse(str(image_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Imagem não encontrada.")


@app.get(
    "/telemetry",
    summary="Métricas de Telemetria do Sistema",
    description="Retorna dados agregados de telemetria e recursos do servidor (uso de CPU, uso de memória do processo e do sistema, tempo de atividade, latências recentes de resposta e contagem de requisições baseada nos coletores do Prometheus).",
    response_description="Relatório de performance em tempo real do sistema para diagnóstico e auditoria.",
    tags=["Monitoramento & Diagnóstico"]
)
def telemetry():
    return telemetry_payload()


@app.get(
    "/runs",
    summary="Histórico de Treinamentos no MLflow",
    description="Consulta o servidor MLflow local/remoto e lista o histórico de runs do experimento 'stock_lstm_hypersearch', incluindo parâmetros, métricas de validação, status da execução, tags e linhagem de derivação.",
    response_description="Histórico cronológico das runs rastreadas no MLflow.",
    tags=["Treinamento & MLOps"]
)
def get_runs(limit: int = 100):
    if not ENABLE_TRAINING_API:
        raise HTTPException(status_code=501, detail="API de treinamento desabilitada ou dependencias (mlflow) nao instaladas.")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("stock_lstm_hypersearch")
    if not experiment:
        return {"runs": []}
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        max_results=limit, 
        order_by=["start_time DESC"]
    )
    
    runs_data = []
    for run in runs:
        runs_data.append({
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        })
    return {"runs": runs_data}


@app.delete(
    "/runs/{run_id}",
    summary="Deletar uma Run do MLflow",
    description="Remove/deleta logicamente uma execução específica do MLflow pelo seu run_id.",
    response_description="Status de sucesso da remoção.",
    tags=["Treinamento & MLOps"]
)
def delete_run(run_id: str):
    if not ENABLE_TRAINING_API:
        raise HTTPException(status_code=501, detail="API de treinamento desabilitada ou dependencias (mlflow) nao instaladas.")
    try:
        client = MlflowClient()
        client.delete_run(run_id)
        return {"status": "success", "message": f"Run {run_id} deletada com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao deletar run: {str(e)}")


@app.post(
    "/train",
    summary="Iniciar Pipeline de Treinamento",
    description="Dispara de forma síncrona o pipeline completo de treinamento: download histórico do yfinance, cálculo de features temporais, split de dados, normalização robusta por Anchor Price, treinamento da rede LSTM com AdamW/Early Stopping, validação cega, log no MLflow e salvamento dos artefatos em ONNX/Preprocessor. Se o modelo treinado tiver um MAPE menor que o campeão atual, ele é promovido automaticamente no final do processo.",
    response_description="Status de sucesso do treinamento, métricas finais obtidas e pasta de saída.",
    tags=["Treinamento & MLOps"]
)
def train_model(req: TrainRequest):
    if not ENABLE_TRAINING_API:
        raise HTTPException(status_code=501, detail="API de treinamento desabilitada ou dependencias (mlflow) nao instaladas.")
    cfg = TrainConfig(**req.model_dump())
    try:
        t0 = time.time()
        results = run_training_pipeline(cfg)
        training_time = time.time() - t0
        APP_METRICS["last_training_time_sec"] = training_time
        
        # Limpa o cache da API para carregar o novo modelo automaticamente se ele foi promovido
        load_predictor.cache_clear()
        load_preprocessor.cache_clear()
        load_predictor_multi.cache_clear()
        load_preprocessor_multi.cache_clear()
        
        return {
            "status": "success", 
            "metrics": results["metrics"], 
            "output_dir": results["output_dir"],
            "message": "Treinamento finalizado com sucesso."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante o treinamento: {str(e)}")


@app.post(
    "/predict",
    summary="Realizar Predição (Inferência D+1)",
    description="Recebe uma série cronológica recente de fechamentos do ativo (mínimo de window_size + 1), extrai os retornos logarítmicos ou absolutos em tempo real, normaliza a janela de lookback usando os parâmetros salvos no preprocessor, executa a inferência acelerada na sessão ONNX Runtime e decodifica a previsão para o preço de fechamento do dia seguinte (D+1). Também computa a variação projetada absoluta/percentual e a direção (alta/queda).",
    response_description="Previsão detalhada para o próximo fechamento (D+1) com base no modelo ativo.",
    tags=["Inferência"]
)
def predict(request: PredictRequest):
    try:
        t0 = time.time()
        result = predict_next(request.closes)
        inference_time = time.time() - t0
        APP_METRICS["last_inference_time_sec"] = inference_time
        APP_METRICS["total_inference_time_sec"] += inference_time
        APP_METRICS["inference_requests"] += 1
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = {
        "symbol": request.symbol,
        "prediction_horizon": result["prediction_horizon"],
        "last_close": result["last_close"],
        "predicted_close": result["predicted_close"],
        "predicted_log_return": result["predicted_log_return"],
        "predicted_return_pct": result["predicted_return_pct"],
        "predicted_change_abs": result["predicted_change_abs"],
        "predicted_change_pct": result["predicted_change_pct"],
        "predicted_direction": result["predicted_direction"],
        "baseline_close": result["last_close"],
        "model_version": "v1",
    }
    APP_METRICS["prediction_requests"] += 1
    APP_METRICS["last_prediction"] = response
    return response


@app.post(
    "/predict/ohlcv",
    summary="Realizar Predição Multivariada (Inferência D+1 via OHLCV)",
    description="Recebe uma série temporal de registros OHLCV recentes, calcula as features correspondentes ao modelo ativo (seja univariado ou multivariado), aplica os normalizadores correspondentes e executa a inferência acelerada ONNX Runtime para prever o fechamento do dia seguinte (D+1).",
    response_description="Previsão detalhada para o próximo fechamento (D+1) baseada no modelo ativo.",
    tags=["Inferência"]
)
def predict_ohlcv(request: PredictOhlcvRequest):
    try:
        t0 = time.time()
        rows_dict = [row.model_dump() for row in request.rows]
        result = predict_next_ohlcv(rows_dict)
        inference_time = time.time() - t0
        APP_METRICS["last_inference_time_sec"] = inference_time
        APP_METRICS["total_inference_time_sec"] += inference_time
        APP_METRICS["inference_requests"] += 1
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = {
        "symbol": request.symbol,
        "prediction_horizon": result["prediction_horizon"],
        "last_close": result["last_close"],
        "predicted_close": result["predicted_close"],
        "predicted_log_return": result["predicted_log_return"],
        "predicted_return_pct": result["predicted_return_pct"],
        "predicted_change_abs": result["predicted_change_abs"],
        "predicted_change_pct": result["predicted_change_pct"],
        "predicted_direction": result["predicted_direction"],
        "baseline_close": result["last_close"],
        "model_version": "v1",
    }
    APP_METRICS["prediction_requests"] += 1
    APP_METRICS["last_prediction"] = response
    return response


def apply_dashboard_label_overrides(html: str) -> str:
    replacements = {
        "Target Mode <span class=\"tooltip-icon\" data-tooltip=\"Formato da variável alvo.\">?</span>": (
            "Alvo da previsão <span class=\"tooltip-icon\" "
            "data-tooltip=\"O que o modelo aprende a prever. Ex.: próximo log-retorno ou próximo fechamento bruto.\">?</span>"
        ),
        "Feature Mode <span class=\"tooltip-icon\"\r\n                  data-tooltip=\"Variáveis preditivas da rede. Modos multivariados (diferentes de 'single') são experimentais e não serão promovidos automaticamente para produção.\">?</span>": (
            "Entradas do modelo <span class=\"tooltip-icon\"\r\n                  data-tooltip=\"Quais variáveis entram no X histórico da LSTM. Isto é diferente do alvo: o alvo é o que o modelo tenta prever. Modos multivariados são experimentais e exigem payload OHLCV na inferência.\">?</span>"
        ),
        "<option value=\"log_returns\">log_returns (Recomendado)</option>": (
            "<option value=\"log_returns\">Prever próximo log-retorno (Recomendado)</option>"
        ),
        "<option value=\"raw_close\">raw_close</option>": (
            "<option value=\"raw_close\">Prever próximo fechamento bruto</option>"
        ),
        "<option value=\"single\">single (Univariado - Produção)</option>": (
            "<option value=\"single\">Close/retorno histórico (Univariado - Produção)</option>"
        ),
        "<option value=\"ohlcv\">ohlcv (Multivariado - Experimental)</option>": (
            "<option value=\"ohlcv\">OHLCV bruto: Open, High, Low, Close, Volume (Experimental)</option>"
        ),
        "<option value=\"ohlcv_returns\">ohlcv_returns (Multivariado - Experimental)</option>": (
            "<option value=\"ohlcv_returns\">OHLCV + log-retorno histórico (Experimental)</option>"
        ),
        "<option value=\"technical_features\">technical_features (Multivariado - Experimental)</option>": (
            "<option value=\"technical_features\">Indicadores técnicos calculados (Experimental)</option>"
        ),
        "<option value=\"custom\">custom (Seleção Customizada - Experimental)</option>": (
            "<option value=\"custom\">Features escolhidas manualmente (Experimental)</option>"
        ),
        "<span>Target Mode</span>": "<span>Alvo</span>",
        "<span>Feature Mode</span>": "<span>Entradas</span>",
        "<th>Target Mode</th>": "<th>Alvo</th>",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


@app.get(
    "/dashboard",
    summary="Dashboard Visual Web",
    description="Renderiza e retorna a interface gráfica HTML interativa (Single Page Application) com controle de treinamento, histórico de runs do MLflow, governança via SageMaker Model Card, telemetria do sistema e simulador visual de payloads de inferência.",
    response_description="Interface gráfica completa do portal em HTML/CSS/JS.",
    tags=["Visualização"],
    response_class=HTMLResponse
)
def dashboard():
    sample_closes = []
    for i in range(65):
        base = 43.0 + i * 0.035 + np.sin(i / 4) * 0.45
        sample_closes.append(round(base + np.sin(i / 3) * 0.18, 2))
    sample = {"symbol": "PETR4.SA", "closes": sample_closes}
    sample_text = json.dumps(sample, indent=2, ensure_ascii=False)
    html = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    html = apply_dashboard_label_overrides(html)
    html = html.replace("__SAMPLE_PAYLOAD__", sample_text)
    html = html.replace("__MODEL_DIR__", str(MODEL_DIR))
    html = html.replace("__ENABLE_TRAINING_API__", "true" if ENABLE_TRAINING_API else "false")
    return HTMLResponse(html)
