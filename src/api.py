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

from __future__ import annotations

import json
import os
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import psutil
import onnxruntime as ort
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
}
TELEMETRY_HISTORY = deque(maxlen=100)


class PredictRequest(BaseModel):
    symbol: str = "PETR4.SA"
    closes: List[float] = Field(..., description="Lista cronologica de fechamentos anteriores")


class TrainRequest(BaseModel):
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
    max_epochs: int = 100
    patience: int = 20
    target_mode: str = "log_returns"
    feature_mode: str = "single"
    feature_scaler_type: str = "standard"
    target_scaler_type: str = "standard"
    grad_clip: float | None = 1.0
    device: str = "auto"
    parent_run_id: str | None = None


app = FastAPI(
    title="Tech Challenge LSTM API",
    description="API para prever o proximo fechamento a partir de fechamentos historicos.",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# Garantir que a API aponte para o MLflow correto localmente
if ENABLE_TRAINING_API:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    if not mlflow_uri.startswith(("file://", "http://", "https://")):
        mlflow_path = Path(mlflow_uri)
        if not mlflow_path.is_absolute():
            mlflow_path = Path(__file__).resolve().parent.parent / mlflow_path
        mlflow_uri = mlflow_path.as_uri()
    mlflow.set_tracking_uri(mlflow_uri)


def sync_best_model_from_mlflow() -> None:
    """
    MLOps Automatic Promotion:
    Busca em todas as runs de todas as experiencias do MLflow o modelo com o menor MAPE
    e o promove como o modelo ativo (copiando seus artefatos para MODEL_DIR),
    se for melhor do que o modelo atualmente em disco.
    """
    if not ENABLE_TRAINING_API:
        return

    try:
        from mlflow.tracking import MlflowClient
        import shutil
        client = MlflowClient()
        experiments = client.search_experiments()
        best_run = None
        best_mape = float("inf")

        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for r in runs:
                # Tenta obter a métrica de teste ou validação
                mape = (
                    r.data.metrics.get("lstm_mape_pct") or 
                    r.data.metrics.get("test_lstm_mape_pct") or 
                    r.data.metrics.get("test_lstm_mape")
                )
                if mape is not None and mape > 0:
                    if mape < best_mape:
                        best_mape = mape
                        best_run = r

        if best_run is not None:
            # Verifica o modelo atual no disco
            current_mape = float("inf")
            metrics_path = MODEL_DIR / "metrics.json"
            if metrics_path.exists():
                try:
                    current_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    current_mape = (
                        current_metrics.get("lstm_test", {}).get("mape_pct") or
                        current_metrics.get("test_lstm_mape_pct") or
                        current_metrics.get("lstm_mape_pct") or
                        float("inf")
                    )
                except Exception:
                    pass

            # Se o modelo do MLflow for estritamente melhor (com margem de precisão para evitar float inaccuracies), promove!
            if best_mape < (current_mape - 1e-6):
                print(f"[MLOps] Novo Campeao detectado no MLflow (Run {best_run.info.run_id}) com MAPE {best_mape:.4f}% (anterior em disco: {current_mape:.4f}%)")
                
                # Garante que MODEL_DIR existe
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                
                # Baixa os artefatos da run
                local_path = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id)
                src_path = Path(local_path)
                
                # Copia os arquivos relevantes
                copied_any = False
                for filename in ["model.onnx", "model.pt", "model.safetensors", "preprocessor.joblib", "metadata.json", "metrics.json", "model_performance.png"]:
                    src_file = src_path / filename
                    if src_file.exists():
                        shutil.copy(str(src_file), str(MODEL_DIR / filename))
                        copied_any = True
                
                if copied_any:
                    # Limpa caches da API
                    load_preprocessor.cache_clear()
                    load_predictor.cache_clear()
                    print(f"[MLOps] Modelo do MLflow (Run {best_run.info.run_id}) promovido com sucesso para {MODEL_DIR}.")
                else:
                    print(f"[MLOps] Aviso: Nenhum artefato copiado de {src_path}")
            else:
                print(f"[MLOps] O modelo em disco ({current_mape:.4f}%) ja e o melhor ou empata com o melhor do MLflow ({best_mape:.4f}%)")
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
        raise ValueError(f"Feature mode '{feature_mode}' nao e suportado para inferencia em tempo real na API simplificada. Use feature_mode='single'.")

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
            "latency_ms": latency_ms,
            "cpu_percent": cpu,
            "memory_mb": mem_mb
        }
        TELEMETRY_HISTORY.append(snapshot)
        if response:
            response.headers["X-Process-Time"] = f"{latency:.6f}"


def model_card_payload() -> dict:
    # MLOps: Garante que estamos exibindo e executando o melhor modelo do MLflow
    sync_best_model_from_mlflow()

    metadata_path = MODEL_DIR / "metadata.json"
    metrics_path = MODEL_DIR / "metrics.json"
    metadata = {}
    metrics = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    preprocessor = None
    if (MODEL_DIR / "preprocessor.joblib").exists():
        try:
            preprocessor = load_preprocessor()
        except Exception:
            preprocessor = None

    return {
        "model_type": "LSTM Multivariada (v8)",
        "input": "Lista cronológica de preços (Close, Open, High, Low) e indicadores calculados automaticamente pela API.",
        "target": metadata.get("target_mode", "N/A"),
        "output": "Preço de fechamento projetado e variação percentual / absoluta.",
        "window_size": metadata.get("window_size") or (preprocessor or {}).get("window_size"),
        "symbol": metadata.get("symbol", "PETR4.SA"),
        "run_id": metadata.get("run_id", "Local / Sem MLflow"),
        "metrics": metrics,
        "artifacts_available": {
            "model": (MODEL_DIR / "model.onnx").exists(),
            "model_onnx": (MODEL_DIR / "model.onnx").exists(),
            "preprocessor": (MODEL_DIR / "preprocessor.joblib").exists(),
            "metadata": metadata_path.exists(),
        },
    }


def telemetry_payload() -> dict:
    total_requests = APP_METRICS["total_requests"]
    avg_latency_ms = (APP_METRICS["total_latency_sec"] / total_requests * 1000) if total_requests else 0.0

    # Recupera os dados diretamente do registro do Prometheus para provar a instrumentação e a fonte da verdade!
    cpu = REGISTRY.get_sample_value('api_cpu_usage_percent') or 0.0
    mem_bytes = REGISTRY.get_sample_value('api_memory_usage_bytes') or 0.0
    sys_mem = REGISTRY.get_sample_value('api_system_memory_percent') or 0.0
    last_lat = REGISTRY.get_sample_value('api_last_latency_ms') or 0.0
    errs = REGISTRY.get_sample_value('api_errors_total_total') or REGISTRY.get_sample_value('api_errors_total') or 0.0

    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "api": {
            "total_requests": total_requests,
            "prediction_requests": APP_METRICS["prediction_requests"],
            "total_errors": int(errs),
            "average_response_time_ms": round(avg_latency_ms, 2),
            "last_response_time_ms": round(last_lat, 2),
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


@app.get("/health")
def health():
    return {"status": "ok", "model_dir": str(MODEL_DIR)}


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/model-card")
def model_card():
    return model_card_payload()


@app.get("/model-image")
def get_model_image():
    image_path = MODEL_DIR / "model_performance.png"
    if image_path.exists():
        return FileResponse(str(image_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Imagem não encontrada.")


@app.get("/telemetry")
def telemetry():
    return telemetry_payload()


@app.get("/runs")
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


@app.post("/train")
def train_model(req: TrainRequest):
    if not ENABLE_TRAINING_API:
        raise HTTPException(status_code=501, detail="API de treinamento desabilitada ou dependencias (mlflow) nao instaladas.")
    cfg = TrainConfig(**req.model_dump())
    try:
        results = run_training_pipeline(cfg)
        
        # Limpa o cache da API para carregar o novo modelo automaticamente se ele foi promovido
        load_predictor.cache_clear()
        load_preprocessor.cache_clear()
        
        return {
            "status": "success", 
            "metrics": results["metrics"], 
            "output_dir": results["output_dir"],
            "message": "Treinamento finalizado com sucesso."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante o treinamento: {str(e)}")


@app.post("/predict")
def predict(request: PredictRequest):
    preprocessor = None
    try:
        preprocessor = load_preprocessor()
    except Exception:
        pass
    min_closes = int((preprocessor or {}).get("window_size", 60)) + 1
    if len(request.closes) < min_closes:
        raise HTTPException(status_code=400, detail=f"Envie pelo menos {min_closes} fechamentos.")

    try:
        result = predict_next(request.closes)
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


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    sample_closes = []
    for i in range(65):
        base = 43.0 + i * 0.035 + np.sin(i / 4) * 0.45
        sample_closes.append(round(base + np.sin(i / 3) * 0.18, 2))
    sample = {"symbol": "PETR4.SA", "closes": sample_closes}
    sample_text = json.dumps(sample, indent=2, ensure_ascii=False)
    html = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__SAMPLE_PAYLOAD__", sample_text)
    html = html.replace("__MODEL_DIR__", str(MODEL_DIR))
    html = html.replace("__ENABLE_TRAINING_API__", "true" if ENABLE_TRAINING_API else "false")
    return HTMLResponse(html)
