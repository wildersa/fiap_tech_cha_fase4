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
        description="Tipo de features: 'single' (apenas preço) ou 'ohlcv_returns' (OHLCV completo).",
        examples=["single"]
    )
    feature_scaler_type: str = Field(
        "standard",
        description="Tipo do normalizador das features: 'standard' ou 'minmax'.",
        examples=["standard"]
    )
    target_scaler_type: str = Field(
        "standard",
        description="Tipo do normalizador do alvo (target): 'standard' ou 'minmax'.",
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

    # Carrega textos qualitativos de model_card_template.json se existir
    template_path = Path(__file__).resolve().parent / "model_card_template.json"
    text_defaults = {
        "model_name": "StockLSTM",
        "model_description": "Modelo Deep Learning baseado em LSTM configurável (Univariado/Multivariado) projetado para previsão de séries temporais de ativos financeiros.",
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

    # Monta a estrutura inspirada no AWS SageMaker Model Cards
    return {
        "model_overview": {
            "model_name": text_defaults["model_name"],
            "model_description": text_defaults["model_description"],
            "model_version": "1.0.0",
            "model_status": "Approved" if (MODEL_DIR / "model.onnx").exists() else "Draft",
            "risk_rating": "Medium",
            "intended_uses": text_defaults["intended_uses"]
        },
        "training_details": {
            "symbol": metadata.get("symbol", "PETR4.SA"),
            "data_source": metadata.get("data_source", "yfinance"),
            "window_size": metadata.get("window_size") or (preprocessor or {}).get("window_size") or 60,
            "feature_mode": metadata.get("feature_mode") or (preprocessor or {}).get("feature_mode") or "single",
            "target_mode": metadata.get("target_mode") or (preprocessor or {}).get("target_mode") or "log_returns",
            "mlflow_run_id": metadata.get("run_id", "Treinamento Local / Sem Run ID"),
            "training_observations": text_defaults["training_observations"]
        },
        "evaluation_details": {
            "evaluation_dataset": text_defaults["evaluation_dataset"],
            "metrics": metrics,
            "artifacts_available": {
                "model_onnx": (MODEL_DIR / "model.onnx").exists(),
                "model_safetensors": (MODEL_DIR / "model.safetensors").exists(),
                "preprocessor_joblib": (MODEL_DIR / "preprocessor.joblib").exists(),
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
    description="Retorna a ficha técnica detalhada do modelo em formato JSON, estruturada no padrão do AWS SageMaker Model Cards (seções Model Overview, Training Details, Evaluation Details e Additional Information).",
    response_description="Payload formatado no padrão AWS SageMaker Model Cards contendo governança, parâmetros de treino, métricas de validação e limites de responsabilidade.",
    tags=["Monitoramento & Diagnóstico"]
)
def model_card():
    return model_card_payload()


@app.get(
    "/model-image",
    summary="Gráfico de Performance do Modelo",
    description="Retorna o gráfico de avaliação offline em formato PNG (gerado durante a fase de validação pós-treino do modelo de produção ativo).",
    response_description="Arquivo de imagem PNG do gráfico de performance e curva de perdas.",
    tags=["Monitoramento & Diagnóstico"]
)
def get_model_image():
    image_path = MODEL_DIR / "model_performance.png"
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


@app.post(
    "/predict",
    summary="Realizar Predição (Inferência D+1)",
    description="Recebe uma série cronológica recente de fechamentos do ativo (mínimo de window_size + 1), extrai os retornos logarítmicos ou absolutos em tempo real, normaliza a janela de lookback usando os parâmetros salvos no preprocessor, executa a inferência acelerada na sessão ONNX Runtime e decodifica a previsão para o preço de fechamento do dia seguinte (D+1). Também computa a variação projetada absoluta/percentual e a direção (alta/queda).",
    response_description="Previsão detalhada para o próximo fechamento (D+1) com base no modelo ativo.",
    tags=["Inferência"]
)
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
    html = html.replace("__SAMPLE_PAYLOAD__", sample_text)
    html = html.replace("__MODEL_DIR__", str(MODEL_DIR))
    html = html.replace("__ENABLE_TRAINING_API__", "true" if ENABLE_TRAINING_API else "false")
    return HTMLResponse(html)
