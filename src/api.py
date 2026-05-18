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
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from model import StockLSTM


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


class PredictRequest(BaseModel):
    symbol: str = "PETR4.SA"
    closes: List[float] = Field(..., description="Lista cronologica de fechamentos anteriores")


app = FastAPI(
    title="Tech Challenge LSTM API",
    description="API para prever o proximo fechamento a partir de fechamentos historicos.",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@lru_cache(maxsize=1)
def load_preprocessor() -> dict:
    path = MODEL_DIR / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessador nao encontrado: {path}. Execute src/train.py primeiro.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_predictor():
    model_path = MODEL_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}. Execute src/train.py primeiro.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = StockLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


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
    required_closes = window_size + 1
    if len(closes) < required_closes:
        raise ValueError(f"Envie pelo menos {required_closes} fechamentos para gerar {window_size} log-retornos.")

    log_returns = np.log(closes[1:] / closes[:-1]).reshape(-1, 1)
    latest_returns = log_returns[-window_size:]
    scaled_window = preprocessor["scaler"].transform(latest_returns)
    return scaled_window.reshape(1, window_size, 1).astype(np.float32)


def predict_next(closes_payload: List[float]) -> dict:
    preprocessor = load_preprocessor()
    model, device = load_predictor()
    closes = prepare_closes(closes_payload)
    X = build_latest_window(closes, preprocessor)
    last_close = float(closes[-1])

    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        predicted_log_return = float(model(x_tensor).detach().cpu().numpy().reshape(-1)[0])

    predicted_close = last_close * float(np.exp(predicted_log_return))
    change_abs = predicted_close - last_close
    change_pct = (change_abs / last_close) * 100 if last_close else 0.0
    return {
        "prediction_horizon": "next_trading_day",
        "last_close": last_close,
        "predicted_close": float(predicted_close),
        "predicted_log_return": predicted_log_return,
        "predicted_return_pct": float((np.exp(predicted_log_return) - 1) * 100),
        "predicted_change_abs": float(change_abs),
        "predicted_change_pct": float(change_pct),
        "predicted_direction": "alta" if change_abs >= 0 else "queda",
    }


@app.middleware("http")
async def collect_app_metrics(request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        APP_METRICS["total_errors"] += 1
        raise

    latency = time.time() - start
    if request.url.path != "/metrics":
        APP_METRICS["total_requests"] += 1
        APP_METRICS["total_latency_sec"] += latency
        APP_METRICS["last_latency_sec"] = latency
        if response.status_code >= 400:
            APP_METRICS["total_errors"] += 1
    response.headers["X-Process-Time"] = f"{latency:.6f}"
    return response


def model_card_payload() -> dict:
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
        "model_type": "LSTM univariada",
        "input": "Lista cronologica de fechamentos; a API calcula log-retornos automaticamente.",
        "target": "Proximo log-retorno.",
        "output": "Preco previsto = ultimo fechamento * exp(log-retorno previsto).",
        "window_size": metadata.get("window_size") or (preprocessor or {}).get("window_size"),
        "symbol": metadata.get("symbol", "PETR4.SA"),
        "metrics": metrics,
        "artifacts_available": {
            "model": (MODEL_DIR / "model.pt").exists(),
            "preprocessor": (MODEL_DIR / "preprocessor.joblib").exists(),
            "metadata": metadata_path.exists(),
        },
    }


def telemetry_payload() -> dict:
    proc = psutil.Process()
    total_requests = APP_METRICS["total_requests"]
    avg_latency_ms = (APP_METRICS["total_latency_sec"] / total_requests * 1000) if total_requests else 0.0
    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "api": {
            "total_requests": total_requests,
            "prediction_requests": APP_METRICS["prediction_requests"],
            "total_errors": APP_METRICS["total_errors"],
            "average_response_time_ms": round(avg_latency_ms, 2),
            "last_response_time_ms": round(APP_METRICS["last_latency_sec"] * 1000, 2),
        },
        "resources": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "process_memory_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            "system_memory_percent": psutil.virtual_memory().percent,
        },
        "model": {
            "model_dir": str(MODEL_DIR),
            "loaded": (MODEL_DIR / "model.pt").exists() and (MODEL_DIR / "preprocessor.joblib").exists(),
        },
        "last_prediction": APP_METRICS["last_prediction"],
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


@app.get("/telemetry")
def telemetry():
    return telemetry_payload()


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
    return HTMLResponse(html)
