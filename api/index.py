"""Lightweight Vercel entrypoint for StockLSTM inference.

This serverless API intentionally avoids importing src.api to keep the Vercel
bundle below the Python function size limit. The full local API remains in
src/api.py; this file serves the production inference-only path.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "lstm_petr4"
MODEL_PATH = MODEL_DIR / "model.onnx"

WINDOW_SIZE = 20
TARGET_MODE = "log_returns"
FEATURE_MODE = "single"
FEATURE_COLS = ["Log_Return"]

# Exported from models/lstm_petr4/preprocessor.joblib.
# RobustScaler: (x - center_) / scale_
FEATURE_SCALER_CENTER = np.asarray([0.00131539], dtype=np.float32)
FEATURE_SCALER_SCALE = np.asarray([0.02854158], dtype=np.float32)

# StandardScaler inverse: x * scale_ + mean_
TARGET_SCALER_MEAN = np.asarray([0.00040944], dtype=np.float32)
TARGET_SCALER_SCALE = np.asarray([0.03097156], dtype=np.float32)


class PredictRequest(BaseModel):
    symbol: str = Field("PETR4.SA", description="Código do ativo financeiro.")
    closes: List[float] = Field(
        ...,
        description="Lista cronológica de fechamentos. Envie pelo menos window_size + 1 valores.",
        examples=[[
            30.1, 30.2, 30.5, 30.4, 30.7, 30.9, 31.0, 31.2, 31.5, 31.4,
            31.8, 31.9, 32.1, 32.0, 32.4, 32.5, 32.7, 32.6, 32.9, 33.1, 33.0,
        ]],
    )


app = FastAPI(
    title="Tech Challenge LSTM API",
    description="API serverless de inferência D+1 para o modelo LSTM univariado.",
    version="1.0.0-vercel-light",
)


@lru_cache(maxsize=1)
def load_predictor() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo ONNX não encontrado em {MODEL_PATH}")
    return ort.InferenceSession(str(MODEL_PATH))


def prepare_closes(closes: List[float]) -> np.ndarray:
    arr = np.asarray(closes, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("closes precisa ser uma lista simples de números.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("closes contém valores inválidos.")
    if np.any(arr <= 0):
        raise ValueError("Todos os fechamentos precisam ser maiores que zero.")
    if len(arr) < WINDOW_SIZE + 1:
        raise ValueError(f"Envie pelo menos {WINDOW_SIZE + 1} fechamentos.")
    return arr


def build_latest_window(closes: np.ndarray) -> np.ndarray:
    log_returns = np.log(closes[1:] / closes[:-1]).reshape(-1, 1)
    latest_feat = log_returns[-WINDOW_SIZE:]
    scaled_window = (latest_feat - FEATURE_SCALER_CENTER) / FEATURE_SCALER_SCALE
    return scaled_window.reshape(1, WINDOW_SIZE, 1).astype(np.float32)


def inverse_target_scale(predicted_scaled: float) -> float:
    return float(predicted_scaled * TARGET_SCALER_SCALE[0] + TARGET_SCALER_MEAN[0])


def predict_next(closes_payload: List[float]) -> dict:
    session = load_predictor()
    closes = prepare_closes(closes_payload)
    x = build_latest_window(closes)
    last_close = float(closes[-1])

    ort_inputs = {session.get_inputs()[0].name: x}
    ort_outs = session.run(None, ort_inputs)
    predicted_scaled = float(ort_outs[0][0][0])
    predicted_log_return = inverse_target_scale(predicted_scaled)

    predicted_close = last_close * float(math.exp(predicted_log_return))
    predicted_return_pct = float((math.exp(predicted_log_return) - 1.0) * 100.0)
    change_abs = predicted_close - last_close
    change_pct = (change_abs / last_close) * 100.0 if last_close else 0.0

    return {
        "symbol": "PETR4.SA",
        "prediction_horizon": "next_trading_day",
        "last_close": last_close,
        "predicted_close": float(predicted_close),
        "predicted_log_return": float(predicted_log_return),
        "predicted_return_pct": predicted_return_pct,
        "predicted_change_abs": float(change_abs),
        "predicted_change_pct": float(change_pct),
        "predicted_direction": "alta" if change_abs >= 0 else "queda",
        "runtime": "vercel_light_inference_only",
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "Tech Challenge LSTM API",
        "runtime": "vercel_light_inference_only",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "runtime": "vercel_light_inference_only",
    }


@app.get("/model")
def model_info() -> dict:
    return {
        "model_type": "single",
        "feature_mode": FEATURE_MODE,
        "target_mode": TARGET_MODE,
        "window_size": WINDOW_SIZE,
        "feature_cols": FEATURE_COLS,
        "required_closes": WINDOW_SIZE + 1,
        "runtime": "vercel_light_inference_only",
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        result = predict_next(request.closes)
        result["symbol"] = request.symbol
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/ohlcv")
def predict_ohlcv_disabled() -> dict:
    raise HTTPException(
        status_code=501,
        detail=(
            "O endpoint multivariado foi desabilitado no deploy Vercel leve para reduzir o bundle. "
            "Use /predict com fechamentos ou execute a API completa via Docker/local."
        ),
    )
