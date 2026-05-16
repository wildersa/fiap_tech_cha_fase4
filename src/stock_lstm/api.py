from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import Predictor


MODEL_DIR = os.getenv("MODEL_DIR", "models/lstm_petr4")

app = FastAPI(
    title="Tech Challenge LSTM API",
    description="API para previsão do próximo fechamento de ação com LSTM.",
    version="0.1.0",
)


class PricePoint(BaseModel):
    date: str
    open: float = Field(..., description="Preço de abertura")
    high: float = Field(..., description="Maior preço do pregão")
    low: float = Field(..., description="Menor preço do pregão")
    close: float = Field(..., description="Preço de fechamento")
    volume: float = Field(..., description="Volume negociado")


class PredictRequest(BaseModel):
    symbol: str = "PETR4.SA"
    historical_prices: List[PricePoint]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(MODEL_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/predict")
def predict(request: PredictRequest):
    if len(request.historical_prices) < 40:
        raise HTTPException(
            status_code=400,
            detail="Envie dados históricos suficientes. Recomenda-se pelo menos 60 pregões.",
        )

    rows = []
    for p in request.historical_prices:
        rows.append({
            "Date": p.date,
            "Open": p.open,
            "High": p.high,
            "Low": p.low,
            "Close": p.close,
            "Volume": p.volume,
        })

    df = pd.DataFrame(rows)

    try:
        result = get_predictor().predict_next(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "symbol": request.symbol,
        "prediction_horizon": result["prediction_horizon"],
        "last_close": result["last_close"],
        "predicted_close": result["predicted_close"],
        "baseline_close": result["last_close"],
        "model_version": "v1",
    }
