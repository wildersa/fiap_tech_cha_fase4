from __future__ import annotations

import os
from dataclasses import dataclass
from src.features import FEATURE_REGISTRY, FEATURE_PRESETS


@dataclass
class TrainConfig:
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
    max_epochs: int = 150
    patience: int = 20
    grad_clip: float | None = 1.0
    output_dir: str = os.getenv("MODEL_DIR", "models/lstm_petr4")
    seed: int = 42
    target_mode: str = "log_returns"
    feature_mode: str = "single"
    feature_scaler_type: str = "standard"
    target_scaler_type: str = "standard"
    device: str = "auto"
    parent_run_id: str | None = None
    selected_features: list[str] | None = None
    feature_preset: str | None = None


def resolve_target_column(target_mode: str) -> str:
    mapping = {"log_returns": "Log_Return", "raw_close": "Close", "returns": "Return"}
    if target_mode not in mapping:
        raise ValueError(f"Target mode desconhecido: {target_mode}")
    return mapping[target_mode]


def resolve_feature_columns(cfg: TrainConfig, target_col: str) -> list[str]:
    valid_names = {f["name"] for f in FEATURE_REGISTRY}
    
    if cfg.feature_mode == "single":
        return [target_col]
    elif cfg.feature_mode == "ohlcv":
        return ["Open", "High", "Low", "Close", "Volume"]
    elif cfg.feature_mode == "ohlcv_returns":
        return ["Open", "High", "Low", "Close", "Volume", "Log_Return"]
    elif cfg.feature_mode == "technical_features":
        preset = cfg.feature_preset or "technical_complete"
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"Preset desconhecido: {preset}")
        features = cfg.selected_features or FEATURE_PRESETS[preset]
    elif cfg.feature_mode == "custom":
        if not cfg.selected_features:
            raise ValueError("O modo 'custom' exige que 'selected_features' nao esteja vazio.")
        features = cfg.selected_features
    else:
        raise ValueError(f"Feature mode desconhecido: {cfg.feature_mode}")
        
    invalid = [f for f in features if f not in valid_names]
    if invalid:
        raise ValueError(f"Features invalidas/desconhecidas no registro: {invalid}")
    return list(features)
