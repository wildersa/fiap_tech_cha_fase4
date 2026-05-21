"""
Exports leves do pacote de treino.

O modo somente inferencia importa `src.train.champion_selection` pela API, mas
nao deve depender de PyTorch. Por isso, objetos que puxam `torch` ficam lazy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from src.train.config import TrainConfig, resolve_target_column, resolve_feature_columns


_LAZY_EXPORTS = {
    "add_features": ("src.data_loader", "add_features"),
    "load_csv": ("src.data_loader", "load_csv"),
    "load_yfinance": ("src.data_loader", "load_yfinance"),
    "set_seed": ("src.train.data_prep", "set_seed"),
    "create_windowed_sequences": ("src.train.data_prep", "create_windowed_sequences"),
    "make_loader": ("src.train.data_prep", "make_loader"),
    "regression_metrics": ("src.train.trainer", "regression_metrics"),
    "directional_accuracy": ("src.train.trainer", "directional_accuracy"),
    "predict_numpy": ("src.train.trainer", "predict_numpy"),
    "NpEncoder": ("src.train.trainer", "NpEncoder"),
    "write_json": ("src.train.artifacts", "write_json"),
    "plot_performance": ("src.train.artifacts", "plot_performance"),
    "run_training_pipeline": ("src.train.pipeline", "run_training_pipeline"),
    "main": ("src.train.pipeline", "main"),
}


__all__ = [
    "TrainConfig",
    "resolve_target_column",
    "resolve_feature_columns",
    "add_features",
    "load_csv",
    "load_yfinance",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'src.train' has no attribute {name!r}")

    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
