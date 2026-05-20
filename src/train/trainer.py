import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


def predict_numpy(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(preds).reshape(-1)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Evita divisão por zero no MAPE
    safe_true = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100)
    
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": mape,
    }


def directional_accuracy(y_true, y_pred, last_close) -> float:
    true_dir = np.sign(np.asarray(y_true) - np.asarray(last_close))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(last_close))
    return float(np.mean(true_dir == pred_dir) * 100)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)
