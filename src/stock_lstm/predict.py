from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .model import build_model
from .preprocessing import WindowPreprocessor, build_feature_frame


class Predictor:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)

        checkpoint_path = self.artifact_dir / "model.pt"
        preprocessor_path = self.artifact_dir / "preprocessor.pkl"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {checkpoint_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessador não encontrado: {preprocessor_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model = build_model(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.preprocessor = WindowPreprocessor.load(str(preprocessor_path))

    def predict_next(self, recent_prices: pd.DataFrame) -> dict:
        df_feat = build_feature_frame(recent_prices)

        X, anchor_price, last_close = self.preprocessor.build_latest_window(df_feat)

        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred_ratio = float(self.model(x_tensor).detach().cpu().numpy().reshape(-1)[0])

        predicted_close = pred_ratio * anchor_price

        return {
            "prediction_horizon": "next_trading_day",
            "anchor_price": float(anchor_price),
            "last_close": float(last_close),
            "predicted_close": float(predicted_close),
            "predicted_ratio": float(pred_ratio),
        }
