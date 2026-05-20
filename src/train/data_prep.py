import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_windowed_sequences(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: list[str],
    target_col: str,
    train_end_row: int,
    feature_scaler_type: str = "standard",
    target_scaler_type: str = "standard"
):
    X_values = df[feature_cols].values.astype(np.float32)
    y_values = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    closes = df["Close"].values.astype(np.float32)
    dates = df.index.to_numpy()

    def get_scaler(scaler_type: str):
        if scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    feature_scaler = get_scaler(feature_scaler_type)
    feature_scaler.fit(X_values[:train_end_row])
    scaled_X = feature_scaler.transform(X_values).astype(np.float32)
    
    target_scaler = get_scaler(target_scaler_type)
    target_scaler.fit(y_values[:train_end_row])
    scaled_y = target_scaler.transform(y_values).astype(np.float32)

    X, y, last_closes, target_closes, target_dates, target_rows = [], [], [], [], [], []

    for i in range(window_size, len(df)):
        X.append(scaled_X[i - window_size : i])
        y.append(scaled_y[i, 0])
        last_closes.append(closes[i - 1])
        target_closes.append(closes[i])
        target_dates.append(dates[i])
        target_rows.append(i)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
        np.asarray(last_closes, dtype=np.float32),
        np.asarray(target_dates),
        np.asarray(target_rows),
        feature_scaler,
        target_scaler
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
