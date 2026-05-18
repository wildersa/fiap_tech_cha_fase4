"""
Definicao do modelo LSTM usado no Tech Challenge.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    LSTM univariada para prever o proximo log-retorno de uma serie de fechamentos.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
