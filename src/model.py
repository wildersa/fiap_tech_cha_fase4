"""
Definicao do modelo LSTM usado no Tech Challenge.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    LSTM com LayerNorm e cabeça Linear dedicada para prever features contínuas.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Pega o ultimo elemento da sequencia de cada batch
        out = out[:, -1, :]
        return self.head(out)
