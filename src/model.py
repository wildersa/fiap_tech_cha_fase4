"""
Definição da Arquitetura de Deep Learning (LSTM) para o Tech Challenge.

Este arquivo contém a classe StockLSTM, implementada em PyTorch, que serve como
o núcleo do modelo de previsão de séries temporais de ativos financeiros.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Arquitetura LSTM customizada para séries temporais financeiras.
    
    Características:
    - Camada LSTM empilhável.
    - LayerNorm para estabilização de gradientes em janelas temporais.
    - Cabeça Linear para regressão (previsão de valores contínuos).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Inicializa o modelo.

        Args:
            input_size: Número de features por passo temporal.
            hidden_size: Número de neurônios na camada oculta da LSTM.
            num_layers: Quantidade de camadas LSTM empilhadas.
            dropout: Fator de regularização Dropout (aplicado se num_layers > 1).
        """
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
        """
        Executa a passagem para frente (forward pass).

        Args:
            x: Tensor de entrada com formato (batch_size, window_size, input_size).

        Returns:
            Tensor de saída com a previsão (batch_size, 1).
        """
        out, _ = self.lstm(x)
        # Pega apenas o último passo temporal da sequência processada
        out = out[:, -1, :]
        return self.head(out)
