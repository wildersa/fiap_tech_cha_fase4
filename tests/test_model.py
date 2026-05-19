import torch
import pytest
from src.model import StockLSTM

def test_stock_lstm_initialization():
    input_size = 5
    hidden_size = 32
    num_layers = 2
    dropout = 0.5

    model = StockLSTM(input_size, hidden_size, num_layers, dropout)

    assert model.lstm.input_size == input_size
    assert model.lstm.hidden_size == hidden_size
    assert model.lstm.num_layers == num_layers
    assert model.lstm.dropout == dropout
    assert isinstance(model.head[2], torch.nn.Linear)
    assert model.head[2].in_features == hidden_size
    assert model.head[2].out_features == 1

def test_stock_lstm_initialization_single_layer():
    model = StockLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.2)
    # PyTorch LSTM sets dropout to 0 if num_layers=1
    assert model.lstm.dropout == 0.0

def test_stock_lstm_forward_pass():
    batch_size = 8
    seq_len = 60
    input_size = 1
    hidden_size = 32

    model = StockLSTM(input_size, hidden_size, num_layers=1, dropout=0.0)
    x = torch.randn(batch_size, seq_len, input_size)

    output = model(x)

    assert output.shape == (batch_size, 1)

def test_stock_lstm_different_batch_sizes():
    input_size = 1
    hidden_size = 16
    model = StockLSTM(input_size, hidden_size, num_layers=1, dropout=0.0)

    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 30, input_size)
        output = model(x)
        assert output.shape == (batch_size, 1)
