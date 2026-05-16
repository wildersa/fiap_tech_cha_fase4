# Tech Challenge Fase 4 — LSTM para previsão de fechamento

Código base modular para o Tech Challenge: coleta/preprocessamento, treino LSTM, avaliação, salvamento dos artefatos e API FastAPI.

## Instalação com Poetry

```bash
poetry install
```

Se for usar GPU NVIDIA, instale o PyTorch CUDA compatível com seu ambiente antes de rodar o treino.

## Treinar usando CSV local

O CSV precisa conter pelo menos:

```text
Date, Open, High, Low, Close, Volume
```

Também aceita nomes em minúsculo.

```bash
poetry run python -m stock_lstm.train \
  --csv data/raw/petr4_long_term.csv \
  --symbol PETR4.SA \
  --output-dir models/lstm_petr4
```

## Treinar usando yfinance

```bash
poetry run python -m stock_lstm.train \
  --symbol PETR4.SA \
  --start-date 2018-01-01 \
  --output-dir models/lstm_petr4
```

## Rodar API

```bash
poetry run uvicorn stock_lstm.api:app --reload
```

## Endpoint

```http
POST /predict
```

A API recebe dados recentes, monta a última janela temporal e prevê o próximo fechamento.

O modelo não é retreinado na API. A API só faz inferência.

## Teste via curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Estrutura

```text
src/stock_lstm/
  api.py
  config.py
  data_loader.py
  model.py
  predict.py
  preprocessing.py
  train.py
  utils.py
```

## Observação

O baseline “amanhã = último fechamento” é calculado junto com a LSTM porque, em séries de preço de ações, esse baseline costuma ser muito competitivo.
