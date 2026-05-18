# Tech Challenge Fase 4 - LSTM para previsão de fechamento

Projeto didático para treinar uma rede LSTM com dados históricos de ações e servir o modelo em uma API FastAPI containerizada.

## Estrutura simplificada

```text
src/
  data_loader.py            # coleta e preparação da série de fechamento
  model.py                  # definição da LSTM
  train.py                 # pipeline completo de treinamento
  api.py                   # API de inferência, dashboard e telemetria
  dashboard.html           # interface web do dashboard
  pipeline_sandbox.ipynb   # notebook com as mesmas etapas do treino
models/
  lstm_petr4/
    model.pt
    preprocessor.joblib
    metrics.json
    metadata.json
    history.json
    test_predictions.csv
    model_performance.png
```

## Instalação

```bash
pip install -r requirements.txt
```

Com Poetry:

```bash
poetry install
```

## Treinamento

O treinamento é offline. Ele baixa dados via `yfinance`, usa a coluna `Close`, transforma os fechamentos em log-retornos, monta janelas temporais, treina a LSTM, avalia com MAE/RMSE/MAPE e salva os artefatos em `models/lstm_petr4`.
O pipeline também registra parâmetros, perdas por época, métricas finais e artefatos no MLflow.

```bash
PYTHONPATH=src python src/train.py \
  --symbol PETR4.SA \
  --start-date 2018-01-01 \
  --output-dir models/lstm_petr4
```

Também é possível treinar com CSV local:

```bash
PYTHONPATH=src python src/train.py \
  --csv data/raw/petr4.csv \
  --symbol PETR4.SA \
  --output-dir models/lstm_petr4
```

O CSV deve conter pelo menos:

```text
Date,Close
```

## API de inferência

Depois de treinar, suba a API:

```bash
PYTHONPATH=src uvicorn api:app --reload
```

Endpoints principais:

- `GET /docs`: Swagger do FastAPI.
- `POST /predict`: recebe fechamentos recentes em JSON e retorna a previsão do próximo fechamento.
- `GET /dashboard`: tela visual para colar JSON, executar inferência e ver gráfico.
- `GET /model-card`: resumo técnico do modelo e dos artefatos.
- `GET /telemetry`: telemetria JSON da API, incluindo chamadas, latência, erros, CPU e memória.
- `GET /metrics`: métricas Prometheus expostas pelo `prometheus-fastapi-instrumentator`.
- `GET /health`: status básico da API.

Payload de inferência:

```json
{
  "symbol": "PETR4.SA",
  "closes": [45.10, 45.20, 45.00, 45.35]
}
```

Com a configuração padrão, envie pelo menos 61 fechamentos: a API calcula 60 log-retornos, aplica o scaler salvo no treino e converte o log-retorno previsto de volta para preço.

## Containerização

A imagem Docker empacota a API de inferência e os artefatos treinados. O treinamento não roda dentro do container por padrão.

```bash
docker build -t tech-challenge-lstm .
docker run --rm -p 8000:8000 tech-challenge-lstm
```

## Monitoramento

O projeto usa duas formas de observabilidade:

- `/metrics`: endpoint técnico em formato Prometheus, gerado pelo `prometheus-fastapi-instrumentator`.
- `/dashboard` e `/telemetry`: visualização didática das métricas operacionais.
- MLflow: acompanhamento dos experimentos de treino offline.

As métricas de produção medem a saúde da API: latência, total de requisições, erros, CPU e RAM. A eficácia do modelo é medida offline após o treino, no conjunto de teste, usando MAE, RMSE e MAPE.

Para abrir a interface do MLflow:

```bash
mlflow ui
```

Por padrão, acesse `http://localhost:5000`.

Referências:
Aulas
https://github.com/FIAP/Pos_Tech_MLET
https://www.kaggle.com/code/farzadnekouei/gold-price-prediction-lstm-96-accuracy
