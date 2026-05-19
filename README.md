# Tech Challenge Fase 4 - LSTM & MLOps Pipeline

Projeto avançado para previsão de preços de fechamento de ações utilizando Deep Learning (LSTM), arquitetura multivariável dinâmica e integração total com MLOps (MLflow + ONNX). O sistema é empacotado para deploy serverless (FastAPI) com um Dashboard Premium integrado.

## Arquitetura Simplificada

```text
src/
  api.py                   # API de inferência, dashboard web e telemetria.
  train.py                 # Pipeline dinâmico de Treinamento (Feature/Target modes).
  data_loader.py           # Coleta (yfinance) e cálculo de feature engineering.
  model.py                 # Definição modular da arquitetura PyTorch LSTM.
  dashboard.html           # Interface web Premium (Inferencia, Treino, MLflow, Telemetria).
scripts/
  run_experiments.py       # Script de Grid Search automatizado p/ testes massivos.
models/
  lstm_petr4/              # Diretório do Modelo "Campeão" atual (Empacotado no Docker)
    model.onnx             # Modelo exportado (Inferencia Ultrarrápida)
    preprocessor.joblib    # Dicionários independentes de Scalers (Features vs Target)
    metadata.json          # Artifact manifest e hiperparâmetros
```

## Instalação (Poetry)

O projeto gerencia dependências inteligentemente usando o **Poetry**.

Para Desenvolvimento / Treinamento (Instala todo o ecossistema de Deep Learning):
```bash
poetry install
```

Para Produção / Serverless (Apenas dependências de inferência como FastAPI e ONNX):
```bash
poetry install --only main
```

## Pipeline de Treinamento e MLOps

O sistema de treinamento foi refatorado para suportar **Múltiplos Modos Dinâmicos**:
- **Feature Modes**: `single` (apenas target), `ohlcv`, `ohlcv_returns`, `technical_features`.
- **Target Modes**: `log_returns` (recomendado) ou `raw_close`.

### Otimizador Avançado (AdamW)
Em vez de utilizar o Adam padrão, a rede é otimizada exclusivamente através do **AdamW**. Esta decisão arquitetural visa desacoplar o decaimento de pesos (L2 regularization) da atualização do gradiente de momento, prevenindo overfittings abruptos e garantindo uma generalização matemática muito superior ao prever o ruído estocástico das ações da Petrobras.

### Champion / Challenger (Promoção Automática)
Sempre que um treinamento é finalizado, o script compara o novo modelo (Challenger) com o modelo atualmente em produção (Champion). Se o novo modelo obtiver um `MAPE` menor na base de teste, ele substituirá os arquivos da pasta `models/lstm_petr4` e se tornará o novo Campeão.

### Interface Gráfica ou CLI
Você pode treinar o modelo diretamente pela interface web (`/dashboard` -> Aba Treino) ou executar o script automatizado de Grid Search, que iterará sobre um roteiro de testes e registrará tudo no MLflow:

```bash
poetry run python scripts/run_experiments.py
```

*Para visualizar os resultados no MLflow UI:*
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Containerização Modular (Docker)

O `Dockerfile` utiliza uma arquitetura robusta e adaptativa baseada em **Build Arguments**, projetada especificamente para contornar limites agressivos de tamanho de bundle em serviços Serverless (ex: Vercel).

**1. Build Leve (Para Produção/Inferência):**
O Poetry ignora módulos pesados (`torch`, `mlflow`, etc.) instalando apenas o motor ONNX. O modelo treinado já é empacotado (`COPY models/`) para gerar um contêiner 100% Stateless de inicialização instantânea.
```bash
docker build --build-arg ENV=prod -t stock-api:prod .
```

**2. Build Pesado (Worker de Treinamento/GPU):**
Para rodar jobs de treino pesados no Kubernetes ou na Cloud.
```bash
docker build --build-arg ENV=dev -t stock-api:dev .
```

## Endpoints e Dashboard Premium

Para subir a API de inferência:
```bash
poetry run uvicorn src.api:app --reload
```

- `GET /dashboard`: O coração visual do projeto! Tela para rodar inferências, treinar novos modelos, visualizar a Loss Curve, tabela ao vivo do MLflow e gráficos em tempo real.
- `POST /predict`: Ponto de entrada de inferência (Recebe JSON de série histórica e retorna a previsão).
- `POST /train`: Dispara um fluxo de treinamento em Background diretamente da API.
- `GET /model-card` & `/runs`: Resumos técnicos das arquiteturas e do MLflow para consumo do Frontend.
- `GET /telemetry`: Retorna as métricas de latência, CPU e RAM que abastecem os gráficos da interface.

## Monitoramento e Telemetria

### 1. Telemetria In-Memory (Vercel-ready)
Toda requisição HTTP passa por um middleware que coleta latência, status e carga na CPU/RAM via `psutil`. O armazenamento ocorre em memória via filas (`deque` limitadas), ideal para instâncias serverless efêmeras, sem dependência de banco de dados para demonstrar a saúde básica.

### 2. Prometheus (Scrape contínuo)
A API expõe o endpoint padrão `/metrics` integrado ao `prometheus-fastapi-instrumentator` para coletar requisições globais e monitoramento de performance de longo prazo utilizando instâncias externas (ex: Grafana/Prometheus Stack).

---
*Projeto desenvolvido para o Tech Challenge FIAP.*
