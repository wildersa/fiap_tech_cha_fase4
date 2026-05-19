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

O projeto gerencia dependências usando o **Poetry**. Para garantir a portabilidade total e otimização do tamanho da imagem de produção, as dependências foram divididas:
- **Dependências de Produção (Grupo Principal)**: Apenas o motor ONNX Runtime, FastAPI, Uvicorn, Pandas, Numpy, Scikit-Learn e Joblib para inferência rápida e leve.
- **Dependências de Treinamento (Grupo `train`)**: Contém `torch` (PyTorch CUDA), `mlflow`, `yfinance`, `matplotlib`, `onnx` e `onnxscript`.

Para instalar apenas a versão enxuta de produção (somente inferência):

```bash
poetry install --only main
```

Para instalar todo o ecossistema (API de inferência + pipeline de treinamento + testes):

```bash
poetry install
```

## Pipeline de Treinamento e MLOps

O sistema de treinamento suporta **Múltiplos Modos Dinâmicos** de análise:

- **Feature Modes**:
  - `single` (Univariado): Utiliza apenas o preço de fechamento como entrada. Pode ser consumido tanto no endpoint `/predict` quanto no novo endpoint multivariado `/predict/ohlcv`.
  - `ohlcv`, `ohlcv_returns`, `technical_features`, `custom` (Multivariados): Calculam features a partir de dados históricos OHLCV completos. São totalmente suportados em produção através do novo endpoint `/predict/ohlcv`.
- **Target Modes**: `log_returns` (recomendado) ou `raw_close`.

### Otimizador Avançado (AdamW)

Em vez de utilizar o Adam padrão, a rede é otimizada exclusivamente através do **AdamW**. Esta decisão arquitetural visa desacoplar o decaimento de pesos (L2 regularization) da atualização do gradiente de momento, prevenindo overfittings abruptos e garantindo uma generalização matemática muito superior ao prever o ruído estocástico das ações da Petrobras.

### Champion / Challenger (Promoção Baseada em Validação)

Sempre que qualquer pipeline de treinamento é finalizado, o script compara o novo modelo (Challenger) com o modelo atualmente em produção (Champion) usando o **MAPE de Validação** (`metrics_val["mape_pct"]`).

- A promoção automática é totalmente **independente do `feature_mode`**. Modelos multivariados também são promovidos automaticamente para produção se obtiverem o menor MAPE de validação geral.
- O conjunto de validação é a única métrica usada para guiar a seleção/promoção de modelos.
- O conjunto de teste (`metrics_test`) permanece estritamente isolado para documentação final e model cards, garantindo que seja um verdadeiro "futuro não visto" e prevenindo vazamentos (*validation leakage*).
- Se o novo modelo obtiver um MAPE de validação menor, ele substituirá os arquivos da pasta `models/lstm_petr4` e se tornará o novo Campeão.

### Interface Gráfica ou CLI

Você pode treinar o modelo diretamente pela interface web (`/dashboard` -> Aba Treino) ou executar o script de sintonia fina/busca de hiperparâmetros, registrando tudo no banco local do MLflow:

```bash
$env:PYTHONPATH="." ; poetry run python src/tune.py --n-trials 5 --max-epochs 30
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
- `POST /predict`: Ponto de entrada de inferência univariada (Recebe JSON com array de preços de fechamento e retorna a previsão). Disponível apenas se o modelo campeão atual for `feature_mode="single"`.
- `POST /predict/ohlcv`: Novo ponto de entrada para inferência multivariada (Recebe JSON contendo linhas históricas com dados OHLCV completos: `date`, `open`, `high`, `low`, `close`, `volume`). Suporta qualquer tipo de modelo (univariado ou multivariado) ao calcular as features correspondentes dinamicamente.
- `POST /train`: Dispara um fluxo de treinamento em Background diretamente da API.
- `GET /model-card` & `/runs`: Resumos técnicos das arquiteturas e do MLflow para consumo do Frontend.
- `GET /telemetry`: Retorna as métricas de latência, CPU e RAM que abastecem os gráficos da interface.

## Segurança e Proteção contra RCE (Remote Code Execution)

Devido às vulnerabilidades de segurança do carregamento padrão do PyTorch (`torch.load()`), que depende do módulo `pickle` do Python e permite execução arbitrária de código (mesmo com a flag `weights_only=True`), este projeto adota práticas defensivas rigorosas:

1. **Inferência Segura com ONNX Runtime:** A API de produção (`src/api.py`) carrega e executa o modelo exclusivamente através do **ONNX Runtime** (`model.onnx`). O ONNX Runtime usa um formato estático que não executa código de serialização Python/Pickle, sendo completamente imune a RCEs na etapa de inferência.
2. **Exportação de Pesos via Safetensors:** No pipeline de treinamento (`src/train.py`), além do arquivo `.pt` legado, exportamos os pesos do modelo no formato seguro **`model.safetensors`** usando a biblioteca `safetensors`. Esse formato armazena puramente os tensores binários e metadados, eliminando riscos de desserialização insegura.
3. **Ausência de `torch.load()`:** O código-fonte do projeto não executa chamadas à função vulnerável `torch.load()`.

## Monitoramento e Telemetria

### 1. Telemetria In-Memory (Vercel-ready)

Toda requisição HTTP passa por um middleware que coleta latência, status e carga na CPU/RAM via `psutil`. O armazenamento ocorre em memória via filas (`deque` limitadas), ideal para instâncias serverless efêmeras, sem dependência de banco de dados para demonstrar a saúde básica.

### 2. Prometheus (Scrape contínuo)

A API expõe o endpoint padrão `/metrics` integrado ao `prometheus-fastapi-instrumentator` para coletar requisições globais e monitoramento de performance de longo prazo utilizando instâncias externas (ex: Grafana/Prometheus Stack).

## Rascunhos e anotações

<https://ijisae.org/index.php/IJISAE/article/view/5396/4121>
<https://etasr.com/index.php/ETASR/article/view/12685/5689>
<https://arxiv.org/abs/2303.02223>

---
*Projeto desenvolvido para o Tech Challenge FIAP.*
