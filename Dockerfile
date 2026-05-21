# Definir a imagem base através de ARG. Padrão é python:3.11-slim para prod e dev-cpu.
# Para dev-cuda, utilize --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Evita arquivos pyc e garante output limpo no terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala o poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copia os arquivos de dependencia do poetry
COPY pyproject.toml poetry.lock ./

# Opções de build:
# - prod: somente inferência, empacota os artefatos de MODEL_BUNDLE_DIR dentro da imagem.
# - dev-cpu/dev-cuda: desenvolvimento/treino; os diretórios podem ser sobrescritos por env ou volume no docker run.
ARG ENV=prod
ARG MODEL_BUNDLE_DIR=models
ARG ENABLE_TRAINING_API=false
ARG MODEL_DIR=/app/models/lstm_petr4
ARG MODEL_DIR_MULTI=/app/models/lstm_petr4_multi

# Instala dependencias condicionalmente
RUN poetry config virtualenvs.create false && \
    if [ "$ENV" = "prod" ]; then \
      echo ">> Construindo imagem de PRODUCAO (Sem dependencias de treino)" && \
      poetry install --only main --no-interaction --no-ansi; \
    elif [ "$ENV" = "dev-cpu" ]; then \
      echo ">> Construindo imagem de DESENVOLVIMENTO CPU (Instalando dependencias + Torch CPU)" && \
      poetry install --no-interaction --no-ansi --without train && \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir "mlflow>=3.0.0,<4.0.0" "yfinance>=0.2.50" "matplotlib>=3.9.0" "onnx>=1.21.0" "onnxscript>=0.7.0"; \
    else \
      echo ">> Construindo imagem de DESENVOLVIMENTO CUDA/GPU (Ignorando Torch do PyPI, assumindo base image PyTorch)" && \
      poetry install --no-interaction --no-ansi --without train && \
      pip install --no-cache-dir "mlflow>=3.0.0,<4.0.0" "yfinance>=0.2.50" "matplotlib>=3.9.0" "onnx>=1.21.0" "onnxscript>=0.7.0"; \
    fi

COPY src/ ./src/
# No modo prod/inferencia, esta pasta vira parte do pacote imutavel.
# Ela deve conter ao menos lstm_petr4/ e lstm_petr4_multi/ com model.onnx e preprocessor.joblib.
COPY ${MODEL_BUNDLE_DIR}/ ./models/
COPY assets/ ./assets/

ENV PYTHONPATH=/app
ENV ENABLE_TRAINING_API=${ENABLE_TRAINING_API}
ENV MODEL_DIR=${MODEL_DIR}
ENV MODEL_DIR_MULTI=${MODEL_DIR_MULTI}

EXPOSE 8000

# Inicializa o Uvicorn referenciando o modulo absoluto (src.api)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
