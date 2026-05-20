FROM python:3.11-slim

# Evita arquivos pyc e garante output limpo no terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala o poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copia os arquivos de dependencia do poetry
COPY pyproject.toml poetry.lock ./

# Opções de build: prod (produção), dev-cpu (treino CPU leve) ou dev-cuda (treino GPU pesado)
ARG ENV=prod

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
      echo ">> Construindo imagem de DESENVOLVIMENTO CUDA/GPU (Instalando tudo, incluindo PyTorch CUDA)" && \
      poetry install --no-interaction --no-ansi; \
    fi

COPY src/ ./src/
COPY models/ ./models/
COPY assets/ ./assets/

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models/lstm_petr4

EXPOSE 8000

# Inicializa o Uvicorn referenciando o modulo absoluto (src.api)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
