FROM python:3.11-slim

# Evita arquivos pyc e garante output limpo no terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala o poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copia os arquivos de dependencia do poetry
COPY pyproject.toml poetry.lock ./

# Define o parametro de ambiente (prod ou dev). O padrao eh prod (API/Inferencia apenas).
ARG ENV=prod

# Instala dependencias condicionalmente (sem criar virtualenv)
RUN poetry config virtualenvs.create false && \
    if [ "$ENV" = "prod" ]; then \
      echo ">> Construindo imagem de PRODUCAO (Ignorando dependencias de treino: Torch, MLflow...)" && \
      poetry install --only main --no-interaction --no-ansi; \
    else \
      echo ">> Construindo imagem de DESENVOLVIMENTO (Instalando ecosistema completo...)" && \
      poetry install --no-interaction --no-ansi; \
    fi

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models/lstm_petr4

EXPOSE 8000

# Inicializa o Uvicorn referenciando o modulo absoluto (src.api)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
