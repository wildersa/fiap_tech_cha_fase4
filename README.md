# StockPredict-LSTM: Bolsa de Valores com Deep Learning

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange.svg)

Este projeto faz parte do **Tech Challenge - Fase 4** da Pós-Tech FIAP (Machine Learning Engineering). O objetivo principal é desenvolver e implantar um modelo de Deep Learning capaz de prever os preços de fechamento de ações da bolsa de valores utilizando redes neurais recorrentes do tipo **LSTM (Long Short-Term Memory)**.

## 🎯 Objetivo

Desenvolver um pipeline completo de Machine Learning, desde a coleta de dados financeiros em tempo real até o deploy de uma API escalável, focando na predição de séries temporais.

## 🚀 Funcionalidades

- **Coleta Automatizada**: Integração com a API do Yahoo Finance (`yfinance`) para download de dados históricos.
- **Storage Flexível**: Suporte para salvamento de dados Local, AWS S3 ou Azure Blob Storage.
- **Modelo de Deep Learning**: Arquitetura LSTM otimizada para capturar dependências temporais em séries financeiras.
- **API REST**: Interface para consumo das predições desenvolvida em FastAPI/Flask.
- **Monitoramento**: Rastreamento de performance e recursos em tempo real.

## 🏗️ Estrutura do Projeto

```text
├── data/               # Conjuntos de dados (brutos e processados)
├── docs/               # Documentação e PDFs de referência
├── models/             # Pesos dos modelos treinados (.pth ou .h5)
├── notebooks/          # Experimentos e Análise Exploratória (EDA)
├── scripts/            # Scripts utilitários
├── src/                # Código-fonte principal
│   ├── data/           # Scripts de coleta e ingestão
│   ├── features/       # Pré-processamento e engenharia de features
│   ├── models/         # Definição e treinamento da rede LSTM
│   └── api/            # API REST para deploy do modelo
├── tests/              # Testes unitários e de integração
├── pyproject.toml      # Gerenciamento de dependências
└── README.md
```

## 🛠️ Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/fiap-tech-challenge-fase4.git
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   # Ou use poetry se disponível
   poetry install
   ```

## 💻 Uso

### Coleta de Dados

Para coletar dados de uma ação específica:
```bash
python src/data/collect_data.py --symbol "PETR4.SA" --interval "1h" --storage "local"
```

## 📊 Avaliação do Modelo

O desempenho do modelo será medido utilizando as seguintes métricas:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---
*Desenvolvido como parte do currículo da Pós-Tech FIAP.*
