# 📈 Previsão do Preço de Ações com LSTM | Deep Learning Pipeline

Este projeto implementa um pipeline completo de *deep learning* para prever o preço de fechamento de ações utilizando uma rede neural LSTM. \
O pipeline abrange desde a coleta de dados brutos até a disponibilização do modelo em produção, passando por versionamento, *serving* via API REST e monitoramento em tempo real.\
Desenvolvido como entrega final do **Tech Challenge #04** da Pós-Graduação em Machine Learning Engineering da FIAP.

---

## 📂 Estrutura de Diretórios

```
tech-challenge-deep-learning-pipeline/
├── config/                  # Configurações globais (tickers, janela, splits)
├── data/                    # Dados brutos e processados (gerado automaticamente)
├── docker/                  # Configurações de Prometheus e Grafana
├── models/                  # Modelo campeão exportado (gerado pelo script)
├── notebooks/               # Análises exploratórias
├── reports/                 # Figuras, métricas e relatórios de drift
├── scripts/                 # Scripts utilitários de suporte
├── src/
│   ├── data/               # Ingestão e pré-processamento
│   ├── model/              # Definição, treino, avaliação e registro do LSTM
│   ├── api/                # Aplicação FastAPI
│   └── monitoring/         # Métricas Prometheus e detecção de drift
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 🛠️ Tecnologias

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.13 |
| Dados & ML | `PyTorch`, `yfinance`, `pandas`, `scikit-learn` |
| API | FastAPI + Uvicorn |
| Rastreamento & Registro de Modelos | MLflow |
| Monitoramento | Prometheus, Grafana, Evidently AI |
| Containerização | Docker + Docker Compose |
| Gerenciamento de Dependências | `pyproject.toml` + `pip` (venv) |

---

## 🚀 Instalação e Execução

### Pré-requisitos

- Python 3.13+
- Docker e Docker Compose

```
Extração de dados  →  Treinamento, avaliação e registro dos modelos  →  API e Monitoramento
```

```bash
# Clonar o repositório
git clone https://github.com/TheElectron/tech-challenge-deep-learning-pipeline.git

cd tech-challenge-deep-learning-pipeline

# Criar e ativar o ambiente virtual
python3 -m venv .venv

source .venv/bin/activate
# venv\Scripts\activate No Windows

# Instalar todas as dependências
pip install -e ".[dev]"

# Executar o pipeline de dados
python -m src.data.pipeline

# Executar o pipeline de treino, avaliação e registro dos modelos
python -m src.model.pipeline

# Cria o dataset de referência
python scripts/build_reference.py

# Exporta o melhor modelo para a pasta /models
python scripts/export_champion.py   

# Inicializa os containers
docker compose up --build

```

| Serviços | Endereços |
|---|---|
| API | http://localhost:8000, http://localhost:8000/docs, http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000  (usuário: admin / senha: admin)|
| MLflow UI | http://localhost:5000 |

---

## ⚙️ Detalhes de Implementação

### 💾 Pipeline de Dados

**Objetivo:** Gerar um dataset reprodutível e versionado, capaz de treinar um modelo que prevê o preço de fechamento de diferentes ativos.

#### Arquivos principais

| Arquivo | Responsabilidade |
|---|---|
| `config/config.yaml` | Tickers, intervalo de datas, lista de features, tamanho da janela, proporções do split, configurações do MLflow |
| `src/data/ingest.py` | Download de dados OHLCV via `yf.Ticker.history()`; salva um CSV por ativo em `data/raw/` |
| `src/data/preprocess.py` | Divisão cronológica → ajuste do scaler no treino → escalonamento de todos os splits → geração de sequências → salvamento de `.npy` + `scaler.pkl` |
| `src/data/pipeline.py` | Orquestrador: executa ingestão → pré-processamento para todos os tickers, registra no MLflow e escreve `data/processed/metadata.json` |

- **Multi-ativo por padrão** — qualquer número de tickers pode ser listado em `config/config.yaml`; o pipeline faz download, pré-processa e salva cada ativo de forma independente.
- **MinMaxScaler por ativo** — cada ativo possui seu próprio scaler ajustado exclusivamente na porção de treino, evitando vazamento de informações futuras para as janelas de validação e teste. O scaler é persistido em disco para que as predições possam ser desnormalizadas no momento de inferência.
- **Divisão cronológica (80 / 10 / 10)** — os dados são divididos em ordem temporal, sem embaralhamento. As sequências são construídas dentro de cada split de forma independente, de modo que nenhuma janela cruze uma fronteira.
- **Sequências com janela deslizante** — uma janela de lookback configurável (padrão: 60 dias de negociação) gera pares `(X, y)`, onde `X` é a janela de valores escalados e `y` é o preço de fechamento escalado do dia seguinte.
- **Rastreamento MLflow** — cada execução do pipeline registra parâmetros (tickers, datas, tamanho da janela, proporções do split), contagens de amostras por ativo e artefatos leves (JSON de metadados e scalers por ativo).

---

### 💻 Modelos

**Objetivo:** Treinar diferentes modelos com avaliação rigorosa. Todos modelos treinados são versionados e promovidos ou rebaixados automaticamente com base na performance no conjunto de teste.

#### Configurações dos modelos

| | StackedLSTM | AttentionLSTM |
|---|---|---|
| Arquitetura | LSTM(128) → Dropout → LSTM(64) → Dropout → Linear(1) | LSTM(128, 2 camadas) → Atenção Multi-Cabeça (4 cabeças) + residual + LayerNorm → Linear(64) → Linear(1) |
| Parâmetros treináveis | ~100 k | ~350 k |
| Vantagem | Captura comprovada de padrões sequenciais; eficiente com poucos dados | Ponderação explícita por passo; mapas de atenção interpretáveis |
| Risco | Pode perder dependências não-locais em sequências muito longas | Overfitting com dados limitados; overhead de atenção em janelas curtas |

#### Detalhes de treinamento

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Função de perda | MSE |
| Taxa de aprendizado | 0,001 com ReduceLROnPlateau (×0,5, patience=5) |
| Clipping de gradiente | max_norm=1,0 |
| Early stopping | patience=15 na perda de validação |
| Tamanho do batch | 64 |
| Estratégia de treino | Modelo universal — todos os ativos concatenados em um único conjunto de treino |

#### Resultados no conjunto de teste — AAPL

| Modelo | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| **StackedLSTM** | **4.2753** | **3.5131** | **1,89%** | **0.5656** |
| AttentionLSTM | 4.6133 | 4.1708 | 2,22% | 0.4942 |

#### Configuração Recomendada: StackedLSTM

O StackedLSTM supera o AttentionLSTM em todas as métricas. Os motivos são estruturais:

1. **Dados de treino limitados** — ~744 amostras por ativo (~4.000 no total). O número 3,5× maior de parâmetros da Config B representa risco significativo de overfitting que o dropout sozinho não absorve.
2. **Sequências curtas reduzem a vantagem da atenção** — a atenção multi-cabeça é mais benéfica em sequências com mais de 200 passos, onde apenas um subconjunto esparso de posições é informativo. Uma janela de 60 passos é curta o suficiente para que o estado oculto do LSTM carregue todo o contexto relevante.
3. **Normalização por ativo já comprime a escala** — entradas já estão em [0, 1] por ativo, tornando o passo de compressão do segundo LSTM suficiente sem necessidade de mecanismo explícito de ponderação.
4. **Generalização entre ativos** — um viés indutivo mais simples generaliza melhor entre ativos com perfis de volatilidade muito diferentes (ex.: AAPL vs PETR4.SA).
5. **Custo em produção** — 3,5× menos parâmetros → artefatos menores, inferência mais rápida, sem penalidade de acurácia.

> O AttentionLSTM seria preferível com sequências acima de 200 passos, 10× mais dados de treino (dados em minutos ou 50+ ativos), ou quando a interpretabilidade da atenção for um requisito.

#### Arquivos principais

| Arquivo | Responsabilidade |
|---|---|
| `src/model/lstm.py` | Definições dos modelos `StackedLSTM` e `AttentionLSTM` |
| `src/model/dataset.py` | `StockDataset` — encapsula arrays numpy como PyTorch Dataset |
| `src/model/train.py` | Loop de treino, early stopping, registro MLflow, serialização do modelo |
| `src/model/evaluate.py` | Cálculo de RMSE, MAE, MAPE, R²; gráficos real vs. predito |
| `src/model/pipeline.py` | Treina ambas as configurações, avalia no conjunto de teste e exibe tabela comparativa |

#### Ciclo de vida

Cada execução de treino registra uma nova **versão** (v1, v2, v3…). As versões são permanentes — nada é deletado; em vez disso, **aliases** nomeados apontam para a versão atualmente relevante:

```
lstm-stock-predictor
  ├── v1  ← @champion    (melhor RMSE registrado → utilizado pela API)
  └── v2  ← @challenger  (último modelo que NÃO superou o champion)
```

A API sempre carrega `models:/lstm-stock-predictor@champion`. Quando um novo modelo supera o champion, apenas o alias é movido — a API passa a servir o novo modelo no próximo restart, sem nenhuma mudança no código.

#### Lógica de promoção

```
treino → avaliação → registrar mean_test_rmse → registrar versão
                                                      │
                          ┌───────────────────────────┴──────────────────────────┐
                          │     comparar RMSE com @champion                       │
                          └───────────────────────────┬──────────────────────────┘
                                 │                                │
                         novo < champion                  novo ≥ champion
                         champion antigo → @challenger    nova versão → @challenger
                         nova versão    → @champion       @champion inalterado
```

#### Estado atual do registry

| Versão | Tipo de modelo | mean_test_rmse | Alias |
|---|---|---|---|
| v1 | stacked_lstm | 3.6475 | **@champion** |
| v2 | attention_lstm | 8.1692 | @challenger |

---

### 📉 API REST

**Objetivo:** Disponibilizar o modelo `@champion` para o usuário final.

#### Decisões de design

- **Modelo carregado uma única vez na inicialização** via hook `lifespan` do FastAPI — sem cold starts por requisição.
- **Inferência universal** — a API aceita qualquer ticker; a normalização é feita dinamicamente usando um MinMaxScaler ajustado na janela de entrada fornecida. Isso significa que a API funciona para qualquer ativo sem necessitar de um scaler pré-ajustado em disco.
- **Respaldada pelo registry** — a API sempre serve o alias `@champion`. Promover um novo modelo no registry é suficiente para atualizar o que a API serve no próximo restart; nenhuma mudança de código é necessária.

#### Endpoints

| Método | Caminho | Descrição |
|---|---|---|
| `GET` | `/health` | Verificação de liveness — confirma que a API está no ar e o modelo está carregado |
| `GET` | `/model/info` | Metadados do registry para o champion carregado (versão, tipo, RMSE) |
| `POST` | `/predict` | Prediz o próximo preço de fechamento dado ≥ 60 fechamentos históricos |
| `GET` | `/docs` | Swagger UI (gerado automaticamente) |
| `GET` | `/redoc` | ReDoc (gerado automaticamente) |

#### Requisição e Resposta

```json
// POST /predict
{
  "ticker": "AAPL",
  "close_prices": [176.38, 177.31, 175.73, ..., 168.16]  // ≥ 60 valores
}

// Resposta
{
  "ticker": "AAPL",
  "predicted_close": 171.694,
  "model_version": "1",
  "model_alias": "champion",
  "prediction_timestamp": "2026-05-17T15:42:49.089105Z"
}
```

---

### 🔍 Monitoramento

**Objetivo:** Monitorar o modelo em produção em duas camadas complementares — métricas de infraestrutura e detecção de drift específica de ML.

#### Camadas de monitoramento

| Camada | Ferramenta | O que rastreia |
|---|---|---|
| Infraestrutura | Prometheus + Grafana | Latência de requisições (p50/p95/p99), taxa de erros, throughput, RMSE do champion |
| Métricas de ML | Evidently AI | Drift de predições, drift de features de entrada, qualidade dos dados |
| Alertas | Alertas do Grafana | Pico de latência, alta taxa de erros, degradação do RMSE do modelo |
| Relatórios de drift | HTML do Evidently | Relatório sob demanda comparando predições em produção com o conjunto de referência |

#### Decisões de design

- **Monitoramento em duas camadas** — Prometheus/Grafana rastreiam o que a infraestrutura *está fazendo* (latência, erros, throughput); Evidently rastreia o que o *modelo está fazendo* (distribuição de entradas e saídas). Nenhum dos dois é suficiente sozinho.
- **Middleware Prometheus** — um único middleware FastAPI intercepta todas as requisições e registra contagem + latência, mantendo os handlers de rota limpos. O endpoint `/metrics` é excluído da instrumentação para não poluir os dados.
- **Registro de drift em tempo real** — cada chamada a `/predict` adiciona um registro em `data/predictions_log.jsonl` (quatro features: `last_close`, `predicted_close`, `price_mean`, `price_range`). Isso é best-effort; uma falha no registro nunca é exposta ao chamador.
- **Dataset de referência** — o Evidently requer uma distribuição de referência estável. A referência é construída a partir das predições do modelo no **split de teste** (execute `scripts/build_reference.py` uma vez após cada promoção de champion). Usar predições do conjunto de teste em vez do conjunto de treino mantém a referência honesta: ela reflete a distribuição de saída real do modelo em dados não vistos.
- **RMSE como proxy de drift** — o gauge Prometheus `model_rmse` (definido na inicialização a partir dos metadados do registry) fornece um sinal rápido e sempre disponível para a qualidade do modelo, sem necessitar de um relatório do Evidently. O threshold de alerta em 7,0 foi escolhido porque o challenger AttentionLSTM rejeitado obteve 8,17; qualquer valor acima de 7,0 justifica investigação.
- **Provisionamento do Grafana** — datasource, dashboard e regras de alerta são todos provisionados via YAML na inicialização do container. O stack é totalmente reprodutível com `docker compose up`; nenhuma configuração manual pela UI é necessária.

#### Regras de alerta

| Alerta | Condição | `for` | Severidade |
|---|---|---|---|
| Latência P95 de Predição Alta | Latência p95 em `/predict` > 1 s | 5 min | warning |
| Taxa de Erros da API Alta | Taxa de erros 5xx > 0,05 req/s | 2 min | critical |
| RMSE do Modelo Champion Degradado | `model_rmse` > 7,0 | 5 min | warning |

#### Acionando um relatório de drift

```bash
# Via API (requer ≥ 10 predições registradas e referência construída)
curl -X POST http://localhost:8000/monitoring/report

# Visualizar predições recentes registradas
curl http://localhost:8000/monitoring/predictions?n=20

# Relatórios são salvos em reports/drift/drift_<timestamp>.html
```

---

## 📁 Estrutura Detalhada do Código-Fonte

```
src/
├── data/
│   ├── ingest.py           # download de dados brutos via yfinance
│   └── preprocess.py       # normalização, geração de sequências, split treino/val/teste
├── model/
│   ├── lstm.py             # definição dos modelos LSTM
│   ├── train.py            # loop de treino + registro MLflow
│   └── evaluate.py         # RMSE, MAE, MAPE, R², gráficos de predição
├── api/
│   ├── main.py             # app FastAPI, lifespan, registro de rotas
│   ├── routes.py           # endpoints /health, /model/info, /predict, /monitoring/*
│   └── schemas.py          # modelos Pydantic de requisição/resposta
└── monitoring/
    ├── metrics.py          # definições de métricas Prometheus
    └── drift.py            # registro de predições + relatórios de drift Evidently AI

docker/
├── prometheus.yml          # config de scrape (api:8000/metrics a cada 15 s)
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── prometheus.yml      # provisiona automaticamente o datasource Prometheus
        ├── dashboards/
        │   ├── provider.yml        # config do provedor de arquivos de dashboard
        │   └── api_dashboard.json  # dashboard de produção com 6 painéis
        └── alerting/
            └── alerts.yml          # 3 regras de alerta (latência, erros, RMSE)

scripts/
├── export_champion.py      # copia @champion do registry MLflow para models/
└── build_reference.py      # constrói data/reference_predictions.csv a partir do split de teste
```

---

## 🚀 Entregáveis

- Descrição completa do projeto ✅
- Diagrama visual do projeto ✅
- Link do vídeo ✅
