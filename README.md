# LSTM Stock Price Prediction Pipeline

End-to-end deep learning pipeline to predict stock closing prices using an LSTM neural network, served via a REST API with model versioning and production monitoring.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
  - [Phase 1 — Data Pipeline](#phase-1--data-pipeline)
  - [Phase 2 — LSTM Model](#phase-2--lstm-model)
  - [Phase 3 — Model Registry & Versioning](#phase-3--model-registry--versioning)
  - [Phase 4 — REST API](#phase-4--rest-api)
  - [Phase 5 — Monitoring](#phase-5--monitoring)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## Architecture

```
tech-challenge-deep-learning-pipeline/
├── data/                    # raw & processed datasets
├── src/
│   ├── data/               # ingestion + preprocessing
│   ├── model/              # LSTM definition, training, evaluation
│   ├── api/                # FastAPI application
│   └── monitoring/         # metrics & drift detection
├── mlruns/                 # MLflow experiment tracking (auto-generated)
├── notebooks/              # exploratory analysis
├── docker/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.13 |
| ML | PyTorch |
| Data | `yfinance`, `pandas`, `scikit-learn` |
| Experiment tracking & registry | MLflow |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus, Grafana, Evidently AI |
| Containerization | Docker + Docker Compose |
| Dependency management | `pyproject.toml` + `pip` (venv) |

---

## Roadmap

### Phase 1 — Data Pipeline ✅

**Goal:** reproducible, versioned dataset capable of training a model that predicts closing prices across different assets.

#### Design decisions

- **Multi-asset by default** — any number of tickers can be listed in `config/config.yaml`; the pipeline downloads, preprocesses, and saves each one independently.
- **Per-asset MinMaxScaler** — each asset has its own scaler fitted exclusively on the training portion, preventing any future price information from leaking into validation or test windows. The scaler is persisted to disk so predictions can be denormalized at inference time without knowing the original price range.
- **Chronological split (80 / 10 / 10)** — data is split in time order, never shuffled. Sequences are built within each split independently so no window ever straddles a boundary.
- **Sliding-window sequences** — a configurable lookback window (default: 60 trading days) is used to produce `(X, y)` pairs where `X` is the window of scaled feature values and `y` is the next day's scaled Close price.
- **MLflow tracking** — every pipeline run logs parameters (tickers, dates, window size, split ratios), per-asset sample counts, and lightweight artifacts (metadata JSON + per-asset scalers).

#### Key files

| File | Responsibility |
|---|---|
| `config/config.yaml` | Tickers, date range, feature list, window size, split ratios, MLflow settings |
| `src/data/ingest.py` | Download OHLCV via `yf.Ticker.history()`; save one CSV per asset to `data/raw/` |
| `src/data/preprocess.py` | Chronological split → fit scaler on train → scale all splits → generate sequences → save `.npy` + `scaler.pkl` |
| `src/data/pipeline.py` | Orchestrator: runs ingest → preprocess for all tickers, logs to MLflow, writes `data/processed/metadata.json` |

#### Output structure

```
data/
├── raw/
│   ├── AAPL.csv
│   ├── PETR4.SA.csv
│   └── ...
└── processed/
    ├── metadata.json          # window_size, features, split_ratios, per-ticker stats
    ├── AAPL/
    │   ├── X_train.npy        # shape (N, 60, 1)
    │   ├── y_train.npy        # shape (N, 1)
    │   ├── X_val.npy
    │   ├── y_val.npy
    │   ├── X_test.npy
    │   ├── y_test.npy
    │   └── scaler.pkl         # per-asset MinMaxScaler
    └── PETR4.SA/
        └── ...
```

#### Running the pipeline

```bash
# activate the virtual environment first
source .venv/bin/activate

# run with settings from config/config.yaml
python -m src.data.pipeline
```

---

### Phase 2 — LSTM Model ✅

**Goal:** trained model with rigorous evaluation across two candidate architectures.

#### Configurations evaluated

| | Config A — StackedLSTM | Config B — AttentionLSTM |
|---|---|---|
| Architecture | LSTM(128) → Dropout → LSTM(64) → Dropout → Linear(1) | LSTM(128, 2 layers) → Multi-Head Attention (4 heads) + residual + LayerNorm → Linear(64) → Linear(1) |
| Trainable params | ~100 k | ~350 k |
| Strength | Proven sequential pattern capture; data-efficient | Explicit per-step weighting; interpretable attention maps |
| Risk | May miss non-local dependencies in very long sequences | Overfits with limited data; attention overhead on short windows |

#### Results on AAPL test set

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| **StackedLSTM** | **4.2753** | **3.5131** | **1.89%** | **0.5656** |
| AttentionLSTM | 4.6133 | 4.1708 | 2.22% | 0.4942 |

#### Recommended: StackedLSTM

StackedLSTM outperforms AttentionLSTM on every metric. The reasons are structural:

1. **Limited training data** — ~744 samples per asset (≈4 000 across all assets). Config B's 3.5× parameter count is a significant overfitting risk that dropout alone cannot absorb.
2. **Short sequences reduce attention's advantage** — multi-head attention is most beneficial on sequences > 200 steps where only a sparse subset of positions is informative. A 60-step window is short enough for the LSTM hidden state to carry all relevant context.
3. **Per-asset normalisation already compresses scale** — inputs are already in [0, 1] per asset, so the second LSTM's compression step is sufficient without an explicit weighting mechanism.
4. **Generalisation across assets** — a simpler inductive bias generalises better across assets with very different volatility profiles (e.g. AAPL vs PETR4.SA).
5. **Production cost** — 3.5× fewer parameters → smaller artifacts, faster inference, no accuracy penalty.

> AttentionLSTM would be preferred with sequences > 200 steps, 10× more training data (minute-level or 50+ assets), or when attention interpretability is a requirement.

#### Training details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | MSE |
| Learning rate | 0.001 with ReduceLROnPlateau (×0.5, patience=5) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=15 on validation loss |
| Batch size | 64 |
| Training strategy | Universal model — all assets concatenated into one training set |

#### Key files

| File | Responsibility |
|---|---|
| `src/model/lstm.py` | `StackedLSTM` and `AttentionLSTM` model definitions |
| `src/model/dataset.py` | `StockDataset` — wraps numpy arrays as a PyTorch Dataset |
| `src/model/train.py` | Training loop, early stopping, MLflow logging, model serialisation |
| `src/model/evaluate.py` | RMSE, MAE, MAPE, R² computation; actual-vs-predicted plots |
| `src/model/pipeline.py` | Trains both configs, evaluates on test set, prints comparison table |

#### Running the model pipeline

```bash
source .venv/bin/activate

# Train both configs, evaluate, and compare
python -m src.model.pipeline
```

Outputs written to:
- `reports/figures/` — actual vs predicted plots per model per ticker
- `reports/comparison.json` — metrics for all models and tickers
- MLflow — every run tracked at `sqlite:///mlruns.db`

---

### Phase 3 — Model Registry & Versioning ✅

**Goal:** every trained model is versioned, tagged, and automatically promoted or demoted based on test performance.

#### What is the Model Registry?

Without a registry, a trained model is just a file on disk — there is no answer to "which model is currently in production?" or "how do I roll back?". The **MLflow Model Registry** is a versioned catalog of model artifacts with a managed lifecycle on top.

MLflow has three layers:

```
Experiment  →  groups related training runs
    Run     →  one training execution: params + metrics + artifacts (model file)
 Registry   →  named, versioned model pulled from a run's artifact store
```

#### Lifecycle and aliases

Every training run registers a new **version** (v1, v2, v3 …). Versions are permanent — nothing is deleted; instead, named **aliases** act as pointers to whichever version is currently relevant:

```
lstm-stock-predictor
  ├── v1  ← @champion    (best RMSE seen so far → used by the API)
  └── v2  ← @challenger  (latest model that did NOT beat the champion)
```

The API always loads `models:/lstm-stock-predictor@champion`. When a new model beats the champion, only the alias moves — the API picks it up on next restart with zero code changes.

#### Promotion logic

```
train → evaluate → log mean_test_rmse → register version
                                              │
                        ┌─────────────────────┴──────────────────────┐
                        │     compare RMSE with @champion             │
                        └─────────────────────┬──────────────────────┘
                               │                           │
                       new < champion               new ≥ champion
                       old champion → @challenger   new version → @challenger
                       new version  → @champion     @champion unchanged
```

#### Current registry state

| Version | Model type | mean_test_rmse | Alias |
|---|---|---|---|
| v1 | stacked_lstm | 3.6475 | **@champion** |
| v2 | attention_lstm | 8.1692 | @challenger |

#### Key files

| File | Responsibility |
|---|---|
| `src/model/registry.py` | `register(run_id)` — versions, tags, and promotes the model; `load_champion()` — loads `@champion` from the registry |
| `src/model/pipeline.py` | Logs `mean_test_rmse`, captures `run_id`, calls `registry.register()` after each run |

#### Loading the champion elsewhere

```python
from src.model.registry import load_champion

model = load_champion()   # loads models:/lstm-stock-predictor@champion
```

#### Running the registry pipeline

```bash
source .venv/bin/activate

# Train, evaluate, register, and promote automatically
python -m src.model.pipeline
```

---

### Phase 4 — REST API ✅

**Goal:** production-grade inference endpoint that serves the `@champion` model from the MLflow registry.

#### Design decisions

- **Model loaded once at startup** via FastAPI's `lifespan` hook — no per-request cold starts.
- **Universal inference** — the API accepts any ticker; normalization is done on-the-fly using a MinMaxScaler fitted on the provided input window. This means the API works for any asset without needing a pre-fitted scaler stored on disk.
- **Registry-backed** — the API always serves the `@champion` alias. Promoting a new model in the registry is enough to update what the API serves on next restart; no code changes needed.

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — confirms API is up and model is loaded |
| `GET` | `/model/info` | Registry metadata for the loaded champion (version, type, RMSE) |
| `POST` | `/predict` | Predict the next closing price given ≥ 60 historical closes |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc (auto-generated) |

#### Request / Response

```json
// POST /predict
{
  "ticker": "AAPL",
  "close_prices": [176.38, 177.31, 175.73, ..., 168.16]  // ≥ 60 values
}

// Response
{
  "ticker": "AAPL",
  "predicted_close": 171.694,
  "model_version": "1",
  "model_alias": "champion",
  "prediction_timestamp": "2026-05-17T15:42:49.089105Z"
}
```

#### Key files

| File | Responsibility |
|---|---|
| `src/api/main.py` | FastAPI app definition and `lifespan` startup hook |
| `src/api/state.py` | Module-level singleton — loads and holds the champion model and its metadata |
| `src/api/routes.py` | Handlers for `/health`, `/model/info`, `/predict` |
| `src/api/schemas.py` | Pydantic request/response models with input validation |

#### Running the API

**Local development (loads model from MLflow registry):**
```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

**Docker (see Phase 4 — Docker below):**
```bash
python scripts/export_champion.py   # one-time export
docker compose up
```

- API: http://localhost:8000  
- Swagger UI: http://localhost:8000/docs

---

### Phase 4 — Docker ✅

**Goal:** containerise the API and MLflow UI so the full stack runs with a single command.

#### Architecture

```
docker compose up
│
├── mlflow  (python:3.13-slim)     port 5000  — experiment tracking UI
│   └── mounts ./mlruns.db + ./mlruns  (read-only, for the UI)
│
└── api     (Dockerfile)           port 8000  — FastAPI prediction service
    └── mounts ./models  (read-only, contains the exported champion model)
```

The API does **not** connect to the MLflow server at runtime. The champion model is pre-exported to `models/` and loaded from disk — no network dependency at inference time.

#### Two loading modes

| Mode | Trigger | Use case |
|---|---|---|
| Registry | `MODEL_DIR` unset | Local dev — loads `@champion` from MLflow SQLite DB |
| Local dir | `MODEL_DIR=/app/models` | Docker — loads from pre-exported `models/champion/` |

Switching between them requires no code change — only the environment variable.

#### Key files

| File | Responsibility |
|---|---|
| `Dockerfile` | Multi-layer build: CPU torch → API deps → source code |
| `docker-compose.yml` | Defines `mlflow` (UI) and `api` (inference) services |
| `.dockerignore` | Excludes data, artifacts, venv, and notebooks from the build context |
| `requirements.txt` | API runtime deps only (no CUDA torch, no dev extras) |
| `scripts/export_champion.py` | Copies the `@champion` model from MLflow registry to `models/champion/` |

#### Running with Docker

```bash
# 1. Export the current champion model (re-run whenever champion changes)
python scripts/export_champion.py

# 2. Start both services
docker compose up --build

# API:          http://localhost:8000
# Swagger UI:   http://localhost:8000/docs
# MLflow UI:    http://localhost:5000
```

---

### Phase 5 — Monitoring ✅

**Goal:** observe model health in production across two complementary layers — infrastructure metrics and ML-specific drift detection.

#### Monitoring layers

| Layer | Tool | What it tracks |
|---|---|---|
| Infrastructure | Prometheus + Grafana | Request latency (p50/p95/p99), error rate, throughput, champion RMSE |
| ML metrics | Evidently AI | Prediction drift, input feature drift, data quality |
| Alerting | Grafana alerts | Latency spike, high error rate, model RMSE degradation |
| Drift reports | Evidently HTML | On-demand report comparing live predictions to test-set reference |

#### Design decisions

- **Two-layer monitoring** — Prometheus/Grafana track what the infrastructure *is doing* (latency, errors, throughput); Evidently tracks what the *model is doing* (distribution of inputs and outputs). Neither alone is sufficient.
- **Prometheus middleware** — a single FastAPI middleware intercepts every request and records count + latency, keeping route handlers clean. The `/metrics` endpoint is excluded from instrumentation to avoid polluting the data.
- **On-the-fly drift logging** — every `/predict` call appends a record to `data/predictions_log.jsonl` (four features: `last_close`, `predicted_close`, `price_mean`, `price_range`). This is best-effort; a failure to log never surfaces to the caller.
- **Reference dataset** — Evidently requires a stable reference distribution. The reference is built from model predictions on the held-out **test split** (run `scripts/build_reference.py` once after each champion promotion). Using test-set predictions rather than training-set predictions keeps the reference honest: it reflects the model's actual output distribution on unseen data.
- **RMSE as drift proxy** — the `model_rmse` Prometheus gauge (set at startup from registry metadata) provides a fast, always-available signal for model quality without requiring an Evidently report. The alert threshold of 7.0 is chosen because the rejected AttentionLSTM challenger scored 8.17; anything above 7.0 warrants investigation.
- **Grafana provisioning** — datasource, dashboard, and alert rules are all provisioned via YAML at container start. The stack is fully reproducible with `docker compose up`; no manual UI configuration is needed.

#### Alert rules

| Alert | Condition | `for` | Severity |
|---|---|---|---|
| High P95 Prediction Latency | p95 `/predict` latency > 1 s | 5 m | warning |
| High API Error Rate | 5xx rate > 0.05 req/s | 2 m | critical |
| Champion Model RMSE Degraded | `model_rmse` > 7.0 | 5 m | warning |

#### Key files

| File | Responsibility |
|---|---|
| `src/monitoring/metrics.py` | Prometheus metric definitions (counter, histogram, gauge) |
| `src/monitoring/drift.py` | Prediction logging, reference building, Evidently report generation |
| `src/api/main.py` | HTTP middleware wiring metrics; `/metrics` scrape endpoint |
| `src/api/routes.py` | `/monitoring/predictions` and `/monitoring/report` endpoints |
| `scripts/build_reference.py` | One-shot script to build `data/reference_predictions.csv` from test split |
| `docker/prometheus.yml` | Prometheus scrape config (target: `api:8000/metrics`, interval: 15 s) |
| `docker/grafana/provisioning/datasources/prometheus.yml` | Auto-provisions the Prometheus datasource in Grafana |
| `docker/grafana/provisioning/dashboards/api_dashboard.json` | 6-panel Grafana dashboard (request rate, error rate, latency, RMSE, ticker counts, price distribution) |
| `docker/grafana/provisioning/alerting/alerts.yml` | Three provisioned alert rules (latency, error rate, RMSE) |

#### Running the monitoring stack

```bash
# 1. Build the reference dataset (once per champion promotion)
source .venv/bin/activate
python scripts/build_reference.py

# 2. Start the full stack
python scripts/export_champion.py   # export champion to models/
docker compose up --build

# Services:
#   API:          http://localhost:8000        (+ /docs, /metrics)
#   Prometheus:   http://localhost:9090
#   Grafana:      http://localhost:3000        (admin / admin)
#   MLflow UI:    http://localhost:5000
```

#### Triggering a drift report

```bash
# Via API (requires ≥ 10 logged predictions and a built reference)
curl -X POST http://localhost:8000/monitoring/report

# View recent logged predictions
curl http://localhost:8000/monitoring/predictions?n=20

# Reports are saved to reports/drift/drift_<timestamp>.html
```

---

### Execution Order

```
Phase 1 (Data)  →  Phase 2 (LSTM)  →  Phase 3 (Registry)
                                              ↓
                              Phase 4 (API)  →  Phase 5 (Monitoring)
```

Phases 4 and 5 can be developed in parallel once Phase 3 produces a registered model.

---

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd tech-challenge-deep-learning-pipeline

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -e ".[dev]"

# --- Phase 1: Data Pipeline ---
# Edit config/config.yaml to choose tickers and date range, then:
python -m src.data.pipeline

# --- Phase 2: Model Training ---
python -m src.model.pipeline

# --- Phase 4: API ---
uvicorn src.api.main:app --reload

# --- Full stack with Docker (Phases 4 + 5) ---
python scripts/export_champion.py
docker compose up --build
```

---

## Project Structure

```
src/
├── data/
│   ├── ingest.py           # download raw stock data via yfinance
│   └── preprocess.py       # normalization, sequence generation, train/val/test split
├── model/
│   ├── lstm.py             # LSTM model definition
│   ├── train.py            # training loop + MLflow logging
│   └── evaluate.py         # RMSE, MAE, MAPE, R², prediction plots
├── api/
│   ├── main.py             # FastAPI app, lifespan, router registration
│   ├── routes.py           # /health, /model/info, /predict, /monitoring/* endpoints
│   └── schemas.py          # Pydantic request/response models
└── monitoring/
    ├── metrics.py          # Prometheus metric definitions
    └── drift.py            # prediction logging + Evidently AI drift reports

docker/
├── prometheus.yml          # scrape config (api:8000/metrics every 15 s)
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── prometheus.yml      # auto-provisions Prometheus datasource
        ├── dashboards/
        │   ├── provider.yml        # dashboard file provider config
        │   └── api_dashboard.json  # 6-panel production dashboard
        └── alerting/
            └── alerts.yml          # 3 alert rules (latency, errors, RMSE)

scripts/
├── export_champion.py      # copies @champion from MLflow registry to models/
└── build_reference.py      # builds data/reference_predictions.csv from test split
```
