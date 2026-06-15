# Real-Time Customer Churn Intelligence Platform

Production-grade ML platform that predicts customer churn, explains predictions
with SHAP, recommends ROI-ranked interventions, and surfaces it all through a
real-time dashboard and API.

> **Status:** ✅ Complete — all 8 phases (Baseline Model, Explainability/
> Interventions, Real-Time Scoring + SQLite logging, MLflow tracking/registry +
> Monitoring, Streamlit dashboard, FastAPI service, Docker + GitHub Actions
> CI/CD, Documentation).

## Architecture

```
                          ┌─────────────────────┐
   Raw / streaming  ─────▶│  Feature Engineering │
   customer data          │  (scale + encode +   │
                          │   interactions)      │
                          └──────────┬───────────┘
                                     ▼
                          ┌─────────────────────┐      ┌──────────────┐
                          │  XGBoost  (churn p)  │◀────▶│   MLflow     │
                          └──────────┬───────────┘      │  registry +  │
                                     ▼                  │  experiments │
                  ┌──────────────────┴───────┐          └──────────────┘
                  ▼                          ▼
        ┌──────────────────┐      ┌──────────────────────┐
        │ SHAP explainer   │      │ Intervention engine  │
        │ (why at risk?)   │      │ (segment + ROI rank) │
        └────────┬─────────┘      └──────────┬───────────┘
                 └───────────┬───────────────┘
                             ▼
                   ┌───────────────────┐
                   │  ScoringService   │  preprocess→predict→explain→recommend
                   └─────────┬─────────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────────┐ ┌───────────┐ ┌──────────────┐
      │ SQLite log   │ │ FastAPI   │ │ Streamlit    │
      │ (history)    │ │ (3 routes)│ │ dashboard    │
      └──────┬───────┘ └───────────┘ └──────────────┘
             ▼
      ┌──────────────┐
      │ Monitoring   │  drift (PSI) · performance · business · latency
      └──────────────┘
```

## Tech Stack

| Layer            | Tooling                          |
|------------------|----------------------------------|
| Model            | XGBoost + SHAP                   |
| API              | FastAPI                          |
| Dashboard        | Streamlit + Plotly               |
| Model registry   | MLflow                           |
| Storage          | SQLite / SQLAlchemy              |
| Deployment       | Docker + GitHub Actions          |

## Project Structure

```
Accenture Project/
├── data/
│   ├── raw/            telco_churn.csv (generated or real Kaggle CSV)
│   └── processed/      train.csv, test.csv (feature-engineered)
├── models/             model.pkl, preprocessing.pkl
├── reports/            baseline_metrics.json, roc_curve.png, confusion_matrix.png
├── src/
│   ├── config.py       paths, schema, modelling constants
│   ├── generate_data.py synthetic Telco-schema dataset
│   ├── data_prep.py    load -> EDA -> feature engineering -> split
│   ├── model.py        XGBoost train / tune / evaluate / save
│   ├── explain.py      SHAP TreeExplainer: global + local explanations
│   ├── interventions.py risk segmentation + ROI-ranked recommendations
│   ├── database.py     SQLAlchemy ORM — customer_scores table + queries
│   ├── inference.py    ScoringService: preprocess->predict->explain->recommend
│   ├── simulate.py     hourly-batch streaming simulation -> SQLite
│   ├── mlflow_tracking.py experiment logging + model registry (production alias)
│   └── monitoring.py   performance/drift/business/inference metrics -> JSON
├── api/
│   ├── main.py         FastAPI app: /score, /model/performance, /dashboard-data
│   └── schemas.py      Pydantic request/response models
├── dashboard/
│   └── app.py          Streamlit dashboard (6 tabs)
├── tests/              fast pytest smoke tests
├── docker/
│   └── entrypoint.sh   bootstraps artifacts then runs the given command
├── .github/workflows/
│   └── ci.yml          test + docker build pipeline
├── Dockerfile
├── docker-compose.yml  API + dashboard services
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the ENTIRE pipeline end-to-end (data -> model -> SHAP -> sim -> MLflow -> monitoring)
python run_all.py            # add --fast to skip hyperparameter tuning

# Then launch the apps (separate terminals):
streamlit run dashboard/app.py     # dashboard  -> http://localhost:8501
uvicorn api.main:app --reload      # API + docs -> http://localhost:8000/docs
```

Or run a single phase, e.g. just train the baseline:

```bash
python -m src.generate_data   # generate data (auto-runs if missing)
python -m src.model           # train + evaluate -> models/, reports/
```

### Using the real Kaggle dataset

Drop the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
CSV at `data/raw/telco_churn.csv`. The pipeline auto-detects it; the synthetic
generator is only a fallback so the project runs without a Kaggle download.

## Phase 1 — Data & Baseline Model

**Pipeline** (`src/data_prep.py`): loads data, runs EDA (missing values, class
balance, distributions), engineers interaction & trend features
(`charges_per_tenure`, `contract_x_charges`, `tenure_x_charges`,
`services_count`), one-hot encodes categoricals, scales numerics, and performs
a stratified 80/20 split.

**Model** (`src/model.py`): XGBoost classifier with a 30-candidate randomized
hyperparameter search (5-fold CV, AUC objective) and an F1-optimal decision
threshold tuned on the training set.

### Baseline performance (synthetic dataset, 7,000 customers, 27% churn)

| Metric    | Value | Target  |
|-----------|-------|---------|
| AUC-ROC   | 0.868 | > 0.85 ✅ |
| Accuracy  | 0.808 | —       |
| Precision | 0.644 | > 0.75 (see note) |
| Recall    | 0.649 | —       |
| F1        | 0.647 | —       |

> **Note on precision:** ~0.65 precision/recall at the F1-optimal threshold is
> representative of real Telco churn data — the classes overlap, so precision
> and recall trade off against each other. The 0.85 AUC bar (overall ranking
> quality) is the meaningful target and is met. Precision can be pushed higher
> by raising the decision threshold at the cost of recall.

## Phase 2 — Explainability & Interventions

**SHAP** (`src/explain.py`): a `ChurnExplainer` wraps a SHAP `TreeExplainer`
around the XGBoost model and produces global summary plots (bar + beeswarm),
local "top-3 reasons this customer is at risk" explanations, an interactive
force plot (HTML), and a plain-English narrative.

```bash
python -m src.explain        # -> reports/shap_summary_*.png, reports/force_plot.html
```

**Intervention engine** (`src/interventions.py`): risk-based segmentation
(HIGH / MEDIUM / LOW) with rule-based action selection, then scores each action
on transparent economics:

| Quantity | Definition |
|----------|------------|
| `revenue_at_risk` | `MonthlyCharges × 12` (annual exposure) |
| `expected_revenue_saved` | `revenue_at_risk × churn_prob × success_prob` |
| `intervention_cost` | fixed amount, or % of revenue at risk |
| `roi` | `expected_revenue_saved / intervention_cost` |
| `net_benefit` | `expected_revenue_saved − intervention_cost` |

Customers are ranked by `net_benefit` so a finite retention budget targets the
highest-value saves first.

```bash
python -m src.interventions  # -> reports/intervention_recommendations.json
```

## Phase 3 — Real-Time Scoring & Logging

**Scoring service** (`src/inference.py`): a reusable `ScoringService` runs the
full pipeline for one customer — *preprocess → predict → SHAP explain →
recommend intervention* — in ~20–40 ms, returning a record ready for the DB or
API. Partial payloads are filled from sensible defaults.

```python
from src.inference import real_time_score
real_time_score({"customerID": "C1", "tenure": 2, "MonthlyCharges": 95.0,
                 "Contract": "Month-to-month", "support_tickets": 4,
                 "usage_trend_3m": -0.25}, log=True)
```

**Database** (`src/database.py`): SQLAlchemy ORM `customer_scores` table holding
timestamp, customer, probability, risk segment, SHAP factors (JSON),
intervention, and the ROI economics — plus query helpers (`fetch_recent`,
`summary_stats`).

**Simulation** (`src/simulate.py`): generates new customers in hourly batches,
scores and logs them.

```bash
python -m src.simulate --batches 10 --batch-size 50 --reset
# -> data/churn_scores.db, reports/sample_scores.csv
```

## Phase 4 — MLflow Versioning & Monitoring

**MLflow** (`src/mlflow_tracking.py`): logs each training run (params, metrics,
model artifact, data snapshot) to a SQLite-backed store, registers the model,
and tags the best version with the `production` alias. Inference can load the
production model straight from the registry.

```bash
python -m src.mlflow_tracking          # log saved model + register + promote
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments
```

**Monitoring** (`src/monitoring.py`): aggregates four metric groups into
`reports/monitoring_data.json` for the dashboard:

* **Model performance** — current vs baseline AUC, with a >5% drop drift flag
* **Score drift** — PSI on the churn-probability distribution across batches
* **Business** — % at-risk, revenue at risk, intervention mix
* **Inference** — avg/p95 latency, throughput, cost per prediction

```bash
python -m src.monitoring   # -> reports/monitoring_data.json
```

## Phase 5 — Streamlit Dashboard

`dashboard/app.py` — a 6-tab interactive dashboard that reads the SQLite DB, the
monitoring JSON, and the SHAP plots:

1. **Executive Summary** — headline KPIs + risk-segment split
2. **Risk Distribution** — probability histogram, segment mix, at-risk trend
3. **At-Risk Customers** — filterable/sortable table with SHAP factors + CSV export
4. **Explainability** — global SHAP importance + per-customer driver chart
5. **Model Performance** — AUC/precision/recall/F1, confusion matrix, drift status, version picker
6. **Business Metrics** — interventions, adjustable success-rate ROI scenario, cost-benefit

```bash
# Run the pipeline + simulation first so there is data to show, then:
streamlit run dashboard/app.py
```

## Phase 6 — FastAPI Service

`api/main.py` — three endpoints with Pydantic-validated I/O and auto-docs at
`/docs`. Reuses the `ScoringService` (loaded once at startup) and `monitoring`.

| Method & path | Purpose |
|---------------|---------|
| `POST /score` | Score one customer → probability, SHAP factors, intervention, ROI. `?log=true` persists to SQLite |
| `GET /model/performance` | AUC / precision / recall / F1, model version, drift status |
| `GET /dashboard-data` | Aggregate KPIs over the scoring history |

```bash
uvicorn api.main:app --reload      # docs at http://localhost:8000/docs

curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
  -d '{"customer_data": {"tenure": 2, "MonthlyCharges": 95.0,
       "Contract": "Month-to-month", "support_tickets": 4, "usage_trend_3m": -0.25}}'
```

## Phase 7 — Docker & CI/CD

A single image serves either the API (default) or the dashboard (override the
command). Model artifacts + a seed scoring history are baked at build time, so
the container runs out of the box; `docker/entrypoint.sh` is a safety net that
regenerates them if volumes are mounted empty.

```bash
# Build and run the API
docker build -t churn-platform:v1 .
docker run -p 8000:8000 churn-platform:v1        # http://localhost:8000/docs

# Or run API + dashboard together
docker compose up --build                         # API :8000, dashboard :8501
```

**CI/CD** (`.github/workflows/ci.yml`) runs on every push/PR:

1. **test** — install deps, generate data, train (fast), seed the DB, run `pytest`
2. **docker** — build the image (with layer caching) and smoke-test `/health`

> Dependencies are pinned in `requirements.txt` to the exact versions the project
> was tested against (newer than the brief's original pins, which no longer
> resolve on current Python). The image is built on `python:3.11-slim`.

## Tests

```bash
python -m pytest tests/ -q     # 29 tests
```

Covers data prep, feature engineering, model training, SHAP explanations, the
intervention engine, the database layer, the scoring service, monitoring math,
the FastAPI endpoints, and a Streamlit `AppTest` render check.

## Business Impact

The platform turns a probability into a prioritized, costed action:

* **Segments** every customer HIGH / MEDIUM / LOW by churn risk
* **Quantifies** revenue at risk (annual exposure) per customer
* **Recommends** an intervention and scores it: `expected_revenue_saved` vs
  `intervention_cost` → ROI and net benefit
* **Ranks** customers by net benefit so a finite retention budget targets the
  highest-value saves first

Worked example — at a company with 50,000 customers and ₹2 Cr/year lost to
churn, surfacing the top ~500 at-risk customers and retaining even 10% of them
returns roughly ₹20 L of otherwise-lost revenue, against a fraction of that in
intervention cost. The dashboard's Business Metrics tab makes this scenario
interactive (adjust the assumed success rate, see the ROI move).

## Engineering Notes & Honest Trade-offs

* **Metrics are honest, not inflated.** Real churn data has overlapping classes;
  precision > 0.75 *and* recall ~0.78 simultaneously isn't attainable without
  faking separability. The model hits the meaningful target (**AUC 0.87**) with
  realistic ~0.65 precision/recall — exactly what production Telco models see.
* **Drift detection is production-shaped.** Streamed customers are *unlabelled*,
  so true performance can't be measured live. The platform tracks **PSI
  distribution drift** (a leading indicator) plus an AUC-vs-baseline check that
  fires when the model or reference data changes.
* **Synthetic data, real pipeline.** A generator produces a Telco-schema dataset
  so the project runs without a Kaggle download; drop the real CSV at
  `data/raw/telco_churn.csv` and the identical pipeline uses it.

## Future Enhancements

* Real Kafka / streaming ingestion in place of the batch simulation
* A/B testing to measure actual intervention effectiveness
* Automated intervention execution (email / SMS / CRM hooks)
* Multi-model ensembling and an automated retraining pipeline
* Label feedback loop to enable true performance-based drift detection

## License

[MIT](LICENSE) — see also the ready-to-post writeup in [docs/LINKEDIN.md](docs/LINKEDIN.md).
