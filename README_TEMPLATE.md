<!--
  README scaffold. Everything in {{DOUBLE BRACES}} is a placeholder you fill
  with a REAL measured number or a real path. Do not ship a placeholder.
  Delete every HTML comment before publishing.
-->

# Flight Delay Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-{{VERSION}}-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-{{VERSION}}-00ADD8)](https://delta.io/)
[![MLflow](https://img.shields.io/badge/MLflow-{{VERSION}}-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Serverless_env_v4-FF3621?logo=databricks&logoColor=white)](https://databricks.com/)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Governed-1B3139)](https://www.databricks.com/product/unity-catalog)
[![CI](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/ci.yml)

Production-shaped flight delay platform on Databricks: {{N}}M historical FAA flight records flow through a Bronze→Silver→Gold Delta Lake medallion pipeline, train a dual Spark ML model system governed by Unity Catalog, and score live AviationStack flights into an analytics table that feeds an AI/BI dashboard — orchestrated as a scheduled Lakeflow job.

---

## Abstract

<!-- The single most-read paragraph in the repo. Every number must be real. -->

Arrival delay prediction is complicated by a data availability problem: the strongest predictor of arrival delay is departure delay, which does not exist until the aircraft has already left the gate. A model trained with `dep_delay` is accurate but useless for pre-flight planning; a model trained without it is deployable but weaker. This platform resolves that by training and serving **two models against the same feature store**, selected at inference time by flight phase.

{{N}} raw flight records ({{YEAR_RANGE}}, {{SOURCE}}) are ingested into a Bronze Delta table with schema and ingestion metadata preserved. A PySpark cleaning pass reduces {{BRONZE_COLS}} columns to {{SILVER_COLS}} and enriches each record with temporal intelligence — US federal holiday proximity, holiday-period membership, season, quarter, week-of-year, weekend flags — producing {{SILVER_ROWS}} Silver records. A Gold transformation builds the ML feature store: categorical indexing and one-hot encoding, HHMM→hour-of-day normalization, standardized numerics, and a {{GOLD_DIM}}-dimensional assembled feature vector, of which the top {{TOP_K}} features are retained by {{SELECTION_METHOD}}, preserving {{RETENTION}}% of information.

Four Spark ML classifiers — Random Forest and Gradient-Boosted Trees, each in pre-departure and in-flight variants — are tuned with Hyperopt Bayesian optimization over {{N_EVALS}} evaluations with {{K}}-fold cross-validation. The best in-flight model reaches **{{AUC_IN}} AUC-ROC / {{F1_IN}} F1**; the best pre-departure model, operating without `dep_delay`, reaches **{{AUC_PRE}} AUC-ROC / {{F1_PRE}} F1**. All runs are tracked in MLflow; the champions are registered in Unity Catalog under `workspace.flights.*` with a `@champion` alias and loaded by alias at inference, so retraining promotes a new model without a code change.

A separate ingestion path pulls live flights from the AviationStack API, validates them against the Silver schema, projects them into the identical Gold feature space, and scores them with both models. Results land in `flight_delay_predictions` alongside an alternative-flight recommender that surfaces lower-risk options within ±3 hours on the same route. The full pipeline runs daily as a Lakeflow job on Databricks serverless compute.

---

## Demo

<!-- Recruiters scroll ~2 screens. Put the proof here, not at the bottom. -->

| | |
|---|---|
| ![Dashboard](docs/screenshots/09_dashboard.png) | ![MLflow](docs/screenshots/04_mlflow_experiment.png) |
| **AI/BI dashboard** — delay risk by airline, route, and hour | **MLflow** — {{N}} tracked runs across 4 model variants |
| ![Lineage](docs/screenshots/02_lineage.png) | ![Job DAG](docs/screenshots/08_job_dag.png) |
| **Unity Catalog lineage** — bronze → silver → gold | **Lakeflow job** — scheduled daily run |

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│  Historical Flights     │         │   AviationStack API      │
│  {{N}} records          │         │   live flight lookup     │
│  {{SOURCE}}             │         │   (fixture fallback)     │
└───────────┬─────────────┘         └────────────┬─────────────┘
            │                                     │
            ▼                                     │
┌─────────────────────────┐                       │
│  BRONZE                 │                       │
│  workspace.flights      │                       │
│    .bronze_flights      │                       │
│  {{BRONZE_ROWS}} rows   │                       │
│  {{BRONZE_COLS}} cols   │                       │
│  raw + _ingested_at     │                       │
└───────────┬─────────────┘                       │
            │  dedup · null policy · rename        │
            │  holiday / season / temporal feats    │
            ▼                                     ▼
┌─────────────────────────┐         ┌──────────────────────────┐
│  SILVER                 │         │  API SILVER              │
│  {{SILVER_ROWS}} rows   │◄────────┤  schema-validated        │
│  {{SILVER_COLS}} cols   │ schema  │  against Silver contract │
└───────────┬─────────────┘ contract└────────────┬─────────────┘
            │  StringIndexer · OneHotEncoder       │
            │  StandardScaler · VectorAssembler    │
            ▼                                     ▼
┌─────────────────────────┐         ┌──────────────────────────┐
│  GOLD (feature store)   │         │  API GOLD                │
│  {{GOLD_DIM}}-dim vector│         │  same feature space      │
│  top {{TOP_K}} selected │         │  via feature_manifest    │
└───────────┬─────────────┘         └────────────┬─────────────┘
            │                                     │
            ▼                                     │
┌─────────────────────────┐                       │
│  TRAINING               │                       │
│  RF + GBT × pre/in      │                       │
│  Hyperopt TPE, {{K}}-CV │                       │
│  MLflow tracking        │                       │
└───────────┬─────────────┘                       │
            │  register + alias                    │
            ▼                                     │
┌─────────────────────────┐                       │
│  UNITY CATALOG MODELS   │──────────────────────►│
│  @champion aliases      │      load by alias    │
└─────────────────────────┘                       │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │  flight_delay_predictions│
                                    │  + alternative_flights   │
                                    │  → AI/BI Dashboard       │
                                    └──────────────────────────┘
```

### Component table

| Module | Path | Responsibility |
|---|---|---|
| Config | `src/config.py` | Single source of truth for catalog/schema/table/volume names |
| Bronze ingest | `notebooks/01_bronze.ipynb` | Raw CSV → Delta with ingestion metadata; idempotent |
| EDA | `notebooks/02_eda.ipynb` | Distribution, correlation, and delay-pattern analysis on Bronze |
| Silver | `notebooks/03_silver.ipynb` | Cleaning, column reduction, temporal + holiday feature engineering |
| Gold | `notebooks/04_gold.ipynb` | Encoding, scaling, vector assembly, feature-store write |
| Training | `notebooks/05_train.ipynb` | Hyperopt tuning, MLflow tracking, UC registration + aliasing |
| API ingest | `src/API_pipeline.py` | AviationStack fetch, schema validation, Silver/Gold projection |
| Scoring | `notebooks/07_score.ipynb` | Load champions by alias, batch score, write predictions |
| Dashboard SQL | `docs/Dashboard_SQL_queries.sql` | {{N}} queries backing the AI/BI dashboard |
| Job bundle | `databricks.yml` | Asset Bundle defining the scheduled pipeline |

---

## Tech stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Compute | Databricks Serverless | env v4 | Spark ML + `mlflow.spark` support on serverless |
| Processing | PySpark | {{VERSION}} | Distributed transformation across all layers |
| Storage | Delta Lake | {{VERSION}} | ACID tables, time travel, schema enforcement |
| Governance | Unity Catalog | — | Table + model governance, lineage, volumes |
| ML | Spark MLlib | {{VERSION}} | RF / GBT classifiers, feature pipeline |
| Tuning | Hyperopt | {{VERSION}} | TPE Bayesian hyperparameter search |
| Tracking | MLflow | {{VERSION}} | Experiments, model registry, aliases |
| Orchestration | Lakeflow Jobs | — | Daily scheduled DAG |
| Source | AviationStack API | v1 | Live flight lookup |
| Holidays | `holidays` | {{VERSION}} | US federal holiday calendar, 2020–2027 |

---

## Data pipeline

### Bronze — raw ingestion
{{ROWS}} rows × {{COLS}} columns, written as-is from `{{SOURCE_FILE}}` with `_ingested_at`. Nothing is dropped or cast at this layer so the pipeline can be replayed from source without re-downloading.

### Silver — cleaned and enriched
Reduced to {{SILVER_COLS}} columns. Cleaning: {{describe dedup / null policy / cancelled-flight handling}}. Enrichment adds:

| Feature | Type | Derivation |
|---|---|---|
| `is_holiday` | bool | Exact match against US federal calendar |
| `is_near_holiday` | bool | Within ±{{N}} days of a federal holiday |
| `is_holiday_period` | bool | Inside a defined high-travel window |
| `season` | string | Meteorological season from month |
| `day_of_week`, `week_of_year`, `quarter`, `is_weekend` | int/bool | Calendar decomposition |
| `dep_hour`, `arr_hour` | int | HHMM → hour of day |

<!-- One paragraph on WHY these features: holiday proximity is a real driver of network congestion. -->

### Gold — feature store
Categorical indexing + one-hot encoding, standardized numerics, assembled into a {{GOLD_DIM}}-dimensional vector. The top {{TOP_K}} indices are selected by {{METHOD}} and persisted to `workspace.flights.feature_manifest`, so scoring reconstructs the exact training feature space instead of relying on hardcoded indices.

Target: `is_delayed = arr_delay >= 15` minutes (FAA on-time standard). Class balance: {{X}}% / {{Y}}%.

---

## ML system

### Why two models

`dep_delay` dominates arrival-delay prediction, but it only exists after pushback. Training one model that uses it produces good metrics and an unusable product; training one that omits it discards the best signal for post-departure flights. Two models, one feature store, phase-selected at inference:

| Model | Features | Available | Use case |
|---|---|---|---|
| Pre-departure | {{N}} (no `dep_delay`) | Before pushback | Booking, scheduling, connection planning |
| In-flight | {{N}} (with `dep_delay`) | After pushback | Live arrival risk, gate/crew reallocation |

### Training configuration

| Setting | Value |
|---|---|
| Algorithms | RandomForestClassifier, GBTClassifier |
| Search | Hyperopt TPE, {{N}} evaluations |
| Validation | {{K}}-fold cross-validation |
| Split | {{X}}/{{Y}} train/test, seed {{SEED}} |
| Constraints | SparkML on serverless caps model size at 100 MB; search space bounded to `numTrees ≤ {{N}}`, `maxDepth ≤ {{N}}` |

### Results

| Model | Variant | AUC-ROC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Random Forest | Pre-departure | {{}} | {{}} | {{}} | {{}} | {{}} |
| GBT | Pre-departure | {{}} | {{}} | {{}} | {{}} | {{}} |
| Random Forest | In-flight | {{}} | {{}} | {{}} | {{}} | {{}} |
| GBT | In-flight | {{}} | {{}} | {{}} | {{}} | {{}} |

<!-- Then 2–3 sentences interpreting it. Where does it fail? Which class? What's the baseline
     (majority-class accuracy)? Naming your own weaknesses is the highest-trust move in a README. -->

![Confusion matrix](docs/screenshots/{{FILE}}.png)

### Registry and promotion

Champions are registered as `workspace.flights.{{model}}` and aliased `@champion`. Inference loads by alias:

```python
mlflow.set_registry_uri("databricks-uc")
model = mlflow.spark.load_model("models:/workspace.flights.gbt_pre_departure@champion")
```

Retraining registers a new version; moving the alias promotes it. No code change, no redeploy.

---

## Reproducing this project

### Prerequisites
- A [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition) workspace
- Serverless **environment version 4** (required for `pyspark.ml` and `mlflow.spark`)
- `{{SOURCE_FILE}}` from {{LINK}}
- Optional: an AviationStack API key. Without one, the pipeline runs against a committed fixture.

### Setup
```sql
CREATE SCHEMA IF NOT EXISTS workspace.flights;
CREATE VOLUME IF NOT EXISTS workspace.flights.raw;
```
Upload the source CSV to `/Volumes/workspace/flights/raw/`, clone this repo as a Databricks Git folder, then run `notebooks/01` → `07` in order. Expected row counts: Bronze {{N}} → Silver {{N}} → Gold {{N}}.

### Without a Databricks workspace
```bash
pip install -r requirements.txt
pytest tests/                 # unit tests for the pure transformation logic
```
Exported notebook HTML with full outputs is in `docs/notebooks/`.

---

## Design decisions

### 1. Medallion architecture over a single transformation script
{{2–4 sentences. Replayability, independently queryable layers, cleaning vs feature engineering separated.}}

### 2. Two phase-specific models over one general model
{{Target leakage framing — the pre-departure model deliberately gives up the strongest feature to be deployable.}}

### 3. Unity Catalog model registry over the workspace registry
Three-level naming, aliases, and lineage mean scoring code never references a version number. {{Note that the earlier iteration of this project logged to the workspace registry while inference read from UC — an unresolvable mismatch, and the reason the original pipeline never closed the loop.}}

### 4. Feature manifest over hardcoded indices
{{Selected feature indices are written to a table by the training run and read by scoring, so the two can never drift.}}

### 5. Recorded API fixture alongside live ingestion
{{Free-tier keys expire and rate-limit. The fixture keeps the pipeline demonstrable and unit-testable offline.}}

---

## Limitations

<!-- Do not skip this section. It is the difference between a student project and an engineer's project. -->

- {{Free-tier API returns ~100 records per call with limited fields; live scoring covers a narrow slice of traffic.}}
- {{SparkML's 100 MB serverless model cap bounds tree depth and count, which likely costs some AUC.}}
- {{Weather is a major delay driver and is absent from the feature set.}}
- {{Class imbalance / threshold choice — the 0.5 cutoff is not cost-optimal for an operational alerting use case.}}
- {{No drift monitoring; a retrained champion is promoted on offline metrics alone.}}

## Future work

- {{Weather API join at Silver}}
- {{Model serving endpoint + latency benchmark}}
- {{Lakehouse Monitoring on the predictions table}}
- {{Expectation-based data quality gates that fail the job, not just log}}

---

## Author

**Omar Elkady** — B.S. Data Science, Georgia State University
[GitHub](https://github.com/OmaryElkady) · [LinkedIn](https://www.linkedin.com/in/omar-elkady-847b051ba/)

## References
{{Dataset citation, AviationStack docs, FAA on-time definition, relevant Databricks docs}}
