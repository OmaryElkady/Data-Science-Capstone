# Flight Delay Prediction Platform

[![Databricks](https://img.shields.io/badge/Databricks-Serverless_env_v4-FF3621?logo=databricks&logoColor=white)](https://databricks.com/)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Governed-1B3139)](https://www.databricks.com/product/unity-catalog)
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-Medallion-00ADD8)](https://delta.io/)
[![MLflow](https://img.shields.io/badge/MLflow-UC_Registry-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-39_passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Production-shaped flight-delay platform on **Databricks Free Edition**. Historical FAA flight records flow through a Bronze → Silver → Gold Delta Lake medallion pipeline, train two paired Spark ML models governed by Unity Catalog, and score live AviationStack flights into a predictions table plus an alternative-flight recommender — orchestrated as a scheduled Lakeflow Job defined by a Databricks Asset Bundle.

> This is a rebuild of an earlier Databricks Community Edition project that hit four stacked failures (blocked outbound egress, workspace/UC registry mismatch, no `pyspark.ml` on serverless, broken table lineage). See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) for the root-cause analysis that shaped this rewrite.

---

## What's in the box

| Layer | Path | Responsibility |
|---|---|---|
| Config | [`src/config.py`](src/config.py) | Single source of truth for every catalog/schema/table/volume/model name. |
| Features | [`src/features.py`](src/features.py) | Pure temporal + holiday helpers. Unit-tested, no Spark required. |
| API ingest lib | [`src/api_pipeline.py`](src/api_pipeline.py) | AviationStack fetch + retry/backoff + Silver projection + DQ check. |
| Bronze | [`notebooks/01_bronze.ipynb`](notebooks/01_bronze.ipynb) | Raw CSV → Delta with ingestion metadata. Idempotent overwrite. |
| EDA | [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb) | Distribution and delay-pattern audit on Bronze. |
| Silver | [`notebooks/03_silver.ipynb`](notebooks/03_silver.ipynb) | Column reduction, 2020 filter, calendar + holiday enrichment. |
| Gold | [`notebooks/04_gold.ipynb`](notebooks/04_gold.ipynb) | Feature pipeline fit + Delta feature store + feature manifest. |
| Training | [`notebooks/05_train.ipynb`](notebooks/05_train.ipynb) | 4 models × Hyperopt TPE, MLflow tracking, UC registration + `@champion`. |
| API ingest | [`notebooks/06_api_ingest.ipynb`](notebooks/06_api_ingest.ipynb) | Live fetch or fixture fallback → API Silver, DQ-gated. |
| Scoring | [`notebooks/07_score.ipynb`](notebooks/07_score.ipynb) | Load champions by alias, ensemble, write predictions + alternatives. |
| Orchestration | [`databricks.yml`](databricks.yml) | Asset Bundle with the daily Lakeflow Job. |
| Dashboard SQL | [`docs/Dashboard_SQL_queries.sql`](docs/Dashboard_SQL_queries.sql) | 29 queries backing the AI/BI dashboard. |
| Runbook | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Step-by-step of what to run on Databricks to reproduce this. |

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│  Historical Flights     │         │   AviationStack API      │
│  Kaggle 3M sample       │         │   live lookup or         │
│  flights_sample_3m.csv  │         │   committed fixture      │
└───────────┬─────────────┘         └────────────┬─────────────┘
            │                                     │
            ▼                                     │
┌─────────────────────────┐                       │
│  BRONZE                 │                       │
│  bronze_flights         │                       │
└───────────┬─────────────┘                       │
            │ column reduction · 2020 drop        │
            │ calendar + US holiday features       │
            ▼                                     ▼
┌─────────────────────────┐         ┌──────────────────────────┐
│  SILVER                 │         │  API SILVER              │
│  silver_flights         │◄────────┤  schema-validated        │
└───────────┬─────────────┘ contract│  (data_quality_log)      │
            │ StringIndexer · OneHotEncoder       │
            │ VectorAssembler · StandardScaler     │
            ▼                                     ▼
┌─────────────────────────┐         ┌──────────────────────────┐
│  GOLD (feature store)   │         │  API features            │
│  gold_ml_features       │         │  (same fitted pipeline)  │
│  + feature_manifest     │         │                          │
└───────────┬─────────────┘         └────────────┬─────────────┘
            │                                     │
            ▼                                     │
┌─────────────────────────┐                       │
│  TRAINING               │                       │
│  RF + GBT × pre/in      │                       │
│  Hyperopt TPE           │                       │
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

---

## The ML system: why two models

Arrival-delay prediction has a data-availability twist: the strongest single predictor is `dep_delay`, which does not exist until the aircraft has left the gate. Training a model with `dep_delay` scores well but is useless for pre-flight planning; training a model without it is deployable but weaker.

This platform serves **two models against the same feature store**, selected at inference time by flight phase:

| Model | Uses `dep_delay` | Available at | Use case |
|---|---|---|---|
| **Pre-departure** | No | Before pushback | Booking, connections, scheduling |
| **In-flight** | Yes | After pushback | Live arrival risk, gate reallocation |

The training notebook fits Random Forest and GBT variants of each. Champions are registered under 3-level Unity Catalog names (e.g. `workspace.flights.gbt_pre_departure`) and given the `@champion` alias. Scoring loads by alias, so retraining promotes a new version without a code change.

```python
mlflow.set_registry_uri("databricks-uc")
model = mlflow.spark.load_model("models:/workspace.flights.gbt_pre_departure@champion")
```

Search space is bounded (`numTrees ≤ 50`, `maxDepth ≤ 8`, `maxIter ≤ 30`) to respect the SparkML-on-serverless 100 MB model size cap.

---

## Reproducing this project

### Fast path: unit tests on your laptop (no Databricks)

```bash
pip install -r requirements.txt
pytest
```

This runs 39 tests over the pure temporal / holiday / API-projection logic in `src/`. The AviationStack fixture at [`tests/fixtures/aviationstack_sample.json`](tests/fixtures/aviationstack_sample.json) makes the tests hermetic — no API key, no network.

### Full pipeline on Databricks Free Edition

Read [`docs/RUNBOOK.md`](docs/RUNBOOK.md). The short version:

1. Complete **LinkedIn verification** on Free Edition to unlock outbound internet (this is the "blocked egress" fix from the migration plan).
2. Set your notebook to **serverless environment version 4** (required for `pyspark.ml` + `mlflow.spark`).
3. Create schema + volumes:
   ```sql
   CREATE SCHEMA IF NOT EXISTS workspace.flights;
   CREATE VOLUME IF NOT EXISTS workspace.flights.raw;
   CREATE VOLUME IF NOT EXISTS workspace.flights.artifacts;
   ```
4. Upload `flights_sample_3m.csv` to `/Volumes/workspace/flights/raw/`.
5. Clone this repo as a Databricks **Git folder** on the `migration/v2` branch.
6. Store the AviationStack key in the `flights` secret scope (optional — omit and set the widget `USE_FIXTURE=true` in `06_api_ingest`).
7. Run notebooks `01` → `07` in order. Deploy the Lakeflow Job with `databricks bundle deploy`.

---

## Design decisions

**Medallion over a single script.** Bronze is replayable from source with no re-download; Silver is independently queryable for the EDA and dashboard; Gold is the training contract. Failures don't have to reset the world.

**Unity Catalog registry over workspace registry.** The single biggest defect in the original project was training against `mlflow.set_registry_uri("databricks")` while scoring read `models:/workspace.default.model@flight` — an unresolvable 3-level name against a workspace-registry model. Both sides now use `databricks-uc` and 3-level UC names.

**Feature manifest over hardcoded indices.** The original scoring script pinned a 40-integer `SELECTED_INDICES_IN` list — unreproducible if anyone changed the feature engineering. The Gold notebook now writes the assembled column order to `workspace.flights.feature_manifest`, so training and scoring can never disagree on layout.

**Committed API fixture.** The AviationStack free tier caps at 100 requests. The fixture at [`tests/fixtures/aviationstack_sample.json`](tests/fixtures/aviationstack_sample.json) keeps the pipeline demoable when the key is missing, expired, or throttled — and gives the unit tests something to exercise the parser against.

**Asset Bundle for orchestration.** `databricks.yml` puts the daily Lakeflow Job under source control instead of the workspace UI. `bundle deploy` from CI would promote it to any other workspace unchanged.

---

## Results

<!-- Filled in after training runs. See docs/RUNBOOK.md for what to send back. -->

| Layer | Rows |
|---|---|
| Bronze | _TBD after 01_bronze run_ |
| Silver | _TBD after 03_silver run_ |
| Gold | _TBD after 04_gold run_ |

| Model | Variant | AUC-ROC (test) |
|---|---|---|
| Random Forest | Pre-departure | _TBD_ |
| GBT | Pre-departure | _TBD_ |
| Random Forest | In-flight | _TBD_ |
| GBT | In-flight | _TBD_ |

Screenshots in `docs/screenshots/` (see the runbook checklist for what to capture).

---

## Limitations

- Free-tier AviationStack returns ~100 records per call; live scoring covers a narrow slice of live traffic. The fixture path exists so demos don't depend on the quota.
- Serverless SparkML caps model artifacts at 100 MB (1 GB session RAM). The Hyperopt space is bounded to respect this, which likely costs a few AUC points versus deeper trees.
- Weather is a major delay driver and is absent from the feature set. Adding a Silver-layer join against a weather API is the highest-value follow-up.
- Threshold is 0.5 out of the box. For an operational alerting use case, the cost-optimal cut is likely lower.
- No drift monitoring; champion promotion is on offline test AUC alone.

## Future work

- Weather join at Silver
- Model Serving endpoint + latency benchmark (CPU custom models are allowed on Free Edition)
- Lakehouse Monitoring on the predictions table
- Expectation-based DQ gates that fail the job, not just log to a table

---

## Author

**Omar Elkady** — B.S. Data Science, Georgia State University
[GitHub](https://github.com/OmaryElkady) · [LinkedIn](https://www.linkedin.com/in/omar-elkady-847b051ba/)

## References

- FAA on-time performance definition (arrival delay ≥ 15 minutes)
- [AviationStack API docs](https://aviationstack.com/documentation)
- [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition)
- Root-cause + migration writeup: [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md)
