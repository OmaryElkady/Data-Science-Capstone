# Databricks Runbook

This is the exact sequence to bring the platform up on Databricks Free Edition and the artifacts to send back so the README can be finalized with real numbers.

Reading order: this file, then `MIGRATION_PLAN.md` for the "why" behind each step.

---

## Phase 0 — Workspace prep (~45 min)

1. Sign up at [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition).
2. **Complete LinkedIn verification** in Account Settings. This unlocks general outbound internet — required for AviationStack.
3. Open a scratch notebook. In the right-hand **Environment** panel, set **Environment version 4**. Verify:
   ```python
   import pyspark.ml, mlflow.spark, sys
   print(sys.version, pyspark.__version__, mlflow.__version__)
   ```
4. Create the schema and volumes:
   ```sql
   CREATE SCHEMA IF NOT EXISTS workspace.flights;
   CREATE VOLUME IF NOT EXISTS workspace.flights.raw;
   CREATE VOLUME IF NOT EXISTS workspace.flights.artifacts;
   ```
5. Download the [Kaggle 2019–2023 flights sample](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023) (`flights_sample_3m.csv`) to your laptop, then upload it into `/Volumes/workspace/flights/raw/`.
6. In the workspace, add this repo as a **Git folder** on branch `migration/v2` (needs a GitHub PAT with `repo` scope).
7. Create the secret scope + AviationStack key (optional — skip and use `USE_FIXTURE=true`):
   ```bash
   databricks secrets create-scope flights
   databricks secrets put-secret flights aviationstack_key
   ```

**Exit criteria:** the environment probe cell runs clean; the CSV is visible in Catalog Explorer under the volume.

---

## Phase 1 — Medallion pipeline

Run in order:

| # | Notebook | Report back |
|---|---|---|
| 1 | `notebooks/01_bronze.ipynb` | Bronze row count |
| 2 | `notebooks/02_eda.ipynb` | Screenshot the delay-by-hour chart |
| 3 | `notebooks/03_silver.ipynb` | Silver row count (after 2020 drop) |
| 4 | `notebooks/04_gold.ipynb` | Gold row count; positive-class rate |

Every layer uses full overwrite so reruns are idempotent — you'll see it in `DESCRIBE HISTORY` as sequential versions with matching row counts.

---

## Phase 2 — Training + Unity Catalog registry (~4–5 h)

Run `notebooks/05_train.ipynb`. Watch for:

- MLflow experiment appears at `/Shared/flight-delay-platform`.
- Four registered models appear in Catalog Explorer under `workspace.flights.*` each with a `@champion` alias.
- The final cell of the notebook is the Phase 2 exit test — loading the champion by alias from within the same session. It should print a `label` / `prediction` / `probability` table.

**Send back:** test AUC for each of the four models; **screenshot the exit-test output** (`07_model_load_success.png`).

---

## Phase 3 — Live API ingestion (~2–3 h)

Run `notebooks/06_api_ingest.ipynb`. Widget `USE_FIXTURE`:

- `true` → committed fixture. No key, no network. Use this first to confirm the projection works.
- `false` → live AviationStack. Requires the secret scope from Phase 0.

**Send back:** `data_quality_log` sample row; row count in `api_silver_flights`.

---

## Phase 4 — Scoring

Run `notebooks/07_score.ipynb`. It applies the fitted feature pipeline, loads all four champions by alias, ensembles RF+GBT, and writes:

- `flight_delay_predictions`
- `alternative_flight_recommendations`

**Send back:** row count in each; three sample rows from `flight_delay_predictions`.

---

## Phase 5 — Asset Bundle / Lakeflow Job

From your laptop (Databricks CLI installed and authenticated):

```bash
databricks bundle deploy -t dev
databricks bundle run flight_delay_pipeline -t dev
```

**Send back:** screenshot of the successful job DAG (`08_job_dag.png`).

---

## Phase 6 — Screenshots for the README

Save all at consistent width, light theme, browser chrome cropped, no email addresses in frame. Drop into `docs/screenshots/`. When you push them I'll wire them into the README.

| File | Shot |
|---|---|
| `01_catalog_explorer.png` | UC tree: catalog → schema → tables + volumes |
| `02_lineage.png` | Lineage tab on `gold_ml_features` |
| `03_pipeline_run.png` | Notebook cell output with row counts |
| `04_mlflow_experiment.png` | Experiment runs list sorted by AUC |
| `05_mlflow_compare.png` | Run comparison / parallel coordinates |
| `06_uc_models.png` | Registered models with `@champion` aliases |
| `07_model_load_success.png` | **Phase 2 exit test output** |
| `08_job_dag.png` | Lakeflow job graph, successful run |
| `09_dashboard.png` | AI/BI dashboard full view |
| `10_predictions_table.png` | Sample rows from `flight_delay_predictions` |
| `11_serving_endpoint.png` | (Optional) curl against a serving endpoint |

Also export each notebook to HTML (`File → Export → HTML`) into `docs/notebooks/`. That fills the promise the README makes about being viewable without Databricks.

---

## What to send me when you're done

Paste this block back into a message:

```
Row counts:
  Bronze: ____
  Silver: ____
  Gold:   ____
  Positive class rate: ____

Test AUC:
  RF pre-departure:  ____
  GBT pre-departure: ____
  RF in-flight:      ____
  GBT in-flight:     ____

API silver rows: ____
Prediction rows: ____
Recommendation rows: ____
```

And push the screenshots + notebook HTML exports to `docs/`. I'll finalize the README numbers, the model comparison paragraph, and the limitations section based on what actually happened.
