# Flight Delay Platform — Free Edition Migration & Completion Plan

Working document for rebuilding `Data-Science-Capstone` on **Databricks Free Edition**.
Drop this at the repo root so Claude Code has the full context in every session.

---

## 0. Root-cause analysis (what actually broke)

Four separate failures were stacked on top of each other. Only #1 was Community Edition's fault.

| # | Failure | Evidence in repo | Fix |
|---|---|---|---|
| 1 | **Outbound HTTP to AviationStack blocked by Databricks-side egress control** | `BASE_URL = 'http://api.aviationstack.com/v1/'` in `src/API_pipeline.py:90` | Free Edition restricts outbound internet to a trusted-domain allowlist. **LinkedIn identity verification unlocks general outbound internet access.** Do this first. |
| 2 | **Registry mismatch — models were written to one registry and read from another** | Training notebook sets `MLFLOW_REGISTRY_URI = "databricks"` (workspace registry, with the comment `# Changed from databricks-uc`), but `API_to_ML_Prediction_Dashboard.py:95-98` loads `models:/workspace.default.model_gbt_pre@flight` and `...Dashboardtest.py:102` sets `databricks-uc`. A workspace-registry model is not addressable by a 3-level UC name, so `mlflow.spark.load_model` fails no matter which compute runs it. | Register to UC with 3-level names from the training run itself. This is the actual "models saved in one place, code ran in another" bug. |
| 3 | **Spark ML was not supported on serverless** | Whole project is `pyspark.ml` + `mlflow.spark` | Serverless **environment version 5** added `pyspark.ml` and `mlflow.spark` support. Must be explicitly selected in the notebook Environment panel. |
| 4 | **Broken table lineage** | `Gold_table.ipynb` writes `default.gold_ml_features`; training reads `default.gold_ml_features_experimental` | Single config module, one source of truth for names. |

Also worth knowing before you start:
- **SparkML on serverless caps model size at 100 MB** (1 GB total in-memory per session). Tree training silently stops early if a model is about to exceed it. Your hyperopt space must be bounded accordingly.
- Free Edition quotas: 1 SQL warehouse (2X-Small), **5 concurrent job tasks**, 1 active pipeline per type, limited serving endpoints (no GPU, no batch inference on custom models), up to 3 Databricks Apps (auto-stop after 24h).
- No R/Scala, serverless only, one workspace + one metastore.

---

## Phase 0 — Workspace prep (~45 min)

- [ ] Create Free Edition account; **complete LinkedIn verification** (unlocks outbound internet — this is the unblock for Phase 3).
- [ ] Open a notebook → **Environment** side panel → set **environment version 5**. Confirm with:
  ```python
  import pyspark.ml, mlflow.spark, sys
  print(sys.version, pyspark.__version__, mlflow.__version__)
  ```
- [ ] Create the namespace. Stop using `default`; a real schema reads better in every screenshot:
  ```sql
  CREATE SCHEMA IF NOT EXISTS workspace.flights;
  CREATE VOLUME IF NOT EXISTS workspace.flights.raw;
  CREATE VOLUME IF NOT EXISTS workspace.flights.artifacts;
  ```
  Note: the old path `/Volumes/workspace/default/ds-capstone/...` uses a **hyphen**, which needs backticking everywhere and is a recurring source of errors. Use underscores.
- [ ] Upload the Kaggle `flights_sample_3m.csv` to `/Volumes/workspace/flights/raw/`. (Do this from your laptop — do not rely on `kagglehub` inside the notebook.)
- [ ] Connect the GitHub repo as a **Git folder** in the workspace (PAT with `repo` scope). All work happens on a branch, not in the workspace-only file tree.
- [ ] Create a secret scope for the API key (workspace-level APIs work on Free Edition):
  ```bash
  databricks secrets create-scope flights
  databricks secrets put-secret flights aviationstack_key
  ```

**Exit criteria:** env v5 notebook runs `import pyspark.ml` clean; CSV visible in Catalog Explorer under the volume.

---

## Phase 1 — Restore the medallion pipeline (~3–4 h)

- [ ] Add `src/config.py` — one place for catalog/schema/volume/table names, imported by every notebook:
  ```python
  CATALOG, SCHEMA = "workspace", "flights"
  BRONZE = f"{CATALOG}.{SCHEMA}.bronze_flights"
  SILVER = f"{CATALOG}.{SCHEMA}.silver_flights"
  GOLD   = f"{CATALOG}.{SCHEMA}.gold_ml_features"
  ```
  Kills failure #4 permanently. Every hardcoded `default.*` string in the repo gets replaced.
- [ ] Renumber notebooks so the run order is obvious to a reviewer:
  `01_bronze` → `02_eda` → `03_silver` → `04_gold` → `05_train` → `06_api_ingest` → `07_score` .
- [ ] Make each layer **idempotent** — re-running must not duplicate rows. Use `MERGE` or a deterministic full overwrite, and say which you chose in the README.
- [ ] Record row counts at each layer as you go. You need the real numbers for the README abstract:
  `Bronze ____ → Silver ____ → Gold ____`.
- [ ] Add a `DESCRIBE HISTORY` cell on the Gold table — cheap, and it demonstrates you understand Delta's transaction log.

**Exit criteria:** three Delta tables in UC, row counts recorded, full rerun from cold produces identical counts.

---

## Phase 2 — Training + Unity Catalog registry (~4–5 h) — *the real fix*

- [ ] In the training notebook, set the registry **before** the first run and register with a 3-level name:
  ```python
  mlflow.set_registry_uri("databricks-uc")
  mlflow.set_experiment("/Users/<you>/flight-delay-platform")

  with mlflow.start_run(run_name="gbt_pre_departure"):
      mlflow.spark.log_model(
          model,
          name="model",
          input_example=example_df,          # infers signature; drop the manual TensorSpec
          registered_model_name=f"{CATALOG}.{SCHEMA}.gbt_pre_departure",
      )
  ```
- [ ] Set the alias in the same run, right after registration, and use one consistent alias name (`champion`, not `flight`):
  ```python
  from mlflow import MlflowClient
  c = MlflowClient()
  v = c.get_registered_model(f"{CATALOG}.{SCHEMA}.gbt_pre_departure").latest_versions[0].version
  c.set_registered_model_alias(f"{CATALOG}.{SCHEMA}.gbt_pre_departure", "champion", v)
  ```
  Delete the whole `try/except` alias block sitting inside the `Config` class body — side effects in a class definition is the single worst code smell a reviewer will hit in this repo.
- [ ] Delete the CE-era workarounds: `MLFLOW_DFS_TMP` / `SPARKML_TEMP_DFS_PATH` env vars and `setup_uc_volume()`. Not needed on env v5; test after removing.
- [ ] **Bound the hyperopt space for the 100 MB cap**: RF `numTrees ≤ 50`, `maxDepth ≤ 8`; GBT `maxIter ≤ 30`, `maxDepth ≤ 6`. Log the serialized model size as a tag so the constraint is visible and defensible.
- [ ] **Stop hardcoding `SELECTED_INDICES_IN`.** That 40-integer list in `Config` is unreproducible — nobody, including future you, can regenerate it. Have the feature-selection step write the indices + feature names to `{CATALOG}.{SCHEMA}.feature_manifest` (or a JSON artifact on the run) and have scoring read them back.
- [ ] Log a confusion matrix and ROC curve as artifacts on each run — they become README images for free.

**Exit criteria — this is the test that closes the original defect.** Open a **brand-new notebook**, fresh session, and run only:
```python
import mlflow
mlflow.set_registry_uri("databricks-uc")
m = mlflow.spark.load_model("models:/workspace.flights.gbt_pre_departure@champion")
m.transform(spark.table("workspace.flights.gold_ml_features").limit(10)).show()
```
If that returns predictions, the thing that blocked you for the entire original project is dead. Screenshot it.

---

## Phase 3 — Live API ingestion (~2–3 h)

- [ ] Switch `BASE_URL` to **https**. It is currently `http://` — plaintext, and likely to be rejected by egress filtering even after verification.
- [ ] Pull the key from the secret scope: `dbutils.secrets.get("flights", "aviationstack_key")`. Never a literal.
- [ ] Add retry with exponential backoff + explicit handling for the free tier's 100-request cap and rate-limit responses.
- [ ] **Ship a recorded fixture.** Save one real API response to `tests/fixtures/aviationstack_sample.json` and add a `USE_FIXTURE` flag. Two reasons: your pipeline stays demoable when the key expires or quota runs out, and it gives you something to unit-test the parser against. Recruiters can clone and run it with no key at all — call this out in the README.
- [ ] Keep `validate_schema()`; log its results to a `{CATALOG}.{SCHEMA}.data_quality_log` table. Data-quality gates are exactly what a data-engineering interviewer probes for.

**Exit criteria:** a live lookup writes API silver + gold tables; the same notebook runs green with `USE_FIXTURE=True` and no network.

---

## Phase 4 — Scoring, consolidated (~3 h)

Right now there are three overlapping scripts, one of which literally says in a header cell:

> *"The three notebooks ... all are non-functional due to the continued restrictions we faced due to utilizing Databricks Community Edition"*

That sentence is currently rendered on GitHub for anyone who opens the file.

- [ ] **Delete** `API_to_ML_Prediction_Dashboardtest.py` and `API_to_ML_Ready_Complete.py`. Two dead near-duplicates (one of which defines `class Config` twice, at lines 59 and 126) do more damage to a reviewer's impression than the third file adds.
- [ ] Rewrite the survivor as `07_score_flights` : load feature manifest → apply the fitted feature pipeline → load both champion models → write `flight_delay_predictions` and `alternative_flight_recommendations`.
- [ ] Remove the "non-functional" disclaimer once it runs.

**Exit criteria:** `SELECT * FROM workspace.flights.flight_delay_predictions` returns scored rows from a live API lookup.

---

## Phase 5 — Orchestration (~1–2 h)

- [ ] Build one **Lakeflow Job** with dependent tasks: `bronze → silver → gold → score`, daily schedule, email on failure. Stay within the 5-concurrent-task quota (sequential deps are fine).
- [ ] Commit it as a **Databricks Asset Bundle** (`databricks.yml`) so the job is code, not clicks. For data-platform roles this is one of the highest-signal things in the repo.
- [ ] Screenshot the job DAG after a successful run.

---

## Phase 6 — Optional flourishes (only after 1–5 are green)

- [ ] **Model serving endpoint** for the pre-departure model (CPU custom models are permitted within Free Edition endpoint quota; no GPU, no batch inference). A `curl` against a live endpoint is a strong README block.
- [ ] **Databricks App** — a small Streamlit-style lookup UI. 3 apps allowed; auto-stops after 24h, so record a GIF rather than promising a live link.
- [ ] **AI/BI Dashboard** from `docs/Dashboard_SQL_queries.sql` (29 queries already written — this is nearly free value).

---

## Phase 7 — Evidence capture

Create `docs/screenshots/` and shoot these, in this order, at consistent window width, light theme, browser chrome cropped out:

| File | Shot |
|---|---|
| `01_catalog_explorer.png` | UC tree: catalog → schema → 3 medallion tables + volumes |
| `02_lineage.png` | Catalog Explorer **Lineage** tab on Gold — shows bronze→silver→gold visually |
| `03_pipeline_run.png` | Notebook cell output with layer row counts |
| `04_mlflow_experiment.png` | Experiment runs list sorted by AUC, all 4 models visible |
| `05_mlflow_compare.png` | MLflow run comparison / parallel-coordinates on hyperparameters |
| `06_uc_models.png` | Registered models with `@champion` aliases |
| `07_model_load_success.png` | **The Phase-2 exit test returning predictions** — the money shot |
| `08_job_dag.png` | Lakeflow job graph, successful run |
| `09_dashboard.png` | AI/BI dashboard, full view |
| `10_predictions_table.png` | `flight_delay_predictions` sample rows |
| `11_serving_endpoint.png` | Endpoint ready + curl response (if Phase 6) |

Rules: no personal email or workspace URL in frame; keep each under ~400 KB; reference them from the README with relative paths.

---

## Phase 8 — Repo hygiene (~2–3 h)

- [ ] `tests/` is empty while `.pre-commit-config.yaml` and `.flake8` exist — that gap reads as abandoned tooling. The pure functions in `API_pipeline.py` (`get_season`, `check_near_holiday`, `check_holiday_period`, `convert_utc_to_hhmm`, `fix_timestamp_smart`) are trivially testable with no Spark. Ten tests closes it.
- [ ] `.github/workflows/ci.yml`: ruff/black + pytest on push. Badge it. Your misinformation-lakehouse repo has a CI badge; this one should match.
- [ ] Export each notebook to HTML into `docs/` — **the current README already promises this and the folder only contains a SQL file.** Broken promises in a README are worse than no promise.
- [ ] Add `LICENSE` (MIT).
- [ ] Repo description + topics on GitHub: `databricks`, `delta-lake`, `mlflow`, `pyspark`, `medallion-architecture`, `mlops`.
- [ ] Decide on `src/kaggle_decision_tree.py` — it's a 725-line standalone sklearn/GPU track that isn't wired to the Databricks pipeline. Either frame it explicitly as the "local baseline before the Spark rewrite" with its numbers in the results table, or cut it. Unexplained orphan files invite the wrong question.
- [ ] Dial back the emoji in cell output and headers. Your misinformation-lakehouse README has zero emoji and reads senior; keep the portfolio consistent.

---

## Suggested sequencing

| Session | Phases | Outcome |
|---|---|---|
| 1 (3 h) | 0 + start 1 | Workspace live, data loaded, bronze rebuilt |
| 2 (4 h) | Finish 1 | Silver + Gold rebuilt, counts recorded |
| 3 (5 h) | 2 | **Models load from UC — original blocker dead** |
| 4 (3 h) | 3 | Live API ingestion working, fixture committed |
| 5 (3 h) | 4 + 5 | Predictions table + scheduled job |
| 6 (3 h) | 7 + 8 | Screenshots, tests, CI, README |
| 7 (opt) | 6 | Serving endpoint / app / dashboard |

## Working with Claude Code

Point it at this file and work one phase per session. Useful framing for each:
> "Phase 2 of MIGRATION_PLAN.md. Refactor `notebooks/05_train.ipynb` to register to Unity Catalog per the plan. Do not change the feature engineering logic. Show me the diff before writing."

Keep the Databricks side authoritative — edit in the Git folder, run, commit from the workspace, then pull locally. Editing the same notebook in both places at once is how you lose an afternoon.
