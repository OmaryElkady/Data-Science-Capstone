# Review Findings — migration/v2

> **Closed.** Every item below was found in a review of this branch and has since been
> fixed; the file is kept as a record of what the review caught, not as outstanding work.
> `P1` refers to `MIGRATION_PLAN.md` and `README_TEMPLATE.md`, which were moved to
> `docs/migration-notes.md` and replaced by a written `README.md` respectively.

Punch-list from a full read of the branch. Ordered by severity. Hand to Claude Code one section at a time.

---

## Blocking

### B1. `databricks.yml` points at Community Edition
```yaml
workspace:
  host: https://community.cloud.databricks.com   # ← CE. Deprecated. Not your workspace.
```
Replace with your Free Edition workspace URL (`https://dbc-xxxxxxxx-xxxx.cloud.databricks.com`). `databricks bundle deploy` fails until this is fixed. Better: drop `host` from the bundle entirely and let it resolve from your CLI profile, so the URL isn't hardcoded in a public repo.

Also verify `notebook_path: ../notebooks/01_bronze` resolves — bundle paths are relative to the YAML file, and your notebooks are `.ipynb`. Deploy once and read the plan output before running.

---

## Correctness — these change your reported numbers

### C1. Hyperparameters are tuned on the test set
In `05_train`, `objective()` returns `-test_auc`, `best_state` is selected by `test_auc`, and `test_auc` is then reported as the headline metric. The test set has been used for model selection, so every AUC in your README would be optimistically biased.

Fix — three-way split:
```python
train, val, test = gold.randomSplit([0.7, 0.15, 0.15], seed=config.RANDOM_SEED)
# objective() evaluates on `val` and returns -val_auc
# after fmin finishes, evaluate the winning model on `test` exactly once
```
Log both `val_auc` and `test_auc`. The gap between them is a talking point, not an embarrassment.

### C2. Random split on temporally ordered data
`gold.randomSplit(...)` puts April 2023 flights in train and March 2023 flights in test. Delay patterns are autocorrelated — same day, same airport, same weather system — so this leaks.

Fix — temporal holdout:
```python
train = gold.filter(F.col("flight_year") <= 2022)
test  = gold.filter(F.col("flight_year") == 2023)
```
This will lower your AUC. That is the point: the number becomes real, and "I used a temporal holdout because flight delays are autocorrelated within a day" is a sentence that ends the methodology part of an interview well.

### C3. `dep_delay` position is a hardcoded magic number
```python
pre_indices = [i for i in range(n_features) if i != 11]   # ← index 11 by comment only
```
This is the same failure class as the original `SELECTED_INDICES_IN`. If anyone reorders `numerical_cols` in `04_gold`, the pre-departure model silently trains **with** `dep_delay` — target leakage that shows up as a suspiciously good AUC, not as an error.

Fix — read the index out of the vector's own metadata and assert:
```python
attrs = gold.schema["features"].metadata["ml_attr"]["attrs"]
idx = {a["name"]: a["idx"] for group in attrs.values() for a in group}
dep_idx = idx["dep_delay"]
assert dep_idx not in pre_indices, "dep_delay leaked into the pre-departure view"
```

### C4. Feature manifest stores the wrong kind of position
`04_gold` writes `position=i` over `assembled_cols`, but one-hot columns expand into many vector slots. `position` equals the true vector index only for the numeric/boolean block at the front, and silently diverges after that. Anything downstream that trusts it for a categorical will be wrong.

Fix: build the manifest from the same `ml_attr` metadata as C3, storing real `vector_index`, `name`, and `attr_type`.

---

## Integrity — claims that don't match code

### I1. Dead config constants
| Constant | Uses outside `config.py` |
|---|---|
| `CV_FOLDS = 2` | **0** — there is no `CrossValidator` anywhere |
| `TOP_K_FEATURES = 40` | **0** — there is no feature selector anywhere |

Either implement them or delete them. Critically: **do not let the README claim k-fold CV or top-40 feature selection.** A reviewer who opens `05_train` and finds neither has just learned your README overstates. That costs more than the features were worth.

### I2. `HYPEROPT_MAX_EVALS = 4` is not a search
Four evaluations of a TPE sampler is barely more than random guessing — TPE needs ~20 startup trials before its surrogate model does anything. Either raise it to 20–25 (each RF fit on bounded depth is cheap), or keep it low and say plainly in the README: "search bounded to N evaluations by Free Edition compute quota."

### I3. Dead code
`_log_metrics()` in `05_train` is defined and never called. Delete.

### I4. `MLFLOW_EXPERIMENT = f"/Shared/flight-delay-platform"` — f-string with no placeholder (flake8 F541).

---

## Reproducibility

### R1. Hidden session state in `04_gold`
```python
try:
    _ = pipeline_model
    print("Reusing existing pipeline_model from previous run")
except NameError:
    ...
```
Notebook behavior now depends on whether a variable survives from an earlier execution. Same notebook, same inputs, two different code paths. This is the single most reviewer-visible smell left in the branch.

Fix — make it explicit and stateless:
```python
dbutils.widgets.dropdown("refit_pipeline", "auto", ["auto", "always", "never"])
pipeline_path = f"{config.ARTIFACT_VOLUME}/feature_pipeline"
# "auto": load from the volume if it exists, else fit and save
```
Keep the comment about the 1 GB serverless ML cache limit — it's a legitimate constraint and explaining it is a plus. Just don't implement it with `NameError`.

### R2. Notebooks committed without outputs
`02_eda` has 14 code cells and **zero** saved outputs. `03_silver` likewise. On GitHub these render as empty grey boxes. The EDA notebook is the one a hiring manager is most likely to open, and right now it shows nothing.

Run them clean top-to-bottom and commit with outputs.

### R3. `.databricks/commit_outputs` is `**` — decide deliberately
Committing all outputs is *good* for a portfolio repo (charts and tables render on GitHub without anyone running Databricks). The risk is `06_api_ingest`, whose outputs contain raw API payloads. Keep `**` and add an exclusion:
```
**
!notebooks/06_api_ingest*
```

---

## Repo presentation

### P1. Scaffolding is committed at the root
`MIGRATION_PLAN.md`, `README_TEMPLATE.md`, and `docs/RUNBOOK.md` are visible on the repo landing page. `README_TEMPLATE.md` in particular tells a visitor the README is unfinished.

Before the final push:
- `README_TEMPLATE.md` → becomes `README.md`, then delete the template
- `MIGRATION_PLAN.md` → `docs/migration-notes.md`, and keep it. The root-cause analysis is a genuine asset; it's evidence you can debug a platform, not just call `.fit()`. Just don't put it on the landing page.
- `docs/RUNBOOK.md` → keep, it's fine where it is

### P2. `tests/.gitkeep` alongside real test files — delete the placeholder.

---

## Optional, real improvement

### O1. Replace holiday UDFs with a broadcast join
`03_silver` calls four Python UDFs per row across ~2M rows. Each one serializes to a Python worker. A tiny date-dimension table (every date 2019–2024 with the four flags precomputed) broadcast-joined to Silver does the same work in a fraction of the time, entirely in the JVM.

This is worth doing not because the current version is too slow, but because "I profiled it, found Python UDF serialization was the bottleneck, and replaced it with a broadcast join against a date dimension" is a real data-engineering answer, and you'd have the before/after timing to prove it.
