# Screenshots

Evidence that the platform runs, for a reader who will not clone the repo and will not open
Databricks. Four shots, in descending order of what they prove.

Drop the PNGs in this directory using **exactly these filenames** — the README's image section
references them by name, and `.gitignore` whitelists `docs/screenshots/*.png` against the
repo-wide `*.png` rule.

| File | What to capture | Why this one |
|---|---|---|
| `01_uc_champion.png` | Unity Catalog → `workspace.flights.rf_pre_departure` → the version carrying `@champion`, with the **tags panel open** showing `decision_threshold`, `advisory_threshold` and `calibrated` | The most important shot in the set. A probability model without its threshold is half a decision system, and this is the proof the threshold travels with the artifact rather than living in a notebook someone has to read |
| `02_job_dag.png` | Jobs & Pipelines → `[dev <you>] flight-delay-scoring` → a successful run → the task graph showing `api_ingest → score → monitor` | Proof the Asset Bundle deployed and the jobs are real. This is the shot a Databricks engineer looks at first |
| `03_mlflow_runs.png` | The MLflow experiment, run list sorted by ROC-AUC, with the nested stage runs expanded | Shows the selection protocol left a trail: the search trials, the confirming CV runs and the champion are all one lineage rather than a single logged number |
| `04_exit_test.png` | `05_train`'s exit-test cell output — both champions reloaded by 3-level UC name and `@champion` alias, scoring rows, showing `p1` beside `p_calibrated` | A round trip, not a coincidence: the model reloads by alias in a fresh session and serves the calibrated column its threshold was selected against |

## Capture notes

- **Dark or light is fine, but pick one** and use it for all four. A set that switches theme
  halfway reads as four unrelated screenshots rather than one walkthrough.
- **Crop to the pane that matters.** A full 4K desktop shrunk into a README column is
  unreadable, and unreadable evidence is not evidence.
- **Do not crop out the workspace URL** in `02_job_dag.png` — the `[dev <you>]` prefix is what
  shows the job came from `databricks bundle deploy` rather than from a form.
- Check for anything that should not be public before committing: tokens, secret values, other
  people's names. The workspace hostname is already in the repo's git history, so that one is
  not worth redacting.
