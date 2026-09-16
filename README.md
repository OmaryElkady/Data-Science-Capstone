# Flight Delay Prediction Platform

[![CI](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/ci.yml)

![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-FF3621?logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Medallion-00ADD8?logo=delta&logoColor=white)
![Spark](https://img.shields.io/badge/Spark%20ML-3.5-E25A1C?logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.8-0194E2?logo=mlflow&logoColor=white)
![Unity Catalog](https://img.shields.io/badge/Unity%20Catalog-%40champion-1B3139)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-136%20passing-success)

A Bronze→Silver→Gold Delta pipeline over 3M BTS flight records, feeding two Spark ML models
registered in Unity Catalog and served against live flights from two APIs.

---

## Abstract

Two models answer the same question at different moments. The **in-flight** model knows the
actual departure delay; the **pre-departure** model does not, and is the only one that is
useful before an aircraft leaves the gate.

The in-flight model scores **0.9256 test ROC-AUC**. A logistic regression on `dep_delay`
*alone* scores **0.9261**. The 818 other engineered features buy **`-0.0004`** — they make it
very slightly worse than the one-feature baseline.

That result is the project's spine. It would have been easy to report "0.93 AUC flight delay
predictor" and stop; the baseline is what makes that claim indefensible, and running it is
what redirected the work. The pre-departure variant — where the features have to do the
work — scores **0.6214 test ROC-AUC, 0.3974 F1** against a majority-class floor of 0.5.

Everything downstream follows from taking that honestly: a temporal split because delay
cascades within an operating day, a third window because the decision threshold cannot be
chosen on data the model trained on, a tie rule because a margin inside fold noise is not a
result, and a threshold of 0.19 rather than 0.5 because at 0.5 the pre-departure model
predicts almost nothing and scores **F1 = 0.0016**.

---

## Architecture

```
  BTS extract (3,000,000 rows, 33 cols)          AeroDataBox            OpenSky Network
         │                                    schedules + gate           /states/all
         │                                        times                       │
         ▼                                           │                        │
  ┌─────────────┐                                    │              7,540 aircraft, 1 call
  │   BRONZE    │  raw, replayable                   │              667 ground / 6,873 air
  │  3,000,000  │  no casts, no drops                │                        │
  └──────┬──────┘                                    ▼                        ▼
         │                                    ┌──────────────────────────────────┐
         ▼                                    │        API BRONZE / SILVER       │
  ┌─────────────┐                             │  flight + alternatives + phase   │
  │   SILVER    │  2,520,650 rows             └────────────────┬─────────────────┘
  │  cleaned +  │  2020 dropped (COVID)                        │
  │  enriched   │  broadcast date dimension                    │
  └──────┬──────┘  CHECK constraints, ZORDER                   │
         │                                                     │
         ▼                                                     │
  ┌─────────────┐                                              │
  │    GOLD     │  2,463,979 labelled rows                     │
  │  feature    │  819-slot vector                             │
  │   store     │  feature manifest (the contract)             │
  └──────┬──────┘                                              │
         │                                                     │
         ▼                                                     │
  ┌──────────────────────┐                                     │
  │      05_train        │   CV ≤2021 │ threshold 2022 │ test 2023
  │  blocked CV, TPE     │                                     │
  └──────────┬───────────┘                                     │
             ▼                                                 ▼
    ┌──────────────────┐                          ┌────────────────────────┐
    │  Unity Catalog   │◄──── models:/…@champion ─│       07_score         │
    │  rf_pre_departure│      threshold read off  │  verdict + alternatives│
    │  rf_in_flight    │      the model version   └────────────────────────┘
    └──────────────────┘
```

### Component table

| Layer | Notebook | Writes | Row count |
|---|---|---|---|
| Bronze | `01_bronze` | `bronze_flights` | 3,000,000 |
| Silver | `02_silver` | `silver_flights` | 2,520,650 |
| Gold | `04_gold` | `gold_ml_features`, `feature_manifest` | 2,463,979 / 819 |
| Training | `05_train` | 2 UC models + `@champion` aliases | — |
| Ingest | `06_api_ingest` | `api_silver_flights`, `opensky_states` | per run |
| Scoring | `07_score` | `flight_delay_predictions`, `alternative_flight_recommendations` | per run |
| Monitoring | `08_monitor` | `prediction_monitoring` | resolved forecasts |
| EDA | `03_eda` | nothing — analysis only | — |

---

## Why Delta, not Parquet

Every one of these is used in the pipeline, not listed aspirationally.

| Feature | Where | Why |
|---|---|---|
| `CHECK` constraints | Silver, Gold | The filters are invariants. `ALTER TABLE ADD CONSTRAINT` validates existing rows, so a clean run *is* the data-quality assertion |
| `OPTIMIZE … ZORDER` | Silver, Gold | Every training split filters `flight_year`; Z-ordering lets those filters skip files |
| `MERGE` | Predictions | Idempotent upsert on (flight, date, run) — re-running after a failure updates rather than duplicates |
| Time travel | Silver | `DESCRIBE HISTORY` answers "did this number move because the model changed or the data did?" |
| `autoOptimize` | Silver, Gold | Reasonable file sizes on write instead of a compaction job later |

---

## Results

All figures from `fast_mode=false` runs. Test window is **2023-01-01 to 2023-08-31** — the
source extract ends there, so the holdout is 8 of 12 months and contains neither Thanksgiving
nor Christmas. It is reported that way rather than as "the 2023 holdout".

| Model | Test ROC-AUC | PR-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| Majority class | 0.5000 | — | 0.0000 | — | — | — |
| **`dep_delay` alone (logistic)** | **0.9261** | — | 0.8067 | — | — | 0.50 |
| In-flight (RF, 819 slots) | 0.9256 | 0.8848 | 0.8136 | 0.8739 | 0.7610 | 0.41 |
| **Pre-departure (RF)** | **0.6214** | 0.3294 | 0.3974 | 0.2906 | 0.6282 | 0.19 |

**Lift of the in-flight model over one feature: `-0.0004`.** Not "small" — *negative*. Every
piece of feature engineering in this project, applied to the variant that already knows the
departure delay, produces a model fractionally worse than a logistic regression on that one
column. Both numbers sit well inside noise, which is the point: they are the same model.

Pre-departure cross-validates at **0.6159 ± 0.0315** against a test score of 0.6214 — a gap of
`+0.0055`, well inside one fold standard deviation. It generalises.

The in-flight CV mean is **0.8461 ± 0.1165** against a test score of 0.9256, and the notebook
refuses to quote the mean on its own: the spread is larger than most of the differences anyone
would want to read off it. Stratifying the folds within each year brought that spread down
from 0.1517 but did not remove it, and `05_train` checks the obvious explanation and rules it
out — the folds differ by only **3.34%** in delay rate and **3.96 minutes** in mean
`dep_delay`, so fold composition is not what is moving the score.

### Threshold

Selected on the 2022 window, measured there, then applied unchanged to 2023.

| | Pre-departure | In-flight |
|---|---|---|
| F1 at the tuned threshold | 0.3773 @ 0.19 | 0.8098 @ 0.41 |
| F1 at Spark's default 0.50 | **0.0016** | 0.8038 |
| Cost of the default | **+0.3757 F1** | +0.0059 F1 |
| Within 1% of best F1 | 0.16 – 0.21 | 0.29 – 0.52 |

At 0.5 the pre-departure model predicts almost nothing. Both optima are plateaus rather than
peaks, which is worth more than the point estimate: anything in those bands performs the same,
so the exact cut is not load-bearing and a reader should not treat 0.19 as precise.

The window is carved out *before* cross-validation, because `CrossValidator.fit()` refits
`bestModel` on its entire input and any CV fold is therefore in-sample by the time a champion
exists.

**The advisory cut exists for one model and not the other.** It is defined as the lowest
threshold whose precision clears 50% *and* whose recall clears 5% — a cut can pass the first
test by flagging almost nothing, and precision measured on a handful of rows is noise wearing
a guarantee. In-flight has one at **0.07** (precision 0.517, recall 0.888). Pre-departure has
none: the best threshold reaching 50% precision is 0.49 and it catches **0.1%** of delayed
flights. `07_score` falls back to the F1 cut and says so. A 0.62-AUC model is simply never
reliably right when it calls a flight late, and that is worth stating rather than hiding
behind a cut chosen for a different purpose.

### Calibration

Isotonic regression, fitted on 2022 and applied to 2023. **It works for one model and not
the other, and the honest reading is not the same in both columns.**

| | Pre-departure raw | Pre-departure calibrated | In-flight raw | In-flight calibrated |
|---|---|---|---|---|
| Largest bin deviation | 0.0750 | 0.0925 | 0.0918 | **0.0247** |
| Brier score | 0.1752 | **0.1712** | 0.0637 | **0.0632** |
| Test ROC-AUC | 0.6214 | 0.6214 | 0.9256 | 0.9256 |

For the in-flight model it does what it is supposed to: bin deviation falls by a factor of
almost four.

For the pre-departure model the two measures **disagree**. The Brier score improves, and Brier
is a proper scoring rule — it cannot be gamed by a calibration map that makes the summary
statistic look better. The largest single bin deviation gets worse. The most likely reading is
that the miscalibration is not stable between 2022 and 2023, so a map fitted on one year is
partly mis-applied to the next; `05_train` says exactly that rather than quietly reporting
whichever number flatters the method.

It ships anyway, on the strength of the proper scoring rule and because the alternative is
serving a probability the threshold was not selected against. That is a judgement, and it is
recorded here as one.

ROC-AUC cannot move in either column: isotonic regression is monotonic, so it corrects
confidence without reordering anything.

**What reading the wrong column would have cost.** The threshold is selected on
`p_calibrated`. Applying that same 0.19 to the raw probability instead — one number, two
scales, no error anywhere — gives F1 **0.3457** against **0.3974**, and flags 24.1% of flights
instead of 49.8%. An earlier version of `05_train` did precisely that: it selected on raw
scores, reported on raw scores, then registered a calibrated pipeline tagged with the raw-scale
cut.

The calibrator is a *stage of the registered model*, not a measurement beside it. `05_train`
appends it to the champion's own fitted stages and registers the combined pipeline, so the
artifact in Unity Catalog serves `p_calibrated` and `07_score` reads that column. The
decision threshold is selected on the same scale, before any metric is reported: choosing a
cut against raw scores and then serving remapped ones is wrong on every row and raises
nothing.

---

## What the EDA decided

`03_eda` sizes each opportunity before anything is built.

**Delay causes**, over 489,792 delayed flights and 33,315,131 delay minutes:

| Cause | Share of delay minutes | Share of delayed flights |
|---|---|---|
| Late aircraft | **38.4%** | 49.8% |
| Carrier | 36.2% | 56.1% |
| NAS (airspace/ATC) | 19.3% | 47.7% |
| Weather (extreme) | **5.8%** | 5.8% |
| Security | 0.2% | 0.5% |

Late aircraft is delay *propagating*, and the share climbs from **15.2% before 09:00 to 44.4%
after 18:00** — the cascade made visible. Late aircraft plus NAS is **57.7%** of delay
minutes and needs no external data, which is why congestion features came before any weather
API. Extreme weather is 5.8%, and a pre-departure model would have a *forecast* at inference
time rather than the observation it trained on.

**Other findings:** hour-of-day risk spans 8.5% at 05:00 to 28.1% at 19:00 (3.3×);
`dep_delay` correlates 0.967 with arrival delay against 0.097 for the next feature; a
do-nothing classifier scores 80.1% accuracy, which is why accuracy is never the headline.

**A negative result, kept:** the three engineered holiday flags move the delay rate by
+0.21pp, −0.42pp and +0.87pp against a ~20% base rate. `is_near_holiday` is *negative*. They
remain in Silver as a general cleaned layer, and the feature selector discards them. Nothing
in this README claims they helped.

---

## Design decisions

**Temporal split, not random.** Delay cascades within an operating day, so a random split
puts the 07:00 departure that caused a delay in train and the 14:00 flight it delayed in
test. CV ≤2021, threshold 2022, test 2023.

**Blocked CV folds.** `CrossValidator` partitions randomly by default, which reintroduces
exactly the leak the outer split removed. `foldCol` supplies contiguous quarters instead.
Two fold columns exist, at `CV_FOLDS` and `SEARCH_CV_FOLDS` granularity, because the values
must be exactly `[0, numFolds)` and the search stages use fewer folds than the confirming one.

**The one-standard-error rule.** The K sweep's argmax moved between runs (10, then 40) while
every candidate stayed inside one standard deviation. The selected K did not move. Picking
the argmax would have crowned fold noise.

**A tie rule that is applied, not printed.** Both variants tied on margin against pooled fold
sd (0.0030 vs 0.0151; 0.0031 vs 0.2195), so both champions are random forests — bagging is
parallel and cheaper to serve than sequential boosting. Registering the argmax of a margin
under a fifth of a standard deviation would be registering noise.

**Feature manifest as a contract.** `ml_attr` metadata does not survive `StandardScaler` plus
a Delta round-trip — confirmed empirically. `04_gold` writes one row per vector slot and
`05_train` reads it, so a magic index cannot drift into target leakage.

**Two APIs, by phase.** AeroDataBox answers *how late* on gate semantics — the same quantity
BTS records as `DEP_DELAY`. OpenSky answers *where and what phase*: one call returned 7,540
aircraft, and `on_ground` is the split the two models need. Deriving delay from ADS-B would
give wheels-off, which differs by taxi-out and is worst at the congested airports where delay
matters most.

**Airframe matching.** AeroDataBox returns `aircraft.modeS`, which is OpenSky's `icao24`.
Matching on it identifies a specific aeroplane rather than a flight number — codeshares share
a number but never an aircraft.

**Local clocks for clock features.** BTS records `CRS_DEP_TIME` as local time at the origin,
so the model learned hour-of-day risk on a local clock. Durations and delays use UTC, where
subtracting two instants is unambiguous across a DST boundary.

---

## Engineering

**Python UDFs → broadcast join.** Four UDFs ran per row across 2.5M rows; all four flags are
functions of the date, and there are only 1,704 distinct dates. Materialising a date
dimension in the driver and broadcasting it moved the work into the JVM:

```
4 Python UDFs   :   37.79s
broadcast join  :    1.38s
speedup         :   27.29x          (200,000-row sample, .write.format("noop"))
```

`build_date_dimension` is unit-tested against the very helpers it replaces across every day
of 2019, so the refactor is proven rather than hoped for. `spark_day_of_week` pins Spark's
1=Sunday convention against Python's 0=Monday — a silent one-day shift otherwise.

**136 tests**, no cluster required. Fixtures are recorded live payloads, so the parsers are
tested against the shape the APIs actually return.

---

## CI

[![CI](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/ci.yml)

Everything here *runs* on Databricks. Nothing runs in GitHub Actions, and the split is
deliberate:

| | GitHub Actions | Databricks |
|---|---|---|
| Checks | code, notebook structure, job definitions | the actual pipeline |
| Needs | nothing | Spark, Unity Catalog, a 3M-row volume, three API secrets |
| Cost | free, every commit | metered, deliberately triggered |
| Time | under a minute | minutes to hours |

CI therefore never executes `01`-`08`. It cannot. What it can do is catch every failure that
does not need a cluster, and four jobs do that:

- **Lint** - `flake8` on `src/` and `tests/`, `nbqa flake8` on the notebooks
- **Unit tests** - 136 tests with coverage. `src/*.py` holds no module-level `SparkSession`,
  by design, which is what lets the suite run without installing PySpark
- **Notebook contracts** - `tools/validate_notebooks.py`: every cell parses, every
  `config.X` resolves, every notebook carries its contract header, and no cell reads a name
  that no earlier cell binds
- **Databricks bundle** - `databricks bundle validate` when workspace credentials are
  present, plus `tools/validate_bundle_paths.py`, which needs none

That last check exists because the bundle really did break. The notebooks were renumbered,
`databricks.yml` kept pointing its silver task at `./notebooks/03_silver.ipynb`, and
`databricks bundle deploy` - the command this README tells you to run - failed. The
credential-free half of the check catches that on a fork, on an outside PR, and on any clone
without a Databricks account.

The test suite is Spark-free for the same reason the notebooks are not run here: a CI job
that needs a cluster is a CI job nobody keeps green.

---

## Reproducing

**Prerequisites:** Databricks Free Edition, serverless environment version 5 (required for
`pyspark.ml` and `mlflow.spark`), `flights_sample_3m.csv` in `/Volumes/workspace/flights/raw/`.

```bash
databricks secrets create-scope flights
databricks secrets put-secret flights aerodatabox_key
databricks secrets put-secret flights opensky_client_id
databricks secrets put-secret flights opensky_client_secret
```

Run `00_setup` - it makes one live call per API and reports quota. Then `01_bronze` -> `02_silver`
-> `04_gold` -> `05_train`. `03_eda` reads and writes nothing and can run at any point.

For live scoring, `06_api_ingest` takes **one required input** - a flight number - plus a
date and two optional tie-breakers:

```
FLIGHT_NUMBER   DL1572          (dl1572, DL 1572, DL-1572 all work)
FLIGHT_DATE     2026-09-14      today is the only date a live aircraft match can succeed
ORIGIN          ATL             optional - only needed if the number flies >1 leg that day
DESTINATION     IAH             optional - same
```

Origin, destination, times, distance, aircraft and both OpenSky join keys are derived from
the flight number. `ORIGIN`/`DESTINATION` do not cost a call and are not a lookup - they
only pick between legs the same response already returned. See **Using it** below.

### Jobs are defined in code, not in the UI

`databricks.yml` is a Databricks Asset Bundle, and it is the source of truth for both jobs.
Deploying it creates them in the workspace, where they behave exactly like jobs built by
clicking: they appear under **Jobs & Pipelines**, run from the UI, and show the same run
history and task graph. The difference is where the definition lives - a job built in the
form exists only in one workspace, cannot be reviewed, and cannot be validated by CI.

```bash
pip install databricks-cli          # or: brew install databricks/tap/databricks
databricks auth login --host https://<your-workspace>.cloud.databricks.com

databricks bundle validate -t dev   # parses, resolves every notebook_path
databricks bundle deploy   -t dev   # creates/updates both jobs in the workspace
```

The jobs then appear as `[dev <your-username>] flight-delay-training` and
`[dev <your-username>] flight-delay-scoring`. Run them from the UI, or:

```bash
databricks bundle run flight_delay_training -t dev
```

**What `mode: development` does for you.** It prefixes job names so a deploy cannot collide
with anything else in the workspace, and it force-pauses every schedule. The scoring job is
declared `PAUSED` as well, deliberately: a live schedule on a metered free tier consumes
quota every morning whether or not anyone is watching.

**One gotcha worth knowing.** `bundle deploy` uploads the notebooks from your *local working
tree* to a bundle-managed workspace path, and the deployed job runs that copy - not the
Databricks Git folder you edit in. So the loop is: edit in the Git folder, commit and push
from there, `git pull` locally, then `bundle deploy`. Skipping the pull deploys whatever your
local checkout last had. (A job can instead be pointed at a `git_source` so it pulls the
branch at run time, which removes the step at the cost of making every run depend on GitHub.)

Training is manual and expensive - a full `05_train` run is a 25-trial TPE search plus blocked
CV on ~1.3M rows. Scoring is cheap and its schedule ships paused.

---

## Using it

The pipeline answers one question - *will this specific flight arrive 15+ minutes late* - and
the answer depends on **when you ask relative to the flight**. That is not a quirk of the
implementation; it is the whole finding. Before pushback nobody knows the departure delay, and
that single fact is worth 0.30 ROC-AUC.

### Run 06, then 07. Always in that order.

`06_api_ingest` fetches. `07_score` scores what was fetched. `07` on its own re-scores the
last ingestion, which is useful for re-reading a result and useless for asking about a new
flight.

### When to run it, and what you get

| When you run `06` + `07` | `dep_delay` | Model used | What you get |
|---|---|---|---|
| Days before departure | unknown | pre-departure | The honest forecast: schedule, route, carrier, time of day. **This is the intended use.** |
| Within a few hours of departure | usually still unknown | pre-departure | Same forecast, plus live FAA airspace conditions for the origin and destination |
| After pushback, before landing | known | in-flight | A much stronger estimate - the model that knows how late the aircraft actually left |
| After it lands | known | in-flight | No longer a forecast. The `OUTCOME` block reports whether the earlier call was right |

**So: no, do not wait until the flight has landed.** Landing is when the answer stops being a
prediction. Run it before departure for the number you can act on; run it again afterwards
only if you want to see the forecast checked against what happened. `07_score` prints that
check automatically whenever `arrival_delay` has been filled in, so a second run after arrival
is how the project closes its own loop - one flight at a time, accumulating in
`flight_delay_predictions`.

### The API cannot see far ahead

AeroDataBox serves a window around today - a few days back, fewer forward. Outside it, a flight
number returns no legs and `06` stops with a message naming the date and the number rather than
writing an empty table. Practically:

- **A date more than a few days out will not resolve.** Ask again closer to the day.
- **Today is the only date the OpenSky match can succeed.** A live ADS-B snapshot contains
  aircraft that are moving now; a flight from another day is not in it at all. On any other
  date the phase comes back `unknown` and everything routes to the pre-departure model, which
  is correct behaviour, not a failure.
- **A past date works** and comes back with actual times filled in.

### When the flight is not found

`06` raises with the number and the date rather than guessing. In order of likelihood:

1. **Date outside the provider's window.** Move it nearer to today.
2. **The number does not operate that day.** Many are not daily.
3. **Wrong `ORIGIN`/`DESTINATION`.** If the number flies that day but not that leg, the error
   lists the legs it *does* fly. Clear both widgets to take the next one due to depart.

### Why you give a flight number and not a route

A flight number plus a date identifies a flight; a route does not. `ATL -> IAH` on a Tuesday is
thirty flights. So the route is **derived** from
`/flights/number/{number}/{date}`, which returns origin, destination, times, aircraft and both
OpenSky join keys in a single response - the one call this notebook had to make regardless.
Deriving the route costs nothing extra, and `06` makes exactly two AeroDataBox calls per run:
one for the flight, one for the same-route alternatives.

`ORIGIN`/`DESTINATION` exist for one case only. A flight number can operate several legs in a
day - DL1572 might fly ATL->IAH in the morning and IAH->ATL in the afternoon - and "DL1572
today" does not say which. Left blank, `06` takes the **next leg due to depart**, which is
usually what a traveller means. Filled in, they pin the leg outright, and an explicit
instruction beats the heuristic. They filter legs already returned; they never trigger a
lookup.

### Reading the output

`07_score` ends with the flight in plain language. Five things to read, in order:

- **`prediction`** - the model's call at its own tuned threshold. Not 0.5: at Spark's default
  the pre-departure model predicts almost nothing and scores F1 = 0.0016. For a pre-departure
  flight this is the F1 cut, because that variant has no usable advisory cut - see
  **Threshold** above. For an in-flight one it is the advisory cut at 0.07.
- **`vs_route`** - this flight against the median flight on the same route that day. Usually
  the more actionable of the two. A 25% risk is bad news when the alternatives sit at 12% and
  simply the price of the route when they sit at 24%.
- **`AIRSPACE CONDITIONS`** - live FAA NAS status, printed beside the forecast and explicitly
  **not** a model input. There is no historical archive, so the column cannot be built for
  2019-2023 and a model that never saw it cannot be scored on it.
- **`FLIGHTS WITH A BETTER CHANCE`** - same route, within a few hours, codeshares excluded.
- **`OUTCOME`** - only once the flight has arrived.

One subject per run. `06` stamps an `ingest_run_id` and clears the flag on every earlier row,
so `is_flight_of_interest` means "the flight being asked about right now" rather than "was
asked about once". If `07` reports more than one, the table predates that behaviour - re-run
`06`.

---

## Monitoring, and why it is not retraining

`08_monitor` grades forecasts against outcomes. A prediction resolves when the flight lands
and a later run of `06` re-fetches it, filling in `arrival_delay`; `08` joins the two and
writes `prediction_monitoring`.

**It is deliberately not a retraining loop.** The obvious move is to accumulate live rows and
feed them back into Gold. The numbers say not to. AeroDataBox's free tier is 400 units a
month and `06` spends two calls a run, so the live path yields a handful of flights per run
against **2,463,979** Gold rows - adding one percent to the training set would take years.

Volume is not even the binding objection. Those rows are whichever flights someone typed into
a widget, so they are a biased sample of one or two routes. Retraining on them would pull the
model toward those routes and report it as improvement. Measuring that and declining to build
it is the result.

The same rows answer three questions that nothing else in this repo can, because every other
evaluation here is against a held-out slice of the same 2019-2023 extract:

1. **Is it still calibrated?** The isotonic stage was fitted on 2022 and confirmed on 2023.
   Calibration is a property of a distribution, not of a model, and distributions move.
2. **Which cut was right?** The F1 and advisory thresholds disagree by construction. With
   outcomes the disagreement becomes scorable.
3. **Does it fail when the NAS is degraded?** `07` records FAA conditions at prediction time
   and the model has never seen them. If misses concentrate under active conditions, that is
   the evidence for collecting NAS history and building the feature. If they do not, the
   caption stays a caption - and that is equally a result.

`08` reports per variant and never pools them: re-scoring a flight after pushback replaces
the pre-departure forecast with an in-flight one, and pooling would credit the 0.62 model
with the 0.93 model's accuracy. Below `MONITORING_MIN_SAMPLE` resolved flights it prints the
count and **refuses to draw a reliability diagram**, because one drawn on eight flights looks
exactly as authoritative as one drawn on eight thousand.

---

## Limitations

**The pre-departure model is weak, and that is the finding.** 0.6214 ROC-AUC. Every model
family tied inside fold noise — a linear model matching a boosted ensemble means the features
are exhausted, not that the search was inadequate. The honest conclusion is that scheduled
attributes of a flight do not determine whether it will be late; what happens on the day
does, and the schedule barely proxies it.

**The source is a ~10% extract, and it costs more than volume.** 3M rows against roughly
26M real BTS movements for these years. Counting co-occurring flights in a sample counts
*sampled* flights, which is why the congestion features are expressed as shares: a ratio
survives uniform sampling and a count does not. `09_rotation_probe` measures the size of that
distortion on one month of full-population data — see below. The sample also drops `TAIL_NUM`
entirely, which is what puts the largest single delay cause out of reach.

**The test year is 8 months.** 2023-01-01 to 2023-08-31, missing the Thanksgiving and
Christmas peaks.

**Base-rate drift.** 18.23% delayed in the CV window against 23.03% in the test year, `+4.80`
points. Some of the test-set degradation is distribution shift rather than overfitting.

**Calibration helps one model and is ambiguous on the other.** In-flight bin deviation
falls 0.0918 → 0.0247. Pre-departure improves on Brier (0.1752 → 0.1712) and worsens on largest
bin deviation (0.0750 → 0.0925), which most likely means the miscalibration is not stable
between 2022 and 2023. The calibrator and the threshold are both fitted on the 2022 window, so
the selection sweep is in-sample for the calibration step; the 2023 evaluation is the only
clean measurement of either.

**The in-flight fold spread is unexplained.** 0.8461 ± 0.1165 in cross-validation against
0.9256 on test. Stratifying folds within each year brought the spread down from 0.1517 but did
not remove it, and the obvious cause is ruled out: the folds differ by 3.34% in delay rate and
3.96 minutes in mean `dep_delay`. The CV mean is reported with that caveat attached rather
than quoted on its own.

**Feature importance is concentrated and the measure is biased.** The top 5 of 819 slots carry
91.0% of Gini importance, and Gini is biased toward high-cardinality features, which inflates
the one-hot airport columns. Permutation importance on the test set is the unbiased
alternative and was not run.

**Live gate-delay data is a paid product.** This runs on free tiers: AeroDataBox's free plan
is metered at 400 API units per month, and OpenSky's coverage is community ADS-B. Codeshares
are filtered because one aircraft sold under three flight numbers would otherwise appear as
three alternatives to itself.

---

## Future work

### Aircraft rotation — measured, costed, and not built

Late aircraft is **38.4% of delay minutes**, the largest single cause, and the feature that
would capture it needs `TAIL_NUM`, which this extract does not have. That was an unmeasured
claim for as long as this README existed. `09_rotation_probe` settles it on one month of
full-population BTS data, downloaded with `tools/fetch_bts.py`, without retraining anything.

**Scheduled turnaround** — the gap between the previous leg's scheduled arrival and this
leg's scheduled departure, same airframe — separates delay rates further than any feature
currently in the model:

| Scheduled turnaround | Arrived 15+ min late |
|---|---|
| under 30 min | **54.47%** |
| over 120 min | **23.28%** |
| *spread* | ***+31.2pp (2.34×)*** |

For scale, hour of day — the strongest feature the model currently has — spans 8.5% to 28.1%,
a spread of 19.6pp. On 386,017 paired legs from January 2023, with 0% missing tail numbers.

It is **schedule-only**, so it is knowable days ahead and carries no leakage. The inbound
aircraft's *actual* arrival delay is the stronger signal and is deliberately excluded: it is
not available before pushback, so a model trained on it would be served without it.

**Why it is not built.** The full BTS table is roughly **twelve times** the rows — about 29M
across these four years against 2,463,979 in Gold. `05_train` already runs about ten hours on
Databricks Free Edition. That is not a feature addition on this platform, and the probe was
built so the decision could be made on evidence rather than on the ten hours.

Two secondary results fall out of the same month. The aircraft's leg number that day separates
only 20.86% to 26.13%, so the carrier-level `dep_sequence_in_day` proxy already in Gold was
not missing much. And the sampling distortion the share features exist to route around is
visible at hubs rather than in a global mean, where small airports dominate.

### The rest

Ranked by the decomposition rather than by interest. Derived congestion features next — late
aircraft plus NAS is 57.7% of delay minutes and needs no new data. Time-aware route target
encoding would replace ~800 one-hot columns with a handful of dense rates, fitted inside CV
folds on prior time only, since it is the easiest way to leak. NAS as a real feature requires
collecting FAA status daily until there is enough history; `08_monitor` is the measurement
that would justify it. Weather last, if at all, at 5.8% of delay minutes and with
forecast-versus-observation skew.

**Lakeflow Declarative Pipelines.** The medallion layers here are imperative notebooks driven
by jobs, with data quality enforced by 20 Delta `CHECK` constraints across Silver and Gold
rather than by pipeline expectations. The declarative form would express the same constraints
as first-class quality rules with per-rule violation metrics. It is a rewrite of `01`–`04`
rather than an addition, which is why it is here and not above.

---

## Author

**Omar Elkady** — B.S. Data Science, Georgia State University

## References

- Bureau of Transportation Statistics, On-Time Performance
- [OpenSky Network](https://opensky-network.org/) — community ADS-B
- [AeroDataBox](https://aerodatabox.com/) — schedules and gate times
- Delta Lake, MLflow, Spark ML documentation
