# Flight Delay Prediction Platform

![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-FF3621?logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Medallion-00ADD8?logo=delta&logoColor=white)
![Spark](https://img.shields.io/badge/Spark%20ML-3.5-E25A1C?logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.8-0194E2?logo=mlflow&logoColor=white)
![Unity Catalog](https://img.shields.io/badge/Unity%20Catalog-%40champion-1B3139)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-119%20passing-success)

A Bronze→Silver→Gold Delta pipeline over 3M BTS flight records, feeding two Spark ML models
registered in Unity Catalog and served against live flights from two APIs.

---

## Abstract

Two models answer the same question at different moments. The **in-flight** model knows the
actual departure delay; the **pre-departure** model does not, and is the only one that is
useful before an aircraft leaves the gate.

The in-flight model scores **0.9262 test ROC-AUC**. A logistic regression on `dep_delay`
*alone* scores **0.9261**. The 817 engineered features buy **0.0001**.

That result is the project's spine. It would have been easy to report "0.93 AUC flight delay
predictor" and stop; the baseline is what makes that claim indefensible, and running it is
what redirected the work. The pre-departure variant — where the features have to do the
work — scores **0.6321 test ROC-AUC, 0.4134 F1** against a majority-class floor of 0.5.

Everything downstream follows from taking that honestly: a temporal split because delay
cascades within an operating day, a third window because the decision threshold cannot be
chosen on data the model trained on, a tie rule because a margin inside fold noise is not a
result, and a threshold of 0.17 rather than 0.5 because at 0.5 the pre-departure model
predicts **zero** delays and scores **F1 = 0.0000**.

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
  │  feature    │  816-slot vector                             │
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
| Silver | `03_silver` | `silver_flights` | 2,520,650 |
| Gold | `04_gold` | `gold_ml_features`, `feature_manifest` | 2,463,979 / 816 |
| Training | `05_train` | 2 UC models + `@champion` aliases | — |
| Ingest | `06_api_ingest` | `api_silver_flights`, `opensky_states` | per run |
| Scoring | `07_score` | `flight_delay_predictions`, `alternative_flight_recommendations` | per run |
| EDA | `02_eda` | nothing — analysis only | — |

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
| In-flight (RF, 817 features) | 0.9262 | 0.8846 | 0.8137 | 0.8758 | 0.7597 | 0.39 |
| **Pre-departure (RF)** | **0.6321** | 0.3266 | 0.4134 | 0.2969 | 0.6801 | 0.17 |

**Lift of the in-flight model over one feature: `+0.0001`.**

Cross-validated 0.6177 ± 0.0099 against a test score of 0.6321 — a gap of `+0.0154`, inside
about 1.5 fold standard deviations. The pre-departure model generalises.

The in-flight CV mean is **0.7684 ± 0.1517** against a test score of 0.9262. That spread is
larger than the quantity being measured, so the notebook flags it as unusable rather than
quoting it: the blocked folds straddle 2019 and 2021, which are different operating regimes.

### Threshold

| | Pre-departure | In-flight |
|---|---|---|
| F1 at the tuned threshold | 0.3842 @ 0.17 | 0.8097 @ 0.38 |
| F1 at Spark's default 0.50 | **0.0000** | 0.8029 |
| Cost of the default | **+0.3842 F1** | +0.0068 F1 |

At 0.5 the pre-departure model predicts no delays at all. The threshold is selected on the
2022 window — carved out *before* cross-validation, because `CrossValidator.fit()` refits
`bestModel` on its entire input and any CV fold is therefore in-sample by the time a champion
exists.

### Calibration

Isotonic regression, fitted on 2022 and applied to 2023:

| | Raw | Calibrated |
|---|---|---|
| Largest bin deviation | 0.1241 | **0.0247** |
| Brier score | 0.1766 | 0.1708 |
| Test ROC-AUC | 0.6321 | 0.6321 — unchanged |

ROC-AUC cannot move: isotonic regression is monotonic, so it corrects confidence without
reordering anything.

---

## What the EDA decided

`02_eda` sizes each opportunity before anything is built.

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

**119 tests**, no cluster required. Fixtures are recorded live payloads, so the parsers are
tested against the shape the APIs actually return.

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

Run `00_setup` — it makes one live call per API and reports quota. Then `01` → `03` → `04` →
`05`. `02_eda` reads and writes nothing and can run any time.

For live scoring, `06_api_ingest` takes **two inputs** — a flight number and a date:

```
FLIGHT_NUMBER   DL1572          (dl1572, DL 1572, DL-1572 all work)
FLIGHT_DATE     2026-09-14      today is the only date a live aircraft match can succeed
```

Origin, destination, times, distance, aircraft and both OpenSky join keys are derived.

```bash
databricks bundle deploy -t dev
databricks bundle run flight_delay_training -t dev
```

Training is manual; scoring is scheduled and ships paused, because a live schedule on a
metered free tier consumes quota whether or not anyone is watching.

---

## Limitations

**The pre-departure model is weak, and that is the finding.** 0.6321 ROC-AUC. Every model
family tied inside fold noise — a linear model matching a boosted ensemble means the features
are exhausted, not that the search was inadequate. The honest conclusion is that scheduled
attributes of a flight do not determine whether it will be late; what happens on the day
does, and the schedule barely proxies it.

**The source is a ~10% extract.** 3M rows against roughly 26M real BTS movements for these
years. Two congestion features had to be dropped for exactly this reason: counting
co-occurring flights counts *sampled* flights, and `sched_deps_origin_hour` came back at a
mean of 3.3 per airport-hour where a hub runs 50–80.

**The test year is 8 months.** 2023-01-01 to 2023-08-31, missing the Thanksgiving and
Christmas peaks.

**Base-rate drift.** 18.23% delayed in the CV window against 23.03% in the test year, `+4.80`
points. Some of the test-set degradation is distribution shift rather than overfitting.

**Calibration is improved but imperfect.** 0.0247 largest bin deviation after isotonic. The
calibrator is measured but not yet logged with the model, so `07_score` still serves raw
probabilities.

**Live gate-delay data is a paid product.** This runs on free tiers: AeroDataBox's free plan
is metered at 400 API units per month, and OpenSky's coverage is community ADS-B. Codeshares
are filtered because one aircraft sold under three flight numbers would otherwise appear as
three alternatives to itself.

---

## Future work

Ranked by the decomposition rather than by interest. Derived congestion features first —
late aircraft plus NAS is 57.7% of delay minutes and needs no new data. Aircraft rotation
chaining would be the strongest addition available, but `TAIL_NUM` is absent from this
extract, so `dep_sequence_in_day` is the closest proxy. Time-aware route target encoding
would replace ~800 one-hot columns with a handful of dense rates. Logging the isotonic
calibrator with the model would close the gap between measured and served probabilities.
Weather last, if at all, at 5.8% of delay minutes and with forecast-versus-observation skew.

---

## Author

**Omar Elkady** — B.S. Data Science, Georgia State University

## References

- Bureau of Transportation Statistics, On-Time Performance
- [OpenSky Network](https://opensky-network.org/) — community ADS-B
- [AeroDataBox](https://aerodatabox.com/) — schedules and gate times
- Delta Lake, MLflow, Spark ML documentation
