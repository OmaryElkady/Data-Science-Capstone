"""Single source of truth for catalog, table, volume and model names, and tuning knobs.

Every notebook imports from here, so the pipeline has one place to change layout.
A constant lives here only if something imports it.
"""

from __future__ import annotations

CATALOG = "workspace"
SCHEMA = "flights"

# Delta tables
BRONZE = f"{CATALOG}.{SCHEMA}.bronze_flights"
SILVER = f"{CATALOG}.{SCHEMA}.silver_flights"
GOLD = f"{CATALOG}.{SCHEMA}.gold_ml_features"

# Live path. There is no API gold table: scoring applies the Gold feature
# pipeline in memory.
API_BRONZE = f"{CATALOG}.{SCHEMA}.api_bronze_flights"
API_SILVER = f"{CATALOG}.{SCHEMA}.api_silver_flights"

FEATURE_MANIFEST = f"{CATALOG}.{SCHEMA}.feature_manifest"

PREDICTIONS = f"{CATALOG}.{SCHEMA}.flight_delay_predictions"
ALTERNATIVES = f"{CATALOG}.{SCHEMA}.alternative_flight_recommendations"

# Graded forecasts. Monitoring only, never a retraining set: live rows are a
# small, route-biased sample (see README, "Monitoring").
MONITORING = f"{CATALOG}.{SCHEMA}.prediction_monitoring"

# Below this many graded forecasts, 08 reports the count instead of drawing a
# reliability diagram: ten bins over 30 flights is three flights a bin.
MONITORING_MIN_SAMPLE = 30

# Volumes
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
ARTIFACT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts"
SOURCE_CSV = f"{RAW_VOLUME}/flights_sample_3m.csv"

# Unity Catalog models (3-level names)
MODEL_RF_PRE = f"{CATALOG}.{SCHEMA}.rf_pre_departure"
MODEL_GBT_PRE = f"{CATALOG}.{SCHEMA}.gbt_pre_departure"
MODEL_RF_IN = f"{CATALOG}.{SCHEMA}.rf_in_flight"
MODEL_GBT_IN = f"{CATALOG}.{SCHEMA}.gbt_in_flight"
CHAMPION_ALIAS = "champion"

MLFLOW_REGISTRY_URI = "databricks-uc"
MLFLOW_EXPERIMENT = "/Shared/flight-delay-platform"

# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
DELAY_THRESHOLD_MINUTES = 15   # US DOT on-time definition
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Split protocol (README, "Design decisions")
# ---------------------------------------------------------------------------
# Three windows by year. The threshold year stays out of CV because
# CrossValidator refits the champion on every fold, so any CV fold is in-sample.
TRAIN_END_YEAR = 2021          # CV window: 2019 + 2021 (2020 dropped in Silver)
THRESHOLD_YEAR = 2022          # used only to pick the decision threshold
TEST_YEAR = 2023               # touched once, for the reported metrics

CV_FOLDS = 4                   # 4 x 2 quarters; 5 would make uneven folds
SEARCH_CV_FOLDS = 3            # cheaper folds for the broad search
SEARCH_SAMPLE_FRACTION = 0.25  # the search runs on a sample; the winner refits on all

# ---------------------------------------------------------------------------
# Feature selection and hyperparameter search
# ---------------------------------------------------------------------------
# K comes out of the 05_train sweep; 0 means keep every feature, as a control.
# The selector is fitted inside each CV fold, never on the full data.
TOP_K_CANDIDATES = [5, 10, 20, 40, 80, 0]

# TPE needs ~20 startup trials before it beats random search.
HYPEROPT_MAX_EVALS = 25
STAGE2_TOP_N = 3               # configurations re-scored on the full CV window

# Serverless SparkML caps a model at 100 MB and a session at 1 GB.
RF_MAX_TREES = 60
RF_MAX_DEPTH = 10
GBT_MAX_ITER = 40
GBT_MAX_DEPTH = 7

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# ROC-AUC tunes (ranking); F-beta reports (decisions). beta=1 until the cost of a
# missed delay against a false alarm is decided.
TUNING_METRIC = "areaUnderROC"
REPORTING_BETA = 1.0

# 0.01 steps up to 0.60, where the optimum lives (base rate ~20%); 0.05 above.
# 0.50 stays on the grid for the "cost of Spark's default" comparison.
THRESHOLD_GRID = (
    [round(0.01 * i, 2) for i in range(2, 61)]      # 0.02 .. 0.60
    + [round(0.05 * i, 2) for i in range(13, 20)]   # 0.65 .. 0.95
)

# ---------------------------------------------------------------------------
# Live data sources. Credentials live in the `flights` secret scope, never here.
# ---------------------------------------------------------------------------
# AeroDataBox: schedules, gate times, status, and both OpenSky join keys
# (callSign and aircraft.modeS). Free tier: 400 units a month.
AERODATABOX_SECRET_SCOPE = "flights"
AERODATABOX_SECRET_KEY = "aerodatabox_key"

# OpenSky: live aircraft state, one /states/all call for the whole US.
OPENSKY_SECRET_SCOPE = "flights"
OPENSKY_CLIENT_ID_KEY = "opensky_client_id"
OPENSKY_CLIENT_SECRET_KEY = "opensky_client_secret"
OPENSKY_STATES = f"{CATALOG}.{SCHEMA}.opensky_states"

# ---------------------------------------------------------------------------
# Alternative-flight recommender
# ---------------------------------------------------------------------------
ALTERNATIVE_SEARCH_HOURS = 4          # +/- around the flight; one query caps at 12h
ALTERNATIVE_WINDOW_MINUTES = 240
ALTERNATIVE_TOP_N = 5
# Same-route flights score close together, so a 10-point bar never cleared.
# 3 points is about a sixth of the base rate.
ALTERNATIVE_MIN_IMPROVEMENT_PCT = 3.0
# An alternative must still be catchable: this far ahead of now, not of the
# original flight's departure.
ALTERNATIVE_MIN_LEAD_MINUTES = 60

# ---------------------------------------------------------------------------
# Scheduled route watch (notebook 10, job flight-delay-route-watch)
# ---------------------------------------------------------------------------
WATCH_ROUTES = "ATL-JFK,ORD-LGA,LAX-SFO"
# The airport query takes local time, so each watched origin needs its zone.
WATCH_ORIGIN_TZ = {
    "ATL": "America/New_York",
    "ORD": "America/Chicago",
    "LAX": "America/Los_Angeles",
}
WATCH_MAX_PER_ROUTE = 3      # forecasts per route per day, spread across the window
WATCH_WINDOW_HOURS = 12      # the provider's cap for one airport query
WATCH_LOOKBACK_DAYS = 3      # how long to keep trying to collect an outcome
# One flight per route already in the air, for the in-flight model: the morning query
# reaches back this far, and a departed flight counts only if departure plus the
# route's median block time, less the margin, is still ahead.
WATCH_AIRBORNE_PER_ROUTE = 1
WATCH_AIRBORNE_LOOKBACK_MINUTES = 120
WATCH_AIRBORNE_MARGIN_MINUTES = 20
# Below this many units the schedule stops calling AeroDataBox, leaving budget
# for manual runs of 06.
AERODATABOX_QUOTA_RESERVE = 40
