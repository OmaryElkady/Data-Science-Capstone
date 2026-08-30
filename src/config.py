"""Single source of truth for catalog, schema, table, and volume names.

Every notebook and script imports from this module so that the pipeline
has exactly one place to change layout. Prevents the class of failure
where Gold_table writes to `default.gold_ml_features` and training reads
from `default.gold_ml_features_experimental`.
"""

from __future__ import annotations

CATALOG = "workspace"
SCHEMA = "flights"

# Delta tables
BRONZE = f"{CATALOG}.{SCHEMA}.bronze_flights"
SILVER = f"{CATALOG}.{SCHEMA}.silver_flights"
GOLD = f"{CATALOG}.{SCHEMA}.gold_ml_features"

# API pipeline tables (live scoring path)
API_BRONZE = f"{CATALOG}.{SCHEMA}.api_bronze_flights"
API_SILVER = f"{CATALOG}.{SCHEMA}.api_silver_flights"
API_GOLD = f"{CATALOG}.{SCHEMA}.api_gold_features"

# Governance + reproducibility artifacts
FEATURE_MANIFEST = f"{CATALOG}.{SCHEMA}.feature_manifest"
DATA_QUALITY_LOG = f"{CATALOG}.{SCHEMA}.data_quality_log"

# Scoring outputs
PREDICTIONS = f"{CATALOG}.{SCHEMA}.flight_delay_predictions"
ALTERNATIVES = f"{CATALOG}.{SCHEMA}.alternative_flight_recommendations"

# Volumes
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
ARTIFACT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts"

# Source file (uploaded manually into RAW_VOLUME)
SOURCE_CSV = f"{RAW_VOLUME}/flights_sample_3m.csv"

# Unity Catalog registered model names (3-level)
MODEL_RF_PRE = f"{CATALOG}.{SCHEMA}.rf_pre_departure"
MODEL_GBT_PRE = f"{CATALOG}.{SCHEMA}.gbt_pre_departure"
MODEL_RF_IN = f"{CATALOG}.{SCHEMA}.rf_in_flight"
MODEL_GBT_IN = f"{CATALOG}.{SCHEMA}.gbt_in_flight"

CHAMPION_ALIAS = "champion"

# MLflow
MLFLOW_REGISTRY_URI = "databricks-uc"
MLFLOW_EXPERIMENT = f"/Shared/flight-delay-platform"

# Modeling constants
DELAY_THRESHOLD_MINUTES = 15  # FAA on-time definition
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8
CV_FOLDS = 2
HYPEROPT_MAX_EVALS = 4  # bounded for Free Edition 100MB serverless cap
TOP_K_FEATURES = 40

# AviationStack
AVIATIONSTACK_BASE_URL = "https://api.aviationstack.com/v1/"
AVIATIONSTACK_SECRET_SCOPE = "flights"
AVIATIONSTACK_SECRET_KEY = "aviationstack_key"
