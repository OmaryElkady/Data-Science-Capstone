"""Single source of truth for catalog, schema, table, and volume names.

Every notebook and script imports from this module so the pipeline has exactly
one place to change layout. Prevents the class of failure where Gold writes to
`gold_ml_features` and training reads `gold_ml_features_experimental`.

Rule for this file: a constant lives here only if something imports it. An
unused constant is a claim the code does not make.
"""

from __future__ import annotations

CATALOG = "workspace"
SCHEMA = "flights"

# Delta tables
BRONZE = f"{CATALOG}.{SCHEMA}.bronze_flights"
SILVER = f"{CATALOG}.{SCHEMA}.silver_flights"
GOLD = f"{CATALOG}.{SCHEMA}.gold_ml_features"

# API pipeline tables (live scoring path). There is no API_GOLD: the live path
# applies the Gold feature pipeline in memory at scoring time rather than
# persisting a second feature table, so the name described a table that was
# never written.
API_BRONZE = f"{CATALOG}.{SCHEMA}.api_bronze_flights"
API_SILVER = f"{CATALOG}.{SCHEMA}.api_silver_flights"

# Governance + reproducibility artifacts
FEATURE_MANIFEST = f"{CATALOG}.{SCHEMA}.feature_manifest"

# Scoring outputs
PREDICTIONS = f"{CATALOG}.{SCHEMA}.flight_delay_predictions"
ALTERNATIVES = f"{CATALOG}.{SCHEMA}.alternative_flight_recommendations"

# Volumes
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
ARTIFACT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts"
SOURCE_CSV = f"{RAW_VOLUME}/flights_sample_3m.csv"

# Unity Catalog registered model names (3-level)
MODEL_RF_PRE = f"{CATALOG}.{SCHEMA}.rf_pre_departure"
MODEL_GBT_PRE = f"{CATALOG}.{SCHEMA}.gbt_pre_departure"
MODEL_RF_IN = f"{CATALOG}.{SCHEMA}.rf_in_flight"
MODEL_GBT_IN = f"{CATALOG}.{SCHEMA}.gbt_in_flight"
CHAMPION_ALIAS = "champion"

# MLflow
MLFLOW_REGISTRY_URI = "databricks-uc"
MLFLOW_EXPERIMENT = "/Shared/flight-delay-platform"

# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
DELAY_THRESHOLD_MINUTES = 15   # US DOT / FAA on-time definition — see 02_eda §1
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Split protocol — see 02_eda §3 for the evidence
# ---------------------------------------------------------------------------
# Delay cascades within an operating day, so a random split leaks: the 07:00
# departure that caused the delay lands in train while the 14:00 flight it
# delayed lands in test. Outer split is a temporal holdout; the inner CV folds
# are contiguous time blocks, not random rows.
#
# Three windows, not two. `CrossValidator.fit()` refits `bestModel` on its
# entire input, so any fold of the CV window has been trained on by the time
# the champion exists. Selecting the decision threshold on such a fold reports
# an optimistically biased cut. THRESHOLD_YEAR is therefore carved out *before*
# cross-validation and never enters CV.
#
# `03_silver` drops 2020 as a COVID anomaly, so the usable inventory is
# 2019, 2021, 2022, 2023 — four years, not five. Holding 2022 out of CV costs a
# third of the training window (12 quarters -> 8). That is the price of an
# unbiased threshold, and it is worth paying: a threshold picked in-sample makes
# every precision/recall number downstream unfalsifiable.
TRAIN_END_YEAR = 2021         # CV window: 2019 + 2021 = 8 quarters
THRESHOLD_YEAR = 2022         # held out of CV; used only to pick the threshold
TEST_YEAR = 2023              # touched exactly once, in Stage 3

# 4, not 5: the CV window is 8 quarters, so 4 folds divide it evenly into 2-quarter
# blocks. 5 folds would produce uneven folds (2,2,1,2,1 quarters), which inflates
# the fold-to-fold standard deviation that Stage 2a uses to call ties.
CV_FOLDS = 4                   # passed to CrossValidator via foldCol
SEARCH_CV_FOLDS = 3            # cheaper folds during the broad search stage
SEARCH_SAMPLE_FRACTION = 0.25  # stage-1 search runs on a sample; winner refits on all

# ---------------------------------------------------------------------------
# Feature selection — K is chosen by the sweep in 05_train, not assumed
# ---------------------------------------------------------------------------
# UnivariateFeatureSelector (ANOVA F-test) is fitted INSIDE each CV fold as a
# Pipeline stage. Selecting on the full dataset before splitting would let the
# selector see validation rows and inflate every score that follows.
# K=5 added after the first full run, where every one of [10, 20, 40, 80, all]
# landed inside one standard deviation of the best. A sweep where nothing is
# distinguishable has not found the point where feature count starts to matter,
# it has only shown that the point is below the smallest value tried.
TOP_K_CANDIDATES = [5, 10, 20, 40, 80, 0]   # 0 = keep all features (the control)
# There is deliberately no TOP_K_FEATURES here. It was a magic 40 that nothing
# derived and the README should not claim (REVIEW_FINDINGS I1). K is now the
# output of the Stage 1 sweep, held in the notebook as SELECTED_K.

# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
# Stage 1: Hyperopt TPE explores broadly, scored by K-fold CV on a sample.
# Stage 2: the top configurations are re-scored by CrossValidator on the full
#          training window, so the winner is confirmed on all the data.
# TPE needs roughly 20 startup trials before its surrogate beats random search,
# so anything below ~20 evals is random search wearing a Bayesian label.
HYPEROPT_MAX_EVALS = 25
STAGE2_TOP_N = 3

# Bounded to respect the serverless SparkML limits: a single model must stay
# under 100 MB and a session under 1 GB, and tree training halts early if a
# model is about to exceed the cap.
RF_MAX_TREES = 60
RF_MAX_DEPTH = 10
GBT_MAX_ITER = 40
GBT_MAX_DEPTH = 7

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# ROC-AUC tunes (threshold-independent, measures ranking). F-beta reports
# (threshold-dependent, measures decisions). beta=1 because the cost ratio of a
# missed delay to a false alarm is a product decision that has not been made;
# raise it when it is. See 02_eda §2.
TUNING_METRIC = "areaUnderROC"
REPORTING_BETA = 1.0

# Resolution matched to where the optimum actually lives. The positive rate is
# ~20%, and the first full run put the pre-departure cut at 0.15, so a uniform
# 0.05 grid was stepping by a third of the answer's own magnitude and could not
# distinguish 0.13 from 0.17. Fine below 0.60, coarse above it where nothing
# competes. 0.50 stays on the grid so the "cost of the default" line still has
# an exact point to compare against.
# `evaluation.confusion_at` evaluates every threshold in a single pass, so the
# extra resolution costs columns in one aggregation rather than extra scans.
THRESHOLD_GRID = (
    [round(0.01 * i, 2) for i in range(2, 61)]      # 0.02 .. 0.60 at 0.01
    + [round(0.05 * i, 2) for i in range(13, 20)]   # 0.65 .. 0.95 at 0.05
)

# AviationStack was the original live-data source. It is gone rather than
# deprecated: its plan returned schedules roughly two weeks old, which makes
# matching against a live ADS-B snapshot impossible rather than merely lossy.
# AeroDataBox replaced it — see docs/API_STRATEGY.md for the measurements.

# ---------------------------------------------------------------------------
# OpenSky Network (live aircraft state)
# ---------------------------------------------------------------------------
# Supplies the ground/air phase split in bulk: one /states/all call returned
# 8,166 aircraft over the continental US, of which 2,893 carried callsigns
# belonging to carriers present in the BTS data. Credentials are OAuth2 client
# credentials and live in the same secret scope as the AviationStack key —
# never in this file, which is public. See docs/API_STRATEGY.md.
OPENSKY_SECRET_SCOPE = "flights"
OPENSKY_CLIENT_ID_KEY = "opensky_client_id"
OPENSKY_CLIENT_SECRET_KEY = "opensky_client_secret"

# Raw snapshot landing table for the live state feed.
OPENSKY_STATES = f"{CATALOG}.{SCHEMA}.opensky_states"

# ---------------------------------------------------------------------------
# Alternative-flight recommender
# ---------------------------------------------------------------------------
# 10 percentage points was too coarse. The pre-departure model's probabilities
# on a single route cluster tightly — flights sharing an origin, destination and
# hour see nearly identical features — so a 10pp gap essentially never occurs and
# the recommender returned nothing. 3pp is meaningful against a ~20% base rate
# (roughly a sixth of the base rate) while still clearing the model's own noise.
# ---------------------------------------------------------------------------
# AeroDataBox (live schedules and gate times)
# ---------------------------------------------------------------------------
# Replaces AviationStack on the live path. The AviationStack plan in use returns
# schedules dated roughly two weeks in the past, which makes matching against a
# live ADS-B snapshot impossible rather than merely lossy: a flight from a
# fortnight ago cannot be airborne now.
#
# AeroDataBox also supplies both OpenSky join keys directly — `callSign`
# ("DAL1572") and `aircraft.modeS` ("A34729", which is OpenSky's icao24) — so no
# IATA-to-ICAO mapping table is needed anywhere in the project.
#
# Free RapidAPI tier is metered in API units as well as requests, and the two
# limits differ (400 units against 1,600 requests when measured). Budget by units.
AERODATABOX_SECRET_SCOPE = "flights"
AERODATABOX_SECRET_KEY = "aerodatabox_key"

# Half-width of the window searched for alternatives at the origin airport. The
# API caps a single query at 12 hours, so this must stay under 6.
ALTERNATIVE_SEARCH_HOURS = 4

ALTERNATIVE_MIN_IMPROVEMENT_PCT = 3.0
ALTERNATIVE_WINDOW_MINUTES = 240   # +/- 4 hours around the flight of interest
ALTERNATIVE_TOP_N = 5
