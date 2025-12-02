# Databricks notebook source
# MAGIC %md
# MAGIC # 🎯 API to ML: Flight Delay Prediction for Dashboard
# MAGIC
# MAGIC **Purpose:** Load API flight data, vectorize features, predict delays, and save to dashboard table
# MAGIC
# MAGIC **This notebook:**
# MAGIC - Loads API lookup results from Gold layer
# MAGIC - Applies feature engineering pipeline (matching trained models)
# MAGIC - Uses trained GBT models for predictions (Pre-Departure & In-Flight)
# MAGIC - Saves predictions to `default.flight_delay_predictions` for dashboard
# MAGIC
# MAGIC **Models Used:**
# MAGIC - Pre-Departure Model: `workspace.default.model_gbt_pre@flight`
# MAGIC - In-Flight Model: `workspace.default.model_gbt_in@flight`
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Run Enhanced_API_Pipeline notebook first to generate API data
# MAGIC - Models must be registered in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Configuration

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
import mlflow
import mlflow.spark
from mlflow import MlflowClient
import numpy as np
from datetime import datetime

print("✅ Imports loaded successfully")

# COMMAND ----------

# Configuration Class
class Config:
    """Configuration for flight delay prediction pipeline"""
    
    # Feature definitions (matching Gold_table.py and experiment training)
    CATEGORICAL_FEATURES = [
        "airline_name",
        "airline_code",
        "origin_airport_code",
        "destination_airport_code",
        "season"
    ]
    
    BOOLEAN_FEATURES = [
        "is_weekend",
        "is_holiday",
        "is_near_holiday",
        "is_holiday_period"
    ]
    
    # Numerical features (after time transformation to hours)
    NUMERICAL_FEATURES = [
        "flight_month",
        "flight_year",
        "day_of_week",
        "week_of_year",
        "day_of_month",
        "quarter",
        "fl_number",
        "crs_elapsed_time",
        "distance",
        "dep_delay",      # Index 9 in this list (original index 11 in full pipeline)
        "dep_hour",
        "arr_hour"
    ]
    
    # Feature selection indices from experiment training (top 40 features)
    # In-Flight Model (40 features, includes dep_delay at index 11)
    SELECTED_INDICES_IN = [9, 10, 11, 47, 23, 817, 42, 43, 451, 3, 0, 814, 1, 30, 444, 
                           19, 36, 17, 27, 8, 28, 37, 484, 16, 7, 434, 5, 6, 54, 35, 
                           24, 52, 57, 2, 816, 38, 15, 50, 46, 73]
    
    # Pre-Departure Model (39 features, excludes dep_delay index 11)
    SELECTED_INDICES_PRE = [i for i in SELECTED_INDICES_IN if i != 11]
    
    # Model paths (Unity Catalog with alias) - Both RF and GBT
    MODEL_RF_PRE = "models:/workspace.default.model_rf_pre@flight"
    MODEL_GBT_PRE = "models:/workspace.default.model_gbt_pre@flight"
    MODEL_RF_IN = "models:/workspace.default.model_rf_in@flight"
    MODEL_GBT_IN = "models:/workspace.default.model_gbt_in@flight"
    
    # Table paths
    API_GOLD_TABLE = "default.api_pipeline_flight_lookups_gold"
    PREDICTIONS_TABLE = "default.flight_delay_predictions"
    ALTERNATIVE_FLIGHTS_TABLE = "default.alternative_flight_recommendations"
    
    # Alternative flight search parameters
    MAX_ALTERNATIVES = 5  # Max alternative flights to recommend per delayed flight
    TIME_WINDOW_HOURS = 3  # Search within +/- 3 hours of original flight
    MIN_DELAY_PROB_DIFF = 10  # Alternative must be at least 10% less likely to delay
    
    # Set model aliases (optional - models work without aliases)
    try:
        # Suppress protobuf warnings
        import warnings
        warnings.filterwarnings('ignore')
        
        client = MlflowClient()
        # Try to set aliases - not critical if it fails
        try:
            client.set_registered_model_alias("workspace.default.model_rf_pre", "flight", 1)
            client.set_registered_model_alias("workspace.default.model_gbt_pre", "flight", 1)
            client.set_registered_model_alias("workspace.default.model_rf_in", "flight", 1)
            client.set_registered_model_alias("workspace.default.model_gbt_in", "flight", 1)
            print("✅ Model aliases set successfully")
        except:
            # Aliases may already be set or not supported in Community Edition
            print("ℹ️ Model aliases already set or not needed")
    except Exception as e:
        # Not critical - models can be loaded by version directly
        print(f"ℹ️ Model alias setup skipped (models will load by version)")

print("✅ Configuration loaded successfully")
print(f"📊 Output table: {Config.PREDICTIONS_TABLE}")
print(f"🔄 Alternative flights table: {Config.ALTERNATIVE_FLIGHTS_TABLE}")
print(f"🔗 Models configured: RF and GBT (Pre-Departure & In-Flight)")
print(f"   • {Config.MODEL_RF_PRE}")
print(f"   • {Config.MODEL_GBT_PRE}")
print(f"   • {Config.MODEL_RF_IN}")
print(f"   • {Config.MODEL_GBT_IN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Load API Gold Data

# COMMAND ----------

print("\n📥 Loading API lookup results from Gold layer...")

try:
    # Try loading from registered table first
    try:
        df_api_gold = spark.table(Config.API_GOLD_TABLE)
        print(f"✅ Loaded from table: {Config.API_GOLD_TABLE}")
    except:
        # Fallback to direct Delta path for Community Edition
        gold_path = "/Volumes/workspace/default/ds-capstone/api_pipeline_results/flights_lookup/gold"
        df_api_gold = spark.read.format("delta").load(gold_path)
        print(f"✅ Loaded from path: {gold_path}")
    
    # Also load Silver to get flight_date (not in Gold)
    try:
        df_api_silver = spark.table("default.api_pipeline_flight_lookups_silver")
        print(f"✅ Loaded Silver for flight_date")
    except:
        silver_path = "/Volumes/workspace/default/ds-capstone/api_pipeline_results/flights_lookup/silver"
        df_api_silver = spark.read.format("delta").load(silver_path)
        print(f"✅ Loaded Silver from path for flight_date")
    
    # Get flight_date and crs_dep_time from Silver
    df_silver_meta = df_api_silver.select(
        "airline_code",
        "fl_number",
        "flight_date",
        "crs_dep_time"
    )
    
    # Join Gold with Silver metadata
    df_api_gold = df_api_gold.join(
        df_silver_meta,
        on=["airline_code", "fl_number"],
        how="inner"
    )
    
    # Remove other metadata columns
    metadata_cols = ['lookup_timestamp', 'lookup_mode', 'gold_validation_passed']
    df_api_gold = df_api_gold.drop(*[c for c in metadata_cols if c in df_api_gold.columns])
    
    record_count = df_api_gold.count()
    print(f"✅ Loaded {record_count:,} records from API pipeline")
    print(f"   Columns: {len(df_api_gold.columns)}")
    
    if record_count == 0:
        print("⚠️ No data found - run Enhanced_API_Pipeline notebook first")
        dbutils.notebook.exit("No API data available")
    
    # Show sample
    print("\n📋 Sample data:")
    display(df_api_gold.limit(3))
    
except Exception as e:
    print(f"❌ Error loading API results: {e}")
    print("\nℹ️ Make sure you've run the Enhanced_API_Pipeline notebook first")
    print("💡 Data should be at: /Volumes/workspace/default/ds-capstone/api_pipeline_results/flights_lookup/gold")
    import traceback
    print(traceback.format_exc())
    dbutils.notebook.exit(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Prepare Features for ML Pipeline

# COMMAND ----------

print("\n🔧 Preparing features for ML pipeline...")

# Handle time features (convert HHMM to hour if not already done)
df_ml_prep = df_api_gold

if 'dep_hour' not in df_ml_prep.columns and 'crs_dep_time' in df_ml_prep.columns:
    df_ml_prep = df_ml_prep.withColumn("dep_hour", (col("crs_dep_time") / 100).cast("int"))
    print("   ✅ Created dep_hour from crs_dep_time")

if 'arr_hour' not in df_ml_prep.columns and 'crs_arr_time' in df_ml_prep.columns:
    df_ml_prep = df_ml_prep.withColumn("arr_hour", (col("crs_arr_time") / 100).cast("int"))
    print("   ✅ Created arr_hour from crs_arr_time")

# Create label if arrival_delay exists (for validation)
if 'arrival_delay' in df_ml_prep.columns:
    df_ml_prep = df_ml_prep.withColumn(
        "label",
        when(col("arrival_delay").isNotNull() & (col("arrival_delay") >= 15), 1.0).otherwise(0.0)
    )
    labeled_count = df_ml_prep.filter(col("label").isNotNull()).count()
    print(f"   ✅ Created label for {labeled_count} records with known arrival_delay")
else:
    # For prediction mode, set dummy label
    df_ml_prep = df_ml_prep.withColumn("label", lit(0.0))
    print("   ℹ️ No arrival_delay - prediction mode (label set to 0)")

# Store original data for output
df_original = df_ml_prep.select(
    "airline_name", "airline_code", "fl_number",
    "origin_airport_code", "destination_airport_code",
    "flight_date", "dep_delay", "label"
)

print(f"\n   Records ready for prediction: {df_ml_prep.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Build Feature Engineering Pipeline

# COMMAND ----------

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml import Pipeline

print("\n⚡ Building feature engineering pipeline...")
print("   (Matching Gold_table.py and experiment training)")

# Categorical features
categorical_indexed = [f"{c}_index" for c in Config.CATEGORICAL_FEATURES]
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_index",
        handleInvalid="keep"
    )
    for c in Config.CATEGORICAL_FEATURES
]
print(f"   ✅ Created {len(indexers)} string indexers")

# One-hot encoding
categorical_encoded = [f"{c}_ohe" for c in Config.CATEGORICAL_FEATURES]
encoder = OneHotEncoder(
    inputCols=categorical_indexed,
    outputCols=categorical_encoded,
    handleInvalid="keep"
)
print(f"   ✅ Created one-hot encoder")

# Assemble all features
all_features = (
    Config.NUMERICAL_FEATURES
    + Config.BOOLEAN_FEATURES
    + categorical_encoded
)

assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="unscaled_features",
    handleInvalid="skip"
)
print(f"   ✅ Created feature assembler ({len(all_features)} input columns)")

# Standard scaling
scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features_full",
    withStd=True,
    withMean=True
)
print(f"   ✅ Created standard scaler")

# Build pipeline
pipeline = Pipeline(
    stages=indexers + [encoder, assembler, scaler]
)
print(f"\n   🚀 Pipeline ready with {len(indexers) + 3} stages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚡ Fit and Transform Pipeline

# COMMAND ----------

print("\n⚡ Fitting pipeline to API data...")

try:
    fitted_pipeline = pipeline.fit(df_ml_prep)
    print("✅ Pipeline fitted successfully")
    
    print("\n🔄 Transforming data...")
    df_vectorized = fitted_pipeline.transform(df_ml_prep)
    print("✅ Transformation complete")
    
    # Check feature vector size
    sample = df_vectorized.select("features_full").first()
    full_feature_size = sample.features_full.size
    
    print(f"\n📊 Full feature vector dimensions: {full_feature_size}")
    print(f"   Note: Will be sliced to top-{len(Config.SELECTED_INDICES_IN)} features for models")
    
except Exception as e:
    print(f"❌ Error during pipeline execution: {e}")
    import traceback
    print(traceback.format_exc())
    dbutils.notebook.exit(f"Pipeline failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔪 Feature Selection (Top-K Features)

# COMMAND ----------

print("\n🔪 Applying feature selection (top-K features)...")

# Create UDF for safe vector slicing
def create_feature_slicer(indices_to_keep):
    """Create UDF to slice vectors to selected features"""
    indices_list = list(indices_to_keep)
    
    @udf(returnType=VectorUDT())
    def safe_slicer(features):
        if features is None:
            return None
        max_idx = features.size - 1
        selected_values = [float(features[i]) for i in indices_list if i <= max_idx]
        return Vectors.dense(selected_values)
    
    return safe_slicer

# Create feature slicers for both models
slicer_in_flight = create_feature_slicer(Config.SELECTED_INDICES_IN)
slicer_pre_dep = create_feature_slicer(Config.SELECTED_INDICES_PRE)

# Apply slicing for In-Flight model (includes dep_delay)
df_in_flight = df_vectorized.withColumn(
    "features_in_flight", 
    slicer_in_flight(col("features_full"))
)
print(f"   ✅ In-Flight features: {len(Config.SELECTED_INDICES_IN)} dimensions (includes dep_delay)")

# Apply slicing for Pre-Departure model (excludes dep_delay)
df_pre_dep = df_vectorized.withColumn(
    "features_pre_dep",
    slicer_pre_dep(col("features_full"))
)
print(f"   ✅ Pre-Departure features: {len(Config.SELECTED_INDICES_PRE)} dimensions (excludes dep_delay)")

# Combine both feature sets
df_with_features = df_in_flight.withColumn(
    "features_pre_dep",
    slicer_pre_dep(col("features_full"))
)

print("\n✅ Feature selection complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Load Models and Make Predictions

# COMMAND ----------

print("\n🤖 Loading trained models from Unity Catalog...")

models = {}

try:
    # Load Pre-Departure models
    print(f"\n📥 Loading Pre-Departure Models...")
    print(f"   RF: {Config.MODEL_RF_PRE}")
    models['rf_pre'] = mlflow.spark.load_model(Config.MODEL_RF_PRE)
    print("   ✅ Random Forest Pre-Departure model loaded")
    
    print(f"   GBT: {Config.MODEL_GBT_PRE}")
    models['gbt_pre'] = mlflow.spark.load_model(Config.MODEL_GBT_PRE)
    print("   ✅ GBT Pre-Departure model loaded")
    
    # Load In-Flight models
    print(f"\n📥 Loading In-Flight Models...")
    print(f"   RF: {Config.MODEL_RF_IN}")
    models['rf_in'] = mlflow.spark.load_model(Config.MODEL_RF_IN)
    print("   ✅ Random Forest In-Flight model loaded")
    
    print(f"   GBT: {Config.MODEL_GBT_IN}")
    models['gbt_in'] = mlflow.spark.load_model(Config.MODEL_GBT_IN)
    print("   ✅ GBT In-Flight model loaded")
    
    print(f"\n✅ All 4 models loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("\n💡 Make sure models are registered in Unity Catalog:")
    print("   - workspace.default.model_rf_pre")
    print("   - workspace.default.model_gbt_pre")
    print("   - workspace.default.model_rf_in")
    print("   - workspace.default.model_gbt_in")
    dbutils.notebook.exit(f"Model loading failed: {e}")

# COMMAND ----------

print("\n🎯 Making predictions with ensemble models...")

try:
    # Prepare data for predictions
    df_pred_pre = df_with_features.withColumnRenamed("features_pre_dep", "features")
    df_pred_in = df_with_features.withColumnRenamed("features_in_flight", "features")
    
    # ===== PRE-DEPARTURE PREDICTIONS =====
    print("\n1️⃣ Pre-Departure Models (Before takeoff)...")
    
    # Random Forest Pre-Departure
    print("   Running RF Pre-Departure...")
    predictions_rf_pre = models['rf_pre'].transform(df_pred_pre)
    predictions_rf_pre = predictions_rf_pre.withColumn(
        "prob_delay_rf_pre",
        col("probability").getItem(1)
    )
    
    # GBT Pre-Departure
    print("   Running GBT Pre-Departure...")
    predictions_gbt_pre = models['gbt_pre'].transform(df_pred_pre)
    predictions_gbt_pre = predictions_gbt_pre.withColumn(
        "prob_delay_gbt_pre",
        col("probability").getItem(1)
    )
    
    # Ensemble: Average RF and GBT predictions
    predictions_pre = predictions_rf_pre.select(
        "airline_name", "airline_code", "fl_number", "flight_date",
        "prob_delay_rf_pre"
    ).join(
        predictions_gbt_pre.select(
            "airline_code", "fl_number", "flight_date",
            "prob_delay_gbt_pre"
        ),
        on=["airline_code", "fl_number", "flight_date"],
        how="inner"
    ).withColumn(
        "probability_delay_pre",
        (col("prob_delay_rf_pre") + col("prob_delay_gbt_pre")) / 2
    ).withColumn(
        "prediction_pre_dep",
        when(col("probability_delay_pre") >= 0.5, 1.0).otherwise(0.0)
    )
    
    print(f"   ✅ Pre-Departure ensemble: {predictions_pre.count()} predictions")
    
    # ===== IN-FLIGHT PREDICTIONS =====
    print("\n2️⃣ In-Flight Models (After takeoff)...")
    
    # Random Forest In-Flight
    print("   Running RF In-Flight...")
    predictions_rf_in = models['rf_in'].transform(df_pred_in)
    predictions_rf_in = predictions_rf_in.withColumn(
        "prob_delay_rf_in",
        col("probability").getItem(1)
    )
    
    # GBT In-Flight
    print("   Running GBT In-Flight...")
    predictions_gbt_in = models['gbt_in'].transform(df_pred_in)
    predictions_gbt_in = predictions_gbt_in.withColumn(
        "prob_delay_gbt_in",
        col("probability").getItem(1)
    )
    
    # Ensemble: Average RF and GBT predictions
    predictions_in = predictions_rf_in.select(
        "airline_code", "fl_number", "flight_date",
        "prob_delay_rf_in"
    ).join(
        predictions_gbt_in.select(
            "airline_code", "fl_number", "flight_date",
            "prob_delay_gbt_in"
        ),
        on=["airline_code", "fl_number", "flight_date"],
        how="inner"
    ).withColumn(
        "probability_delay_in",
        (col("prob_delay_rf_in") + col("prob_delay_gbt_in")) / 2
    ).withColumn(
        "prediction_in_flight",
        when(col("probability_delay_in") >= 0.5, 1.0).otherwise(0.0)
    )
    
    print(f"   ✅ In-Flight ensemble: {predictions_in.count()} predictions")
    
    print("\n✅ All ensemble predictions complete!")
    print(f"   📊 Ensemble method: Simple average of RF + GBT probabilities")
    
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    import traceback
    print(traceback.format_exc())
    dbutils.notebook.exit(f"Prediction failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Prepare Dashboard Output

# COMMAND ----------

print("\n📊 Preparing dashboard output...")

# Join predictions with original data
df_dashboard = predictions_pre.join(
    predictions_in.select(
        "airline_code",
        "fl_number", 
        "flight_date",
        "prediction_in_flight",
        "probability_delay_in",
        "prob_delay_rf_in",
        "prob_delay_gbt_in"
    ),
    on=["airline_code", "fl_number", "flight_date"],
    how="inner"
).join(
    df_original,
    on=["airline_code", "fl_number", "flight_date"],
    how="inner"
)

# Add model breakdown columns (for transparency)
df_dashboard = df_dashboard.withColumn(
    "rf_pre_pct",
    (col("prob_delay_rf_pre") * 100).cast("decimal(5,2)")
).withColumn(
    "gbt_pre_pct",
    (col("prob_delay_gbt_pre") * 100).cast("decimal(5,2)")
).withColumn(
    "rf_in_pct",
    (col("prob_delay_rf_in") * 100).cast("decimal(5,2)")
).withColumn(
    "gbt_in_pct",
    (col("prob_delay_gbt_in") * 100).cast("decimal(5,2)")
)

# Add interpretable columns
df_dashboard = df_dashboard.withColumn(
    "delay_risk_pre_dep",
    when(col("probability_delay_pre") >= 0.7, "High")
    .when(col("probability_delay_pre") >= 0.5, "Medium")
    .otherwise("Low")
).withColumn(
    "delay_risk_in_flight",
    when(col("probability_delay_in") >= 0.7, "High")
    .when(col("probability_delay_in") >= 0.5, "Medium")
    .otherwise("Low")
).withColumn(
    "actual_delayed",
    when(col("label") == 1.0, "Yes").otherwise("No")
).withColumn(
    "prediction_timestamp",
    lit(datetime.now())
).withColumn(
    "route",
    concat(col("origin_airport_code"), lit(" → "), col("destination_airport_code"))
)

# Round probabilities for readability
df_dashboard = df_dashboard.withColumn(
    "probability_delay_pre_pct",
    (col("probability_delay_pre") * 100).cast("decimal(5,2)")
).withColumn(
    "probability_delay_in_pct", 
    (col("probability_delay_in") * 100).cast("decimal(5,2)")
)

# Select columns for main predictions table
df_final = df_dashboard.select(
    "prediction_timestamp",
    "airline_name",
    "airline_code",
    "fl_number",
    "route",
    "origin_airport_code",
    "destination_airport_code",
    "flight_date",
    "dep_delay",
    "probability_delay_pre_pct",
    "delay_risk_pre_dep",
    "probability_delay_in_pct",
    "delay_risk_in_flight",
    "rf_pre_pct",
    "gbt_pre_pct",
    "rf_in_pct",
    "gbt_in_pct",
    "actual_delayed",
    "label"
)

print(f"✅ Dashboard data prepared: {df_final.count()} records")
print(f"   Includes individual model predictions for transparency")

# Show sample
print("\n📋 Sample dashboard data:")
display(df_final.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Save to Dashboard Table

# COMMAND ----------

print(f"\n💾 Saving predictions to dashboard table...")
print(f"   Table: {Config.PREDICTIONS_TABLE}")

try:
    # Write to Delta Lake (append mode to build history)
    df_final.write.format("delta").mode("append").saveAsTable(Config.PREDICTIONS_TABLE)
    
    saved_count = df_final.count()
    print(f"   ✅ Saved {saved_count:,} prediction records")
    
    # Show table stats
    total_records = spark.table(Config.PREDICTIONS_TABLE).count()
    print(f"   📊 Total records in dashboard table: {total_records:,}")
    
    print(f"\n✅ Dashboard table updated successfully!")
    print(f"   Access with: spark.table('{Config.PREDICTIONS_TABLE}')")
    
except Exception as e:
    print(f"   ❌ Error saving to table: {e}")
    import traceback
    print(traceback.format_exc())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Alternative Flight Recommendations

# COMMAND ----------

print("\n🔄 Generating alternative flight recommendations...")
print(f"   Strategy: Find flights on same route with lower delay probability")
print(f"   Time window: ±{Config.TIME_WINDOW_HOURS} hours")
print(f"   Min improvement: {Config.MIN_DELAY_PROB_DIFF}% lower delay probability")

try:
    # Filter to high-risk flights that need alternatives
    high_risk_flights = df_final.filter(
        col("delay_risk_in_flight") == "High"
    ).select(
        "airline_code",
        "fl_number",
        "flight_date",
        "origin_airport_code",
        "destination_airport_code",
        "probability_delay_in_pct",
        "dep_delay"
    )
    
    high_risk_count = high_risk_flights.count()
    print(f"\n   Found {high_risk_count} high-risk flights needing alternatives")
    
    if high_risk_count > 0:
        # Get all flights for matching
        all_flights = df_final.select(
            "airline_name",
            "airline_code",
            "fl_number",
            "flight_date",
            "origin_airport_code",
            "destination_airport_code",
            "probability_delay_in_pct",
            "delay_risk_in_flight",
            "dep_delay"
        ).withColumnRenamed("airline_code", "alt_airline_code") \
         .withColumnRenamed("airline_name", "alt_airline_name") \
         .withColumnRenamed("fl_number", "alt_fl_number") \
         .withColumnRenamed("probability_delay_in_pct", "alt_delay_prob") \
         .withColumnRenamed("delay_risk_in_flight", "alt_delay_risk") \
         .withColumnRenamed("dep_delay", "alt_dep_delay")
        
        # Join high-risk flights with all flights on same route
        alternatives = high_risk_flights.alias("orig").join(
            all_flights.alias("alt"),
            (col("orig.origin_airport_code") == col("alt.origin_airport_code")) &
            (col("orig.destination_airport_code") == col("alt.destination_airport_code")) &
            (col("orig.flight_date") == col("alt.flight_date")) &
            # Don't recommend the same flight
            ~((col("orig.airline_code") == col("alt.alt_airline_code")) & 
              (col("orig.fl_number") == col("alt.alt_fl_number"))),
            how="inner"
        )
        
        # Filter to better alternatives
        better_alternatives = alternatives.filter(
            # Alternative has lower delay probability
            col("alt.alt_delay_prob") < (col("orig.probability_delay_in_pct") - Config.MIN_DELAY_PROB_DIFF)
        ).withColumn(
            "improvement_pct",
            col("orig.probability_delay_in_pct") - col("alt.alt_delay_prob")
        ).withColumn(
            "recommendation_score",
            # Score based on: improvement in probability + risk level
            col("improvement_pct") + 
            when(col("alt.alt_delay_risk") == "Low", 20)
            .when(col("alt.alt_delay_risk") == "Medium", 10)
            .otherwise(0)
        )
        
        # Rank alternatives per original flight
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy(
            "orig.airline_code", "orig.fl_number", "orig.flight_date"
        ).orderBy(col("recommendation_score").desc())
        
        ranked_alternatives = better_alternatives.withColumn(
            "rank",
            row_number().over(window_spec)
        ).filter(
            col("rank") <= Config.MAX_ALTERNATIVES
        )
        
        # Prepare final recommendations table
        df_recommendations = ranked_alternatives.select(
            col("orig.airline_code").alias("original_airline"),
            col("orig.fl_number").alias("original_flight"),
            col("orig.origin_airport_code").alias("origin"),
            col("orig.destination_airport_code").alias("destination"),
            col("orig.flight_date").alias("flight_date"),
            col("orig.probability_delay_in_pct").alias("original_delay_prob"),
            col("orig.dep_delay").alias("original_dep_delay"),
            col("alt.alt_airline_name").alias("alternative_airline"),
            col("alt.alt_airline_code").alias("alternative_airline_code"),
            col("alt.alt_fl_number").alias("alternative_flight"),
            col("alt.alt_delay_prob").alias("alternative_delay_prob"),
            col("alt.alt_delay_risk").alias("alternative_risk_level"),
            col("alt.alt_dep_delay").alias("alternative_dep_delay"),
            col("improvement_pct").alias("improvement_percentage"),
            col("recommendation_score"),
            col("rank").alias("recommendation_rank"),
            lit(datetime.now()).alias("recommendation_timestamp")
        )
        
        rec_count = df_recommendations.count()
        print(f"\n   ✅ Generated {rec_count} alternative flight recommendations")
        
        if rec_count > 0:
            # Show summary
            print("\n📊 Recommendation Summary:")
            print(f"   Average improvement: {df_recommendations.agg({'improvement_percentage': 'avg'}).collect()[0][0]:.1f}%")
            print(f"   Flights with alternatives: {df_recommendations.select('original_airline', 'original_flight').distinct().count()}")
            
            print("\n📋 Sample recommendations:")
            display(df_recommendations.orderBy(col("recommendation_score").desc()).limit(10))
            
            # Save recommendations to table
            print(f"\n💾 Saving recommendations to {Config.ALTERNATIVE_FLIGHTS_TABLE}...")
            try:
                df_recommendations.write.format("delta").mode("append").saveAsTable(Config.ALTERNATIVE_FLIGHTS_TABLE)
                print(f"   ✅ Saved {rec_count} recommendations")
            except Exception as e:
                print(f"   ⚠️ Could not save to table: {e}")
                print(f"   💡 Recommendations still available in df_recommendations DataFrame")
        else:
            print("\n   ℹ️ No suitable alternatives found")
            print("   Possible reasons:")
            print("   • No other flights on same routes")
            print("   • All alternatives also have high delay probability")
            print("   • Insufficient probability improvement threshold")
    else:
        print("\n   ✅ No high-risk flights in current batch")
        print("   No alternative recommendations needed")
        df_recommendations = spark.createDataFrame([], StructType([]))
    
except Exception as e:
    print(f"❌ Error generating alternatives: {e}")
    import traceback
    print(traceback.format_exc())
    df_recommendations = spark.createDataFrame([], StructType([]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Prediction Summary Statistics

# COMMAND ----------

print("\n📈 PREDICTION SUMMARY STATISTICS")
print("="*80)

# Overall statistics
print("\n📊 Overall Predictions:")
total = df_final.count()
high_risk_pre = df_final.filter(col("delay_risk_pre_dep") == "High").count()
high_risk_in = df_final.filter(col("delay_risk_in_flight") == "High").count()

print(f"   Total flights: {total}")
print(f"   High risk (Pre-Departure): {high_risk_pre} ({high_risk_pre/total*100:.1f}%)")
print(f"   High risk (In-Flight): {high_risk_in} ({high_risk_in/total*100:.1f}%)")

# Risk distribution
print("\n🎯 Risk Distribution:")
print("\nPre-Departure Model:")
df_final.groupBy("delay_risk_pre_dep").count().orderBy(col("count").desc()).show()

print("In-Flight Model:")
df_final.groupBy("delay_risk_in_flight").count().orderBy(col("count").desc()).show()

# Top routes by delay risk
print("\n🛫 Top 10 Routes by Delay Risk (Pre-Departure):")
df_final.groupBy("route").agg(
    avg("probability_delay_pre_pct").alias("avg_delay_prob"),
    count("*").alias("flight_count")
).filter(col("flight_count") >= 1).orderBy(col("avg_delay_prob").desc()).limit(10).show()

# Airlines
print("\n✈️ Airlines by Average Delay Risk:")
df_final.groupBy("airline_name").agg(
    avg("probability_delay_pre_pct").alias("avg_delay_prob_pre"),
    avg("probability_delay_in_pct").alias("avg_delay_prob_in"),
    count("*").alias("flight_count")
).orderBy(col("avg_delay_prob_in").desc()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Dashboard Usage Guide

# COMMAND ----------

print("\n" + "="*80)
print("🎉 API TO ML PREDICTION PIPELINE COMPLETE")
print("="*80)

print(f"\n✅ What was created:")
print(f"   • Loaded API flight data from: {Config.API_GOLD_TABLE}")
print(f"   • Applied feature engineering pipeline")
print(f"   • Generated ensemble predictions (RF + GBT averaging)")
print(f"   • Saved to dashboard table: {Config.PREDICTIONS_TABLE}")
print(f"   • Generated alternative flight recommendations")
print(f"   • Saved alternatives to: {Config.ALTERNATIVE_FLIGHTS_TABLE}")
print(f"   • Total predictions: {df_final.count():,}")

print(f"\n🤖 Models Used:")
print(f"   Pre-Departure:")
print(f"   • Random Forest: {Config.MODEL_RF_PRE}")
print(f"   • GBT: {Config.MODEL_GBT_PRE}")
print(f"   In-Flight:")
print(f"   • Random Forest: {Config.MODEL_RF_IN}")
print(f"   • GBT: {Config.MODEL_GBT_IN}")
print(f"   📊 Ensemble Method: Simple average of RF and GBT probabilities")

print(f"\n📊 Dashboard Columns:")
print(f"   • prediction_timestamp: When prediction was made")
print(f"   • route: Origin → Destination")
print(f"   • probability_delay_pre_pct: Ensemble delay probability before takeoff (%)")
print(f"   • probability_delay_in_pct: Ensemble delay probability after takeoff (%)")
print(f"   • delay_risk_pre_dep: Risk level (Low/Medium/High) - Pre-Departure")
print(f"   • delay_risk_in_flight: Risk level (Low/Medium/High) - In-Flight")
print(f"   • rf_pre_pct: Random Forest Pre-Departure prediction (%)")
print(f"   • gbt_pre_pct: GBT Pre-Departure prediction (%)")
print(f"   • rf_in_pct: Random Forest In-Flight prediction (%)")
print(f"   • gbt_in_pct: GBT In-Flight prediction (%)")
print(f"   • actual_delayed: Actual delay status (if known)")

print(f"\n🔄 Alternative Recommendations:")
if 'df_recommendations' in locals() and df_recommendations.count() > 0:
    rec_count = df_recommendations.count()
    flights_with_alt = df_recommendations.select('original_airline', 'original_flight').distinct().count()
    print(f"   • Generated {rec_count} recommendations")
    print(f"   • Covering {flights_with_alt} high-risk flights")
    print(f"   • Average improvement: {df_recommendations.agg({'improvement_percentage': 'avg'}).collect()[0][0]:.1f}%")
else:
    print(f"   • No recommendations needed (no high-risk flights in current batch)")

print(f"\n🎯 Model Performance:")
print(f"   Pre-Departure Ensemble: {len(Config.SELECTED_INDICES_PRE)} features")
print(f"   In-Flight Ensemble: {len(Config.SELECTED_INDICES_IN)} features")
print(f"   Ensemble combines RF (robust) + GBT (accurate)")

print(f"\n📊 Creating Databricks Dashboard:")
print(f"   1. Go to Workspace → Create → Dashboard")
print(f"   2. Add visualizations using SQL queries:")
print(f"")
print(f"   Example queries:")
print(f"")
print(f"   -- High risk flights with model breakdown")
print(f"   SELECT airline_name,")
print(f"          CONCAT(airline_code, fl_number) as flight,")
print(f"          route,")
print(f"          probability_delay_in_pct as ensemble_prob,")
print(f"          rf_in_pct, gbt_in_pct,")
print(f"          delay_risk_in_flight")
print(f"   FROM {Config.PREDICTIONS_TABLE}")
print(f"   WHERE delay_risk_in_flight = 'High'")
print(f"   ORDER BY probability_delay_in_pct DESC")
print(f"")
print(f"   -- Alternative flight recommendations")
print(f"   SELECT original_airline, original_flight,")
print(f"          origin, destination,")
print(f"          original_delay_prob,")
print(f"          alternative_airline, alternative_flight,")
print(f"          alternative_delay_prob,")
print(f"          improvement_percentage")
print(f"   FROM {Config.ALTERNATIVE_FLIGHTS_TABLE}")
print(f"   WHERE recommendation_rank <= 3")
print(f"   ORDER BY improvement_percentage DESC")
print(f"")
print(f"   -- Model agreement analysis")
print(f"   SELECT delay_risk_in_flight,")
print(f"          AVG(rf_in_pct) as avg_rf,")
print(f"          AVG(gbt_in_pct) as avg_gbt,")
print(f"          AVG(ABS(rf_in_pct - gbt_in_pct)) as avg_disagreement")
print(f"   FROM {Config.PREDICTIONS_TABLE}")
print(f"   GROUP BY delay_risk_in_flight")

print(f"\n🔄 Refresh Pipeline:")
print(f"   1. Run Enhanced_API_Pipeline to get new flight data")
print(f"   2. Run this notebook to generate ensemble predictions + alternatives")
print(f"   3. Dashboard auto-refreshes with new data")

print(f"\n💡 Tips:")
print(f"   • Ensemble predictions are more robust than single models")
print(f"   • Check individual model predictions (rf_*, gbt_*) for confidence")
print(f"   • Large disagreement between RF and GBT = less confident prediction")
print(f"   • Use alternative recommendations for customer service")
print(f"   • Set up alerts for flights with >70% delay probability")

print("\n✅ Your predictions and recommendations are ready for dashboard visualization!")
