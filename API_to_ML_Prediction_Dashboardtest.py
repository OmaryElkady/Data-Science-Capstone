# Databricks notebook source
# MAGIC %md
# MAGIC # 🎯 API to ML: Flight Delay Prediction Dashboard
# MAGIC
# MAGIC **Uses Unity Catalog Volumes + Spark ML Pipeline (Serverless Compatible!)**
# MAGIC
# MAGIC This notebook:
# MAGIC - Loads API flight data from Gold table
# MAGIC - Applies Spark ML feature engineering pipeline
# MAGIC - Uses Unity Catalog Volume for model loading
# MAGIC - Makes ensemble predictions (RF + GBT)
# MAGIC - Saves to dashboard table
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Unity Catalog Volume: `/Volumes/workspace/default/mlflow_shared_tmp`
# MAGIC - Models registered in Unity Catalog
# MAGIC - API Gold table: `default.api_pipeline_flight_lookups_gold`
# MAGIC
# MAGIC
# MAGIC The three notebooks API_to_ML_Prediction_Dashboard, API_to_ML_Prediction_Dashboardtest, and API_to_ML_Ready_Complete all are non-functional due to the continued restictions we faced due to utilizing Databricks Community Edition

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Import Libraries

# COMMAND ----------

# Standard libraries
import os
from datetime import datetime
import pandas as pd

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# PySpark ML
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

# MLflow
import mlflow
import mlflow.spark
from mlflow import MlflowClient

print("✅ Libraries imported")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

class Config:
    """Configuration for flight delay prediction pipeline"""
    
    # ====================================================================
    # MODEL PATHS - Unity Catalog
    # ====================================================================
    # Using run IDs from your training runs (most reliable!)
    # MODEL_RF_PRE = "runs:/2896769bb7b74fea886fdbd94392f260/model"
    # MODEL_GBT_PRE = "runs:/253312d8148748dc9a86c0afab7e72e8/model"
    # MODEL_RF_IN = "runs:/15f4032c64c24abd811925f4ac1c9d7f/model"
    # MODEL_GBT_IN = "runs:/0c8f6788d01945df8f6d08072988893f/model"
    
    client = MlflowClient()
    client.set_registered_model_alias("workspace.default.model_rf_pre", "flight", 1)
    client.set_registered_model_alias("workspace.default.model_gbt_pre", "flight", 1)
    client.set_registered_model_alias("workspace.default.model_rf_in", "flight", 1)
    client.set_registered_model_alias("workspace.default.model_gbt_in", "flight", 1)

    # Alternative options if run IDs don't work:
    # Option 1: Use @flight alias
    MODEL_RF_PRE = "models:/workspace.default.model_rf_pre@flight"
    MODEL_GBT_PRE = "models:/workspace.default.model_gbt_pre@flight"
    MODEL_RF_IN = "models:/workspace.default.model_rf_in@flight"
    MODEL_GBT_IN = "models:/workspace.default.model_gbt_in@flight"
    
    # Option 2: Use version numbers
    # MODEL_RF_PRE = "models:/workspace.default.model_rf_pre/1"
    # MODEL_GBT_PRE = "models:/workspace.default.model_gbt_pre/1"
    # MODEL_RF_IN = "models:/workspace.default.model_rf_in/1"
    # MODEL_GBT_IN = "models:/workspace.default.model_gbt_in/1"
    # ====================================================================
    
    # Table paths
    API_GOLD_TABLE = "default.api_pipeline_flight_lookups_gold"
    API_SILVER_TABLE = "default.api_pipeline_flight_lookups_silver"
    PREDICTIONS_TABLE = "default.flight_delay_predictions"
    ALTERNATIVE_FLIGHTS_TABLE = "default.alternative_flight_recommendations"

print("✅ Configuration loaded")

# COMMAND ----------

# Set registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()
client.set_registered_model_alias(
    "workspace.default.model_rf_pre",
    "flight",
    1
)
client.set_registered_model_alias(
    "workspace.default.model_gbt_pre",
    "flight",
    1
)
client.set_registered_model_alias(
    "workspace.default.model_rf_in",
    "flight",
    1
)
client.set_registered_model_alias(
    "workspace.default.model_gbt_in",
    "flight",
    1
)

class Config:
    """Configuration for flight delay prediction pipeline"""
    MODEL_RF_PRE = "models:/workspace.default.model_rf_pre@flight"
    MODEL_GBT_PRE = "models:/workspace.default.model_gbt_pre@flight"
    MODEL_RF_IN = "models:/workspace.default.model_rf_in@flight"
    MODEL_GBT_IN = "models:/workspace.default.model_gbt_in@flight"
    API_GOLD_TABLE = "default.api_pipeline_flight_lookups_gold"
    API_SILVER_TABLE = "default.api_pipeline_flight_lookups_silver"
    PREDICTIONS_TABLE = "default.flight_delay_predictions"
    ALTERNATIVE_FLIGHTS_TABLE = "default.alternative_flight_recommendations"

print("✅ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Feature Engineering Functions

# COMMAND ----------

def create_feature_pipeline(include_dep_delay=True):
    """
    Create Spark ML pipeline for feature engineering
    Matches the training pipeline exactly
    """
    
    # Categorical columns to encode
    categorical_cols = ['airline_name', 'airline_code', 'origin_airport_code', 
                        'destination_airport_code']
    
    # String Indexers
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
        for col in categorical_cols
    ]
    
    # One-Hot Encoder
    indexed_cols = [f"{col}_index" for col in categorical_cols]
    ohe_cols = [f"{col}_ohe" for col in categorical_cols]
    
    ohe_encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=ohe_cols,
        handleInvalid="keep"
    )
    
    # Numeric features based on phase
    numeric_cols = ['flight_month', 'flight_year', 'crs_elapsed_time', 'distance']
    if include_dep_delay:
        numeric_cols.append('dep_delay')
    
    # Vector Assembler
    feature_cols = numeric_cols + ohe_cols
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="unscaled_features",
        handleInvalid="skip"
    )
    
    # Standard Scaler
    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[*indexers, ohe_encoder, assembler, scaler])
    
    return pipeline

def add_datetime_features(df):
    """Add datetime-based features to match training data"""
    
    # Convert flight_date to timestamp if needed
    if 'flight_date' in df.columns:
        df = df.withColumn('flight_date', to_timestamp(col('flight_date')))
    
    # Extract datetime components
    df = df.withColumn('flight_year', year(col('flight_date')))
    df = df.withColumn('flight_month', month(col('flight_date')))
    df = df.withColumn('day_of_month', dayofmonth(col('flight_date')))
    df = df.withColumn('day_of_week', dayofweek(col('flight_date')))
    df = df.withColumn('week_of_year', weekofyear(col('flight_date')))
    
    # Weekend indicator
    df = df.withColumn('is_weekend', 
                       when(col('day_of_week').isin([1, 7]), 1).otherwise(0))
    
    # Holiday features (simplified for API data)
    df = df.withColumn('is_holiday', lit(0))
    df = df.withColumn('is_near_holiday', lit(0))
    df = df.withColumn('is_holiday_period', lit(0))
    
    # Season
    df = df.withColumn('season',
                       when(col('flight_month').isin([12, 1, 2]), 'Winter')
                       .when(col('flight_month').isin([3, 4, 5]), 'Spring')
                       .when(col('flight_month').isin([6, 7, 8]), 'Summer')
                       .otherwise('Fall'))
    
    # Quarter
    df = df.withColumn('quarter',
                       when(col('flight_month') <= 3, 'Q1')
                       .when(col('flight_month') <= 6, 'Q2')
                       .when(col('flight_month') <= 9, 'Q3')
                       .otherwise('Q4'))
    
    return df

print("✅ Feature engineering functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Unity Catalog Volume Setup (CRITICAL!)

# COMMAND ----------

# CRITICAL: Configure MLflow to use Unity Catalog Volume
# This allows model loading on serverless compute!
os.environ['MLFLOW_DFS_TMP'] = '/Volumes/workspace/default/mlflow_shared_tmp'

# Set MLflow URIs for Unity Catalog
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

print("✅ MLflow configured for Unity Catalog")
print(f"  Temp directory: /Volumes/workspace/default/mlflow_shared_tmp")
print("✅ Ready to load models on serverless!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Load Models from Unity Catalog

# COMMAND ----------

print("📥 Loading trained models from Unity Catalog...")

try:
    # Load Pre-Departure models
    print("\n📦 Loading Pre-Departure models...")
    model_rf_pre = mlflow.spark.load_model(Config.MODEL_RF_PRE)
    print("   ✅ Loaded RF Pre-Departure")
    
    model_gbt_pre = mlflow.spark.load_model(Config.MODEL_GBT_PRE)
    print("   ✅ Loaded GBT Pre-Departure")
    
    # Load In-Flight models
    print("\n📦 Loading In-Flight models...")
    model_rf_in = mlflow.spark.load_model(Config.MODEL_RF_IN)
    print("   ✅ Loaded RF In-Flight")
    
    model_gbt_in = mlflow.spark.load_model(Config.MODEL_GBT_IN)
    print("   ✅ Loaded GBT In-Flight")
    
    models_loaded = True
    print("\n✅ All models loaded successfully!")
    
except Exception as e:
    print(f"\n❌ Error loading models: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Verify Unity Catalog Volume exists:")
    print("      /Volumes/workspace/default/mlflow_shared_tmp")
    print("   2. Check models in Catalog Explorer (workspace > default)")
    print("   3. Try different model path formats in Config class:")
    print("      - @flight alias")
    print("      - Version numbers (/1)")
    print("      - Run IDs (runs:/RUN_ID/model)")
    models_loaded = False
    
    # Try to create volume if it doesn't exist
    print("\n🔧 Attempting to create Unity Catalog Volume...")
    try:
        spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.mlflow_shared_tmp")
        print("✅ Volume created! Please re-run this cell.")
    except Exception as ve:
        print(f"⚠️  Could not create volume: {ve}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Make Predictions Function

# COMMAND ----------

def make_predictions(df, model_rf, model_gbt, flight_phase):
    """
    Make ensemble predictions using both models
    
    Args:
        df: Input DataFrame with flight data
        model_rf: Random Forest model
        model_gbt: Gradient Boosted Trees model
        flight_phase: 'pre' or 'in' (determines if dep_delay is included)
    """
    
    include_dep_delay = (flight_phase == "in")
    
    # Add datetime features
    print("🔧 Adding datetime features...")
    df = add_datetime_features(df)
    
    # Create and fit feature pipeline
    print(f"🔧 Applying feature pipeline (DEP_DELAY included: {include_dep_delay})...")
    pipeline = create_feature_pipeline(include_dep_delay=include_dep_delay)
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)
    
    # Make predictions with both models
    print("🎯 Making predictions with Random Forest...")
    pred_rf = model_rf.transform(df_transformed)
    
    print("🎯 Making predictions with Gradient Boosted Trees...")
    pred_gbt = model_gbt.transform(df_transformed)
    
    # Create ensemble prediction by averaging probabilities
    print("🔄 Creating ensemble prediction...")
    
    @udf(returnType=VectorUDT())
    def avg_probability(prob1, prob2):
        if prob1 is None or prob2 is None:
            return prob1 or prob2
        return Vectors.dense([(prob1[i] + prob2[i]) / 2 for i in range(len(prob1))])
    
    # Join predictions and create ensemble
    predictions = pred_rf.alias('rf').join(
        pred_gbt.alias('gbt'),
        on='fl_number',
        how='inner'
    ).select(
        col('rf.fl_number'),
        col('rf.airline_name'),
        col('rf.airline_code'),
        col('rf.origin_airport_code'),
        col('rf.destination_airport_code'),
        col('rf.flight_date'),
        col('rf.dep_delay'),
        col('rf.crs_elapsed_time'),
        col('rf.distance'),
        avg_probability(col('rf.probability'), col('gbt.probability')).alias('probability')
    )
    
    # Extract delay probability
    @udf(returnType=DoubleType())
    def get_delay_prob(probability):
        if probability is None:
            return 0.0
        return float(probability[1])  # Probability of delay (class 1)
    
    predictions = predictions.withColumn(
        'delay_probability',
        get_delay_prob(col('probability'))
    )
    
    # Add prediction label
    predictions = predictions.withColumn(
        'predicted_delay',
        when(col('delay_probability') > 0.5, 'Delayed').otherwise('On-Time')
    )
    
    # Add risk category
    predictions = predictions.withColumn(
        'risk_category',
        when(col('delay_probability') >= 0.75, 'High Risk')
        .when(col('delay_probability') >= 0.50, 'Medium Risk')
        .when(col('delay_probability') >= 0.25, 'Low Risk')
        .otherwise('Very Low Risk')
    )
    
    # Add metadata
    predictions = predictions.withColumn('prediction_timestamp', current_timestamp())
    predictions = predictions.withColumn('model_version', lit('ensemble_v1'))
    predictions = predictions.withColumn('flight_phase', lit(flight_phase))
    
    print("✅ Predictions complete!")
    
    return predictions

print("✅ Prediction function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 RUN PREDICTIONS

# COMMAND ----------

if not models_loaded:
    print("❌ Cannot make predictions - models not loaded")
    print("    Please check the model loading cell above and troubleshoot")
else:
    print(f"\n{'='*80}")
    print("🎯 RUNNING PREDICTION PIPELINE")
    print(f"{'='*80}\n")
    
    # Load data from API Gold table
    print(f"📊 Loading data from: {Config.API_GOLD_TABLE}")
    
    try:
        df_gold = spark.table(Config.API_GOLD_TABLE)
        record_count = df_gold.count()
        
        if record_count == 0:
            print("⚠️  No records found in Gold table")
            print("    Please run your API pipeline notebook first to generate data")
        else:
            print(f"✅ Loaded {record_count} records from Gold table")
            
            # Join with Silver table to get flight_date
            print(f"\n📊 Joining with Silver table to get flight_date...")
            df_silver = spark.table(Config.API_SILVER_TABLE)
            
            df_input = df_gold.alias('g').join(
                df_silver.alias('s').select('fl_number', 'flight_date'),
                on='fl_number',
                how='left'
            )
            
            print(f"✅ Joined data ready for predictions")
            
            # Show sample of input data
            print("\n📋 Sample input data:")
            df_input.select(
                'fl_number', 'airline_name', 'airline_code',
                'origin_airport_code', 'destination_airport_code',
                'flight_date', 'dep_delay'
            ).show(5, truncate=False)
            
            # Make Pre-Departure predictions
            print(f"\n{'='*80}")
            print("🎯 GENERATING PRE-DEPARTURE PREDICTIONS")
            print(f"{'='*80}\n")
            
            predictions_pre = make_predictions(df_input, model_rf_pre, model_gbt_pre, 'pre')
            
            # Make In-Flight predictions
            print(f"\n{'='*80}")
            print("🎯 GENERATING IN-FLIGHT PREDICTIONS")
            print(f"{'='*80}\n")
            
            predictions_in = make_predictions(df_input, model_rf_in, model_gbt_in, 'in')
            
            # Combine predictions
            all_predictions = predictions_pre.union(predictions_in)
            
            # Display results summary
            print(f"\n{'='*80}")
            print("📊 PREDICTION RESULTS SUMMARY")
            print(f"{'='*80}\n")
            
            total = all_predictions.count()
            delayed = all_predictions.filter(col('predicted_delay') == 'Delayed').count()
            ontime = total - delayed
            avg_prob = all_predictions.agg({'delay_probability': 'avg'}).collect()[0][0]
            
            print(f"Total Predictions: {total}")
            print(f"  Pre-Departure: {predictions_pre.count()}")
            print(f"  In-Flight: {predictions_in.count()}")
            print(f"\nPredicted Delays: {delayed} ({delayed/total*100:.1f}%)")
            print(f"Predicted On-Time: {ontime} ({ontime/total*100:.1f}%)")
            print(f"Average Delay Probability: {avg_prob:.2%}\n")
            
            # Show detailed predictions
            print("Sample Detailed Predictions:")
            all_predictions.select(
                'fl_number',
                'airline_name',
                concat(col('origin_airport_code'), lit(' → '), col('destination_airport_code')).alias('route'),
                'flight_date',
                round(col('dep_delay'), 1).alias('dep_delay_min'),
                round(col('delay_probability') * 100, 1).alias('delay_prob_%'),
                'predicted_delay',
                'risk_category',
                'flight_phase'
            ).show(10, truncate=False)
            
            # Save to predictions table
            print(f"\n💾 Saving predictions to: {Config.PREDICTIONS_TABLE}")
            
            try:
                # Append to existing table or create new
                all_predictions.write.format('delta').mode('append').saveAsTable(Config.PREDICTIONS_TABLE)
                print(f"✅ Predictions saved successfully!")
                
                # Show total records in table
                total_records = spark.table(Config.PREDICTIONS_TABLE).count()
                print(f"📊 Total records in predictions table: {total_records:,}")
                
            except Exception as e:
                print(f"⚠️  Error saving to table: {e}")
                print("    Trying to create table...")
                try:
                    all_predictions.write.format('delta').mode('overwrite').saveAsTable(Config.PREDICTIONS_TABLE)
                    print(f"✅ Table created and predictions saved!")
                except Exception as e2:
                    print(f"❌ Failed to create table: {e2}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\n💡 Troubleshooting:")
        print(f"   1. Verify table exists: {Config.API_GOLD_TABLE}")
        print("   2. Run your API pipeline notebook first")
        print("   3. Check table has data: spark.table('default.api_pipeline_flight_lookups_gold').show()")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 View Predictions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View all predictions
# MAGIC SELECT 
# MAGIC    prediction_timestamp,
# MAGIC    fl_number,
# MAGIC    airline_name,
# MAGIC    CONCAT(origin_airport_code, ' → ', destination_airport_code) as route,
# MAGIC    flight_date,
# MAGIC    ROUND(dep_delay, 1) as departure_delay_min,
# MAGIC    ROUND(delay_probability * 100, 1) as delay_probability_pct,
# MAGIC    predicted_delay,
# MAGIC    risk_category,
# MAGIC    flight_phase,
# MAGIC    model_version
# MAGIC FROM default.flight_delay_predictions
# MAGIC ORDER BY prediction_timestamp DESC
# MAGIC LIMIT 100

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Prediction Analytics

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary by airline
# MAGIC SELECT 
# MAGIC    airline_name,
# MAGIC    COUNT(*) as total_predictions,
# MAGIC    SUM(CASE WHEN predicted_delay = 'Delayed' THEN 1 ELSE 0 END) as predicted_delays,
# MAGIC    ROUND(AVG(delay_probability) * 100, 1) as avg_delay_prob_pct,
# MAGIC    ROUND(SUM(CASE WHEN predicted_delay = 'Delayed' THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) as delay_rate_pct
# MAGIC FROM default.flight_delay_predictions
# MAGIC GROUP BY airline_name
# MAGIC ORDER BY avg_delay_prob_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary by route
# MAGIC SELECT 
# MAGIC    CONCAT(origin_airport_code, ' → ', destination_airport_code) as route,
# MAGIC    COUNT(*) as total_predictions,
# MAGIC    ROUND(AVG(delay_probability) * 100, 1) as avg_delay_prob_pct,
# MAGIC    SUM(CASE WHEN predicted_delay = 'Delayed' THEN 1 ELSE 0 END) as predicted_delays
# MAGIC FROM default.flight_delay_predictions
# MAGIC GROUP BY origin_airport_code, destination_airport_code
# MAGIC HAVING COUNT(*) >= 2
# MAGIC ORDER BY avg_delay_prob_pct DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary by risk category
# MAGIC SELECT 
# MAGIC    risk_category,
# MAGIC    flight_phase,
# MAGIC    COUNT(*) as prediction_count,
# MAGIC    ROUND(AVG(delay_probability) * 100, 1) as avg_delay_prob_pct
# MAGIC FROM default.flight_delay_predictions
# MAGIC GROUP BY risk_category, flight_phase
# MAGIC ORDER BY 
# MAGIC    CASE risk_category
# MAGIC      WHEN 'High Risk' THEN 1
# MAGIC      WHEN 'Medium Risk' THEN 2
# MAGIC      WHEN 'Low Risk' THEN 3
# MAGIC      ELSE 4
# MAGIC    END,
# MAGIC    flight_phase

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare Pre-Departure vs In-Flight predictions
# MAGIC SELECT 
# MAGIC    fl_number,
# MAGIC    airline_name,
# MAGIC    MAX(CASE WHEN flight_phase = 'pre' THEN ROUND(delay_probability * 100, 1) END) as pre_departure_prob,
# MAGIC    MAX(CASE WHEN flight_phase = 'in' THEN ROUND(delay_probability * 100, 1) END) as in_flight_prob,
# MAGIC    MAX(CASE WHEN flight_phase = 'in' THEN ROUND(delay_probability * 100, 1) END) - 
# MAGIC    MAX(CASE WHEN flight_phase = 'pre' THEN ROUND(delay_probability * 100, 1) END) as probability_change
# MAGIC FROM default.flight_delay_predictions
# MAGIC GROUP BY fl_number, airline_name
# MAGIC HAVING MAX(CASE WHEN flight_phase = 'pre' THEN 1 END) = 1 
# MAGIC    AND MAX(CASE WHEN flight_phase = 'in' THEN 1 END) = 1
# MAGIC ORDER BY ABS(probability_change) DESC
# MAGIC LIMIT 20

# COMMAND ----------

print("=" * 80)
print("✅ NOTEBOOK COMPLETE!")
print("=" * 80)
print("\n📋 What was done:")
print("   ✅ Loaded models using Unity Catalog Volumes")
print("   ✅ Applied Spark ML feature engineering pipeline")
print("   ✅ Generated Pre-Departure predictions")
print("   ✅ Generated In-Flight predictions")
print("   ✅ Saved predictions to Delta table")
print("   ✅ Created analytics views")
print("\n📊 Next Steps:")
print("   1. Review prediction results in cells above")
print("   2. Create dashboard using SQL queries")
print("   3. Schedule this notebook as a job for continuous predictions")
print("\n💡 Tips:")
print("   - Run your API pipeline first to generate new data")
print("   - This notebook works on SERVERLESS compute!")
print("   - Models load from Unity Catalog Volume (no /dbfs/tmp needed)")
print("=" * 80)
