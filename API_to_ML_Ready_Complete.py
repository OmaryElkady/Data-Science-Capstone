# Databricks notebook source
# MAGIC %md
# MAGIC # Transform API Gold Table for ML Inference
# MAGIC
# MAGIC **Purpose:** Prepare `api_pipeline_flight_lookups_gold` so that it matches exactly the feature order of `gold_ml_features_experimental`, including top-40 feature selection, ready for Pre-Departure/In-Flight models.
# MAGIC
# MAGIC
# MAGIC The three notebooks API_to_ML_Prediction_Dashboard, API_to_ML_Prediction_Dashboardtest, and API_to_ML_Ready_Complete all are non-functional due to the continued restictions we faced due to utilizing Databricks Community Edition

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load API gold table
df_api = spark.table("default.api_pipeline_flight_lookups_gold")
print(f"📥 Loaded API table with {df_api.count():,} rows")

# COMMAND ----------

# Drop columns not needed for feature vector (match original Silver schema)
cols_to_keep = [
    "airline_name", "airline_code",
    "origin_airport_code", "destination_airport_code",
    "season",
    "flight_month", "flight_year", "day_of_week", "week_of_year", "day_of_month", "quarter",
    "fl_number", "crs_elapsed_time", "distance", 
    "dep_hour", "arr_hour", "dep_delay",
    "is_weekend", "is_holiday", "is_near_holiday"
]
df_api = df_api.select(*cols_to_keep)

# COMMAND ----------

# Rename columns to match original Silver schema
df_api = df_api.withColumnRenamed("fl_number", "fl_number") \
               .withColumnRenamed("crs_elapsed_time", "crs_elapsed_time") \
               .withColumnRenamed("dep_hour", "dep_hour") \
               .withColumnRenamed("arr_hour", "arr_hour") \
               .withColumnRenamed("dep_delay", "dep_delay")

# COMMAND ----------

# Normalize datatypes
dtype_mapping = {
    "flight_month": IntegerType(),
    "flight_year": IntegerType(),
    "day_of_week": IntegerType(),
    "week_of_year": IntegerType(),
    "day_of_month": IntegerType(),
    "quarter": IntegerType(),
    "fl_number": IntegerType(),
    "crs_elapsed_time": DoubleType(),
    "distance": DoubleType(),
    "dep_hour": IntegerType(),
    "arr_hour": IntegerType(),
    "dep_delay": DoubleType(),
    "is_weekend": IntegerType(),
    "is_holiday": IntegerType(),
    "is_near_holiday": IntegerType()
}

for col_name, dtype in dtype_mapping.items():
    df_api = df_api.withColumn(col_name, col(col_name).cast(dtype))

# Fill missing values for numeric columns
numeric_cols = [
    "flight_month", "flight_year", "day_of_week", "week_of_year", "day_of_month", "quarter",
    "fl_number", "crs_elapsed_time", "distance", "dep_hour", "arr_hour", "dep_delay"
]
for col_name in numeric_cols:
    df_api = df_api.fillna({col_name: 0})

# Fill missing values for boolean columns
boolean_cols = ["is_weekend", "is_holiday", "is_near_holiday"]
for col_name in boolean_cols:
    df_api = df_api.fillna({col_name: 0})

# COMMAND ----------

# Categorical features
categorical_features = ["airline_name", "airline_code", "origin_airport_code", "destination_airport_code", "season"]

# Numerical features (in same order as original Gold)
numerical_features = [
    "flight_month", "flight_year", "day_of_week", "week_of_year", "day_of_month", "quarter",
    "fl_number", "crs_elapsed_time", "distance", "dep_hour", "arr_hour", "dep_delay"
]

# Boolean features
boolean_features = ["is_weekend", "is_holiday", "is_near_holiday"]

# COMMAND ----------


# Create StringIndexers for categorical features
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_features]

# OneHotEncoder
ohe_cols = [f"{c}_ohe" for c in categorical_features]
encoder = OneHotEncoder(inputCols=[f"{c}_index" for c in categorical_features], outputCols=ohe_cols, handleInvalid="keep")

# Assemble all features
all_features = numerical_features + boolean_features + ohe_cols
assembler = VectorAssembler(inputCols=all_features, outputCol="unscaled_features", handleInvalid="skip")

# Standard Scaler
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withMean=True, withStd=True)

# Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler])
fitted_pipeline = pipeline.fit(df_api)
df_features = fitted_pipeline.transform(df_api)


# COMMAND ----------

# Select top-40 indices (exact order as experiments)
selected_indices = [9,10,11,47,23,817,42,43,451,3,0,814,1,30,444,19,36,17,27,8,
                    28,37,484,16,7,434,5,6,54,35,24,52,57,2,816,38,15,50,46,73]

from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

@udf(returnType=VectorUDT())
def slice_features(features):
    if features is None:
        return None
    max_idx = features.size - 1
    sliced = [float(features[i]) for i in selected_indices if i <= max_idx]
    return Vectors.dense(sliced)

df_selected = df_features.withColumn("features", slice_features(col("features")))

# COMMAND ----------

# Create Pre-Departure and In-Flight datasets
# dep_delay original index = 11
dep_delay_idx_in_selected = selected_indices.index(11) if 11 in selected_indices else None

if dep_delay_idx_in_selected is not None:
    @udf(returnType=VectorUDT())
    def pre_dep_features(features):
        return Vectors.dense([float(features[i]) for i in range(len(features)) if i != dep_delay_idx_in_selected])
    
    df_pre_dep = df_selected.withColumn("features", pre_dep_features(col("features")))
else:
    df_pre_dep = df_selected

df_in_flight = df_selected

# COMMAND ----------

# Add dummy label column (required for Spark ML format)
df_pre_dep = df_pre_dep.withColumn("label", col("dep_delay") * 0.0)
df_in_flight = df_in_flight.withColumn("label", col("dep_delay") * 0.0)

# COMMAND ----------

# Save as Delta tables
GOLD_PATH_PRE = "/Volumes/workspace/default/ds-capstone/gold/ml_features_api_pre_dep"
GOLD_PATH_IN = "/Volumes/workspace/default/ds-capstone/gold/ml_features_api_in_flight"

df_pre_dep.select("features", "label").write.format("delta").mode("overwrite").save(GOLD_PATH_PRE)
df_in_flight.select("features", "label").write.format("delta").mode("overwrite").save(GOLD_PATH_IN)

print("✅ API Gold features transformed and saved for ML inference.")
