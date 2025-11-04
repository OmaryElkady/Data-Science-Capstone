#!/usr/bin/env python3
"""
Comprehensive Flight Data Processing Pipeline
ENSURE the code in the Kaggle_Table_Generation notebook has been run before running this script
This updated script takes the original dataset from the delta lake and produces a single cleaned, standardized, and feature engineered dataset, uploading it to the delta lake as SILVER_TABLE
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Extra imports added to facilitate Delta Lake connection
from pyspark.sql import SparkSession
import joblib
warnings.filterwarnings("ignore")

# Initialize Spark
spark = SparkSession.builder.appName("FlightsDataProcessing").getOrCreate()

# Paths and table names
BASE_PATH = "/Volumes/workspace/default/ds-capstone"
BRONZE_PATH = f"{BASE_PATH}/bronze/flights_data"
SILVER_PATH = f"{BASE_PATH}/silver/flights_data"
BRONZE_TABLE = "default.bronze_flights_data"
SILVER_TABLE = "default.silver_flights_data"

# Preprocessing helper functions
def extract_hour_minute(time_int):
    """Extract hour and minute from HHMM format"""
    if pd.isna(time_int):
        return np.nan, np.nan
    time_str = str(int(time_int)).zfill(4)
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    return hour, minute


def get_time_of_day(hour):
    """Get time of day category from hour"""
    if pd.isna(hour):
        return "Unknown"
    elif 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def get_hour_category(hour):
    """Get hour category from hour"""
    if pd.isna(hour):
        return "Unknown"
    elif 6 <= hour < 12:
        return "Early_Morning"
    elif 12 <= hour < 18:
        return "Daytime"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Late_Night"


def clean_data(df):
    """Clean the original dataset by removing unnecessary columns"""
    print("=== CLEANING DATA ===")

    # Define columns to drop
    columns_to_drop = [
        # Post-departure actual times (not available for prediction)
        "DEP_TIME",
        "DEP_DELAY",
        "TAXI_OUT",
        "WHEELS_OFF",
        "WHEELS_ON",
        "TAXI_IN",
        "ARR_TIME",
        "ELAPSED_TIME",
        "AIR_TIME",
        # Post-hoc delay analysis (not predictive)
        "DELAY_DUE_CARRIER",
        "DELAY_DUE_WEATHER",
        "DELAY_DUE_NAS",
        "DELAY_DUE_SECURITY",
        "DELAY_DUE_LATE_AIRCRAFT",
        # Redundant identifiers
        "AIRLINE_DOT",
        "DOT_CODE",
        "ORIGIN_CITY",
        "DEST_CITY",
        # Different targets (not features)
        "CANCELLED",
        "CANCELLATION_CODE",
        "DIVERTED",
        # Redundant airline info
        "AIRLINE_CODE",
    ]

    # Check which columns actually exist and drop them
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_columns_to_drop)

    print(f"Dropped {len(existing_columns_to_drop)} unnecessary columns")
    print(f"Remaining columns: {df_cleaned.shape[1]}")

    return df_cleaned


def handle_missing_values_and_create_target(df):
    """Handle missing values and create binary target variables"""
    print("\n=== HANDLING MISSING VALUES AND CREATING TARGET ===")

    # Check missing values in ARR_DELAY
    missing_count = df["ARR_DELAY"].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"Missing values in ARR_DELAY: {missing_count:,} ({missing_pct:.2f}%)")

    # Impute missing ARR_DELAY with mean value
    mean_delay = df["ARR_DELAY"].mean()
    df["ARR_DELAY"] = df["ARR_DELAY"].fillna(mean_delay)
    print(f"Imputed {missing_count:,} missing values with mean: {mean_delay:.2f}")

    # Create binary target variables
    df["is_delayed_any"] = (df["ARR_DELAY"] > 0).astype(int)
    df["is_delayed_15min"] = (df["ARR_DELAY"] >= 15).astype(int)
    df["is_delayed_30min"] = (df["ARR_DELAY"] >= 30).astype(int)

    print("Binary target distributions:")
    print(f"  Any delay (> 0 min):     {df['is_delayed_any'].mean():.3f}")
    print(f"  15+ min delay:           {df['is_delayed_15min'].mean():.3f}")
    print(f"  30+ min delay:           {df['is_delayed_30min'].mean():.3f}")

    return df


def engineer_temporal_features(df):
    """Create temporal features from FL_DATE and time columns"""
    print("\n=== ENGINEERING TEMPORAL FEATURES ===")

    # Convert FL_DATE to datetime
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    # Extract basic date features
    df["year"] = df["FL_DATE"].dt.year
    df["month"] = df["FL_DATE"].dt.month
    df["day_of_month"] = df["FL_DATE"].dt.day
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["quarter"] = df["FL_DATE"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Create season feature
    df["season"] = df["month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
    )

    # Extract time features from departure and arrival times
    dep_hour_min = df["CRS_DEP_TIME"].apply(extract_hour_minute)
    arr_hour_min = df["CRS_ARR_TIME"].apply(extract_hour_minute)

    df["dep_hour"] = [x[0] for x in dep_hour_min]
    df["dep_minute"] = [x[1] for x in dep_hour_min]
    df["arr_hour"] = [x[0] for x in arr_hour_min]
    df["arr_minute"] = [x[1] for x in arr_hour_min]

    # Create time categories
    df["dep_time_of_day"] = df["dep_hour"].apply(get_time_of_day)
    df["arr_time_of_day"] = df["arr_hour"].apply(get_time_of_day)
    df["dep_hour_category"] = df["dep_hour"].apply(get_hour_category)

    print("Added temporal features: year, month, day, season, time categories")

    return df


def engineer_route_features(df):
    """Create route and airline features"""
    print("\n=== ENGINEERING ROUTE AND AIRLINE FEATURES ===")

    # Create route identifier
    df["route"] = df["ORIGIN"] + "-" + df["DEST"]
    df["airline_route"] = df["AIRLINE"] + "-" + df["route"]

    # Flight duration and distance categories
    df["duration_category"] = pd.cut(
        df["CRS_ELAPSED_TIME"], bins=[0, 90, 180, 300, float("inf")], labels=["Short", "Medium", "Long", "Very_Long"]
    )

    df["distance_category"] = pd.cut(
        df["DISTANCE"], bins=[0, 500, 1000, 2000, float("inf")], labels=["Short", "Medium", "Long", "Very_Long"]
    )

    print("Added route features: route, airline_route, duration/distance categories")

    return df

# These preprocess_for_ml and save_processed_data functions are unnecessary in this new delta lake integrated data prep pipeline but their functionality will still be necessary in the creation of the final ML ready training sets so I commented them out for the time being
"""
def preprocess_for_ml(df, target_column="is_delayed_15min"):
    # Preprocess data for machine learning
    print(f"\n=== PREPROCESSING FOR ML (Target: {target_column}) ===")

    # Separate features and target
    X = df.drop(columns=[target_column, "ARR_DELAY", "is_delayed_any", "is_delayed_30min"])
    y = df[target_column]

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.mean():.3f}")

    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Encode high cardinality categorical features with label encoding
    high_cardinality_threshold = 50
    label_encoders = {}

    for col in categorical_features:
        if X[col].nunique() > high_cardinality_threshold:
            print(f"Label encoding {col} ({X[col].nunique()} unique values)")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            print(f"Keeping {col} for one-hot encoding ({X[col].nunique()} unique values)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")

    # Create preprocessing pipeline
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # Fit and transform data
    print("Fitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed features shape: {X_train_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, label_encoders


def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, target_column):
    # Save the processed data as CSV files
    print("\n=== SAVING PROCESSED DATA ===")

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Create DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Add target variable
    X_train_df[target_column] = y_train.values
    X_test_df[target_column] = y_test.values

    # Save as CSV files
    X_train_df.to_csv("processed_train_data.csv", index=False)
    X_test_df.to_csv("processed_test_data.csv", index=False)

    print("Saved processed data:")
    print("  processed_train_data.csv")
    print("  processed_test_data.csv")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training samples: {len(X_train_df):,}")
    print(f"  Test samples: {len(X_test_df):,}")

    return feature_names
"""

def main():
    print("="*70)
    print("✈️  FLIGHT DATA CLEANING AND FEATURE ENGINEERING")
    print("="*70)

    # Step 1: Load Bronze table
    print(f"\n📥 Loading Bronze Delta table: {BRONZE_TABLE}")
    try:
        df_spark = spark.read.table(BRONZE_TABLE)
        df = df_spark.toPandas()
        print(f"✅ Loaded {len(df):,} records")
    except Exception as e:
        print(f"❌ ERROR: Could not load Bronze table '{BRONZE_TABLE}'")
        print(f"   Error: {str(e)}")
        raise

    # Step 2: Clean + Feature Engineer
    print("\n🧹 Cleaning and engineering features...")
    try:
        df = clean_data(df)
        df = handle_missing_values_and_create_target(df)
        df = engineer_temporal_features(df)
        df = engineer_route_features(df)
        print(f"✅ Feature engineering complete! {len(df):,} records ready")
    except Exception as e:
        print(f"❌ ERROR during feature engineering: {str(e)}")
        raise

    # Step 3: Convert back to Spark
    print("\n🔄 Converting cleaned DataFrame back to Spark...")
    df_silver = spark.createDataFrame(df)
    print(f"✅ Spark DataFrame created with {df_silver.count():,} records")

    # Step 4: Write Silver Delta table
    print("\n💾 Writing Silver Delta table...")
    try:
        df_silver.write.format("delta").mode("overwrite").save(SILVER_PATH)
        print(f"✅ Delta table written to: {SILVER_PATH}")
        print(f"✅ Records written: {df_silver.count():,}")
    except Exception as e:
        print(f"❌ ERROR: Could not write Delta table")
        print(f"   Error: {str(e)}")
        print(f"\n💡 Trying to clean and retry...")
        try:
            dbutils.fs.rm(SILVER_PATH, recurse=True)
            df_silver.write.format("delta").mode("overwrite").save(SILVER_PATH)
            print(f"✅ Successfully wrote Delta table after cleanup")
        except Exception as e2:
            print(f"❌ Still failed: {str(e2)}")
            raise

    # Step 5: Register the Silver Delta table
    print(f"\n📌 Registering Delta table as: {SILVER_TABLE}")
    try:
        # Ensure the database exists
        spark.sql("CREATE DATABASE IF NOT EXISTS default")
        print(f"✅ Database 'default' ready")

        # Drop any previous version of the table
        spark.sql(f"DROP TABLE IF EXISTS {SILVER_TABLE}")
        print("🧹 Dropped existing table (if any)")

        # Try registering with saveAsTable
        df_for_table = spark.read.format("delta").load(SILVER_PATH)
        df_for_table.write.format("delta").mode("overwrite").saveAsTable(SILVER_TABLE)

        print(f"✅ Table registered successfully as '{SILVER_TABLE}'!")

    except Exception as e:
        print(f"⚠️  Could not create table with saveAsTable, trying fallback...")
        try:
            # Add dbfs: prefix if missing for LOCATION clause
            SILVER_PATH_SQL = SILVER_PATH if SILVER_PATH.startswith("dbfs:") else f"dbfs:{SILVER_PATH}"
            spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {SILVER_TABLE}
                USING DELTA
                LOCATION '{SILVER_PATH_SQL}'
            """)
            print(f"✅ Table registered successfully with LOCATION clause!")
        except Exception as e2:
            print(f"⚠️  Table registration failed: {str(e2)}")
            print(f"💡 You can still access the data directly using:")
            print(f"   spark.read.format('delta').load('{SILVER_PATH}')")

    print("\n✅ PIPELINE COMPLETE — Silver table ready for downstream analysis!")
    print("="*70)


if __name__ == "__main__":
    main()
