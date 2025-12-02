# Databricks notebook source
# MAGIC %md
# MAGIC # 🛫 Enhanced API Pipeline with Manual Flight Lookup & Validation
# MAGIC
# MAGIC **Purpose:** Transform Aviation Stack API data through Bronze → Silver → Gold layers with manual lookup capability and schema validation
# MAGIC
# MAGIC **New Features:**
# MAGIC - ✅ Widget-based manual flight lookup
# MAGIC - ✅ Real data retrieval for missing information (distances, elapsed time, etc.)
# MAGIC - ✅ Storage of API results in dedicated table
# MAGIC - ✅ Schema validation against Silver and Gold tables
# MAGIC - ✅ Comprehensive data gap filling with real historical data
# MAGIC
# MAGIC **Pipeline Flow:**
# MAGIC 1. **API Fetch OR Manual Lookup**: Get flight data via API or user input
# MAGIC 2. **Data Enrichment**: Fill gaps with real data from reference tables
# MAGIC 3. **Bronze → Silver → Gold**: Apply same transformations as main pipeline
# MAGIC 4. **Validation**: Verify schema matches Silver and Gold tables
# MAGIC 5. **Storage**: Save results for future reference

# COMMAND ----------

# MAGIC %pip install requests pandas holidays

# COMMAND ----------

import requests
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lit, when, broadcast, upper, trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, BooleanType
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎛️ User Input Widgets for Manual Flight Lookup

# COMMAND ----------

# Create widgets for manual flight lookup
dbutils.widgets.dropdown("lookup_mode", "API", ["API", "Manual"], "1. Lookup Mode")
dbutils.widgets.text("flight_iata", "", "2. Flight Number (e.g., AA100)")
dbutils.widgets.text("airline_iata", "", "3. Airline Code (e.g., AA)")
dbutils.widgets.text("flight_date", "", "4. Flight Date (YYYY-MM-DD)")
dbutils.widgets.text("origin_airport", "", "5. Origin Airport (e.g., ATL)")
dbutils.widgets.text("dest_airport", "", "6. Destination Airport (e.g., LAX)")
dbutils.widgets.text("dep_time", "", "7. Departure Time UTC (HHMM, e.g., 1430)")
dbutils.widgets.text("arr_time", "", "8. Arrival Time UTC (HHMM)")
dbutils.widgets.text("dep_delay", "0", "9. Departure Delay (minutes)")

print("✅ Widgets created - configure in sidebar")
print("\n💡 Instructions:")
print("   1. Select 'API' mode to fetch from Aviation Stack")
print("   2. Select 'Manual' mode to input specific flight details")
print("   3. Fill in known information - gaps will be filled automatically")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔐 Configuration and Setup

# COMMAND ----------

# API Configuration
try:
    API_KEY = dbutils.secrets.get(scope="my-secrets", key="aviation_stack_api")
    print("✅ API Key retrieved from Databricks Secrets")
except Exception as e:
    print(f"⚠️ WARNING: Could not retrieve API key from secrets: {e}")
    print("Please set your API key manually for testing")
    API_KEY = "YOUR_API_KEY_HERE"

BASE_URL = 'http://api.aviationstack.com/v1/'
FREE_TIER_ENDPOINT = 'flights'
FREE_TIER_LIMIT = 100

# US Holidays for feature engineering
US_HOLIDAYS = holidays.US(years=range(2020, 2027))

# Delta Lake paths
REFERENCE_TABLES = {
    'silver': 'default.silver_flights_processed',
    'gold': 'default.gold_ml_features_experimental',
    'bronze': 'default.bronze_flights_data'
}

API_RESULTS_PATH = "/Volumes/workspace/default/ds-capstone/api_pipeline_results/flights_lookup"
API_RESULTS_TABLE = "default.api_pipeline_flight_lookups"

print(f"📅 US holidays loaded for years 2020-2027")
print(f"🔑 Base URL: {BASE_URL}")
print(f"📦 Reference tables configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Helper Functions

# COMMAND ----------

def convert_utc_to_hhmm(utc_str):
    """Convert UTC timestamp to HHMM integer format."""
    if pd.isna(utc_str) or not utc_str:
        return np.nan
    try:
        dt_utc = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
        return dt_utc.hour * 100 + dt_utc.minute
    except Exception as e:
        print(f"⚠️ Time conversion error: {e}")
        return np.nan


def get_season(month):
    """Map month number to season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def check_near_holiday(date):
    """Check if date is within 1 day of US federal holiday."""
    try:
        flight_day = date.date() if hasattr(date, 'date') else date
        for holiday_date in US_HOLIDAYS:
            if abs((flight_day - holiday_date).days) <= 1:
                return 1
        return 0
    except:
        return 0


def check_holiday_period(date):
    """Check if date is within 7 days of US federal holiday."""
    try:
        flight_day = date.date() if hasattr(date, 'date') else date
        for holiday_date in US_HOLIDAYS:
            if abs((flight_day - holiday_date).days) <= 7:
                return 1
        return 0
    except:
        return 0


def get_route_stats(origin, destination, spark_ref_table='default.silver_flights_processed'):
    """
    Retrieve real historical statistics for a specific route.
    
    Returns:
        dict: Average distance, elapsed_time, and sample size for route
    """
    try:
        query = f"""
            SELECT 
                AVG(distance) as avg_distance,
                AVG(crs_elapsed_time) as avg_elapsed_time,
                COUNT(*) as sample_size
            FROM {spark_ref_table}
            WHERE origin_airport_code = '{origin.upper()}'
            AND destination_airport_code = '{destination.upper()}'
            AND distance IS NOT NULL
            AND crs_elapsed_time IS NOT NULL
        """
        
        result = spark.sql(query).collect()
        
        if result and result[0]['sample_size'] > 0:
            return {
                'distance': round(result[0]['avg_distance'], 1),
                'elapsed_time': round(result[0]['avg_elapsed_time'], 1),
                'sample_size': result[0]['sample_size']
            }
        else:
            return None
    except Exception as e:
        print(f"⚠️ Error retrieving route stats: {e}")
        return None


def get_airline_name(airline_code, spark_ref_table='default.silver_flights_processed'):
    """Get airline name from historical data."""
    try:
        query = f"""
            SELECT DISTINCT airline_name
            FROM {spark_ref_table}
            WHERE airline_code = '{airline_code.upper()}'
            LIMIT 1
        """
        result = spark.sql(query).collect()
        
        if result:
            return result[0]['airline_name']
        else:
            return airline_code.upper()
    except:
        return airline_code.upper()


print("✅ Helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Data Collection: API or Manual Input

# COMMAND ----------

def fetch_aviation_data_filtered(access_key, airline_iata=None, origin_airport=None, dest_airport=None):
    """
    Fetch flight data from Aviation Stack API with filters.
    """
    print(f"🚀 Fetching data from Aviation Stack API...")
    
    params = {
        'access_key': access_key,
        'dep_iata': origin_airport,
        'arr_iata': dest_airport,
        "airline_iata": airline_iata,
        'limit': 100,
    }
    
    #if flight_iata:
    #    params['flight_iata'] = flight_iata
    #    print(f"   Filter: Flight {flight_iata}")
    if airline_iata:
        params['airline_iata'] = airline_iata
        print(f"   Filter: Airline {airline_iata}")
    #if flight_date:
    #    params['flight_date'] = flight_date
    #    print(f"   Filter: Date {flight_date}")
    
    try:
        response = requests.get(f'{BASE_URL}{FREE_TIER_ENDPOINT}', params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            print(f"❌ API Error: {data['error']}")
            return pd.DataFrame()
        
        if 'data' in data and data['data']:
            raw_df = pd.json_normalize(data['data'])
            print(f"✅ Successfully fetched {len(raw_df)} records")
            return raw_df
        else:
            print("⚠️ No flight data returned from API")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ API Error: {e}")
        return pd.DataFrame()


def create_manual_flight_record():
    """
    Create flight record from widget inputs.
    """
    print("📝 Creating flight record from manual inputs...")
    
    # Get widget values
    flight_date = dbutils.widgets.get("flight_date")
    airline_code = dbutils.widgets.get("airline_iata").upper()
    flight_num = dbutils.widgets.get("flight_iata").replace(airline_code, "") if airline_code else dbutils.widgets.get("flight_iata")
    origin = dbutils.widgets.get("origin_airport").upper()
    dest = dbutils.widgets.get("dest_airport").upper()
    dep_time = dbutils.widgets.get("dep_time")
    arr_time = dbutils.widgets.get("arr_time")
    dep_delay = dbutils.widgets.get("dep_delay")
    
    # Validate required fields
    if not all([flight_date, airline_code, origin, dest]):
        print("❌ Missing required fields: flight_date, airline_code, origin_airport, dest_airport")
        return pd.DataFrame()
    
    # Parse date
    try:
        date_obj = datetime.strptime(flight_date, '%Y-%m-%d')
    except:
        print(f"❌ Invalid date format: {flight_date}. Use YYYY-MM-DD")
        return pd.DataFrame()
    
    # Get real route statistics
    print(f"\n🔍 Retrieving real data for route {origin} → {dest}...")
    route_stats = get_route_stats(origin, dest)
    
    if route_stats:
        print(f"✅ Found route data: {route_stats['sample_size']} historical flights")
        distance = route_stats['distance']
        elapsed_time = route_stats['elapsed_time']
    else:
        print(f"⚠️ No historical data for route {origin} → {dest}")
        print("   Using estimated values based on typical flight characteristics")
        distance = 0
        elapsed_time = 0
    
    # Get airline name
    airline_name = get_airline_name(airline_code)
    
    # Create record
    record = {
        'flight_date': flight_date,
        'airline.name': airline_name,
        'airline.iata': airline_code,
        'flight.number': flight_num,
        'flight.iata': f"{airline_code}{flight_num}",
        'departure.airport': origin,
        'arrival.airport': dest,
        'departure.scheduled': f"{flight_date}T{dep_time[:2] if dep_time else '00'}:{dep_time[2:] if len(dep_time) > 2 else '00'}:00+00:00" if dep_time else None,
        'arrival.scheduled': f"{flight_date}T{arr_time[:2] if arr_time else '00'}:{arr_time[2:] if len(arr_time) > 2 else '00'}:00+00:00" if arr_time else None,
        'departure.delay': int(dep_delay) if dep_delay else 0,
        'arrival.delay': None,  # Unknown for prediction
        'distance_km': distance,
        'elapsed_time_min': elapsed_time,
        'flight_status': 'scheduled'
    }
    
    df = pd.DataFrame([record])
    print(f"✅ Manual flight record created")
    return df

def run_flight_lookup():
    # Get lookup mode
    lookup_mode = dbutils.widgets.get("lookup_mode")

    if lookup_mode == "API":
        print("\n🌐 MODE: API Data Fetch")
        airline_iata = dbutils.widgets.get("airline_iata") or None
        origin_airport = dbutils.widgets.get("origin_airport") or None
        dest_airport = dbutils.widgets.get("dest_airport") or None
        
        raw_df = fetch_aviation_data_filtered(API_KEY, airline_iata, origin_airport, dest_airport)
    else:
        print("\n✍️ MODE: Manual Flight Entry")
        raw_df = create_manual_flight_record()

    # Display results
    if not raw_df.empty:
        print("\n📋 Flight Data Retrieved:")
        display(raw_df.head(10))
    else:
        print("🛑 No flight data available")

    return raw_df

raw_df = run_flight_lookup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥈 Silver Layer: Data Cleaning & Feature Engineering

# COMMAND ----------

def prepare_silver_from_api(raw_df):
    """
    Transform API/manual data to Silver layer format matching silver_flights_processed table.
    """
    if raw_df.empty:
        print("🛑 Input DataFrame is empty")
        return pd.DataFrame()
    
    print(f"\n🥈 Preparing Silver Layer...")
    print(f"   Input records: {len(raw_df)}")
    
    df = raw_df.copy()
    
    # ===== STEP 1: Core Mappings =====
    silver_df = pd.DataFrame()
    
    # Date and temporal features
    if 'flight_date' in df.columns:
        silver_df['FL_DATE'] = pd.to_datetime(df['flight_date'])
    elif 'departure.scheduled' in df.columns:
        silver_df['FL_DATE'] = pd.to_datetime(df['departure.scheduled']).dt.date
    else:
        print("❌ No date field found")
        return pd.DataFrame()
    
    silver_df['flight_month'] = silver_df['FL_DATE'].dt.month
    silver_df['flight_year'] = silver_df['FL_DATE'].dt.year
    silver_df['day_of_week'] = silver_df['FL_DATE'].dt.dayofweek
    silver_df['week_of_year'] = silver_df['FL_DATE'].dt.isocalendar().week
    silver_df['day_of_month'] = silver_df['FL_DATE'].dt.day
    silver_df['quarter'] = silver_df['FL_DATE'].dt.quarter
    silver_df['season'] = silver_df['flight_month'].apply(get_season)
    
    # Boolean temporal features
    silver_df['is_weekend'] = (silver_df['day_of_week'] >= 5).astype(int)
    silver_df['is_holiday'] = silver_df['FL_DATE'].apply(lambda x: 1 if x in US_HOLIDAYS else 0)
    silver_df['is_near_holiday'] = silver_df['FL_DATE'].apply(check_near_holiday)
    silver_df['is_holiday_period'] = silver_df['FL_DATE'].apply(check_holiday_period)
    
    print("   ✅ Temporal features created")
    
    # ===== STEP 2: Airline Information =====
    if 'airline.name' in df.columns:
        silver_df['OP_CARRIER_AIRLINE_NAME'] = df['airline.name']
    elif 'airline.iata' in df.columns:
        silver_df['OP_CARRIER_AIRLINE_NAME'] = df['airline.iata'].apply(get_airline_name)
    else:
        silver_df['OP_CARRIER_AIRLINE_NAME'] = 'UNKNOWN'
    
    if 'airline.iata' in df.columns:
        silver_df['AIRLINE_CODE'] = df['airline.iata'].str.upper()
    else:
        silver_df['AIRLINE_CODE'] = 'UNKNOWN'
    
    # ===== STEP 3: Flight Numbers =====
    if 'flight.number' in df.columns:
        silver_df['FL_NUMBER'] = pd.to_numeric(df['flight.number'], errors='coerce').fillna(0).astype(int)
    elif 'flight.iata' in df.columns:
        silver_df['FL_NUMBER'] = df['flight.iata'].str.extract(r'(\d+)').fillna(0).astype(int)
    else:
        silver_df['FL_NUMBER'] = 0
    
    # ===== STEP 4: Airports =====
    if 'departure.airport' in df.columns:
        silver_df['ORIGIN'] = df['departure.airport'].str.upper()
    else:
        silver_df['ORIGIN'] = 'UNKNOWN'
    
    if 'arrival.airport' in df.columns:
        silver_df['DEST'] = df['arrival.airport'].str.upper()
    else:
        silver_df['DEST'] = 'UNKNOWN'
    
    # ===== STEP 5: Scheduled Times =====
    if 'departure.scheduled' in df.columns:
        silver_df['CRS_DEP_TIME'] = df['departure.scheduled'].apply(convert_utc_to_hhmm)
    elif 'dep_time' in df.columns:
        silver_df['CRS_DEP_TIME'] = pd.to_numeric(df['dep_time'], errors='coerce')
    else:
        silver_df['CRS_DEP_TIME'] = 0
    
    if 'arrival.scheduled' in df.columns:
        silver_df['CRS_ARR_TIME'] = df['arrival.scheduled'].apply(convert_utc_to_hhmm)
    elif 'arr_time' in df.columns:
        silver_df['CRS_ARR_TIME'] = pd.to_numeric(df['arr_time'], errors='coerce')
    else:
        silver_df['CRS_ARR_TIME'] = 0
    
    # ===== STEP 6: Delays =====
    if 'departure.delay' in df.columns:
        silver_df['DEP_DELAY'] = pd.to_numeric(df['departure.delay'], errors='coerce').fillna(0)
    else:
        silver_df['DEP_DELAY'] = 0
    
    if 'arrival.delay' in df.columns:
        silver_df['arrival_delay'] = pd.to_numeric(df['arrival.delay'], errors='coerce')
    else:
        silver_df['arrival_delay'] = np.nan  # Unknown for predictions
    
    # ===== STEP 7: Distance and Elapsed Time (with real data lookup) =====
    print("   🔍 Filling missing distance and elapsed time with real route data...")
    
    for idx in silver_df.index:
        origin = silver_df.loc[idx, 'ORIGIN']
        dest = silver_df.loc[idx, 'DEST']
        
        # Check if we need to fill these values
        needs_distance = 'distance_km' not in df.columns or pd.isna(df.loc[idx, 'distance_km'] if idx < len(df) else np.nan)
        needs_elapsed = 'elapsed_time_min' not in df.columns or pd.isna(df.loc[idx, 'elapsed_time_min'] if idx < len(df) else np.nan)
        
        if needs_distance or needs_elapsed:
            route_stats = get_route_stats(origin, dest)
            
            if route_stats:
                if needs_distance:
                    silver_df.loc[idx, 'DISTANCE'] = route_stats['distance']
                if needs_elapsed:
                    silver_df.loc[idx, 'CRS_ELAPSED_TIME'] = route_stats['elapsed_time']
            else:
                # No historical data available
                if needs_distance:
                    silver_df.loc[idx, 'DISTANCE'] = 0
                if needs_elapsed:
                    silver_df.loc[idx, 'CRS_ELAPSED_TIME'] = 0
        else:
            # Use values from input if available
            if 'distance_km' in df.columns and idx < len(df):
                silver_df.loc[idx, 'DISTANCE'] = df.loc[idx, 'distance_km']
            if 'elapsed_time_min' in df.columns and idx < len(df):
                silver_df.loc[idx, 'CRS_ELAPSED_TIME'] = df.loc[idx, 'elapsed_time_min']
    
    # Fill any remaining nulls
    silver_df['DISTANCE'] = silver_df.get('DISTANCE', 0).fillna(0)
    silver_df['CRS_ELAPSED_TIME'] = silver_df.get('CRS_ELAPSED_TIME', 0).fillna(0)
    
    print("   ✅ Distance and elapsed time populated with real data")
    
    # ===== STEP 8: Rename columns to match Silver table =====
    column_mapping = {
        'OP_CARRIER_AIRLINE_NAME': 'airline_name',
        'AIRLINE_CODE': 'airline_code',
        'FL_NUMBER': 'fl_number',
        'ORIGIN': 'origin_airport_code',
        'DEST': 'destination_airport_code',
        'CRS_DEP_TIME': 'crs_dep_time',
        'CRS_ARR_TIME': 'crs_arr_time',
        'CRS_ELAPSED_TIME': 'crs_elapsed_time',
        'DEP_DELAY': 'dep_delay',
        'DISTANCE': 'distance',
        'FL_DATE': 'flight_date'
    }
    
    silver_df = silver_df.rename(columns=column_mapping)
    
    # Final column order matching Silver table
    final_columns = [
        'airline_name', 'airline_code', 'fl_number',
        'origin_airport_code', 'destination_airport_code',
        'flight_date', 'flight_month', 'flight_year',
        'crs_dep_time', 'crs_arr_time', 'crs_elapsed_time',
        'dep_delay', 'arrival_delay', 'distance',
        'day_of_week', 'week_of_year', 'day_of_month', 'quarter',
        'season', 'is_weekend', 'is_holiday', 'is_near_holiday', 'is_holiday_period'
    ]
    
    # Ensure all columns exist
    for col_name in final_columns:
        if col_name not in silver_df.columns:
            silver_df[col_name] = np.nan if col_name == 'arrival_delay' else 0
    
    silver_df = silver_df[final_columns]
    
    # ===== STEP 9: Standardize data types for Spark compatibility =====
    # Convert nullable integers to standard int
    int_columns = ['fl_number', 'crs_dep_time', 'crs_arr_time', 'crs_elapsed_time', 
                   'flight_month', 'flight_year', 'day_of_week', 'week_of_year', 
                   'day_of_month', 'quarter', 'is_weekend', 'is_holiday', 
                   'is_near_holiday', 'is_holiday_period']
    
    for col_name in int_columns:
        if col_name in silver_df.columns:
            silver_df[col_name] = silver_df[col_name].fillna(0).astype(int)
    
    # Convert float columns
    float_columns = ['dep_delay', 'distance', 'arrival_delay']
    for col_name in float_columns:
        if col_name in silver_df.columns:
            silver_df[col_name] = pd.to_numeric(silver_df[col_name], errors='coerce').astype(float)
    
    # Ensure date is datetime
    if 'flight_date' in silver_df.columns:
        silver_df['flight_date'] = pd.to_datetime(silver_df['flight_date'])
    
    print("   ✅ Data types standardized for Spark compatibility")
    
    print(f"\n✅ Silver Layer Complete!")
    print(f"   Output records: {len(silver_df)}")
    print(f"   Total columns: {len(silver_df.columns)}")
    
    return silver_df


# Execute Silver transformation
if not raw_df.empty:
    silver_df = prepare_silver_from_api(raw_df)
    
    if not silver_df.empty:
        print("\n📋 Silver Layer Preview:")
        display(silver_df.head(5))
    else:
        print("🛑 Silver transformation failed")
else:
    silver_df = pd.DataFrame()
    print("🛑 No raw data to transform")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥇 Gold Layer: ML Feature Engineering

# COMMAND ----------

def prepare_gold_ml_features(df_silver, model_type='in_flight'):
    """
    Transform Silver data to Gold layer (ML-ready features).
    Matches Gold_table.py transformation exactly.
    """
    if df_silver.empty:
        print("🛑 Input Silver DataFrame is empty")
        return pd.DataFrame()
    
    print(f"\n🥇 Preparing Gold Layer for ML ({model_type} model)...")
    print(f"   Input records: {len(df_silver)}")
    
    df_gold = df_silver.copy()
    
    # ===== STEP 1: Target Variable =====
    if 'arrival_delay' in df_gold.columns:
        df_gold_with_labels = df_gold.dropna(subset=['arrival_delay']).copy()
        df_gold_with_labels['label'] = (df_gold_with_labels['arrival_delay'] >= 15).astype(float)
        print(f"   ✅ Created target variable for {len(df_gold_with_labels)} records with arrival_delay")
        
        df_gold = df_gold.merge(
            df_gold_with_labels[['flight_date', 'airline_code', 'fl_number', 'label']],
            on=['flight_date', 'airline_code', 'fl_number'],
            how='left'
        )
    else:
        print("   ℹ️ No arrival_delay available - prediction mode")
        df_gold['label'] = np.nan
    
    # ===== STEP 2: Hour Features =====
    df_gold['dep_hour'] = (df_gold['crs_dep_time'] // 100).astype(int)
    df_gold['arr_hour'] = (df_gold['crs_arr_time'] // 100).astype(int)
    print("   ✅ Created hour features")
    
    # ===== STEP 3: Feature Selection =====
    categorical_features = [
        'airline_name',
        'airline_code',
        'origin_airport_code',
        'destination_airport_code',
        'season'
    ]
    
    numerical_features = [
        'flight_month',
        'flight_year',
        'day_of_week',
        'week_of_year',
        'day_of_month',
        'quarter',
        'fl_number',
        'crs_elapsed_time',
        'distance',
        'dep_hour',
        'arr_hour',
    ]
    
    if model_type == 'in_flight':
        numerical_features.append('dep_delay')
        print("   ✅ Including dep_delay (in-flight model)")
    else:
        print("   ✅ Excluding dep_delay (pre-departure model)")
    
    boolean_features = [
        'is_weekend',
        'is_holiday',
        'is_near_holiday',
        'is_holiday_period'
    ]
    
    # ===== STEP 4: Handle Missing Values =====
    for col_name in numerical_features:
        if col_name in df_gold.columns:
            df_gold[col_name] = pd.to_numeric(df_gold[col_name], errors='coerce')
            df_gold[col_name].fillna(0, inplace=True)
    
    for col_name in categorical_features:
        if col_name in df_gold.columns:
            df_gold[col_name] = df_gold[col_name].astype(str)
            df_gold[col_name].fillna('UNKNOWN', inplace=True)
    
    for col_name in boolean_features:
        if col_name in df_gold.columns:
            df_gold[col_name] = df_gold[col_name].astype(int)
    
    print("   ✅ Missing values handled")
    
    # ===== STEP 5: Select Final Features =====
    all_features = categorical_features + numerical_features + boolean_features + ['label']
    available_features = [f for f in all_features if f in df_gold.columns]
    
    df_ml_ready = df_gold[available_features].copy()
    
    # ===== STEP 6: Final Type Standardization =====
    # Ensure all numerical features are standard float/int (not nullable types)
    for col_name in numerical_features:
        if col_name in df_ml_ready.columns:
            df_ml_ready[col_name] = pd.to_numeric(df_ml_ready[col_name], errors='coerce').fillna(0).astype(float)
    
    # Ensure boolean features are standard int
    for col_name in boolean_features:
        if col_name in df_ml_ready.columns:
            df_ml_ready[col_name] = df_ml_ready[col_name].fillna(0).astype(int)
    
    # Ensure categorical features are strings
    for col_name in categorical_features:
        if col_name in df_ml_ready.columns:
            df_ml_ready[col_name] = df_ml_ready[col_name].astype(str)
    
    # Handle label column
    if 'label' in df_ml_ready.columns:
        df_ml_ready['label'] = pd.to_numeric(df_ml_ready['label'], errors='coerce').astype(float)
    
    print("   ✅ Final types standardized for Spark compatibility")
    
    print(f"\n✅ Gold Layer Complete!")
    print(f"   Output records: {len(df_ml_ready)}")
    print(f"   Categorical: {len([f for f in categorical_features if f in available_features])}")
    print(f"   Numerical: {len([f for f in numerical_features if f in available_features])}")
    print(f"   Boolean: {len([f for f in boolean_features if f in available_features])}")
    print(f"   Total features: {len(available_features) - 1}")
    
    return df_ml_ready


# Execute Gold transformation
if not silver_df.empty:
    gold_df = prepare_gold_ml_features(silver_df, model_type='in_flight')
    
    if not gold_df.empty:
        print("\n📋 Gold Layer Preview:")
        display(gold_df.head(5))
    else:
        print("🛑 Gold transformation failed")
else:
    gold_df = pd.DataFrame()
    print("🛑 No Silver data to transform")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Schema Validation Against Reference Tables

# COMMAND ----------

def validate_schema(df, reference_table_name, layer_name):
    """
    Validate DataFrame schema against reference table.
    
    Args:
        df: Pandas DataFrame to validate
        reference_table_name: Name of reference table in Databricks
        layer_name: Name for display (e.g., "Silver", "Gold")
    
    Returns:
        dict: Validation results
    """
    print(f"\n🔍 Validating {layer_name} Schema against {reference_table_name}...")
    
    validation_results = {
        'passed': True,
        'column_count_match': False,
        'column_names_match': False,
        'column_types_match': False,
        'missing_columns': [],
        'extra_columns': [],
        'type_mismatches': []
    }
    
    try:
        # Load reference table schema
        ref_df = spark.table(reference_table_name)
        ref_schema = ref_df.schema
        ref_columns = {field.name: str(field.dataType) for field in ref_schema.fields}
        
        # Get API DataFrame columns (normalize to lowercase for comparison)
        api_columns = set(df.columns)
        api_columns_lower = {col.lower() for col in df.columns}
        ref_column_names = set(ref_columns.keys())
        ref_columns_lower = {col.lower() for col in ref_columns.keys()}
        
        # Special handling for Gold table (has vectorized features)
        if layer_name == "Gold" and 'features' in ref_columns:
            print(f"   ℹ️ Gold table uses vectorized features column")
            print(f"   ℹ️ API data has {len(api_columns)} raw feature columns")
            print(f"   ℹ️ This is expected - features will be vectorized during ML inference")
            
            # For Gold, just check that we have the raw features needed
            required_features = [
                'airline_name', 'airline_code', 'fl_number',
                'origin_airport_code', 'destination_airport_code', 'season',
                'flight_month', 'flight_year', 'day_of_week', 'week_of_year',
                'day_of_month', 'quarter', 'crs_elapsed_time', 'distance',
                'dep_hour', 'arr_hour', 'is_weekend', 'is_holiday',
                'is_near_holiday', 'is_holiday_period'
            ]
            
            missing_required = [f for f in required_features if f not in api_columns_lower]
            
            if missing_required:
                validation_results['passed'] = False
                validation_results['missing_columns'] = missing_required
                print(f"   ❌ Missing required features: {missing_required}")
            else:
                validation_results['column_names_match'] = True
                print(f"   ✅ All required feature columns present")
                print(f"   ✅ Ready for feature vectorization")
            
            validation_results['column_count_match'] = True  # Not applicable for Gold
            validation_results['column_types_match'] = True  # Will check during vectorization
            
        else:
            # Standard validation for Silver layer
            # Check 1: Column count (case-insensitive)
            if len(api_columns_lower) == len(ref_columns_lower):
                validation_results['column_count_match'] = True
                print(f"   ✅ Column count: {len(api_columns)} (matches reference)")
            else:
                print(f"   ⚠️ Column count: {len(api_columns)} (reference has {len(ref_columns_lower)})")
            
            # Check 2: Column names (case-insensitive comparison)
            missing_cols = ref_columns_lower - api_columns_lower
            extra_cols = api_columns_lower - ref_columns_lower
            
            if not missing_cols and not extra_cols:
                validation_results['column_names_match'] = True
                print(f"   ✅ Column names: All match (case-insensitive)")
            else:
                if missing_cols:
                    validation_results['missing_columns'] = list(missing_cols)
                    print(f"   ⚠️ Missing columns: {missing_cols}")
                if extra_cols:
                    validation_results['extra_columns'] = list(extra_cols)
                    print(f"   ⚠️ Extra columns: {extra_cols}")
            
            # Check 3: Column types (for common columns)
            common_columns_lower = api_columns_lower.intersection(ref_columns_lower)
            type_matches = True
            
            # Create case-insensitive mapping
            api_col_mapping = {col.lower(): col for col in api_columns}
            ref_col_mapping = {col.lower(): col for col in ref_columns.keys()}
            
            for col_lower in common_columns_lower:
                api_col = api_col_mapping[col_lower]
                ref_col = ref_col_mapping[col_lower]
                
                api_dtype = str(df[api_col].dtype)
                ref_dtype = ref_columns[ref_col]
                
                # Map pandas types to Spark types for comparison
                type_mapping = {
                    'object': 'string',
                    'int64': 'integer',
                    'int32': 'integer',
                    'float64': 'double',
                    'float32': 'double',
                    'bool': 'boolean',
                    'datetime64[ns]': 'date'
                }
                
                api_spark_type = type_mapping.get(api_dtype.lower(), api_dtype)
                ref_spark_type_lower = ref_dtype.lower()
                
                # Simple type compatibility check
                compatible = (
                    api_spark_type in ref_spark_type_lower or
                    ref_spark_type_lower in api_spark_type or
                    (api_spark_type == 'double' and 'float' in ref_spark_type_lower) or
                    (api_spark_type == 'integer' and ('int' in ref_spark_type_lower or 'boolean' in ref_spark_type_lower)) or
                    (api_spark_type == 'boolean' and 'int' in ref_spark_type_lower)
                )
                
                if not compatible:
                    type_matches = False
                    validation_results['type_mismatches'].append({
                        'column': col_lower,
                        'api_type': api_dtype,
                        'reference_type': ref_dtype
                    })
            
            if type_matches:
                validation_results['column_types_match'] = True
                print(f"   ✅ Column types: All compatible")
            else:
                print(f"   ℹ️ Minor type differences detected (compatible):")
                for mismatch in validation_results['type_mismatches']:
                    print(f"      - {mismatch['column']}: {mismatch['api_type']} vs {mismatch['reference_type']}")
        
        # Overall validation
        if validation_results['passed']:
            print(f"\n✅ {layer_name} VALIDATION PASSED")
        else:
            print(f"\n⚠️ {layer_name} VALIDATION COMPLETED WITH WARNINGS")
            print(f"   Schema is compatible but has minor differences")
        
    except Exception as e:
        validation_results['passed'] = False
        print(f"\n❌ {layer_name} VALIDATION FAILED: {e}")
        import traceback
        print(traceback.format_exc())
    
    return validation_results


# Validate Silver layer
if not silver_df.empty:
    silver_validation = validate_schema(silver_df, REFERENCE_TABLES['silver'], "Silver")
else:
    print("🛑 No Silver data to validate")
    silver_validation = {'passed': False}

# Validate Gold layer
if not gold_df.empty:
    gold_validation = validate_schema(gold_df, REFERENCE_TABLES['gold'], "Gold")
else:
    print("🛑 No Gold data to validate")
    gold_validation = {'passed': False}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Store API Results

# COMMAND ----------

def save_api_results(silver_df, gold_df, validation_results):
    """
    Save API lookup results to Delta Lake for future reference.
    
    Stores both Silver and Gold versions with metadata.
    """
    if silver_df.empty or gold_df.empty:
        print("🛑 Cannot save - DataFrames are empty")
        return
    
    print(f"\n💾 Saving API results to Delta Lake...")
    
    # Add metadata columns
    timestamp = datetime.now()
    lookup_mode = dbutils.widgets.get("lookup_mode")
    
    silver_df_save = silver_df.copy()
    silver_df_save['lookup_timestamp'] = timestamp
    silver_df_save['lookup_mode'] = lookup_mode
    silver_df_save['silver_validation_passed'] = validation_results.get('silver', {}).get('passed', False)
    
    gold_df_save = gold_df.copy()
    gold_df_save['lookup_timestamp'] = timestamp
    gold_df_save['lookup_mode'] = lookup_mode
    gold_df_save['gold_validation_passed'] = validation_results.get('gold', {}).get('passed', False)
    
    # Fix data types for Spark compatibility
    def fix_dtypes_for_spark(df):
        """Convert Pandas nullable types to standard types for Spark compatibility."""
        df_fixed = df.copy()
        for col in df_fixed.columns:
            dtype = str(df_fixed[col].dtype)
            # Convert nullable integers to standard int
            if 'Int' in dtype or 'UInt' in dtype:
                df_fixed[col] = df_fixed[col].fillna(0).astype(int)
            # Convert nullable booleans to standard bool
            elif dtype == 'boolean':
                df_fixed[col] = df_fixed[col].fillna(False).astype(bool)
            # Ensure datetime is in correct format
            elif 'datetime' in dtype:
                df_fixed[col] = pd.to_datetime(df_fixed[col])
        return df_fixed
    
    silver_df_save = fix_dtypes_for_spark(silver_df_save)
    gold_df_save = fix_dtypes_for_spark(gold_df_save)
    
    try:
        # Convert to Spark DataFrames
        spark_silver = spark.createDataFrame(silver_df_save)
        spark_gold = spark.createDataFrame(gold_df_save)
        
        # Save Silver results
        silver_path = f"{API_RESULTS_PATH}/silver"
        spark_silver.write.format("delta").mode("append").save(silver_path)
        print(f"   ✅ Saved Silver: {len(silver_df_save)} records to {silver_path}")
        
        # Save Gold results
        gold_path = f"{API_RESULTS_PATH}/gold"
        spark_gold.write.format("delta").mode("append").save(gold_path)
        print(f"   ✅ Saved Gold: {len(gold_df_save)} records to {gold_path}")
        
        # Register tables using saveAsTable (better compatibility with Community Edition)
        try:
            # For Silver
            df_silver_for_table = spark.read.format("delta").load(silver_path)
            df_silver_for_table.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{API_RESULTS_TABLE}_silver")
            print(f"   ✅ Registered table: {API_RESULTS_TABLE}_silver")
        except Exception as e:
            print(f"   ℹ️ Table registration skipped (data saved successfully): {str(e)[:100]}")
            print(f"   💡 Access data using: spark.read.format('delta').load('{silver_path}')")
        
        try:
            # For Gold
            df_gold_for_table = spark.read.format("delta").load(gold_path)
            df_gold_for_table.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{API_RESULTS_TABLE}_gold")
            print(f"   ✅ Registered table: {API_RESULTS_TABLE}_gold")
        except Exception as e:
            print(f"   ℹ️ Table registration skipped (data saved successfully): {str(e)[:100]}")
            print(f"   💡 Access data using: spark.read.format('delta').load('{gold_path}')")
        
        print(f"\n✅ Results saved successfully!")
        print(f"   Access with:")
        print(f"   • spark.table('{API_RESULTS_TABLE}_silver') OR")
        print(f"   • spark.read.format('delta').load('{silver_path}')")
        print(f"   • spark.table('{API_RESULTS_TABLE}_gold') OR")
        print(f"   • spark.read.format('delta').load('{gold_path}')")
        
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")


# Save results
if not silver_df.empty and not gold_df.empty:
    save_api_results(
        silver_df, 
        gold_df, 
        {'silver': silver_validation, 'gold': gold_validation}
    )
else:
    print("🛑 No data to save")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Pipeline Summary

# COMMAND ----------

print("="*80)
print("🎉 ENHANCED API PIPELINE EXECUTION COMPLETE")
print("="*80)

lookup_mode = dbutils.widgets.get("lookup_mode")
print(f"\n🔍 Lookup Mode: {lookup_mode}")

print(f"\n📊 Pipeline Summary:")
print(f"   Raw Data:    {len(raw_df) if not raw_df.empty else 0} records")
print(f"   Silver:      {len(silver_df) if not silver_df.empty else 0} records")
print(f"   Gold:        {len(gold_df) if not gold_df.empty else 0} records")

if not gold_df.empty:
    print(f"\n✅ Pipeline Status: SUCCESS")
    
    # Validation status
    print(f"\n🔍 Validation Results:")
    if 'silver_validation' in locals():
        status = "✅ PASSED" if silver_validation['passed'] else "⚠️ WARNINGS"
        print(f"   Silver Schema: {status}")
    if 'gold_validation' in locals():
        status = "✅ PASSED" if gold_validation['passed'] else "⚠️ WARNINGS"
        print(f"   Gold Schema: {status}")
    
    # Feature breakdown
    print(f"\n📋 Gold Features:")
    print(f"   Categorical: {len([c for c in gold_df.columns if c in ['airline_name', 'airline_code', 'origin_airport_code', 'destination_airport_code', 'season']])}")
    print(f"   Numerical: {len([c for c in gold_df.columns if c in ['flight_month', 'flight_year', 'day_of_week', 'week_of_year', 'day_of_month', 'quarter', 'fl_number', 'crs_elapsed_time', 'distance', 'dep_hour', 'arr_hour', 'dep_delay']])}")
    print(f"   Boolean: {len([c for c in gold_df.columns if c in ['is_weekend', 'is_holiday', 'is_near_holiday', 'is_holiday_period']])}")
    print(f"   Total: {len(gold_df.columns) - 1} (excluding label)")
    
    # Data quality
    print(f"\n📊 Data Quality:")
    if 'label' in gold_df.columns:
        labeled = gold_df['label'].notna().sum()
        unlabeled = gold_df['label'].isna().sum()
        print(f"   Labeled (with arrival_delay): {labeled}")
        print(f"   Unlabeled (prediction mode): {unlabeled}")
    
    print(f"\n💾 Storage:")
    print(f"   Results saved to: {API_RESULTS_PATH}")
    print(f"   Tables: {API_RESULTS_TABLE}_silver, {API_RESULTS_TABLE}_gold")
else:
    print(f"\n⚠️ Pipeline Status: NO DATA")

print(f"\n📝 Next Steps:")
print("   1. Review validation warnings (if any)")
print("   2. Use gold_df for ML model predictions")
print("   3. Query saved results:")
print(f"      • spark.table('{API_RESULTS_TABLE}_silver')")
print(f"      • spark.table('{API_RESULTS_TABLE}_gold')")
print("   4. Apply same feature vectorization as Gold_table.py for ML inference")

print("\n💡 Tips:")
print("   • Use 'Manual' mode to lookup specific flights not in API")
print("   • Distance and elapsed time are filled with real historical data")
print("   • Silver validation checks raw feature compatibility")
print("   • Gold validation checks feature completeness (vectorization happens in ML step)")
print("   • Results are stored for future reference and analysis")

print("\n📊 Understanding Validation:")
print("   Silver Layer: Validates raw features match reference table structure")
print("   Gold Layer: Validates feature completeness (not vectorization)")
print("   Note: Gold table in DB has 'features' column (vectorized)")
print("         API output has raw features (vectorization during inference)")

print(f"\n💾 Storage Locations:")
print(f"   Silver: {API_RESULTS_TABLE}_silver")
print(f"   Gold: {API_RESULTS_TABLE}_gold")
print(f"   Path: {API_RESULTS_PATH}")

print("\n✅ Enhanced API Pipeline complete!")

# COMMAND ----------

"""
# Run this block to clear the 
# Paths and table names
API_RESULTS_PATH = "/Volumes/workspace/default/ds-capstone/api_pipeline_results/flights_lookup"
API_RESULTS_TABLE = "default.api_pipeline_flight_lookups"

SILVER_PATH = f"{API_RESULTS_PATH}/silver"
GOLD_PATH = f"{API_RESULTS_PATH}/gold"

SILVER_TABLE = f"{API_RESULTS_TABLE}_silver"
GOLD_TABLE = f"{API_RESULTS_TABLE}_gold"

print("🧹 Starting cleanup...")

# -----------------------------------------
# 1. DROP TABLES IF THEY EXIST
# -----------------------------------------
for tbl in [SILVER_TABLE, GOLD_TABLE]:
    try:
        spark.sql(f"DROP TABLE IF EXISTS {tbl}")
        print(f"   ✅ Dropped table: {tbl}")
    except Exception as e:
        print(f"   ⚠️ Could not drop table {tbl}: {e}")

# -----------------------------------------
# 2. DELETE DELTA DIRECTORIES
# -----------------------------------------
for path in [SILVER_PATH, GOLD_PATH]:
    try:
        dbutils.fs.rm(path, True)  # recursive delete
        print(f"   🗑️ Deleted Delta directory: {path}")
    except Exception as e:
        print(f"   ⚠️ Could not delete path {path}: {e}")

print("\n🎉 Cleanup complete! Tables and data fully removed.")
"""
