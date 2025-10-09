#!/usr/bin/env python3
"""
Feature Preparation Script for Flight Delay Classification
This script handles missing values, creates binary target, and engineers features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def handle_missing_values_and_create_target(df):
    """
    Handle missing values in ARR_DELAY and create binary target variable.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe with ARR_DELAY column
    
    Returns:
        pd.DataFrame: Dataframe with handled missing values and binary target
    """
    
    print("=== HANDLING MISSING VALUES AND CREATING TARGET ===")
    
    # Check missing values in ARR_DELAY
    missing_count = df['ARR_DELAY'].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    print(f"Missing values in ARR_DELAY: {missing_count:,} ({missing_pct:.2f}%)")
    
    # Strategy: Impute missing ARR_DELAY with mean value
    print("Strategy: Imputing missing ARR_DELAY with mean value...")
    df_clean = df.copy()
    
    # Calculate mean delay (excluding missing values)
    mean_delay = df_clean['ARR_DELAY'].mean()
    print(f"Mean delay (for imputation): {mean_delay:.2f} minutes")
    
    # Fill missing values with mean
    df_clean['ARR_DELAY'] = df_clean['ARR_DELAY'].fillna(mean_delay)
    
    print(f"Rows after imputation: {len(df_clean):,}")
    print(f"Imputed {missing_count:,} missing values")
    
    # Create binary target variables
    print("\n=== CREATING BINARY TARGET VARIABLES ===")
    
    # Option 1: Any delay > 0
    df_clean['is_delayed_any'] = (df_clean['ARR_DELAY'] > 0).astype(int)
    
    # Option 2: Delay >= 15 minutes (industry standard)
    df_clean['is_delayed_15min'] = (df_clean['ARR_DELAY'] >= 15).astype(int)
    
    # Option 3: Delay >= 30 minutes (significant delay)
    df_clean['is_delayed_30min'] = (df_clean['ARR_DELAY'] >= 30).astype(int)
    
    # Show distribution of targets
    print("Binary target distributions:")
    print(f"Any delay (> 0 min):     {df_clean['is_delayed_any'].mean():.3f} ({df_clean['is_delayed_any'].sum():,} flights)")
    print(f"15+ min delay:           {df_clean['is_delayed_15min'].mean():.3f} ({df_clean['is_delayed_15min'].sum():,} flights)")
    print(f"30+ min delay:           {df_clean['is_delayed_30min'].mean():.3f} ({df_clean['is_delayed_30min'].sum():,} flights)")
    
    return df_clean

def engineer_temporal_features(df):
    """
    Create temporal features from FL_DATE and time columns.
    
    Args:
        df (pd.DataFrame): Dataframe with FL_DATE and time columns
    
    Returns:
        pd.DataFrame: Dataframe with additional temporal features
    """
    
    print("\n=== ENGINEERING TEMPORAL FEATURES ===")
    
    # Convert FL_DATE to datetime
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    # Extract temporal features from FL_DATE
    df['year'] = df['FL_DATE'].dt.year
    df['month'] = df['FL_DATE'].dt.month
    df['day_of_month'] = df['FL_DATE'].dt.day
    df['day_of_week'] = df['FL_DATE'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['quarter'] = df['FL_DATE'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Create season feature
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Extract time features from CRS_DEP_TIME (format: HHMM)
    def extract_hour_minute(time_int):
        """Extract hour and minute from HHMM format"""
        if pd.isna(time_int):
            return np.nan, np.nan
        time_str = str(int(time_int)).zfill(4)
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        return hour, minute
    
    # Apply to both departure and arrival times
    dep_hour_min = df['CRS_DEP_TIME'].apply(extract_hour_minute)
    arr_hour_min = df['CRS_ARR_TIME'].apply(extract_hour_minute)
    
    df['dep_hour'] = [x[0] for x in dep_hour_min]
    df['dep_minute'] = [x[1] for x in dep_hour_min]
    df['arr_hour'] = [x[0] for x in arr_hour_min]
    df['arr_minute'] = [x[1] for x in arr_hour_min]
    
    # Create time-of-day categories
    def get_time_of_day(hour):
        if pd.isna(hour):
            return 'Unknown'
        elif 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['dep_time_of_day'] = df['dep_hour'].apply(get_time_of_day)
    df['arr_time_of_day'] = df['arr_hour'].apply(get_time_of_day)
    
    # Create departure hour categories
    def get_hour_category(hour):
        if pd.isna(hour):
            return 'Unknown'
        elif 6 <= hour < 12:
            return 'Early_Morning'
        elif 12 <= hour < 18:
            return 'Daytime'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Late_Night'
    
    df['dep_hour_category'] = df['dep_hour'].apply(get_hour_category)
    
    print("Temporal features created:")
    print("- Year, month, day_of_month, day_of_week, quarter")
    print("- Weekend indicator")
    print("- Season")
    print("- Departure/arrival hour and minute")
    print("- Time of day categories")
    print("- Hour categories")
    
    return df

def engineer_route_features(df):
    """
    Create route and airline features.
    
    Args:
        df (pd.DataFrame): Dataframe with ORIGIN, DEST, AIRLINE columns
    
    Returns:
        pd.DataFrame: Dataframe with additional route features
    """
    
    print("\n=== ENGINEERING ROUTE AND AIRLINE FEATURES ===")
    
    # Create route identifier
    df['route'] = df['ORIGIN'] + '-' + df['DEST']
    
    # Create airline-route combination
    df['airline_route'] = df['AIRLINE'] + '-' + df['route']
    
    # Flight duration categories
    df['duration_category'] = pd.cut(
        df['CRS_ELAPSED_TIME'],
        bins=[0, 90, 180, 300, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Very_Long']
    )
    
    # Distance categories
    df['distance_category'] = pd.cut(
        df['DISTANCE'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Very_Long']
    )
    
    print("Route and airline features created:")
    print("- Route identifier (ORIGIN-DEST)")
    print("- Airline-route combination")
    print("- Flight duration categories")
    print("- Distance categories")
    
    return df

def create_summary_statistics(df):
    """
    Create summary statistics for the processed dataset.
    
    Args:
        df (pd.DataFrame): Processed dataframe
    
    Returns:
        dict: Summary statistics
    """
    
    print("\n=== DATASET SUMMARY ===")
    
    summary = {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
        'categorical_columns': int(len(df.select_dtypes(include=['object', 'category']).columns)),
        'missing_values': int(df.isnull().sum().sum()),
        'target_distributions': {
            'any_delay': float(df['is_delayed_any'].mean()),
            'delay_15min': float(df['is_delayed_15min'].mean()),
            'delay_30min': float(df['is_delayed_30min'].mean())
        }
    }
    
    print(f"Total rows: {summary['total_rows']:,}")
    print(f"Total columns: {summary['total_columns']}")
    print(f"Numeric columns: {summary['numeric_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")
    print(f"Missing values: {summary['missing_values']:,}")
    
    print(f"\nTarget distributions:")
    print(f"Any delay: {summary['target_distributions']['any_delay']:.3f}")
    print(f"15+ min delay: {summary['target_distributions']['delay_15min']:.3f}")
    print(f"30+ min delay: {summary['target_distributions']['delay_30min']:.3f}")
    
    return summary

def main():
    """Main function to process the cleaned data."""
    
    print("Loading cleaned data...")
    df = pd.read_csv('flights_cleaned.csv')
    print(f"Loaded data shape: {df.shape}")
    
    # Step 1: Handle missing values and create targets
    df_processed = handle_missing_values_and_create_target(df)
    
    # Step 2: Engineer temporal features
    df_processed = engineer_temporal_features(df_processed)
    
    # Step 3: Engineer route features
    df_processed = engineer_route_features(df_processed)
    
    # Step 4: Create summary
    summary = create_summary_statistics(df_processed)
    
    # Save processed data
    output_path = 'flights_processed.csv'
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    print(f"Final shape: {df_processed.shape}")
    
    # Save summary
    import json
    with open('processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Processing summary saved to: processing_summary.json")
    
    # Display sample of processed data
    print(f"\nSample of processed data:")
    print(df_processed.head())
    
    print(f"\nColumn names:")
    for i, col in enumerate(df_processed.columns, 1):
        print(f"{i:2d}. {col}")

if __name__ == "__main__":
    main()
