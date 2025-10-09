#!/usr/bin/env python3
"""
Data Cleaning Script for Flight Delay Classification
This script drops unnecessary columns and prepares the dataset for delay prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_and_clean_flights_data(data_path="flights_sample_3m.csv", sample_size=None):
    """
    Load and clean flights data for delay classification.
    
    Args:
        data_path (str): Path to the CSV file
        sample_size (int, optional): If provided, sample this many rows for faster processing
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    
    print("Loading flights data...")
    
    # Load data
    if sample_size:
        print(f"Loading sample of {sample_size:,} rows...")
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        print("Loading full dataset...")
        df = pd.read_csv(data_path)
    
    print(f"Original shape: {df.shape}")
    
    # Define columns to drop
    columns_to_drop = [
        # Post-departure actual times (not available for prediction)
        'DEP_TIME',           # Actual departure time
        'DEP_DELAY',          # Actual departure delay
        'TAXI_OUT',           # Actual taxi out time
        'WHEELS_OFF',         # Actual wheels off time
        'WHEELS_ON',          # Actual wheels on time
        'TAXI_IN',            # Actual taxi in time
        'ARR_TIME',           # Actual arrival time
        'ELAPSED_TIME',       # Actual elapsed time
        'AIR_TIME',           # Actual air time
        
        # Post-hoc delay analysis (not predictive)
        'DELAY_DUE_CARRIER',      # Delay reason - carrier
        'DELAY_DUE_WEATHER',      # Delay reason - weather
        'DELAY_DUE_NAS',          # Delay reason - NAS
        'DELAY_DUE_SECURITY',     # Delay reason - security
        'DELAY_DUE_LATE_AIRCRAFT', # Delay reason - late aircraft
        
        # Redundant identifiers
        'AIRLINE_DOT',        # Redundant with AIRLINE/AIRLINE_CODE
        'DOT_CODE',           # Redundant with AIRLINE_CODE
        'ORIGIN_CITY',        # Redundant with ORIGIN
        'DEST_CITY',          # Redundant with DEST
        
        # Different targets (not features)
        'CANCELLED',          # Cancellation target
        'CANCELLATION_CODE',  # Cancellation reason (97% missing)
        'DIVERTED',           # Diversion target
        
        # Redundant airline info
        'AIRLINE_CODE',       # Keep AIRLINE instead
    ]
    
    # Check which columns actually exist in the dataset
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    
    print(f"Columns to drop: {len(existing_columns_to_drop)}")
    print(f"Columns not found in dataset: {missing_columns}")
    
    # Drop unnecessary columns
    df_cleaned = df.drop(columns=existing_columns_to_drop)
    
    print(f"Shape after dropping columns: {df_cleaned.shape}")
    print(f"Dropped {df.shape[1] - df_cleaned.shape[1]} columns")
    
    # Display remaining columns
    print("\nRemaining columns:")
    for i, col in enumerate(df_cleaned.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Basic info about remaining data
    print(f"\nData types:")
    print(df_cleaned.dtypes)
    
    print(f"\nMissing values:")
    missing = df_cleaned.isnull().sum()
    missing_pct = (missing / len(df_cleaned) * 100).round(2)
    missing_df = pd.concat([missing, missing_pct], axis=1, keys=['count', 'percent'])
    print(missing_df[missing_df['count'] > 0])
    
    # Check target variable (ARR_DELAY)
    if 'ARR_DELAY' in df_cleaned.columns:
        print(f"\nTarget variable (ARR_DELAY) info:")
        print(f"Missing values: {df_cleaned['ARR_DELAY'].isnull().sum():,}")
        print(f"Delay statistics:")
        print(df_cleaned['ARR_DELAY'].describe())
        
        # Delay distribution
        delayed = (df_cleaned['ARR_DELAY'] > 0).sum()
        on_time = (df_cleaned['ARR_DELAY'] <= 0).sum()
        print(f"\nDelay classification:")
        print(f"Delayed flights (> 0 min): {delayed:,} ({delayed/len(df_cleaned)*100:.1f}%)")
        print(f"On-time flights (≤ 0 min): {on_time:,} ({on_time/len(df_cleaned)*100:.1f}%)")
    
    return df_cleaned

def main():
    """Main function to run the cleaning process."""
    
    # Set up paths
    data_path = Path("flights_sample_3m.csv")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return
    
    # Load and clean data
    # Start with a sample for faster processing
    sample_size = 500_000  # Adjust as needed
    df_cleaned = load_and_clean_flights_data(data_path, sample_size=sample_size)
    
    # Save cleaned data
    output_path = "flights_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final shape: {df_cleaned.shape}")
    
    # Save column info
    column_info = {
        'original_columns': 32,
        'dropped_columns': 32 - df_cleaned.shape[1],
        'remaining_columns': df_cleaned.shape[1],
        'remaining_column_names': df_cleaned.columns.tolist(),
        'dropped_column_names': [
            'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN',
            'ARR_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
            'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
            'AIRLINE_DOT', 'DOT_CODE', 'ORIGIN_CITY', 'DEST_CITY', 'CANCELLED',
            'CANCELLATION_CODE', 'DIVERTED', 'AIRLINE_CODE'
        ]
    }
    
    import json
    with open('column_info.json', 'w') as f:
        json.dump(column_info, f, indent=2)
    print(f"Column info saved to: column_info.json")

if __name__ == "__main__":
    main()
