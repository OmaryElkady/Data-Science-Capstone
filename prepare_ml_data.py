#!/usr/bin/env python3
"""
Machine Learning Data Preparation Script
This script splits the data and encodes categorical variables for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path='flights_processed.csv'):
    """
    Load processed data and prepare for machine learning.
    
    Args:
        data_path (str): Path to processed CSV file
    
    Returns:
        pd.DataFrame: Loaded and prepared dataframe
    """
    
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Use 15+ minute delay as our target
    target_column = 'is_delayed_15min'
    print(f"Using target: {target_column}")
    print(f"Target distribution:")
    print(f"  Delayed (1): {df[target_column].sum():,} ({df[target_column].mean():.3f})")
    print(f"  On-time (0): {(1-df[target_column]).sum():,} ({1-df[target_column].mean():.3f})")
    
    return df, target_column

def identify_feature_types(df):
    """
    Identify different types of features for encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Feature type classifications
    """
    
    # Remove target columns and ARR_DELAY (original delay values)
    exclude_cols = ['ARR_DELAY', 'is_delayed_any', 'is_delayed_30min', 'is_delayed_15min']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Categorize features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Identify high-cardinality categorical features
    high_cardinality = []
    low_cardinality = []
    
    for col in categorical_features:
        unique_count = df[col].nunique()
        if unique_count > 50:  # Threshold for high cardinality
            high_cardinality.append(col)
        else:
            low_cardinality.append(col)
    
    feature_types = {
        'numeric': numeric_features,
        'categorical_low_cardinality': low_cardinality,
        'categorical_high_cardinality': high_cardinality,
        'all_features': feature_cols,
        'exclude_cols': exclude_cols
    }
    
    print(f"\nFeature type analysis:")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"  {numeric_features}")
    print(f"Low cardinality categorical: {len(low_cardinality)}")
    print(f"  {low_cardinality}")
    print(f"High cardinality categorical: {len(high_cardinality)}")
    print(f"  {high_cardinality}")
    
    return feature_types

def create_encoding_pipeline(feature_types):
    """
    Create encoding pipeline for different feature types.
    
    Args:
        feature_types (dict): Feature type classifications
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # For low cardinality categorical features, use one-hot encoding
    categorical_low_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # For high cardinality categorical features, use label encoding
    categorical_high_transformer = Pipeline(steps=[
        ('label', LabelEncoder())
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_types['numeric']),
            ('cat_low', categorical_low_transformer, feature_types['categorical_low_cardinality']),
            # Note: High cardinality features will be handled separately due to LabelEncoder limitations
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def encode_high_cardinality_features(df, feature_types):
    """
    Handle high cardinality categorical features with label encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_types (dict): Feature type classifications
    
    Returns:
        pd.DataFrame: Dataframe with encoded high cardinality features
        dict: Label encoders for later use
    """
    
    df_encoded = df.copy()
    label_encoders = {}
    
    print(f"\nEncoding high cardinality features...")
    
    for col in feature_types['categorical_high_cardinality']:
        print(f"Encoding {col} (unique values: {df[col].nunique()})")
        
        # Create label encoder
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Store encoder for later use
        label_encoders[col] = le
        
        # Drop original column
        df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded, label_encoders

def split_and_preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split data and apply preprocessing.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Target column name
        test_size (float): Test set size
        random_state (int): Random seed
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor, label_encoders
    """
    
    print(f"\n=== DATA SPLITTING AND PREPROCESSING ===")
    
    # Identify feature types
    feature_types = identify_feature_types(df)
    
    # Encode high cardinality features first
    df_encoded, label_encoders = encode_high_cardinality_features(df, feature_types)
    
    # Update feature types after encoding
    exclude_cols = feature_types['exclude_cols'] + [col for col in feature_types['categorical_high_cardinality']]
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    # Prepare features and target
    X = df_encoded[feature_cols]
    y = df_encoded[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    print(f"Train target distribution: {y_train.mean():.3f}")
    print(f"Test target distribution: {y_test.mean():.3f}")
    
    # Create preprocessing pipeline for remaining features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Remaining numeric features: {len(numeric_features)}")
    print(f"Remaining categorical features: {len(categorical_features)}")
    
    # Create final preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform training data
    print(f"\nFitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed train features shape: {X_train_processed.shape}")
    print(f"Processed test features shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, label_encoders

def save_preprocessing_artifacts(preprocessor, label_encoders, feature_info):
    """
    Save preprocessing artifacts for later use.
    
    Args:
        preprocessor: Fitted preprocessing pipeline
        label_encoders (dict): Label encoders for high cardinality features
        feature_info (dict): Feature information
    """
    
    print(f"\n=== SAVING PREPROCESSING ARTIFACTS ===")
    
    # Save preprocessor
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("Saved preprocessor to: preprocessor.pkl")
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Saved label encoders to: label_encoders.pkl")
    
    # Save feature info
    import json
    with open('feature_info.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_info = {}
        for key, value in feature_info.items():
            if isinstance(value, (list, tuple)):
                json_info[key] = [str(v) for v in value]  # Convert to string for JSON
            else:
                json_info[key] = str(value)
        json.dump(json_info, f, indent=2)
    print("Saved feature info to: feature_info.json")

def main():
    """Main function to prepare ML data."""
    
    # Load data
    df, target_column = load_and_prepare_data()
    
    # Split and preprocess data
    X_train, X_test, y_train, y_test, preprocessor, label_encoders = split_and_preprocess_data(
        df, target_column, test_size=0.2, random_state=42
    )
    
    # Save preprocessed data
    print(f"\n=== SAVING PREPROCESSED DATA ===")
    
    # Save as numpy arrays for efficient loading
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train.values)
    np.save('y_test.npy', y_test.values)
    
    print("Saved preprocessed data:")
    print("  X_train.npy, X_test.npy")
    print("  y_train.npy, y_test.npy")
    
    # Save preprocessing artifacts
    feature_info = {
        'target_column': target_column,
        'n_features': X_train.shape[1],
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'train_positive_rate': y_train.mean(),
        'test_positive_rate': y_test.mean()
    }
    
    save_preprocessing_artifacts(preprocessor, label_encoders, feature_info)
    
    # Display final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Target: {target_column} (15+ minute delays)")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Positive class rate (train): {y_train.mean():.3f}")
    print(f"Positive class rate (test): {y_test.mean():.3f}")
    print(f"Data ready for machine learning!")

if __name__ == "__main__":
    main()
