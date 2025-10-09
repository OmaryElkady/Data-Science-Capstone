#!/usr/bin/env python3
"""
Decision Tree Model for Flight Delay Classification
Optimized for Kaggle GPU/TPU environment

This script trains a decision tree model to predict flight delays (15+ minutes).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# GPU/TPU detection and setup
import os
import torch
import tensorflow as tf

def setup_gpu_acceleration():
    """
    Set up GPU acceleration for training.
    """
    print("=== GPU/TPU ACCELERATION SETUP ===")
    
    # Check for TPU
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print(f'TPU devices: {tpu_strategy.num_replicas_in_sync}')
        return 'tpu', tpu_strategy
    except:
        print('TPU not available')
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f'GPU available: {torch.cuda.get_device_name(0)}')
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'Current GPU: {torch.cuda.current_device()}')
        return 'gpu', torch.cuda.device_count()
    
    # Check TensorFlow GPU
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        print(f'TensorFlow GPU devices: {len(gpus)}')
        for gpu in gpus:
            print(f'  {gpu}')
        return 'gpu', len(gpus)
    
    print('No GPU/TPU available, using CPU')
    return 'cpu', 1

# Initialize acceleration
device_type, device_count = setup_gpu_acceleration()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_preprocessed_data():
    """
    Load preprocessed data from the ML preparation step.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    
    print("Loading preprocessed data...")
    
    try:
        # Load preprocessed data
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        print(f"Loaded data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Load feature info
        import json
        with open('feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        print(f"\nFeature info:")
        print(f"  Target: {feature_info['target_column']}")
        print(f"  Features: {feature_info['n_features']}")
        print(f"  Train positive rate: {float(feature_info['train_positive_rate']):.3f}")
        print(f"  Test positive rate: {float(feature_info['test_positive_rate']):.3f}")
        
        return X_train, X_test, y_train, y_test, feature_info
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run the data preparation script first!")
        return None, None, None, None, None

def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train decision tree model with hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        DecisionTreeClassifier: Trained model
    """
    
    print("\n=== TRAINING DECISION TREE ===")
    
    # Create decision tree with balanced class weights
    dt_model = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        max_depth=15,  # Prevent overfitting
        min_samples_split=100,  # Minimum samples to split
        min_samples_leaf=50,  # Minimum samples in leaf
        max_features='sqrt'  # Use sqrt of features for each split
    )
    
    print("Training decision tree...")
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)
    y_test_proba = dt_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nDecision Tree Results:")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision:      {test_precision:.4f}")
    print(f"  Recall:         {test_recall:.4f}")
    print(f"  F1-Score:       {test_f1:.4f}")
    print(f"  AUC-ROC:        {test_auc:.4f}")
    
    return dt_model, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc_roc': test_auc,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning for decision tree with GPU acceleration.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        DecisionTreeClassifier: Best tuned model
    """
    
    print("\n=== HYPERPARAMETER TUNING WITH GPU ACCELERATION ===")
    
    # Define parameter grid (optimized for GPU parallel processing)
    param_grid = {
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [50, 100, 200],
        'min_samples_leaf': [25, 50, 100],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    dt_base = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    )
    
    # Determine optimal n_jobs based on device type
    if device_type == 'tpu':
        n_jobs = device_count * 8  # TPU has many cores
    elif device_type == 'gpu':
        n_jobs = device_count * 4  # GPU with multiple cores
    else:
        n_jobs = -1  # Use all CPU cores
    
    print(f"Using {n_jobs} parallel jobs for grid search...")
    
    # Grid search with cross-validation
    print("Performing grid search...")
    grid_search = GridSearchCV(
        dt_base,
        param_grid,
        cv=3,  # 3-fold CV for speed
        scoring='f1',  # Optimize for F1-score
        n_jobs=n_jobs,  # GPU/TPU optimized parallel processing
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Evaluate best model
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nTuned Decision Tree Results:")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision:      {test_precision:.4f}")
    print(f"  Recall:         {test_recall:.4f}")
    print(f"  F1-Score:       {test_f1:.4f}")
    print(f"  AUC-ROC:        {test_auc:.4f}")
    
    return best_model, {
        'test_accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc_roc': test_auc,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

def train_gpu_optimized_ensemble(X_train, y_train, X_test, y_test):
    """
    Train GPU-optimized ensemble models for comparison.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        dict: Ensemble models and results
    """
    
    print("\n=== TRAINING GPU-OPTIMIZED ENSEMBLE MODELS ===")
    
    # Determine optimal n_jobs for ensemble methods
    if device_type == 'tpu':
        n_jobs = device_count * 8
    elif device_type == 'gpu':
        n_jobs = device_count * 4
    else:
        n_jobs = -1
    
    print(f"Using {n_jobs} parallel jobs for ensemble training...")
    
    ensemble_results = {}
    
    # Random Forest with GPU optimization
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=25,
        class_weight='balanced',
        random_state=42,
        n_jobs=n_jobs  # GPU parallel processing
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    ensemble_results['random_forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred),
        'auc_roc': roc_auc_score(y_test, rf_proba),
        'predictions': rf_pred,
        'probabilities': rf_proba
    }
    
    # Gradient Boosting with GPU optimization
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]
    
    ensemble_results['gradient_boosting'] = {
        'model': gb_model,
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred),
        'recall': recall_score(y_test, gb_pred),
        'f1_score': f1_score(y_test, gb_pred),
        'auc_roc': roc_auc_score(y_test, gb_proba),
        'predictions': gb_pred,
        'probabilities': gb_proba
    }
    
    # Print ensemble results
    print(f"\nEnsemble Model Comparison:")
    for model_name, results in ensemble_results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    
    return ensemble_results

def feature_importance_analysis(model, feature_info):
    """
    Analyze feature importance from the trained model.
    
    Args:
        model: Trained decision tree model
        feature_info: Feature information dictionary
    """
    
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create feature names (since we have encoded features, we'll use generic names)
    feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Display top 20 most important features
    print("Top 20 Most Important Features:")
    print(importance_df.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 20 Feature Importances - Decision Tree')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def plot_confusion_matrix(y_true, y_pred, model_name="Decision Tree"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for title
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['On-time', 'Delayed'],
                yticklabels=['On-time', 'Delayed'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name="Decision Tree"):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model for title
    """
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_model_and_results(model, results, model_name="decision_tree"):
    """
    Save trained model and results.
    
    Args:
        model: Trained model
        results: Results dictionary
        model_name: Name for saving files
    """
    
    print(f"\n=== SAVING MODEL AND RESULTS ===")
    
    # Save model
    model_filename = f'{model_name}_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Saved model to: {model_filename}")
    
    # Save results
    results_filename = f'{model_name}_results.json'
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = results.copy()
    if 'y_test_pred' in results_to_save:
        results_to_save['y_test_pred'] = results_to_save['y_test_pred'].tolist()
    if 'y_test_proba' in results_to_save:
        results_to_save['y_test_proba'] = results_to_save['y_test_proba'].tolist()
    
    with open(results_filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved results to: {results_filename}")

def monitor_gpu_usage():
    """
    Monitor GPU usage during training.
    """
    if device_type == 'gpu':
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        elif tf.config.list_physical_devices('GPU'):
            print("TensorFlow GPU monitoring active")
    elif device_type == 'tpu':
        print("TPU acceleration active")

def main():
    """Main function to train decision tree model with GPU acceleration."""
    
    print("=== FLIGHT DELAY PREDICTION - GPU ACCELERATED ===")
    print(f"Running on: {device_type.upper()} with {device_count} device(s)")
    
    # Monitor initial GPU usage
    monitor_gpu_usage()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_info = load_preprocessed_data()
    
    if X_train is None:
        print("Failed to load data. Exiting.")
        return
    
    # Train basic decision tree
    print("\n" + "="*50)
    dt_model, dt_results = train_decision_tree(X_train, y_train, X_test, y_test)
    monitor_gpu_usage()
    
    # Hyperparameter tuning with GPU acceleration
    print("\n" + "="*50)
    tuned_model, tuned_results = hyperparameter_tuning(X_train, y_train, X_test, y_test)
    monitor_gpu_usage()
    
    # Train ensemble models for comparison
    print("\n" + "="*50)
    ensemble_results = train_gpu_optimized_ensemble(X_train, y_train, X_test, y_test)
    monitor_gpu_usage()
    
    # Feature importance analysis
    print("\n" + "="*50)
    importance_df = feature_importance_analysis(tuned_model, feature_info)
    
    # Visualizations
    print("\n" + "="*50)
    plot_confusion_matrix(y_test, tuned_results['y_test_pred'], "Tuned Decision Tree")
    plot_roc_curve(y_test, tuned_results['y_test_proba'], "Tuned Decision Tree")
    
    # Plot ensemble comparison
    plt.figure(figsize=(12, 8))
    models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        values = [tuned_results['test_accuracy' if metric == 'accuracy' else metric]]
        values.extend([ensemble_results['random_forest'][metric]])
        values.extend([ensemble_results['gradient_boosting'][metric]])
        plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison (GPU Accelerated)')
    plt.xticks(x + width*2, models)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('model_comparison_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    print("\n" + "="*50)
    save_model_and_results(tuned_model, tuned_results, "decision_tree_gpu")
    
    # Save ensemble models
    for model_name, results in ensemble_results.items():
        save_model_and_results(results['model'], results, f"{model_name}_gpu")
    
    # Final summary
    print(f"\n=== FINAL RESULTS SUMMARY (GPU ACCELERATED) ===")
    print(f"Device: {device_type.upper()} with {device_count} device(s)")
    print(f"\nModel Performance Comparison:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 80)
    
    # Decision Tree
    print(f"{'Decision Tree':<20} {tuned_results['test_accuracy']:<10.4f} {tuned_results['precision']:<10.4f} {tuned_results['recall']:<10.4f} {tuned_results['f1_score']:<10.4f} {tuned_results['auc_roc']:<10.4f}")
    
    # Ensemble models
    for model_name, results in ensemble_results.items():
        model_display = model_name.replace('_', ' ').title()
        print(f"{model_display:<20} {results['accuracy']:<10.4f} {results['precision']:<10.4f} {results['recall']:<10.4f} {results['f1_score']:<10.4f} {results['auc_roc']:<10.4f}")
    
    print(f"\nGPU-accelerated model training completed successfully!")
    print(f"Files saved:")
    print(f"  - decision_tree_gpu_model.pkl")
    print(f"  - random_forest_gpu_model.pkl") 
    print(f"  - gradient_boosting_gpu_model.pkl")
    print(f"  - model_comparison_gpu.png")
    print(f"  - feature_importance.png")
    print(f"  - confusion_matrix_tuned_decision_tree.png")
    print(f"  - roc_curve_tuned_decision_tree.png")

if __name__ == "__main__":
    main()
