#!/usr/bin/env python3
"""
Kaggle Setup Script for Flight Delay Prediction
This script sets up the environment and prepares data for Kaggle GPU training.
"""

import os
import shutil
import zipfile
from pathlib import Path

def setup_kaggle_environment():
    """
    Set up the Kaggle environment with all necessary files.
    """
    
    print("=== KAGGLE ENVIRONMENT SETUP ===")
    
    # Create kaggle directory
    kaggle_dir = Path("kaggle_workspace")
    kaggle_dir.mkdir(exist_ok=True)
    
    # Files to copy to Kaggle workspace
    files_to_copy = [
        "flights_processed.csv",
        "X_train.npy",
        "X_test.npy", 
        "y_train.npy",
        "y_test.npy",
        "preprocessor.pkl",
        "label_encoders.pkl",
        "feature_info.json",
        "kaggle_decision_tree.py"
    ]
    
    print("Copying files to Kaggle workspace...")
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, kaggle_dir / file)
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
    
    # Create Kaggle requirements.txt with GPU support
    kaggle_requirements = """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
torch>=1.12.0
tensorflow>=2.8.0
xgboost>=1.6.0
lightgbm>=3.3.0
"""
    
    with open(kaggle_dir / "requirements.txt", "w") as f:
        f.write(kaggle_requirements)
    
    print("  ✓ requirements.txt")
    
    # Create Kaggle README
    kaggle_readme = """# Flight Delay Prediction - GPU Accelerated Models

## Overview
This notebook trains multiple ML models (Decision Tree, Random Forest, Gradient Boosting) to predict flight delays (15+ minutes) using airline data with **GPU/TPU acceleration**.

## 🚀 GPU/TPU Acceleration Features
- **Automatic GPU/TPU detection** (NVIDIA GPU, Google TPU)
- **Parallel processing optimization** based on device type
- **Memory monitoring** during training
- **Multi-model comparison** with ensemble methods

## Data
- **Target**: 15+ minute delays (17.8% positive class)
- **Features**: 63 engineered features from flight data
- **Samples**: 400K training, 100K test
- **Preprocessing**: All categorical variables encoded, numeric features scaled

## Models
- **Decision Tree**: With hyperparameter tuning
- **Random Forest**: GPU-optimized parallel processing
- **Gradient Boosting**: Enhanced performance
- **Optimization**: Grid search with GPU-accelerated cross-validation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Files
- `kaggle_decision_tree.py`: GPU-accelerated training script
- `X_train.npy`, `X_test.npy`: Preprocessed features
- `y_train.npy`, `y_test.npy`: Target variables
- `preprocessor.pkl`: Preprocessing pipeline
- `feature_info.json`: Feature metadata

## 🎯 Kaggle Setup Instructions
1. **Upload dataset**: Upload all files to Kaggle as a dataset
2. **Enable GPU**: In notebook settings, select "GPU T4 x2" or "TPU v3-8"
3. **Run script**: Execute `kaggle_decision_tree.py`
4. **Monitor performance**: GPU usage and training progress displayed
5. **Compare models**: Automatic comparison of all trained models

## Expected Results (GPU Accelerated)
- **Training time**: 3-5x faster than CPU
- **Decision Tree**: Accuracy ~85-90%, F1-Score ~0.6-0.7
- **Random Forest**: Improved performance with ensemble
- **Gradient Boosting**: Best overall performance
- **AUC-ROC**: ~0.8-0.9 across all models

## GPU Memory Usage
- **TPU**: Automatically utilizes all TPU cores
- **GPU**: Optimized parallel processing with memory monitoring
- **CPU fallback**: Available if no GPU/TPU detected

## Output Files
- Model files: `*_gpu_model.pkl`
- Results: `*_gpu_results.json`
- Visualizations: Performance comparison charts
- Feature importance: Top 20 most important features
"""
    
    with open(kaggle_dir / "README.md", "w") as f:
        f.write(kaggle_readme)
    
    print("  ✓ README.md")
    
    # Create zip file for easy upload
    zip_filename = "kaggle_flight_delay_prediction.zip"
    print(f"\nCreating zip file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in kaggle_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(kaggle_dir.parent))
    
    print(f"✓ Created {zip_filename}")
    
    return kaggle_dir, zip_filename

def create_kaggle_notebook():
    """
    Create a Kaggle notebook version of the training script.
    """
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Flight Delay Prediction - Decision Tree Model\\n",
    "\\n",
    "This notebook trains a decision tree model to predict flight delays (15+ minutes) using airline data.\\n",
    "\\n",
    "## Key Features:\\n",
    "- **Target**: 15+ minute delays (17.8% positive class)\\n",
    "- **Features**: 63 engineered features\\n",
    "- **Samples**: 400K training, 100K test\\n",
    "- **Model**: Decision Tree with hyperparameter tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages\\n",
    "!pip install -q scikit-learn matplotlib seaborn joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import joblib\\n",
    "from sklearn.tree import DecisionTreeClassifier\\n",
    "from sklearn.metrics import (\\n",
    "    classification_report, confusion_matrix, accuracy_score,\\n",
    "    precision_score, recall_score, f1_score, roc_auc_score, roc_curve\\n",
    ")\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from sklearn.model_selection import GridSearchCV\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "# Set style\\n",
    "plt.style.use('default')\\n",
    "sns.set_palette(\"husl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load preprocessed data\\n",
    "print(\"Loading preprocessed data...\")\\n",
    "\\n",
    "X_train = np.load('/kaggle/input/flight-delay-data/X_train.npy')\\n",
    "X_test = np.load('/kaggle/input/flight-delay-data/X_test.npy')\\n",
    "y_train = np.load('/kaggle/input/flight-delay-data/y_train.npy')\\n",
    "y_test = np.load('/kaggle/input/flight-delay-data/y_test.npy')\\n",
    "\\n",
    "print(f\"Data shapes:\")\\n",
    "print(f\"X_train: {X_train.shape}\")\\n",
    "print(f\"X_test: {X_test.shape}\")\\n",
    "print(f\"y_train: {y_train.shape}\")\\n",
    "print(f\"y_test: {y_test.shape}\")\\n",
    "print(f\"Positive class rate: {y_train.mean():.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train Decision Tree with hyperparameter tuning\\n",
    "print(\"Training Decision Tree...\")\\n",
    "\\n",
    "# Parameter grid for tuning\\n",
    "param_grid = {\\n",
    "    'max_depth': [10, 15, 20],\\n",
    "    'min_samples_split': [50, 100, 200],\\n",
    "    'min_samples_leaf': [25, 50, 100],\\n",
    "    'max_features': ['sqrt', 'log2']\\n",
    "}\\n",
    "\\n",
    "# Base model\\n",
    "dt_base = DecisionTreeClassifier(\\n",
    "    random_state=42,\\n",
    "    class_weight='balanced'\\n",
    ")\\n",
    "\\n",
    "# Grid search\\n",
    "grid_search = GridSearchCV(\\n",
    "    dt_base, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1\\n",
    ")\\n",
    "\\n",
    "grid_search.fit(X_train, y_train)\\n",
    "\\n",
    "print(f\"Best parameters: {grid_search.best_params_}\")\\n",
    "print(f\"Best CV score: {grid_search.best_score_:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate model\\n",
    "best_model = grid_search.best_estimator_\\n",
    "y_test_pred = best_model.predict(X_test)\\n",
    "y_test_proba = best_model.predict_proba(X_test)[:, 1]\\n",
    "\\n",
    "# Calculate metrics\\n",
    "accuracy = accuracy_score(y_test, y_test_pred)\\n",
    "precision = precision_score(y_test, y_test_pred)\\n",
    "recall = recall_score(y_test, y_test_pred)\\n",
    "f1 = f1_score(y_test, y_test_pred)\\n",
    "auc = roc_auc_score(y_test, y_test_proba)\\n",
    "\\n",
    "print(f\"Model Performance:\")\\n",
    "print(f\"Accuracy:  {accuracy:.4f}\")\\n",
    "print(f\"Precision: {precision:.4f}\")\\n",
    "print(f\"Recall:    {recall:.4f}\")\\n",
    "print(f\"F1-Score:  {f1:.4f}\")\\n",
    "print(f\"AUC-ROC:   {auc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature importance\\n",
    "importances = best_model.feature_importances_\\n",
    "feature_names = [f\"Feature_{i}\" for i in range(len(importances))]\\n",
    "\\n",
    "importance_df = pd.DataFrame({\\n",
    "    'feature': feature_names,\\n",
    "    'importance': importances\\n",
    "}).sort_values('importance', ascending=False)\\n",
    "\\n",
    "print(\"Top 10 Most Important Features:\")\\n",
    "print(importance_df.head(10))\\n",
    "\\n",
    "# Plot feature importance\\n",
    "plt.figure(figsize=(12, 8))\\n",
    "top_features = importance_df.head(20)\\n",
    "sns.barplot(data=top_features, x='importance', y='feature')\\n",
    "plt.title('Top 20 Feature Importances - Decision Tree')\\n",
    "plt.xlabel('Importance')\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix\\n",
    "cm = confusion_matrix(y_test, y_test_pred)\\n",
    "\\n",
    "plt.figure(figsize=(8, 6))\\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\\n",
    "            xticklabels=['On-time', 'Delayed'],\\n",
    "            yticklabels=['On-time', 'Delayed'])\\n",
    "plt.title('Confusion Matrix - Decision Tree')\\n",
    "plt.ylabel('True Label')\\n",
    "plt.xlabel('Predicted Label')\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ROC Curve\\n",
    "fpr, tpr, _ = roc_curve(y_test, y_test_proba)\\n",
    "\\n",
    "plt.figure(figsize=(8, 6))\\n",
    "plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {auc:.3f})')\\n",
    "plt.plot([0, 1], [0, 1], 'k--', label='Random')\\n",
    "plt.xlabel('False Positive Rate')\\n",
    "plt.ylabel('True Positive Rate')\\n",
    "plt.title('ROC Curve - Decision Tree')\\n",
    "plt.legend()\\n",
    "plt.grid(True, alpha=0.3)\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    kaggle_dir = Path("kaggle_workspace")
    with open(kaggle_dir / "flight_delay_prediction.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("✓ Created Kaggle notebook: flight_delay_prediction.ipynb")

def main():
    """Main setup function."""
    
    print("Setting up Kaggle environment for flight delay prediction...")
    
    # Setup Kaggle environment
    kaggle_dir, zip_filename = setup_kaggle_environment()
    
    # Create Kaggle notebook
    create_kaggle_notebook()
    
    print(f"\n=== SETUP COMPLETE ===")
    print(f"Kaggle workspace created in: {kaggle_dir}")
    print(f"Zip file ready for upload: {zip_filename}")
    print(f"\nNext steps:")
    print(f"1. Upload {zip_filename} to Kaggle as a dataset")
    print(f"2. Create a new notebook in Kaggle")
    print(f"3. Enable GPU/TPU for faster training")
    print(f"4. Run the training script or notebook")
    
    # List files in Kaggle workspace
    print(f"\nFiles in Kaggle workspace:")
    for file in sorted(kaggle_dir.iterdir()):
        size = file.stat().st_size / 1024 / 1024  # MB
        print(f"  {file.name} ({size:.1f} MB)")

if __name__ == "__main__":
    main()
