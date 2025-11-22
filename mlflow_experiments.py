#!/usr/bin/env python3
"""
MLflow Experiment Tracking for Flight Delay Prediction
Optimized for Databricks Platform

This script demonstrates MLflow experiment tracking for the Flightmasters project,
comparing multiple ML algorithms for flight delay classification.
"""

import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# MLflow Configuration for Databricks
def setup_mlflow():
    """
    Set up MLflow for Databricks platform.
    """
    print("=== MLFLOW SETUP FOR DATABRICKS ===")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # Set experiment name
    experiment_name = "/Users/kshitijmishra231@gmail.com/Flightmasters_Flight_Delay_Prediction"

    # Create or get experiment
    try:
        mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Using existing experiment: {experiment_name}")

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    return experiment_name


def load_flight_data_from_databricks():
    """
    Load flight delay data from Databricks tables and preprocess it for ML.
    """
    print("=== LOADING FLIGHT DATA FROM DATABRICKS ===")

    try:
        print("📊 Loading from Databricks tables...")
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("MLflowFlightDelay").getOrCreate()
        flights_df = spark.sql("SELECT * FROM workspace.google_drive.flights_processed")
        df = flights_df.toPandas()
        print(f"✅ Loaded flights_processed data: {df.shape}")

        # --- START: NEW PREPROCESSING CODE ---

        # 1. Define the target variable
        target_column = "is_delayed_15_min"

        # 2. Select only columns with number-like data types for our features (X)
        # This automatically ignores strings, dates, and timestamps.
        X = df.select_dtypes(include=np.number)

        # 3. Explicitly drop the target column and any other non-feature columns from X
        if target_column in X.columns:
            X = X.drop(columns=[target_column])
        # Also drop other target-like columns to prevent data leakage
        leaky_columns = ["is_delayed_30_min", "is_delayed_any", "arr_delay"]
        X = X.drop(columns=[col for col in leaky_columns if col in X.columns])

        # 4. Define our target data (y)
        y = df[target_column]

        print("✅ Preprocessing complete: Selected only numeric features.")
        print(f"📋 Using {len(X.columns)} features: {list(X.columns)}")

        # --- END: NEW PREPROCESSING CODE ---

        # Handle any missing values in the remaining numeric columns
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"✅ Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")

        # Now the delay rate should be a valid probability (between 0 and 1)
        print(f"📊 Delay rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"⚠️  An error occurred: {e}")
        print("📊 Creating sample data for demonstration...")
        return create_sample_data()  # Fallback to sample data on error


def create_sample_data():
    """
    Create sample data as fallback.
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Generate sample features (flight data simulation)
    X = np.random.randn(n_samples, n_features)

    # Create realistic feature names
    feature_names = [
        "departure_hour",
        "departure_delay",
        "weather_score",
        "airport_congestion",
        "airline_delay_history",
        "route_popularity",
        "day_of_week",
        "month",
        "distance",
        "aircraft_age",
        "crew_experience",
        "fuel_efficiency",
        "maintenance_score",
        "passenger_load",
        "baggage_weight",
        "wind_speed",
        "visibility",
        "temperature",
        "pressure",
        "humidity",
    ]

    X_df = pd.DataFrame(X, columns=feature_names)

    # Create target variable (flight delay probability)
    delay_prob = (
        0.3 * X_df["departure_delay"]
        + 0.2 * X_df["weather_score"]
        + 0.15 * X_df["airport_congestion"]
        + 0.1 * X_df["airline_delay_history"]
        + np.random.normal(0, 0.1, n_samples)
    )

    # Convert to binary classification (delay >= 15 minutes)
    y = (delay_prob > np.percentile(delay_prob, 70)).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

    print(f"📊 Created sample data - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"📈 Delay rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


def log_model_performance(mlflow, model, X_test, y_test, model_name, run_params=None):
    """
    Log model performance metrics to MLflow.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("auc_roc", auc)

    # Log model parameters if provided
    if run_params:
        for param, value in run_params.items():
            mlflow.log_param(param, value)

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_dict = {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }
    mlflow.log_dict(cm_dict, "confusion_matrix.json")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc if y_proba is not None else None,
    }


def experiment_decision_tree(X_train, X_test, y_train, y_test):
    """
    Experiment with Decision Tree models and log to MLflow.
    """
    print("\n=== DECISION TREE EXPERIMENT ===")

    with mlflow.start_run(run_name="Decision_Tree_Baseline"):
        # Train baseline decision tree
        dt_model = DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=15, min_samples_split=100, min_samples_leaf=50
        )

        dt_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "DecisionTreeClassifier",
                "max_depth": 15,
                "min_samples_split": 100,
                "min_samples_leaf": 50,
                "class_weight": "balanced",
            }
        )

        # Log performance
        results = log_model_performance(mlflow, dt_model, X_test, y_test, "Decision Tree")

        # Log model
        mlflow.sklearn.log_model(dt_model, "model")

        print(f"Decision Tree - F1: {results['f1_score']:.4f}, AUC: {results['auc_roc']:.4f}")

        return dt_model, results


def experiment_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """
    Experiment with hyperparameter tuning using GridSearchCV.
    """
    print("\n=== HYPERPARAMETER TUNING EXPERIMENT ===")

    with mlflow.start_run(run_name="Decision_Tree_Hyperparameter_Tuning"):
        # Define parameter grid
        param_grid = {
            "max_depth": [10, 15, 20, 25],
            "min_samples_split": [50, 100, 200],
            "min_samples_leaf": [25, 50, 100],
            "max_features": ["sqrt", "log2", None],
        }

        # Grid search
        dt_base = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        grid_search = GridSearchCV(dt_base, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)

        # Log best parameters
        mlflow.log_params(
            {
                "model_type": "DecisionTreeClassifier_Tuned",
                "best_max_depth": grid_search.best_params_["max_depth"],
                "best_min_samples_split": grid_search.best_params_["min_samples_split"],
                "best_min_samples_leaf": grid_search.best_params_["min_samples_leaf"],
                "best_max_features": str(grid_search.best_params_["max_features"]),
                "cv_folds": 3,
                "scoring": "f1",
            }
        )

        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Log performance
        results = log_model_performance(mlflow, grid_search.best_estimator_, X_test, y_test, "Tuned Decision Tree")

        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

        print(f"Best params: {grid_search.best_params_}")
        print(f"Tuned Decision Tree - F1: {results['f1_score']:.4f}, AUC: {results['auc_roc']:.4f}")

        return grid_search.best_estimator_, results


def experiment_random_forest(X_train, X_test, y_train, y_test):
    """
    Experiment with Random Forest models.
    """
    print("\n=== RANDOM FOREST EXPERIMENT ===")

    with mlflow.start_run(run_name="Random_Forest"):
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=50,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        rf_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 50,
                "min_samples_leaf": 25,
                "class_weight": "balanced",
            }
        )

        # Log performance
        results = log_model_performance(mlflow, rf_model, X_test, y_test, "Random Forest")

        # Log model
        mlflow.sklearn.log_model(rf_model, "model")

        print(f"Random Forest - F1: {results['f1_score']:.4f}, AUC: {results['auc_roc']:.4f}")

        return rf_model, results


def experiment_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Experiment with Gradient Boosting models.
    """
    print("\n=== GRADIENT BOOSTING EXPERIMENT ===")

    with mlflow.start_run(run_name="Gradient_Boosting"):
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

        gb_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(
            {"model_type": "GradientBoostingClassifier", "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        )

        # Log performance
        results = log_model_performance(mlflow, gb_model, X_test, y_test, "Gradient Boosting")

        # Log model
        mlflow.sklearn.log_model(gb_model, "model")

        print(f"Gradient Boosting - F1: {results['f1_score']:.4f}, AUC: {results['auc_roc']:.4f}")

        return gb_model, results


def experiment_feature_engineering(X_train, X_test, y_train, y_test):
    """
    Experiment with feature engineering (scaling).
    """
    print("\n=== FEATURE ENGINEERING EXPERIMENT ===")

    with mlflow.start_run(run_name="Decision_Tree_with_Feature_Scaling"):
        # Apply feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model with scaled features
        dt_model = DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=15, min_samples_split=100, min_samples_leaf=50
        )

        dt_model.fit(X_train_scaled, y_train)

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "DecisionTreeClassifier_Scaled",
                "feature_scaling": "StandardScaler",
                "max_depth": 15,
                "min_samples_split": 100,
                "min_samples_leaf": 50,
                "class_weight": "balanced",
            }
        )

        # Log performance
        results = log_model_performance(mlflow, dt_model, X_test_scaled, y_test, "Decision Tree Scaled")

        # Log both model and scaler
        mlflow.sklearn.log_model(dt_model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")

        print(f"Decision Tree (Scaled) - F1: {results['f1_score']:.4f}, AUC: {results['auc_roc']:.4f}")

        return dt_model, scaler, results


def create_experiment_summary(experiment_results):
    """
    Create a summary of all experiments.
    """
    print("\n=== EXPERIMENT SUMMARY ===")

    summary_data = []
    for run_name, results in experiment_results.items():
        summary_data.append(
            {
                "experiment": run_name,
                "f1_score": results["f1_score"],
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "auc_roc": results["auc_roc"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("f1_score", ascending=False)

    print("\n📊 Model Performance Ranking (by F1-Score):")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # Find best model
    best_model = summary_df.iloc[0]
    print(f"\n🏆 Best Performing Model: {best_model['experiment']}")
    print(f"   F1-Score: {best_model['f1_score']:.4f}")
    print(f"   AUC-ROC: {best_model['auc_roc']:.4f}")

    return summary_df


def main():
    """
    Main function to run all MLflow experiments.
    """
    print("=== FLIGHTMASTERS MLFLOW EXPERIMENT TRACKING ===")
    print("Demonstrating MLflow for Flight Delay Prediction")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup MLflow
    experiment_name = setup_mlflow()

    # Load data from Databricks
    X_train, X_test, y_train, y_test = load_flight_data_from_databricks()

    # Store experiment results
    experiment_results = {}

    # Run experiments
    print(f"\n🚀 Starting experiments in MLflow experiment: {experiment_name}")

    # 1. Baseline Decision Tree
    dt_model, dt_results = experiment_decision_tree(X_train, X_test, y_train, y_test)
    experiment_results["Decision_Tree_Baseline"] = dt_results

    # 2. Hyperparameter Tuning
    tuned_model, tuned_results = experiment_hyperparameter_tuning(X_train, X_test, y_train, y_test)
    experiment_results["Decision_Tree_Tuned"] = tuned_results

    # 3. Random Forest
    rf_model, rf_results = experiment_random_forest(X_train, X_test, y_train, y_test)
    experiment_results["Random_Forest"] = rf_results

    # 4. Gradient Boosting
    gb_model, gb_results = experiment_gradient_boosting(X_train, X_test, y_train, y_test)
    experiment_results["Gradient_Boosting"] = gb_results

    # 5. Feature Engineering
    scaled_model, scaler, scaled_results = experiment_feature_engineering(X_train, X_test, y_train, y_test)
    experiment_results["Decision_Tree_Scaled"] = scaled_results

    # Create summary
    summary_df = create_experiment_summary(experiment_results)

    # Save summary
    summary_df.to_csv("mlflow_experiment_summary.csv", index=False)
    print("\n💾 Experiment summary saved to: mlflow_experiment_summary.csv")

    print("\n✅ All experiments completed successfully!")
    print("\n📋 Next Steps:")
    print("1. View experiments in MLflow UI")
    print("2. Compare model performance metrics")
    print("3. Select best model for deployment")
    print("4. Use MLflow Model Registry for production models")

    return experiment_results, summary_df


if __name__ == "__main__":
    results, summary = main()
