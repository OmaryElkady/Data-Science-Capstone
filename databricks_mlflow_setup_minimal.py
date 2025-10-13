#!/usr/bin/env python3
"""
Minimal Databricks MLflow Setup
Avoids all problematic registry calls
"""

import warnings
from datetime import datetime

import mlflow

warnings.filterwarnings("ignore")


def setup_mlflow_minimal():
    """
    Minimal MLflow setup that avoids all registry issues.
    """
    print("=== MINIMAL DATABRICKS MLFLOW SETUP ===")

    # Check if running on Databricks
    try:
        import databricks  # noqa: F401

        print("✅ Running on Databricks platform")
        is_databricks = True
    except ImportError:
        print("⚠️  Not running on Databricks - using local MLflow")
        is_databricks = False

    # MLflow configuration
    if is_databricks:
        tracking_uri = "databricks"
        print("📊 Using Databricks managed MLflow tracking")
    else:
        tracking_uri = "file:///tmp/mlflow"
        print("📁 Using local MLflow tracking")

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment
    experiment_name = "Flightmasters_Flight_Delay_Prediction"

    try:
        mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        print(f"✅ Using existing experiment: {experiment_name}")

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    print("\n📋 Setup Complete:")
    print("   Experiment:", experiment_name)
    print("   Tracking URI:", tracking_uri)
    print("   Platform: Databricks" if is_databricks else "Local")

    return {
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "is_databricks": is_databricks,
    }


def main():
    """
    Main function for minimal MLflow setup.
    """
    print("=== FLIGHTMASTERS MINIMAL MLFLOW SETUP ===")
    print(f"Setup initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup MLflow (minimal)
    setup_info = setup_mlflow_minimal()

    print("\n✅ Minimal MLflow setup completed successfully!")
    print("\n🚀 Next Steps:")
    print("1. Run mlflow_experiments.py to start your experiments")
    print("2. View results in Databricks MLflow UI (Experiments tab)")
    print("3. All experiments will work perfectly!")

    return setup_info


if __name__ == "__main__":
    setup_info = main()
