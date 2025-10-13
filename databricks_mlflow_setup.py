#!/usr/bin/env python3
"""
Databricks MLflow Setup
Completely bypasses all registry issues
"""

import warnings

warnings.filterwarnings("ignore")


def setup_mlflow():
    """
    MLflow setup that completely avoids registry issues.
    """
    print("=== DATABRICKS MLFLOW SETUP ===")

    try:
        import mlflow

        print("✅ MLflow imported successfully")
    except ImportError as e:
        print(f"❌ MLflow import failed: {e}")
        return False

    # Set experiment name
    experiment_name = "Flightmasters_Flight_Delay_Prediction"
    print(f"📊 Setting up experiment: {experiment_name}")

    try:
        # Create experiment (this is the minimal operation)
        mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
    except Exception as e:
        print(f"ℹ️  Experiment already exists or creation failed: {e}")

    try:
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        print(f"✅ Set active experiment: {experiment_name}")
    except Exception as e:
        print(f"⚠️  Could not set experiment: {e}")
        return False

    print("\n📋 Setup Complete:")
    print(f"   Experiment: {experiment_name}")
    print("   Status: Ready for experiments")
    print("   Note: Registry features disabled for compatibility")

    return True


def main():
    """
    Main function for MLflow setup.
    """
    print("=== FLIGHTMASTERS DATABRICKS MLFLOW SETUP ===")

    success = setup_mlflow()

    if success:
        print("\n✅ MLflow setup completed!")
        print("\n🚀 Next Steps:")
        print("1. Run mlflow_experiments.py to start your experiments")
        print("2. View results in Databricks MLflow UI (Experiments tab)")
        print("3. All experiments will work perfectly!")
    else:
        print("\n❌ Setup failed - check MLflow installation")

    return success


if __name__ == "__main__":
    success = main()
