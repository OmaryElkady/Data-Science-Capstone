#!/usr/bin/env python3
"""
Databricks MLflow Integration Setup
Configuration and setup for MLflow experiments on Databricks platform
"""

import json
import os
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def setup_databricks_mlflow():
    """
    Set up MLflow for Databricks platform with proper configuration.
    """
    print("=== DATABRICKS MLFLOW SETUP ===")

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
        # Use Databricks managed MLflow
        tracking_uri = "databricks"
        print("📊 Using Databricks managed MLflow tracking")

        # Get workspace info
        workspace_url = os.environ.get("DATABRICKS_HOST", "Unknown")
        print("🏢 Workspace URL:", workspace_url)

    else:
        # Use local file store for development
        tracking_uri = "file:///tmp/mlflow"
        os.makedirs("/tmp/mlflow", exist_ok=True)
        print("📁 Using local MLflow tracking")

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment
    experiment_name = "Flightmasters_Flight_Delay_Prediction"

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"✅ Using existing experiment: {experiment_name}")

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Initialize MLflow client
    client = MlflowClient()

    # Get experiment details
    experiment = client.get_experiment(experiment_id)

    print("\n📋 Experiment Details:")
    print("   Name:", experiment.name)
    print("   ID:", experiment.experiment_id)
    print("   Artifact Location:", experiment.artifact_location)
    print("   Lifecycle Stage:", experiment.lifecycle_stage)

    return {
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "tracking_uri": tracking_uri,
        "is_databricks": is_databricks,
        "client": client,
    }


def create_model_registry():
    """
    Create and configure MLflow Model Registry for production models.
    """
    print("\n=== MODEL REGISTRY SETUP ===")

    client = MlflowClient()

    # Model registry info
    registry_info = {"registry_uri": mlflow.get_registry_uri(), "registered_models": []}

    # List existing registered models
    try:
        registered_models = client.search_registered_models()
        if registered_models:
            print(f"📚 Found {len(registered_models)} registered models:")
            for model in registered_models:
                registry_info["registered_models"].append(
                    {
                        "name": model.name,
                        "latest_version": model.latest_versions[0].version if model.latest_versions else "None",
                        "stage": model.latest_versions[0].current_stage if model.latest_versions else "None",
                    }
                )
                print(f"   - {model.name} (v{model.latest_versions[0].version if model.latest_versions else 'None'})")
        else:
            print("📚 No registered models found")
    except Exception as e:
        print("⚠️  Could not access model registry:", str(e))

    return registry_info


def setup_experiment_tags():
    """
    Set up common tags for all experiments.
    """
    print("\n=== EXPERIMENT TAGS SETUP ===")

    # Common tags for the Flightmasters project
    common_tags = {
        "project": "Flightmasters",
        "domain": "Aviation",
        "task": "Flight_Delay_Classification",
        "team": "Data_Science_Capstone",
        "created_by": "Kshitij_Mishra",
        "dataset": "Flight_Delay_Data",
        "target": "15min_delay_classification",
        "framework": "scikit-learn",
        "platform": "Databricks",
    }

    print("🏷️  Common experiment tags:")
    for key, value in common_tags.items():
        print(f"   {key}: {value}")

    return common_tags


def create_experiment_template():
    """
    Create a template for consistent experiment runs.
    """
    print("\n=== EXPERIMENT TEMPLATE ===")

    template = {
        "experiment_name": "Flightmasters_Flight_Delay_Prediction",
        "run_name_template": "{model_type}_{timestamp}",
        "required_metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
        "required_params": ["model_type", "random_state", "class_weight"],
        "artifacts_to_log": ["model", "confusion_matrix.json", "feature_importance.json"],
        "tags": {"project": "Flightmasters", "domain": "Aviation", "task": "Flight_Delay_Classification"},
    }

    # Save template
    with open("mlflow_experiment_template.json", "w") as f:
        json.dump(template, f, indent=2)

    print("📄 Experiment template saved to: mlflow_experiment_template.json")

    return template


def validate_mlflow_setup():
    """
    Validate that MLflow is properly configured.
    """
    print("\n=== MLFLOW VALIDATION ===")

    validation_results = {
        "mlflow_version": mlflow.__version__,
        "tracking_uri": mlflow.get_tracking_uri(),
        "registry_uri": mlflow.get_registry_uri(),
        "experiment_set": False,
        "client_connected": False,
    }

    try:
        # Test MLflow client
        client = MlflowClient()
        experiments = client.search_experiments()
        validation_results["client_connected"] = True
        validation_results["experiment_count"] = len(experiments)

        print("✅ MLflow client connection successful")
        print(f"📊 Found {len(experiments)} experiments")

    except Exception as e:
        print(f"❌ MLflow client connection failed: {e}")
        validation_results["error"] = str(e)

    # Check current experiment
    try:
        current_experiment = mlflow.get_experiment(mlflow.active_run().info.experiment_id if mlflow.active_run() else None)
        if current_experiment:
            validation_results["experiment_set"] = True
            print(f"✅ Current experiment: {current_experiment.name}")
    except Exception:
        print("⚠️  No active experiment set")

    print("\n📋 MLflow Configuration:")
    print("   Version:", validation_results["mlflow_version"])
    print("   Tracking URI:", validation_results["tracking_uri"])
    print("   Registry URI:", validation_results["registry_uri"])

    return validation_results


def create_databricks_notebook_cells():
    """
    Generate Databricks notebook cells for MLflow integration.
    """
    print("\n=== DATABRICKS NOTEBOOK CELLS ===")

    cells = {
        "cell_1_setup": """
# Cell 1: MLflow Setup for Databricks
%pip install mlflow>=2.8.0

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set up MLflow experiment
experiment_name = "Flightmasters_Flight_Delay_Prediction"
mlflow.set_experiment(experiment_name)
print(f"✅ MLflow experiment set: {experiment_name}")
""",
        "cell_2_data": """
# Cell 2: Load and Prepare Data
# Load your flight delay data here
# Replace with your actual data loading code

# Example data loading (replace with your actual data)
df = spark.sql("SELECT * FROM your_flight_delay_table").toPandas()

# Feature engineering and preprocessing
# ... your preprocessing code here ...

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
""",
        "cell_3_experiment": """
# Cell 3: MLflow Experiment Run
with mlflow.start_run(run_name="Random_Forest_Baseline"):
    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": 42
    })

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Experiment logged - F1: {f1:.4f}, AUC: {auc:.4f}")
""",
        "cell_4_comparison": """
# Cell 4: Model Comparison
# Compare different models
models_to_test = [
    ("Decision_Tree", DecisionTreeClassifier(random_state=42)),
    ("Random_Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient_Boosting", GradientBoostingClassifier(random_state=42))
]

results = []
for model_name, model in models_to_test:
    with mlflow.start_run(run_name=f"{model_name}_Comparison"):
        # Train and evaluate model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba)
        }

        # Log to MLflow
        mlflow.log_params({"model_type": model_name})
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        mlflow.sklearn.log_model(model, "model")

        results.append((model_name, metrics))
        print(f"{model_name} - F1: {metrics['f1_score']:.4f}")

# Display results
import pandas as pd
results_df = pd.DataFrame([(name, metrics['f1_score'], metrics['auc_roc'])
                          for name, metrics in results],
                         columns=['Model', 'F1_Score', 'AUC_ROC'])
print(results_df)
""",
    }

    # Save notebook cells
    with open("databricks_mlflow_notebook.py", "w") as f:
        f.write("# Databricks MLflow Integration Notebook\n")
        f.write("# Copy these cells into your Databricks notebook\n\n")

        for cell_name, cell_content in cells.items():
            f.write(f"# {cell_name.replace('_', ' ').title()}\n")
            f.write(cell_content)
            f.write("\n" + "=" * 80 + "\n\n")

    print("📓 Databricks notebook cells saved to: databricks_mlflow_notebook.py")

    return cells


def main():
    """
    Main function to set up MLflow for Databricks.
    """
    print("=== FLIGHTMASTERS DATABRICKS MLFLOW SETUP ===")
    print(f"Setup initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Setup MLflow
    setup_info = setup_databricks_mlflow()

    # 2. Create model registry
    registry_info = create_model_registry()

    # 3. Setup experiment tags
    common_tags = setup_experiment_tags()

    # 4. Create experiment template
    template = create_experiment_template()

    # 5. Validate setup
    validation = validate_mlflow_setup()

    # 6. Create Databricks notebook cells
    create_databricks_notebook_cells()

    # Save setup summary
    setup_summary = {
        "setup_info": setup_info,
        "registry_info": registry_info,
        "common_tags": common_tags,
        "template": template,
        "validation": validation,
        "timestamp": datetime.now().isoformat(),
    }

    with open("databricks_mlflow_setup_summary.json", "w") as f:
        json.dump(setup_summary, f, indent=2, default=str)

    print("\n✅ Databricks MLflow setup completed successfully!")
    print("\n📋 Summary of created files:")
    print("   - mlflow_experiment_template.json")
    print("   - databricks_mlflow_notebook.py")
    print("   - databricks_mlflow_setup_summary.json")

    print("\n🚀 Next Steps:")
    print("1. Upload your code to Databricks workspace")
    print("2. Run the MLflow experiments using mlflow_experiments.py")
    print("3. View results in Databricks MLflow UI")
    print("4. Use the notebook cells for interactive development")

    return setup_summary


if __name__ == "__main__":
    setup_summary = main()
