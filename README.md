# ✈️ Flight Delay Prediction & Analytics Platform

End-to-end flight delay prediction system built on Databricks, using a medallion architecture to ingest, process, and model FAA flight data for real-time delay risk analytics.

## Overview

- Ingests live flight data from the AviationStack API
- Processes data through Bronze/Silver/Gold medallion layers using Delta Lake
- Trains classification and regression models using Spark MLlib and Scikit-learn
- Tracks all experiments and model versions with MLflow
- Visualizes delay risk and contributing factors in interactive Databricks dashboards

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-003366?style=flat&logo=delta&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)

## Architecture

Data flows through a three-layer medallion architecture:

- **Bronze:** raw API ingestion from AviationStack
- **Silver:** cleaned, validated, and joined flight records
- **Gold:** feature-engineered tables ready for modeling and dashboards

## Models

- **Classification:** predicts whether a flight will be delayed (binary)
- **Regression:** predicts delay duration in minutes
- **Framework:** Spark MLlib and Scikit-learn
- **Experiment tracking:** MLflow (model registry, run comparison, metrics logging)

## Notebooks

Exported notebook HTMLs are in the `docs/` folder and can be opened in any browser to view code and outputs without Databricks access.

## Note on Running This Project

This project was built and runs on Databricks Runtime 14.x LTS. The medallion pipeline, dashboards, and MLflow tracking server are Databricks-native. To explore the code and outputs without a Databricks environment, see the exported notebooks in `docs/`.

## Contact

- GitHub: [OmaryElkady](https://github.com/OmaryElkady)
- LinkedIn: [Omar Elkady](https://www.linkedin.com/in/omar-elkady-847b051ba/)
