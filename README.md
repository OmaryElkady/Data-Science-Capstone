# ✈️ Flightmasters: A Cloud-Based Flight Delay Prediction Platform

[![Python Code Quality](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/python-code-quality.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/python-code-quality.yml)
[![Notebook Checks](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/notebook-checks.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/notebook-checks.yml)

> **Senior Year Data Science Capstone Project**

## 📘 Overview
**Flightmasters** is a data science capstone project designed to predict flight delays and enhance the passenger experience through real-time data analytics.

By integrating open-source APIs — **AviationStack** and **Open-Meteo** — into a unified **Databricks** environment, Flightmasters enables end-to-end **data engineering**, **machine learning**, and **visualization** to forecast disruptions *before* they happen.

The ultimate goal is to move beyond reactive delay notifications and provide **proactive flight disruption forecasting**, empowering passengers and airlines to make smarter, data-driven decisions.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

### Installation

```bash
# Clone the repository
git clone https://github.com/OmaryElkady/Data-Science-Capstone.git
cd Data-Science-Capstone

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Jupyter Notebooks

```bash
jupyter notebook
```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

---

## 🎯 Project Objectives

1. **Predict Flight Delays**
   - Forecast the probability and duration of flight delays using historical and live data.

2. **Integrate Real-Time Data Sources**
   - **AviationStack API** – Flight schedules, routes, and historical delays
   - **Open-Meteo API** – Local weather forecasts and atmospheric conditions

3. **Build a Scalable Cloud Pipeline**
   - Develop an **ETL pipeline** in **Databricks** using **Apache Spark** and **Delta Lake** for continuous ingestion, cleaning, and transformation of live flight and weather data.

4. **Develop and Evaluate Predictive Models**
   - Train machine learning models including **Logistic Regression**, **Random Forest**, and **Gradient Boosted Trees** using **Spark MLlib** and **scikit-learn**.
   - Evaluate models using metrics such as **Precision**, **Recall**, and **RMSE**.

5. **Deliver Actionable Insights via Dashboard**
   - Create an interactive **Databricks SQL dashboard** that displays:
     - Real-time delay risk scores (Low / Medium / High)
     - Key contributing factors (weather, seasonal patterns, traffic trends)
     - Suggested alternative flights or airports with lower disruption risk

---

## 🧠 Technical Architecture

### 🗂️ Data Sources
| Source | Description |
|:--------|:-------------|
| 🛫 **AviationStack** | Scheduled and historical flight data |
| 🌤 **Open-Meteo** | Real-time and forecasted weather conditions |

### 🧰 Tech Stack
| Layer | Tools & Technologies |
|:------|:----------------------|
| **Cloud Platform** | Databricks Community Edition |
| **Data Processing** | Apache Spark, Delta Lake |
| **Machine Learning** | Spark MLlib, Scikit-learn, MLflow |
| **Storage** | Delta Tables, Databricks File System (DBFS) |
| **Visualization** | Databricks SQL Dashboards, Power BI / Tableau (optional) |

### ⚙️ Architecture Workflow

```plaintext
        +------------------+
        |   AviationStack  |
        +--------+---------+
                 |
        +--------v---------+
        |    Open-Meteo    |
        +--------+---------+
                 |
        +--------v---------+
        |  Databricks ETL  |  --> Cleans & merges data (Spark + Delta Lake)
        +--------+---------+
                 |
        +--------v---------+
        |   ML Training    |  --> Delay classification & regression (MLlib / Sklearn)
        +--------+---------+
                 |
        +--------v---------+
        | Visualization    |  --> Databricks SQL Dashboard
        +------------------+
```

---

## 📊 Expected Deliverables

- ✅ **Integrated Data Pipeline** — Continuous ETL system merging flight and weather data
- ✅ **Predictive ML Models** — Classification and regression models with explainable insights
- ✅ **Interactive Dashboard** — Real-time visualization of flight delay risk and recommendations

---

## 👥 Team Members

- **Omar Elkady**
- **Aidan Maltby**
- **Kshitij Mishra**

---

## 📈 Impact

By combining aviation and weather data with machine learning in the cloud, Flightmasters demonstrates how data-driven insights can revolutionize passenger experiences and airline operations.

The platform offers a blueprint for:

- Reducing uncertainty and frustration for travelers
- Helping airlines anticipate and manage delays efficiently
- Promoting transparency and smarter decision-making in modern air travel

---

## 📁 Project Structure

```
Data-Science-Capstone/
├── EDA_*.ipynb              # Exploratory Data Analysis notebooks
├── *_table.ipynb            # Data processing notebooks (Bronze, Silver, Gold)
├── "ML FLow with pyspark.ipynb" # Machine Learning pipeline with MLflow
├── mlflow_experiments.py    # MLflow experiment tracking
├── process_flight_data.py   # Data processing scripts
├── requirements.txt         # Python dependencies
├── Makefile                 # Development automation commands
├── docs/                    # Documentation
│   ├── SETUP.md            # Development environment setup
│   └── CODE_QUALITY_GUIDE.md
└── .github/workflows/       # CI/CD pipelines
```

---

## 📚 Documentation

- [**Contributing Guide**](CONTRIBUTING.md) — How to contribute to the project
- [**Setup Guide**](docs/SETUP.md) — Development environment setup
- [**MLflow Guide**](MLFLOW_GUIDE.md) — Machine learning experiment tracking
- [**Data Analysis Report**](DATA_ANALYSIS_REPORT.md) — EDA findings and insights
- [**Implementation Summary**](IMPLEMENTATION_SUMMARY.md) — Technical implementation details

---

## 🛠️ Development

### Available Make Commands

```bash
make install       # Install project dependencies
make install-dev   # Install development dependencies
make format        # Format code with black and isort
make lint          # Run linting checks
make check         # Run all code quality checks
make clean         # Clean up cache files
make notebook      # Start Jupyter notebook server
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## 📄 License

This project is part of a senior year data science capstone course.
