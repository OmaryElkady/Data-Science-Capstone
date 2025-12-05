# Data-Science-Capstone-Project:

This is my senior year data science capstone project!

# ✈️ Flightmasters: A Cloud-Based Flight Delay Prediction Platform

## 📘 Overview
**Flightmasters** is a data science capstone project designed to predict flight delays and enhance the passenger experience through real-time data analytics.

By integrating open-source APIs — **AviationStack** — into a unified **Databricks** environment, Flightmasters enables end-to-end **data engineering**, **machine learning** to forecast disruptions *before* they happen.

The ultimate goal is to move beyond reactive delay notifications and provide **proactive flight disruption forecasting**, empowering passengers and airlines to make smarter, data-driven decisions.

---

## 🚀 Project Objectives

1. **Predict Flight Delays**
   - Forecast the probability and duration of flight delays using historical and live data.

2. **Integrate Real-Time Data Sources**
   - **AviationStack API** – Flight schedules, routes, and historical delays

3. **Build a Scalable Cloud Pipeline**
   - Develop an **ETL pipeline** in **Databricks** using **Apache Spark** and **Delta Lake** for continuous ingestion, cleaning, and transformation of live flight and weather data.

4. **Develop and Evaluate Predictive Models**
   - Train machine learning models including **Random Forest**, and **Gradient Boosted Trees** using **Spark MLlib**
   - Evaluate models using metrics such as **Accuracy** **Precision**, **Recall**, **F1** and **ROC AUC**

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

### 🧰 Tech Stack
| Layer | Tools & Technologies |
|:------|:----------------------|
| **Cloud Platform** | Databricks Community Edition |
| **Data Processing** | Apache Spark, Delta Lake |
| **Machine Learning** | Spark MLlib, MLflow |
| **Storage** | Delta Tables, Databricks File System (DBFS) |
| **Visualization** | Databricks SQL Dashboards, Power BI / Tableau (optional) |

### ⚙️ Architecture Workflow
```plaintext
        +------------------+
        |   AviationStack  |
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

📊 Expected Deliverables

✅ Integrated Data Pipeline — Continuous ETL system merging flight and weather data

✅ Predictive ML Models — Classification and regression models with explainable insights

✅ Interactive Dashboard — Real-time visualization of flight delay risk and recommendations


👥 Team Members

Omar Elkady

Aidan Maltby

Kshitij Minshra


📈 Impact

By combining aviation data with machine learning in the cloud, Flightmasters demonstrates how data-driven insights can revolutionize passenger experiences and airline operations.

The platform offers a blueprint for:

Reducing uncertainty and frustration for travelers

Helping airlines anticipate and manage delays efficiently

Promoting transparency and smarter decision-making in modern air travel
        +------------------+


