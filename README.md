🏦 End-to-End Fraud Detection Pipeline (ML + Data Engineering)
📌 Project Overview

This project simulates a real-world bank fraud detection system using transaction data, machine learning, and a production-style data pipeline.

The goal is not just modeling, but demonstrating how fraud detection works end-to-end:

Data ingestion

Preprocessing

Model training

Threshold tuning

Batch inference

Database integration

Decision logic (auto-approve / manual review / auto-reject)

This mirrors how fraud models are deployed at large financial institutions (e.g., JPMorgan Chase, Capital One, Stripe).

🧠 Problem Statement

Fraud detection is a highly imbalanced classification problem where:

Fraud is rare (<5%)

False positives are costly

Missed fraud is dangerous

Instead of optimizing for accuracy, this project focuses on:

Precision / Recall tradeoffs

Average Precision (PR-AUC)

Decision thresholds aligned with business rules

📂 Dataset

Based on Kaggle IEEE-CIS Fraud Detection dataset

~400+ anonymized features

Includes:

Transaction metadata

Card/device information

Time-based variables

Target label: isFraud

🏗️ Project Architecture
Raw CSV Data
     ↓
Data Cleaning & Feature Processing
     ↓
Preprocessor (Imputers + Encoder) → saved as .pkl
     ↓
LightGBM Fraud Model → saved as .pkl
     ↓
SQLite Mock Bank Database
     ↓
Batch Scoring Pipeline
     ↓
Fraud Decision Engine

⚙️ Tech Stack

Languages & Libraries

Python

Pandas, NumPy

Scikit-learn

LightGBM

Joblib

SQLite3

Concepts Used

Class imbalance handling

Ordinal encoding

Missing value imputation

Model persistence

Batch inference

SQL ↔ ML integration

Business-driven thresholds

🔬 Modeling Approach
Model

LightGBM (Gradient Boosted Trees)

Chosen because:

Handles large feature spaces

Performs well on tabular fraud data

Used widely in industry

Training Strategy

Time-aware train / test split

Class imbalance handled via:

scale_pos_weight = num_nonfraud / num_fraud


Evaluation metric:

Average Precision Score (PR-AUC)

🎯 Threshold-Based Decision System

Instead of a single prediction, the model outputs a fraud probability, which is mapped to actions:

Fraud Probability	Decision
≥ 0.90	AUTO REJECT
0.70 – 0.89	MANUAL REVIEW
< 0.70	AUTO APPROVE

This reflects how real banks operate — ML assists humans, it doesn’t blindly replace them.

🗄️ Mock Bank Database (SQLite)

A SQLite database simulates a production banking system:

Transactions stored in transactions table

Batch processing using LIMIT / OFFSET

Each transaction is:

Pulled from SQL

Preprocessed using saved pipeline

Scored by ML model

Assigned a fraud decision

This demonstrates data engineering + ML integration, not just notebooks.

🔄 Batch Scoring Pipeline
SELECT * FROM transactions
LIMIT batch_size OFFSET offset


For each batch:

Load from database

Apply saved preprocessor

Predict fraud probability

Apply decision rules

Output results (or store back to DB)

This simulates real-world fraud monitoring systems.

📁 Repository Structure
.
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_threshold_analysis.ipynb
│
├── pipeline/
│   ├── batch_scoring.py
│
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│
├── mock_database_bank.db
├── README.md

🚀 How to Run
1️⃣ Train the Model

Run the training notebooks to generate:

fraud_model.pkl

preprocessor.pkl

2️⃣ Create Database
python pipeline/create_database.py

3️⃣ Run Batch Scoring
python pipeline/batch_scoring.py

📈 Sample Output
TransactionID | Fraud Probability | Decision
--------------------------------------------
3032075       | 0.9426            | AUTO REJECT
3032078       | 0.7585            | MANUAL REVIEW
3032070       | 0.0098            | AUTO APPROVE

🧠 Key Learnings

Fraud detection is not about accuracy

Thresholds matter more than models

ML must integrate with databases and pipelines

Interpretability is often traded for performance

Production ML = engineering + modeling

🔮 Future Improvements

Real-time streaming (Kafka)

REST API (FastAPI)

Model monitoring & drift detection

SHAP explainability dashboards

Cloud deployment (AWS/GCP)

👤 Author

Vishruth Gonur
Information Science + Data Science
University of Illinois Urbana-Champaign
