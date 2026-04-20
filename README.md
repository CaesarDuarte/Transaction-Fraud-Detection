# Transaction Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud%20Detection-orange?logo=kaggle)

> An end-to-end Machine Learning Engineering project for real-time transaction fraud detection (covering data engineering, feature engineering, model training, serving via REST API, and data drift monitoring)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Phases & Progress](#phases--progress)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Problem Statement

Financial fraud costs the global economy hundreds of billions of dollars annually. Traditional rule-based systems suffer from high false-positive rates, harming customer experience, and low recall on novel fraud patterns.

This project builds a **supervised binary classification system** capable of flagging fraudulent transactions in near-real-time. The core challenges addressed are:

- **Severe class imbalance** (~3.5% fraud rate in the IEEE-CIS dataset)
- **High dimensionality** (400+ raw features across transaction and identity tables)
- **Temporal leakage risk** during feature engineering and model evaluation
- **Production serving** with low-latency inference and schema validation

---

## Solution Overview

```
Raw transaction data -> Data validation -> Feature engineering pipeline
-> XGBoost/LightGBM classifier -> FastAPI inference endpoint
-> Evidently drift monitoring -> GitHub Actions CI
```

The model is optimized for **PR-AUC** (Precision-Recall Area Under Curve) rather than accuracy, which is the correct metric for imbalanced fraud detection. Threshold tuning is applied post-training to balance precision and recall for the business use case.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                           │
│  IEEE-CIS raw CSVs -> processed / curated                   |                 
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Processing Layer                         │
│  Great Expectations validation -> pandas / polars pipeline  │
│  Missing value imputation -> encoding -> temporal features  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     Modeling Layer                          │
│  XGBoost + LightGBM -> MLflow experiment tracking           │
│  SMOTE / class_weight -> threshold tuning -> SHAP explainer │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Serving Layer                          │
│  FastAPI -> Pydantic schema validation -> Docker container  │
│  POST /predict  ->  { transaction_id, fraud_probability,    │
│                      is_fraud, model_version }              │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   Monitoring Layer                          │
│  Evidently data drift reports -> pytest -> GitHub Actions   │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

**IEEE-CIS Fraud Detection** — Kaggle Competition  
[https://www.kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection)

| Property | Value |
|---|---|
| Source | Vesta Corporation via IEEE-CIS |
| Train transactions | ~590,000 rows |
| Features | 433 (transaction + identity tables) |
| Fraud rate | ~3.5% |
| Target variable | `isFraud` (binary) |
| Time span | ~6 months |

> **Note:** The dataset is not included in this repository. Download it from Kaggle and place the files in `data/raw/`. See [Quickstart](#quickstart).

---

## Project Structure

```
transaction-fraud-detection-system/
│
├── data/
│   ├── raw/                  # Original CSVs from Kaggle (git-ignored)
│   ├── processed/            # Cleaned, merged, validated data
│   └── curated/              # Feature-engineered, model-ready data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py         # Load and merge transaction + identity tables
│   │   └── validation.py     # Great Expectations data contracts
│   ├── features/
│   │   ├── pipeline.py       # sklearn Pipeline for feature engineering
│   │   ├── temporal.py       # Velocity features, time-window aggregations
│   │   └── encoding.py       # Target encoding, frequency encoding
│   ├── models/
│   │   ├── train.py          # Training script with MLflow logging
│   │   ├── evaluate.py       # PR-AUC, F1, confusion matrix, SHAP
│   │   └── threshold.py      # Threshold optimization
│   └── serving/
│       ├── app.py            # FastAPI application
│       ├── schemas.py        # Pydantic input/output models
│       └── predictor.py      # Model loading and inference
│
├── tests/
│   ├── test_features.py
│   ├── test_serving.py
│   └── test_validation.py
│
├── monitoring/
│   └── drift_report.py       # Evidently drift detection
│
├── .github/
│   └── workflows/
│       └── ci.yml            # Lint + test on push
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [conda](https://docs.conda.io/) or `venv`
- Docker (for serving)
- Kaggle account (for dataset download)

### 1. Clone and set up environment

```bash
git clone https://github.com/CaesarDuarte/Transaction-Fraud-Detection-System.git
cd Transaction-Fraud-Detection-System

conda create -n fraud-detection python=3.11
conda activate fraud-detection

pip install -e ".[dev]"
```

### 2. Download the dataset

```bash
# Install Kaggle CLI if needed
pip install kaggle

# Place your kaggle.json in ~/.kaggle/
kaggle competitions download -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

### 3. Run the data pipeline

```bash
python src/ingestion/loader.py
python src/features/pipeline.py
```

### 4. Train the model

```bash
python src/models/train.py
# MLflow UI: mlflow ui --port 5000
```

### 5. Serve the model

```bash
docker compose up --build
# API docs: http://localhost:8000/docs
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Phases & Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Data Engineering (validation, ingestion) | 🔲 Not started |
| 2 | EDA + Feature Engineering | 🔲 Not started |
| 3 | Modeling + Experiment Tracking (MLflow) | 🔲 Not started |
| 4 | Serving (FastAPI + Docker) | 🔲 Not started |
| 5 | Monitoring + CI/CD (Evidently + GitHub Actions) | 🔲 Not started |

---

## Key Results

> *To be updated as the project progresses.*

| Metric | Baseline (Logistic Regression) | Best Model |
|---|---|---|
| PR-AUC | — | — |
| F1-Score (fraud class) | — | — |
| Recall @ 80% Precision | — | — |
| Inference latency (p99) | — | — |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data validation | Great Expectations |
| Data processing | pandas, polars |
| Feature engineering | scikit-learn Pipelines |
| Modeling | XGBoost, LightGBM |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Serving | FastAPI, Pydantic, Uvicorn |
| Containerization | Docker, Docker Compose |
| Monitoring | Evidently |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## License

MIT © César Duarte
