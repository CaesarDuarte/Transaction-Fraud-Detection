# Transaction Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud%20Detection-orange?logo=kaggle)

> A Machine Learning project focused on building a simple and reliable pipeline for fraud detection using the IEEE-CIS dataset

> (PT) Projeto de Machine Learning focado na construção de um pipeline simples e confiável para detecção de fraudes.
---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Phases & Progress](#phases--progress)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Problem Statement

Financial fraud costs the global economy billions of dollars annually. Traditional rule-based systems often struggle with high false-positive rates and limited ability to detect new fraud patterns.

This project focuses on building a **supervised binary classification pipeline** to identify potentially fraudulent transactions using the IEEE-CIS dataset.

The main challenges explored include:

- **Severe class imbalance** (~3.5% fraud rate)
- **High dimensionality** (400+ features across transaction and identity data)
- **Handling missing data**, which is significant in this dataset
- **Basic temporal considerations** during feature engineering

---

## Solution Overview

```
Raw data -> Data validation -> EDA -> Feature engineering -> Model training

```

The goal is to understand the dataset, create useful features, and train a baseline model for fraud detection.

---

## Pipeline

- Load and merge transaction and identity data
- Validate dataset quality
- Perform exploratory data analysis (EDA)
- Apply basic feature engineering
- Train and evaluate a baseline model

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
│   │   └── validation.py     # Basic data validation checks
│   ├── features/
│   │   └── pipeline.py       # sklearn Pipeline for feature engineering
│   └── models/
│       └── train.py          # Model training and evaluation
│
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [conda](https://docs.conda.io/) or `venv`
- Kaggle account (for dataset download) or manual download

### 1. Clone and set up environment

```bash
git clone https://github.com/CaesarDuarte/Transaction-Fraud-Detection.git
cd Transaction-Fraud-Detection

conda create -n fraud-detection python=3.11
conda activate fraud-detection

pip install -e ".[dev]"
```
 ⚠️ Important dependency

This project uses Parquet for efficient storage. Ensure that pyarrow is installed, as it is required for Parquet support.
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
python src/ingestion/validation.py
python src/features/pipeline.py
```

### 4. Train the model

```bash
python src/models/train.py
```

---

## Phases & Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Data Ingestion | 🟩 Finished |
| 2 | Data Validation| 🟩 Finished |
| 3 | EDA  | 🟨 In Progress |
| 4 | Feature Engineering | 🟥 Not started |
| 5 | Modeling | 🟥 Not started |


---

## Key Results

> *To be updated as the project progresses.*

| Metric | Baseline Model |
|---|---|
| PR-AUC | — |
| F1-Score (fraud class) | — |
| Recall | — |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data validation | Great Expectations |
| Data processing | numpy, pandas, polars |
| Data visualization | matplotlib, seaborn |
| Feature engineering | scikit-learn |
| Modeling | XGBoost, LightGBM |

---

## Notes

This project initially aimed to include a full production pipeline (API, monitoring, CI/CD), but the scope was reduced to focus on building a solid foundation in data understanding and modeling.

## License

MIT © César Duarte
