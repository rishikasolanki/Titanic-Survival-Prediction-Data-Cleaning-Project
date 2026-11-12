# Titanic Survival Prediction - Data Preprocessing Project

A Python machine learning data preprocessing project using the Titanic dataset. This project demonstrates how to prepare raw data for model training through exploratory analysis, missing-value treatment, categorical encoding, visualization, and cleaned data export.

## Features

- Titanic dataset loading (local file fallback + automatic download)
- Exploratory Data Analysis (EDA):
  - Data preview
  - Data types inspection
  - Missing values summary
  - Descriptive statistics
- Missing value handling:
  - `Age` filled with median
  - `Embarked` filled with mode
  - `Cabin` dropped due to high missingness
- Categorical encoding:
  - Label Encoding for `Sex`
  - One-Hot Encoding for `Embarked`
- Age distribution visualization using **Seaborn + Matplotlib**
- Export of cleaned data to CSV

## Project Structure

```text
Titanic-Survival-Prediction-Data-Cleaning-Project/
├── data/
│   ├── raw/
│   │   └── titanic.csv
│   └── processed/
│       └── titanic_cleaned.csv
├── outputs/
│   └── age_distribution.png
├── src/
│   └── preprocess_titanic.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Preprocessing Pipeline

```bash
python src/preprocess_titanic.py
```

## Generated Outputs

After execution:

- Cleaned dataset: `data/processed/titanic_cleaned.csv`
- Visualization: `outputs/age_distribution.png`

## Notes

- If `data/raw/titanic.csv` is missing, the script downloads the Titanic dataset and stores it locally.
- The script is designed as a reusable preprocessing baseline for machine learning workflows.
