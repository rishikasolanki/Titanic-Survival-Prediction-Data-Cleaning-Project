# Week 1: ML Fundamentals and Data Preprocessing

This project practices dataset exploration and preprocessing with the Kaggle Titanic passenger dataset. It covers loading a CSV with Pandas, reviewing `.info()` and `.describe()`, handling missing values, encoding categorical features, visualizing age, and exporting a cleaned CSV.

## Project layout

- `data/Titanic-Dataset.csv` — source dataset
- `data/Titanic_Cleaned.csv` — generated cleaned dataset
- `notebooks/Titanic_Data_Cleaning.ipynb` — guided assignment notebook
- `src/titanic_cleaning.py` — reusable end-to-end cleaning script
- `images/age_distribution.png` — generated age histogram

## Setup and run

From the project directory:

```bash
python -m pip install -r requirements.txt
python src/titanic_cleaning.py
```

The script writes `data/Titanic_Cleaned.csv` and `images/age_distribution.png`. Paths are resolved from the script location, so it can also be launched from another working directory.

## Cleaning choices

- Fill missing `Age` and `Fare` values with their medians.
- Fill missing `Embarked` values with its most common category.
- Drop `Cabin`, which is missing for most rows in this dataset.
- Label encode `Sex` and one-hot encode `Embarked`.
- Keep `PassengerId`, `Name`, and `Ticket` as source columns; feature selection is outside this data-cleaning exercise.

For a fair machine-learning evaluation, fit imputers and encoders on training data only, then apply them to test data. The project script demonstrates cleaning the provided dataset as a standalone preprocessing exercise.
