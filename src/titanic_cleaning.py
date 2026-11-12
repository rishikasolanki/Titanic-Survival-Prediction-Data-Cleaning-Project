"""Clean the Titanic passenger dataset and create an age distribution plot.

Run from any directory with ``python src/titanic_cleaning.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "Titanic-Dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "Titanic_Cleaned.csv"
PLOT_PATH = PROJECT_ROOT / "images" / "age_distribution.png"


def clean_titanic_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load, summarize, clean, and encode the Titanic dataset."""
    df = pd.read_csv(input_path)

    print("Original dataset shape:", df.shape)
    print("\nDataset information:")
    df.info()
    print("\nNumeric summary statistics:")
    print(df.describe())
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # Median is robust to skew and outliers; mode is appropriate for a category.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

    # Cabin is missing for most passengers, so remove it for this beginner project.
    df = df.drop(columns=["Cabin"])

    # Label encode the binary Sex feature and one-hot encode Embarked.
    sex_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])

    embarked_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="int64")
    embarked_encoded = embarked_encoder.fit_transform(df[["Embarked"]])
    embarked_columns = embarked_encoder.get_feature_names_out(["Embarked"])
    embarked_df = pd.DataFrame(embarked_encoded, columns=embarked_columns, index=df.index)
    df = pd.concat([df.drop(columns=["Embarked"]), embarked_df], axis=1)

    print("\nMissing values after cleaning:")
    print(df.isna().sum())
    print("\nCleaned dataset shape:", df.shape)
    return df


def main() -> None:
    df = clean_titanic_data()

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(df["Age"], bins=30, color="skyblue", edgecolor="white")
    plt.title("Age Distribution of Titanic Passengers")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of passengers")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned CSV saved to: {OUTPUT_PATH}")
    print(f"Age distribution saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
