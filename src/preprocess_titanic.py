from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = Path("data/raw/titanic.csv")
CLEANED_DATA_PATH = Path("data/processed/titanic_cleaned.csv")
AGE_PLOT_PATH = Path("outputs/age_distribution.png")
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_data() -> pd.DataFrame:
    """Load Titanic data from local file, or download it if missing."""
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TITANIC_URL)
    df.to_csv(RAW_DATA_PATH, index=False)
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Print a compact EDA summary."""
    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Summary statistics ---")
    print(df.describe(include="all"))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and encode categorical variables."""
    cleaned = df.copy()

    cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())
    cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode()[0])

    if "Cabin" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Cabin"])

    if "Sex" in cleaned.columns:
        encoder = LabelEncoder()
        cleaned["Sex"] = encoder.fit_transform(cleaned["Sex"])

    cleaned = pd.get_dummies(cleaned, columns=["Embarked"], prefix="Embarked", drop_first=False)

    return cleaned


def plot_age_distribution(df: pd.DataFrame) -> None:
    """Save age distribution chart with seaborn + matplotlib."""
    AGE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Age"], bins=30, kde=True, color="steelblue")
    plt.title("Titanic Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(AGE_PLOT_PATH)
    plt.close()


def export_cleaned_data(df: pd.DataFrame) -> None:
    """Export cleaned dataset to CSV."""
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)


def main() -> None:
    df = load_data()
    run_eda(df)
    cleaned = preprocess_data(df)
    plot_age_distribution(cleaned)
    export_cleaned_data(cleaned)

    print(f"\nCleaned dataset saved to: {CLEANED_DATA_PATH}")
    print(f"Age distribution plot saved to: {AGE_PLOT_PATH}")


if __name__ == "__main__":
    main()
