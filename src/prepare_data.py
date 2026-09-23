from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SOURCE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
OUTPUT_FILE = DATA_DIR / "titanic_cleaned.csv"

def fetch_and_clean():
    print("[1/4] Downloading Titanic dataset...")
    df = pd.read_csv(SOURCE_URL)
    print(f"Raw shape: {df.shape}")

    print("[2/4] Cleaning missing values...")
    df = df.drop_duplicates().copy()

    # Numerical missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())

    # Categorical missing values
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    print("[3/4] Feature engineering...")
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["sex_male"] = (df["sex"] == "male").astype(int)
    df["embarked_c"] = (df["embarked"] == "C").astype(int)
    df["embarked_q"] = (df["embarked"] == "Q").astype(int)
    df["embarked_s"] = (df["embarked"] == "S").astype(int)

    # Keep a clean, analysis-ready dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("[4/4] Saving cleaned dataset...")
    print(f"Cleaned shape: {df.shape}")
    print(f"Saved: {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    fetch_and_clean()
