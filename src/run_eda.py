from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "titanic_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run `python src/prepare_data.py` first."
        )
    return pd.read_csv(DATA_FILE)

def create_charts(df):
    sns.set_theme(style="whitegrid")

    # 1. Survival rate by class and gender
    survival = (
        df.groupby(["pclass", "sex"], as_index=False)["survived"]
        .mean()
    )
    survival["survival_rate"] = survival["survived"] * 100

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=survival,
        x="pclass",
        y="survival_rate",
        hue="sex"
    )
    ax.set_title("Titanic Survival Rate by Passenger Class and Gender")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_rate_by_class_gender.png", dpi=180)
    plt.close()

    # 2. Age distribution
    plt.figure(figsize=(9, 6))
    ax = sns.histplot(df["age"], bins=30, kde=True)
    ax.set_title("Titanic Passenger Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Passenger Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=180)
    plt.close()

    # 3. Correlation heatmap
    numeric_cols = [
        "survived", "pclass", "age", "sibsp", "parch",
        "fare", "family_size", "is_alone"
    ]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    return survival, corr

def write_insights(df, survival):
    overall = df["survived"].mean() * 100
    female = df.loc[df["sex"] == "female", "survived"].mean() * 100
    male = df.loc[df["sex"] == "male", "survived"].mean() * 100
    first = df.loc[df["pclass"] == 1, "survived"].mean() * 100
    third = df.loc[df["pclass"] == 3, "survived"].mean() * 100

    text = f"""Titanic EDA Insights

1. Overall survival rate in the cleaned dataset: {overall:.1f}%.
2. Female passengers had a substantially higher observed survival rate ({female:.1f}%) than male passengers ({male:.1f}%).
3. First-class passengers had a higher observed survival rate ({first:.1f}%) than third-class passengers ({third:.1f}%).
4. Age is distributed across a broad range, with many passengers concentrated among young and middle-aged adults.
5. Passenger class, fare, and survival show meaningful relationships in the numerical correlation analysis; correlation should be interpreted as association, not causation.

Conclusion:
Survival patterns vary notably by gender and passenger class. The EDA also shows a wide age distribution and relationships among socioeconomic and survival-related numerical variables. These findings identify useful features for further predictive modelling while avoiding causal claims from correlation alone.
"""
    (OUTPUT_DIR / "insights_summary.txt").write_text(text, encoding="utf-8")

if __name__ == "__main__":
    print("[1/3] Loading cleaned data...")
    df = load_data()
    print(f"Loaded {len(df)} rows.")

    print("[2/3] Creating required charts...")
    survival, corr = create_charts(df)

    print("[3/3] Writing insights...")
    write_insights(df, survival)
    print("EDA outputs created successfully.")
