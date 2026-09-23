# Dataset Instructions

The AVIP Task 2 requires a Titanic dataset link and reproducible fetch instructions.

## Source

Public Titanic CSV:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

## Download automatically

From the project root:

```bash
python src/prepare_data.py
```

The script downloads the raw CSV, performs the documented cleaning and feature engineering steps, and saves:

```text
data/titanic_cleaned.csv
```

The raw dataset is intentionally not committed to the repository. This keeps the repository lightweight while preserving reproducibility.
