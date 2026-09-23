# AVIP DS Task 2 - Titanic Dataset Exploratory Data Analysis

This project completes **AVIP 2026 Data Science Basic Task 2 — Titanic Dataset Exploratory Data Analysis (EDA)**.

## Task Requirements

- Clean the dataset and document cleaning steps, including missing-value handling and feature engineering.
- Required visualizations:
  1. Survival rate by passenger class and gender
  2. Age distribution histogram
  3. Correlation heatmap for numerical features
- Include concise EDA observations in Jupyter Notebook Markdown cells.
- Provide at least three actionable observations.
- Deliverables: public GitHub repository, README with dataset link, cleaned dataset or fetch instructions, Jupyter notebook, exported chart images, and a 3–5 line conclusion.

## Dataset

Dataset: Titanic dataset from the public `seaborn-data` repository.

Source:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

The project includes a fetch-and-clean script so the dataset can be reproduced without committing the raw dataset to GitHub.

## Project Structure

```text
AVIP_DS_Task2_Titanic_EDA/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── task_2_titanic_eda.ipynb
├── src/
│   ├── prepare_data.py
│   └── run_eda.py
└── outputs/
    └── README.txt
```

## Run Locally

From the project root:

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py
python src/run_eda.py
```

Then open:

```text
notebooks/task_2_titanic_eda.ipynb
```

For an executed notebook:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/task_2_titanic_eda.ipynb --output task_2_titanic_eda_executed --output-dir notebooks
```

## Generated Outputs

After running the scripts/notebook:

- `data/titanic_cleaned.csv`
- `outputs/survival_rate_by_class_gender.png`
- `outputs/age_distribution.png`
- `outputs/correlation_heatmap.png`
- `outputs/insights_summary.txt`

## Cleaning & Feature Engineering

The workflow:
- removes duplicate rows;
- fills missing `age` values using the median age;
- fills missing `embarked` values using the mode;
- fills missing `fare` values using the median fare;
- converts `sex` and `embarked` into numeric indicator features;
- creates `family_size` from `sibsp + parch + 1`;
- creates `is_alone` to indicate whether the passenger travelled alone.

The raw dataset is not committed by default; the fetch instructions in `data/README.md` are the reproducible source.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook.

## Internship

**Program:** B.Y.T.E by Arithmatrix — AVIP 2026  
**Domain:** Data Science  
**Task:** Basic Task 2 — Titanic Dataset EDA
