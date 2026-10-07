# Credit Risk Model Validation

Pet project for a Data Science / Model Validation internship.

The project demonstrates an end-to-end validation workflow for a synthetic **PD (Probability of Default)** model on tabular data:

- data quality checks and EDA;
- baseline PD model: Logistic Regression;
- alternative model: HistGradientBoostingClassifier;
- cross-validation;
- holdout validation;
- ROC-AUC, Gini, KS, Precision, Recall, F1, LogLoss, Brier Score;
- calibration analysis;
- PSI for score stability;
- model comparison;
- automated validation report;
- SQL examples for validation checks;
- a small PySpark example for data-quality checks.

> The dataset is synthetic and the project is educational. It does not represent a production bank model or regulatory validation.

## Project structure

```text
credit-risk-model-validation/
├── data/
│   └── credit_risk_synthetic.csv
├── docs/
│   └── interview_notes.md
├── notebooks/
│   └── eda.ipynb
├── pyspark/
│   └── validation_pyspark.py
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── validation_report.md
├── sql/
│   └── validation_queries.sql
├── src/
│   ├── generate_data.py
│   ├── metrics.py
│   ├── psi.py
│   └── train_validate.py
├── tests/
│   └── test_metrics.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick start

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

WSL / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run validation

```bash
python src/train_validate.py
```

The script will create / refresh:

- `data/credit_risk_synthetic.csv`
- `reports/metrics.json`
- `reports/validation_report.md`
- plots in `reports/figures/`

### 4. Run tests

```bash
pytest -q
```

## What to show in an interview

Start with `reports/validation_report.md`, then explain:

1. why the Logistic Regression model is the baseline;
2. why a second model is used as an alternative model;
3. how ROC-AUC and Gini measure discriminatory power;
4. how KS measures separation between default / non-default observations;
5. why Brier Score and calibration matter for PD models;
6. what PSI says about score stability;
7. why validation must be done on data not used to fit the model.

## Resume-ready description

**Credit Risk Model Validation — pet project**

- Built a validation pipeline for a synthetic PD model on tabular data using Python, Pandas, NumPy and scikit-learn.
- Compared Logistic Regression with an alternative gradient boosting model using ROC-AUC, Gini, KS, F1, LogLoss and Brier Score.
- Evaluated calibration and score stability with calibration curves and PSI.
- Automated data-quality checks, model comparison and generation of a validation report.
- Added SQL validation queries and a PySpark example for large-scale data checks.

## Tech stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, SQL, pytest, PySpark (basic example).
