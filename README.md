# Credit Risk Model Validation

## Computer Vision — documented, verified CV/MLOps research

[Document Vision Lab](extensions/document-vision-lab/README.md): CNN and ViT training on 660 synthetic images, TorchScript and FastAPI, **DVC 3-stage reproduction**, **two MLflow experiments** and a real-document field extraction benchmark using pretrained **Donut on 20 publicly licensed CORD receipts** (18/19 evaluable totals correct). [DVC/MLflow CI](https://github.com/angelina2006mmmm-lab/pd-model-validation/actions/runs/38380044243) · [Donut vs Tesseract CI](https://github.com/angelina2006mmmm-lab/pd-model-validation/actions/runs/38380950953). Models are educational and Donut inference is not self-fine-tuning.


## New: Retail Credit DS Lifecycle (ML + PySpark)

**[Retail Credit DS Lifecycle — data quality, two-source ETL, Spark, champion/challenger, OOT, model card](extensions/retail-credit-ds-lifecycle/README.md)**

Portfolio demonstration: 4,200 **synthetic** loan applications joined with monthly macroeconomic scenario data, leakage-safe time split, reproducible Logistic Regression vs HistGradientBoosting comparison, JSON metrics, joblib inference artifact and model card. Pandas and PySpark source transformation parity is checked separately in CI. This does not claim a production Spark cluster or real bank decisions.

---


## New flagship: Independent Model Risk Validation & Monitoring

**[Model Risk Validation & Monitoring — challenger ML, OOT, drift, calibration, implementation bug hunting, agent audit](extensions/model-risk-validation/README.md)**

An educational independent risk-governance framework using synthetic 2023–2025 credit data and hand-authored AI-agent traces. Train/validation/OOT leakage controls, challenger ML, monthly monitoring, business-cost proxy and deterministic incident checks; Python tests + GitHub Actions + reproducible report. It is **not** Basel-compliant or production bank monitoring.

---

## New: Credit Risk Model Serving API — Python / FastAPI

**[Credit Risk Model Serving API — source code, model artifact, tests, design](extensions/credit-risk-model-serving/README.md)**

Versioned REST implementation of a PD risk-strategy prototype: trained sklearn logistic model, JSON artifact, Pydantic validation, non-temporal synthetic test ROC-AUC, model version, structured logs with no raw applicant data, and 20 verified unit/integration tests. Nginx/Jenkins/OpenShift examples are templates and have NOT been deployed.

---

## New: Credit Risk Scoring - CatBoost vs Logistic Regression

A reproducible **credit underwriting PD** pet-project with 4,200 synthetic applications, chronological train/validation/out-of-time split, leakage checks, validation-based model selection, CatBoost, Logistic Regression, ROC-AUC/Gini/KS/Brier/PSI, SQL checks and 10 automated tests.

**[Open scoring project and code](extensions/credit-risk-scoring-catboost)** · [OOT report](extensions/credit-risk-scoring-catboost/reports/report.md)

Synthetic 2025 holdout: Logistic Regression (selected by validation ROC-AUC) 0.7103, CatBoost challenger 0.7020. Educational data and metrics only; no real banking outcomes.

---

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
