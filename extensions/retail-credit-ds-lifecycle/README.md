# Retail Credit DS Lifecycle — end-to-end classic ML + PySpark

**A small, reproducible portfolio demonstration for the Sber Data Scientist internship.** All data and macro conditions are artificially generated; this is not a bank dataset and cannot estimate actual bank risk or profitability.

## Full lifecycle

1. Two distinct source tables: **loan applications (4,200 rows)** and **monthly macro scenario (36 months)**, joined by `application_month`.
2. Input quality: unique primary keys, positive loan/income, valid 0/1 targets, many-to-one macro join, complete feature columns.
3. Data preparation implemented independently in **Pandas and PySpark**. Spark `local[2]` parity test verifies the loan-to-income feature on 240 synthetic rows.
4. Chronological train (2023–mid-2024), validation (late-2024), untouched OOT (2025).
5. Logistic Regression vs HistGradientBoosting; champion selection by validation ROC-AUC only.
6. Final OOT evaluation with ROC-AUC, PR-AUC, Brier and LogLoss.
7. Export an actual `joblib` inference artifact, machine-readable metrics, and **model card**. GitHub Actions stores generated artifacts.
8. Unit tests for duplicate source records, missing macro keys, reproducibility, data leakage, model selection and artifact roundtrip.

## Reproduced local run: 4,200 synthetic applications

| Metric | Result |
|---|---:|
| train / validation / OOT | 2,128 / 690 / 1,382 |
| validation Logistic Regression ROC-AUC | 0.69266 |
| validation HistGradientBoosting ROC-AUC | 0.65916 |
| selected model | Logistic Regression |
| OOT ROC-AUC | **0.65472** |
| OOT PR-AUC | 0.19318 |
| OOT Brier | 0.08402 |

A more complex model did not win; the results are *not* hidden or cherry-picked. Metrics are from a simple synthetic learning task and should not be generalized.

## Local execution

```bash
cd extensions/retail-credit-ds-lifecycle
python -m pip install -r requirements.txt
python -m pytest -q
python -m pipeline.experiment
```

Optional Spark ETL parity (requires Java and the extra packages):

```bash
python -m pip install -r requirements-spark.txt
python -m pipeline.spark_etl
```

GitHub Actions runs the ordinary ML tests and Spark integration test in separate jobs. PySpark and its parity results should only be described as **verified** after the corresponding CI job is green.

## Scope

This project demonstrates data-source integration, deterministic feature engineering, leakage control, model comparison and deployment artifacts. It is *not* a regulated Basel scoring model, not a trained Transformer, and not a production Spark cluster. For separate completed experiments with PyTorch Transformer, XGBoost/LightGBM/CatBoost, see the [credit-scoring extension](../credit-risk-scoring-catboost).
