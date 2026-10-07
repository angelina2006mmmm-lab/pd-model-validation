# Model Validation Report

## 1. Scope

Educational validation of a synthetic 12-month **Probability of Default (PD)** model for legal entities.

The purpose is to demonstrate a reproducible validation workflow, not to claim regulatory compliance.

## 2. Data quality

- Rows: **6000**
- Columns: **10**
- Duplicate rows: **0**
- Observed default rate: **21.80%**
- Missing values: **{'current_ratio': 90, 'years_in_business': 60, 'employees': 48}**

Missing numeric values are imputed with the training-sample median inside the sklearn Pipeline to avoid data leakage.

## 3. Baseline model

Baseline: **Logistic Regression**.

5-fold cross-validation ROC-AUC on development data:

- Mean: **0.7274**
- Std: **0.0315**

Holdout metrics:

- **roc_auc**: 0.7296
- **gini**: 0.4592
- **ks**: 0.3503
- **precision**: 0.3609
- **recall**: 0.6684
- **f1**: 0.4687
- **log_loss**: 0.6100
- **brier_score**: 0.2105

## 4. Alternative model

Alternative: **HistGradientBoostingClassifier**.

Holdout metrics:

- **roc_auc**: 0.7156
- **gini**: 0.4312
- **ks**: 0.3320
- **precision**: 0.5956
- **recall**: 0.2066
- **f1**: 0.3068
- **log_loss**: 0.4715
- **brier_score**: 0.1510

On ROC-AUC, the stronger model in this run is **Logistic Regression**.

## 5. Calibration

Calibration is checked using:

- **Brier Score**;
- a calibration curve comparing predicted PD with observed default rate.

See `reports/figures/calibration_curve.png`.

## 6. Discriminatory power

The project reports:

- ROC-AUC;
- Gini = `2 * ROC-AUC - 1`;
- KS statistic;
- Precision / Recall / F1.

See `reports/figures/roc_curve.png` and `reports/figures/score_distribution.png`.

## 7. Stability

PSI between Logistic Regression scores on train and holdout samples:

- **PSI: 0.0047**

PSI is used here as a simple stability diagnostic. Interpretation thresholds should be defined by the model-risk methodology rather than treated as universal rules.

## 8. Validation conclusion

This educational example demonstrates:

- independent holdout validation;
- cross-validation on development data;
- comparison with an alternative model;
- discrimination metrics;
- calibration analysis;
- score stability analysis;
- automated reporting.

A real bank validation would additionally require approved internal methodology, regulatory context, model documentation, portfolio segmentation, backtesting over time, representativeness checks, overrides / expert adjustments where applicable, and governance procedures.
