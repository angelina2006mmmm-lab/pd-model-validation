# Credit Risk Scoring: CatBoost vs Logistic Regression

Educational, reproducible probability-of-default (PD) pipeline for risk modeling. **Synthetic loan data only** (no bank data, no lending decisions).

- Generate 4,200 synthetic applications (seed 42) with temporal shift and without demographic attributes.
- Split chronologically: train Jan 2023-Jun 2024; validation Jul-Dec 2024; **out-of-time test in 2025**.
- Compare logistic baseline and CatBoost gradient boosting on the same features.
- Select model by validation ROC-AUC and choose threshold on validation using hypothetical asymmetric errors (FN:FP = 5:1). The 2025 test is never used for selection.
- Out-of-time ROC-AUC, Gini, KS, PR-AUC, Brier, LogLoss and PSI.
- SQL data quality and vintage checks; pytest regression tests; JSON and Markdown reports.

## Verified results (seed 42, synthetic 4,200 rows)

- Train / validation / OOT: 2,121 / 691 / 1,388.
- Validation ROC-AUC: Logistic Regression 0.7328 vs CatBoost 0.7073. The selected model is **Logistic Regression**, not CatBoost.
- OOT for selected model: ROC-AUC 0.7103, Gini 0.4207, KS 0.3295, Brier 0.1028, PSI 0.0892.
- 10 local pytest tests passed. CatBoost was implemented and evaluated as a challenger model.
- Hypothetical decision cost is an illustrative error metric, **not** financial gain.

## Run

From this folder:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    PYTHONPATH=src python -m creditrisk.experiment
    PYTHONPATH=src pytest -q

The code includes chronological sample isolation, leakage checks, and a deliberately untouched OOT test. It does not establish real-world credit model performance or compliance with bank governance.

## Files
- src/creditrisk/data.py - synthetic generator, features, time split, quality checks
- src/creditrisk/metrics.py - ROC-AUC, Gini, KS, PR-AUC, Brier, PSI, cost-sensitive threshold
- src/creditrisk/experiment.py - model training, validation, OOT report
- tests/test_model.py - 10 automated checks
- sql/data_quality.sql - PostgreSQL checks
- reports/metrics.json and reports/report.md - actual locally generated experiment results
