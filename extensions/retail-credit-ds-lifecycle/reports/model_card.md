# Synthetic Retail Credit ML Model Card

**Educational PoC. No bank data or real lending decisions.**

- Data sources: 4,200 synthetic applications joined to 36 synthetic monthly macro records
- Train / validation / 2025 OOT: 2,128 / 690 / 1,382
- Selected based on validation ROC-AUC: **Logistic Regression** (0.69266 vs 0.65916)
- OOT ROC-AUC: **0.65472**, PR-AUC: 0.19318, Brier: 0.08402
- Features: loan_to_income, debt_to_income, utilization, credit_history_months, late_payments_12m, macro_stress, product
- Application IDs, dates and target are not inputs. OOT data not used for model selection.
- Artifacts: joblib model stored by the CI workflow, machine-readable JSON metrics.
- No production deployment, regulatory validation, lending threshold, assessed business value or real-world data.
