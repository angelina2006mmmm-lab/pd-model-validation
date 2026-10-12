# Synthetic credit risk: Logistic Regression vs XGBoost vs LightGBM

4,200 synthetic applications, chronological train/validation/2025 OOT test. Model selected on validation only.

| Model | Validation ROC-AUC | OOT ROC-AUC | OOT Brier | OOT PSI |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7328 | 0.7103 | 0.1028 | 0.0892 |
| XGBoost | 0.7170 | 0.6780 | 0.1067 | 0.3796 |
| LightGBM | 0.7159 | 0.6799 | 0.1074 | 0.4676 |

The Logistic Regression baseline won. XGBoost and LightGBM were **actually trained**, not merely listed as skills. Metrics are illustrative on synthetic data, never real bank outcomes. Thresholds and model choice used validation rather than OOT.
