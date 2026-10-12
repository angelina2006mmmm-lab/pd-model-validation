# Risk segmentation: synthetic PD scoring

**All data are synthetic. This is a study, not a live bank scorecard.**

On the independent 2025 out-of-time sample: **1,388** applications, Logistic Regression ROC-AUC **0.7103**.

Five risk-group boundaries were derived from the **2024 validation predictions only**, not from OOT data.

| Risk group | OOT applications | Mean predicted PD | Observed default rate |
|---:|---:|---:|---:|
| 1 | 224 | 0.0295 | 0.0446 |
| 2 | 256 | 0.0574 | 0.0547 |
| 3 | 259 | 0.0883 | 0.0888 |
| 4 | 318 | 0.1439 | 0.1321 |
| 5 | 331 | 0.3150 | 0.2598 |

Interpretation: more-risky model-score groups have higher mean forecast PD; observed default rates are noisy and don't fully agree with probability calibration, which should be validated.

**Business and model risk:** a credit cutoff is not set from discrimination metrics alone. False approvals and false declines differ in cost; approval policy, legal requirements, model monitoring, fairness/shift assessment, and independent validation are needed before real use.

Run `PYTHONPATH=src python -m creditrisk.business_brief` to regenerate the report, or `PYTHONPATH=src pytest -q` for regression checks.
