# Decision brief: synthetic credit-risk models

**Question:** Which PD model should the team advance to an independent risk review?

**Recommendation for this educational dataset:** Logistic Regression. It leads on validation ROC-AUC (0.7328 vs 0.7170 XGBoost and 0.7159 LightGBM), stays strongest on 2025 OOT ROC-AUC (0.7103 vs 0.6780 and 0.6799), and has the lowest score PSI (0.0892 vs 0.3796 and 0.4676).

**Trade-off:** Under a wholly hypothetical false-negative:false-positive error cost of 5:1, validation-selected thresholds yield OOT average proxy cost 0.5043 (logistic), 0.5684 (XGBoost), 0.5137 (LightGBM). These are not rubles or profit figures.

**Caveats:** All applications are synthetic, sample size only 4,200, potential dataset shift and instability. This is not a regulatory or lending-ready risk model. Production requires real governed data, fairness and drift checks, independent validation, versioning, monitoring, business approval and reproducible deployment.
