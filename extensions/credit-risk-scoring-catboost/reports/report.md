# Credit Scoring - 4,200 Synthetic Applications (seed 42)

Reproduced locally on a chronological split:
- Train 2,121; validation 691; 2025 out-of-time holdout 1,388.
- Validation ROC-AUC: Logistic Regression **0.7328**, CatBoost **0.7073**. Logistic Regression selected; challenger CatBoost evaluated.
- Holdout selected model: ROC-AUC **0.7103**, Gini **0.4207**, KS **0.3295**, Brier **0.1028**, PR-AUC **0.2978**.
- Population Stability Index (train vs out-of-time prediction scores): **0.0892**.
- Illustration: FN cost 5, FP cost 1, validation-selected threshold 0.17, holdout average proxy cost 0.5043. This is **not real banking income or risk management policy**.
- 10 tests passed locally, including out-of-time split and target leakage checks.
- No actual borrower data, production, regulatory compliance or bank approval is claimed.
