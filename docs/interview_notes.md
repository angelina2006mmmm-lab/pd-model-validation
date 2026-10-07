# Interview notes — Credit Risk Model Validation

Use these notes to explain the project in your own words.

## PD / LGD / EAD

- **PD (Probability of Default)** — probability that a borrower defaults over a defined horizon.
- **LGD (Loss Given Default)** — proportion of exposure that is lost if default occurs.
- **EAD (Exposure at Default)** — exposure amount at the time of default.

This project models **PD only**.

## Why Logistic Regression?

It is a strong baseline for tabular credit-risk tasks because it is simple, stable, interpretable and produces probabilities directly. It also makes it easier to discuss calibration and coefficient direction.

## Why an alternative model?

Validation should not only inspect the original model. An alternative model helps test whether a materially different algorithm captures predictive signal better or exposes weaknesses in the baseline.

Here the alternative is `HistGradientBoostingClassifier`.

## ROC-AUC

Measures ranking ability: how often a randomly selected default observation receives a higher score than a randomly selected non-default observation.

Higher is better, but a high AUC alone does not mean the probabilities are calibrated.

## Gini

Calculated here as:

`Gini = 2 * ROC-AUC - 1`

It is another way of expressing discriminatory power.

## KS

Maximum distance between the cumulative score distributions of default and non-default groups.

Useful for checking separation.

## Calibration

A PD model should not only rank risky borrowers correctly. Predicted probabilities should also correspond reasonably well to observed default frequencies.

This project checks calibration using:

- Brier Score;
- calibration curve.

## PSI

Population Stability Index compares the distribution of scores between two samples.

Here it is used to compare train and validation score distributions. In production, it is often used as part of monitoring across time periods.

Do not present PSI thresholds as universal laws. Thresholds should be defined by the model-risk methodology.

## Cross-validation

Cross-validation estimates model performance across several train/validation folds and helps assess whether the result is stable rather than dependent on one split.

## Data leakage

Leakage happens when information unavailable at prediction time gets into model training or validation.

This project uses sklearn Pipelines so median imputation is fitted only on training data inside each fold / fit.

## What I would add in a production validation

- time-based out-of-time sample;
- portfolio representativeness analysis;
- segment-level performance;
- backtesting;
- drift / stability by period;
- challenger models;
- documentation review;
- implementation verification;
- governance / methodology checks;
- regulatory requirements relevant to the bank and model scope.

## 845-P / regulatory context

For an interview, distinguish clearly between:

1. what you implemented in this educational project;
2. what you studied conceptually about internal ratings / regulatory model validation.

Do not claim that this repository is compliant with 845-P. It is not a regulatory implementation.
