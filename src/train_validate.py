\
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_data import generate_dataset
from metrics import classification_metrics
from psi import population_stability_index

RANDOM_STATE = 42
TARGET = "default_12m"


def data_quality_summary(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "default_rate": float(df[TARGET].mean()),
        "missing_by_column": {
            col: int(v) for col, v in df.isna().sum().items() if int(v) > 0
        },
    }


def make_logistic_pipeline(features):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", numeric, features)])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def make_hgb_pipeline(features):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer([("num", numeric, features)])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=4,
            max_iter=220,
            random_state=RANDOM_STATE,
        )),
    ])


def save_roc(y_true, baseline_score, alt_score, out):
    fpr1, tpr1, _ = roc_curve(y_true, baseline_score)
    fpr2, tpr2, _ = roc_curve(y_true, alt_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr1, tpr1, label="Logistic Regression")
    plt.plot(fpr2, tpr2, label="HistGradientBoosting")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def save_calibration(y_true, baseline_score, alt_score, out):
    p1, m1 = calibration_curve(y_true, baseline_score, n_bins=10, strategy="quantile")
    p2, m2 = calibration_curve(y_true, alt_score, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 5))
    plt.plot(m1, p1, marker="o", label="Logistic Regression")
    plt.plot(m2, p2, marker="o", label="HistGradientBoosting")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed default rate")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def save_score_distribution(y_true, score, out):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    plt.figure(figsize=(7, 5))
    plt.hist(score[y_true == 0], bins=30, alpha=0.6, label="Non-default")
    plt.hist(score[y_true == 1], bins=30, alpha=0.6, label="Default")
    plt.xlabel("Predicted PD")
    plt.ylabel("Count")
    plt.title("Score distribution — Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def format_metrics(metrics):
    return "\n".join(f"- **{k}**: {v:.4f}" for k, v in metrics.items())


def main():
    data_path = ROOT / "data" / "credit_risk_synthetic.csv"
    reports_dir = ROOT / "reports"
    figs_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(data_path, index=False)

    quality = data_quality_summary(df)
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    baseline = make_logistic_pipeline(features)
    alternative = make_hgb_pipeline(features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(baseline, X_train, y_train, scoring="roc_auc", cv=cv)

    baseline.fit(X_train, y_train)
    alternative.fit(X_train, y_train)

    p_train = baseline.predict_proba(X_train)[:, 1]
    p_valid = baseline.predict_proba(X_valid)[:, 1]
    p_alt = alternative.predict_proba(X_valid)[:, 1]

    baseline_metrics = classification_metrics(y_valid, p_valid)
    alternative_metrics = classification_metrics(y_valid, p_alt)
    psi_value = population_stability_index(p_train, p_valid, bins=10)

    save_roc(y_valid, p_valid, p_alt, figs_dir / "roc_curve.png")
    save_calibration(y_valid, p_valid, p_alt, figs_dir / "calibration_curve.png")
    save_score_distribution(y_valid, p_valid, figs_dir / "score_distribution.png")

    results = {
        "data_quality": quality,
        "cross_validation": {
            "baseline_roc_auc_mean": float(cv_auc.mean()),
            "baseline_roc_auc_std": float(cv_auc.std()),
        },
        "baseline_logistic_regression": baseline_metrics,
        "alternative_hist_gradient_boosting": alternative_metrics,
        "score_psi_train_vs_validation": float(psi_value),
    }

    (reports_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    winner = (
        "HistGradientBoostingClassifier"
        if alternative_metrics["roc_auc"] > baseline_metrics["roc_auc"]
        else "Logistic Regression"
    )

    report = f"""# Model Validation Report

## 1. Scope

Educational validation of a synthetic 12-month **Probability of Default (PD)** model for legal entities.

The purpose is to demonstrate a reproducible validation workflow, not to claim regulatory compliance.

## 2. Data quality

- Rows: **{quality['rows']}**
- Columns: **{quality['columns']}**
- Duplicate rows: **{quality['duplicates']}**
- Observed default rate: **{quality['default_rate']:.2%}**
- Missing values: **{quality['missing_by_column']}**

Missing numeric values are imputed with the training-sample median inside the sklearn Pipeline to avoid data leakage.

## 3. Baseline model

Baseline: **Logistic Regression**.

5-fold cross-validation ROC-AUC on development data:

- Mean: **{cv_auc.mean():.4f}**
- Std: **{cv_auc.std():.4f}**

Holdout metrics:

{format_metrics(baseline_metrics)}

## 4. Alternative model

Alternative: **HistGradientBoostingClassifier**.

Holdout metrics:

{format_metrics(alternative_metrics)}

On ROC-AUC, the stronger model in this run is **{winner}**.

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

- **PSI: {psi_value:.4f}**

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
"""

    (reports_dir / "validation_report.md").write_text(report, encoding="utf-8")

    print("Validation complete")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Report: {reports_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
