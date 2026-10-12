# Credit Risk Scoring: CatBoost vs Logistic Regression

## Extended experiment: XGBoost, LightGBM and PyTorch Transformers (October 2026)

**Actually executed locally:** 17 pytest tests pass. Logistic Regression, XGBoost and LightGBM trained on the same chronological split of 4,200 **synthetic** credit applications; CatBoost remains a separate benchmark in this folder. Training and score reporting are reproducible with the fixed seed.

| Model | Validation ROC-AUC | OOT ROC-AUC | Score PSI |
|---|---:|---:|---:|
| Logistic Regression | 0.7328 | 0.7103 | 0.0892 |
| XGBoost | 0.7170 | 0.6780 | 0.3796 |
| LightGBM | 0.7159 | 0.6799 | 0.4676 |

**Selected on validation:** Logistic Regression. Lower OOT scores for the challengers are reported without cherry-picking. Thresholds are validation-selected, with **illustrative** 5:1 false-negative/false-positive costs. See [benchmark](reports/boosting_comparison.md), [business-oriented decision brief](reports/business_brief.md).

A separate small PyTorch `TransformerEncoder` was **trained from scratch** on six synthetic financial-history observations. OOT ROC-AUC 0.5764; this is a learning experiment, not an effective lending model. [Transformer note](reports/transformer_brief.md).

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m creditrisk.boosting_challengers
PYTHONPATH=src python -m creditrisk.transformer_demo
PYTHONPATH=src pytest -q
```

No real banking data, validated decision process, production deployment or regulatory claims are made.


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


## Additional models, DL and business deliverables (verified locally)

- **XGBoost / LightGBM**: both are actually trained in `src/creditrisk/boosting_challengers.py`; validation AUC 0.7170 / 0.7159, 2025 OOT AUC 0.6780 / 0.6799. Baseline logistic regression remains strongest on this synthetic sample.
- **PyTorch TransformerEncoder**: `src/creditrisk/transformer_demo.py` trains a small network over six synthetic monthly feature snapshots. OOT ROC-AUC 0.5764 (not a pretrained model).
- **Business-facing risk grouping**: `src/creditrisk/business_brief.py` computes five predicted-PD segments using boundaries from validation only, then compares predicted vs observed default rates on OOT without leakage. See [risk segmentation report](reports/risk_segmentation.md).
- **Decision summary**: [business decision brief](reports/business_brief.md) explains why a simpler model is preferred, illustrative cost tradeoffs and deployment limitations.
- **Local regression checks**: 17 pytest tests passed (baseline/CatBoost + XGBoost/LightGBM + Transformer + business segmentation). `PySpark` remains an illustrative script in the parent repo, not a claim of a production Spark pipeline.

Run: `PYTHONPATH=src python -m creditrisk.boosting_challengers`, `PYTHONPATH=src python -m creditrisk.transformer_demo`, `PYTHONPATH=src python -m creditrisk.business_brief`, `PYTHONPATH=src pytest -q`.

All evaluation data are simulated, so the reported metrics cannot establish accuracy, fairness or regulatory fitness on real loan portfolios.

## Business presentation: PD risk segmentation

[Five independent out-of-time PD risk groups](reports/risk_segmentation.md) show application counts, average predicted PD and observed synthetic default rates, with conclusions for nontechnical stakeholders. The thresholds/bins were set using validation rather than OOT outcomes. All samples are synthetic; 17 local tests now pass.
