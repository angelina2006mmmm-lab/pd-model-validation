# Model Risk Validation & Monitoring — Independent Challenger Framework

**Portfolio project for a Data Science / Model Risk Management internship.**
An independently reproducible prototype that validates **credit ML models and hypothetical AI-agent traces** and generates a concise report for business/risk stakeholders.

All inputs are **synthetic**; no Sberbank systems, borrower data, approved lending rules, regulatory validation or production monitoring are involved.

## What the reviewer can verify

| Task | How the prototype addresses it |
|---|---|
| Challenger modeling | Fits Logistic Regression and HistGradientBoosting on exactly the same training period; selects on validation ROC-AUC |
| Independent validation | Chronological train (2023–mid-2024), validation (late-2024), untouched OOT (2025); ROC-AUC, Gini, KS, PR-AUC, Brier, LogLoss, ECE |
| Model monitoring | Monthly OOT discrimination/calibration, score PSI versus training, illustrative thresholds and alerts |
| Business impact | Validation-chosen decision threshold, missed-risk/review cost proxy, expected-loss proxy (PD × EAD × assumed LGD) and macro shock |
| Model implementation incidents | Golden-score consistency check detects deliberately swapped serving features (`dti` and `utilization`) |
| AI-agent governance | Audits 8 hand-authored agent traces for unsupported citations, stale document versions, invalid statuses and unauthorized tool calls |
| Documentation | Machine-readable JSON, Markdown stakeholder report, data-quality SQL and methodology notes |
| Reproducible automation | pytest unit/integration tests; GitHub Actions runs validation and publishes reports as artifacts |

**The agent audit checks traces, not actual accuracy of a live LLM.** A separate [RiskOps project](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent) explores RAG/agents; this project targets independent model-risk monitoring.

## Quick start

```bash
cd extensions/model-risk-validation
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m modelrisk.validation
```

Output: `reports/validation_results.json` and `reports/independent_validation_report.md`.

## Research method

1. Validate data quality, ID uniqueness, numeric limits and presence of targets.
2. Chronologically split independent vintages; exclude ID, target and date from features.
3. Fit baseline and challenger; choose by **validation** ROC-AUC before ever inspecting OOT.
4. Evaluate both models on the untouched OOT period; monitor stability and calibration.
5. Compute a transparent sensitivity analysis using **assumed**, not measured bank economics.
6. Detect a deliberately introduced feature-mapping bug in a hypothetical scoring adapter.
7. Audit synthetic agent traces for missing/unapproved tool use and policy-version errors.
8. Publish a concise model-risk report for an independent reviewer.

See [methodology](docs/methodology.md), [model implementation controls](docs/implementation_controls.md) and [SQL](sql/model_validation_checks.sql).


## Reproduced CI results (2026-10-12)

[Successful independent validation workflow](https://github.com/angelina2006mmmm-lab/pd-model-validation/actions/runs/38267898317) — **19 tests passed**.

- Synthetic training / validation / 2025 OOT: **2428 / 770 / 1602** applications
- Validation ROC-AUC: Logistic Regression 0.72646, HistGradientBoosting 0.70326; champion selected on validation only
- Champion OOT: ROC-AUC **0.72165**, Gini 0.4433, KS 0.35882, Brier 0.0968, ECE 0.01715
- Challenger OOT ROC-AUC: **0.69333**
- Score PSI train → OOT: **0.03759**
- Golden-score audit: intentionally swapped two features, **120/120** predictions differed (maximum absolute PD delta 0.152484)
- AI-agent trace audit: **5/8** synthetic traces had planted control failures
- Hypothetical stress input +0.25 raised mean predicted PD 0.10347 → 0.11406; illustrative expected loss proxy 39,818,367.38 → 43,882,613.08, assuming LGD=45%.

[Full verified metrics](reports/validation_results.json) · [Stakeholder report](reports/independent_validation_report.md)

## Scope boundaries

- The data and business costs are synthetic. PD/LGD/EAD vocabulary does **not** make these results Basel/IRB regulatory models.
- Alerts are demonstrative thresholds (AUC 0.65, PSI 0.2, ECE 0.08), not bank policy.
- No comparison of external SOTA models or claims about production financial return.
- No authorized access to a real bank model registry, risk system, governance board or historical incident database.
- LLM/agent evaluation is trace-level, not an end-to-end generative AI benchmark.
