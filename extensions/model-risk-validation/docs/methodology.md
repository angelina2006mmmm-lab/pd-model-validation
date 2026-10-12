# Independent validation methodology

**Object of review:** synthetic default_12m classification on loan applications, plus trace-level agent governance.

Train from Jan 2023 through June 2024; validation from July to December 2024; OOT 2025. The validation sample determines the selected alternative and threshold; the OOT dataset is held back until the final report.

Report discrimination (ROC-AUC, Gini, KS, PR-AUC), calibration (Brier, log loss, ECE), drift (score PSI versus training), OOT monthly checks and stress sensitivity.

**Challenge the developer:** train HistGradientBoosting and Logistic Regression on the same training data; compare validation ROC-AUC, then examine the other approach on OOT without tuning. A more complex challenger need not win.

**Business interpretation:** calculate a hypothetical review-versus-missed-default cost and exposure-weighted PD × assumed LGD. These are proxy calculations, not regulatory capital or true bank economic losses.

**Regulation:** Basel frameworks motivate PD/LGD/EAD risk terminology, but the demo is not a Basel-compliant or independently certified regulatory model. Its input data and validations are intentionally too limited.
