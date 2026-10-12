# Technical integration design (educational)

## Goal
Prototype the integration of a versioned probability-of-default (PD) model into a loan-risk system. All values and records are synthetic; no actual credit decision.

## API contract
- POST /v1/score: applicant ID, debt_to_income [0,1], missed_payments_12m [0,12], credit_history_months [0,600], utilization [0,1]; returns estimated PD, risk band and version.
- GET /v1/model: model 1.0.0 (default) or 1.1.0, features, synthetic holdout ROC-AUC. GET /health: readiness. GET /metrics: in-memory diagnostics without applicant inputs. /docs: OpenAPI.

## Design and deterministic transformations
1. Pydantic validates required fields and ranges; malformed and non-finite values return 422.
2. Trained logistic model in versioned JSON; features scaled using train-only statistics. Train/validation/test 2400/600/600.
3. Model returns deterministic PD and illustrative review band from validation-chosen threshold; score is **not** a real lending decision.
4. Middleware writes JSON with request ID, route, status, model version and latency, excluding applicant IDs and raw features.
5. Unit and API integration tests, CI checks, planned version upgrade/release and rollback.

## Production gaps
No live bank data/infrastructure, no authentication or TLS/permissions, production alerting, drift monitoring or regulator-approved strategy. Infrastructure templates are NOT deployed or benchmarked.

## Release and rollback (demonstrated in tests)
- v1.0.0 and v1.1.0 are separate model artifacts with stable feature contracts; v1.1.0 is trained using C=0.5 regularization.
- Set `PD_MODEL_VERSION=1.1.0` before starting the service to select the candidate.
- Monitor /health, /v1/model, /metrics and validated requests; restore `PD_MODEL_VERSION=1.0.0` and restart to roll back.
- Automated tests simulate the version switch and rollback. This is not a real cluster rollout or hot reload.
- **test_roc_auc** is estimated on a non-temporal synthetic test subset; no out-of-time validation was performed in this service experiment.
