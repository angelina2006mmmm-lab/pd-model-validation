# Credit Risk Model Serving API — Python + FastAPI

**Educational Python model-integration prototype** for risk-strategy systems. All records and thresholds synthetic: no real bank data, approved decisions or production infrastructure.

## Implemented and locally tested
- scikit-learn LogisticRegression trained on 3,600 synthetic applications, train/validation/test 2,400/600/600; model versions 1.0.0/1.1.0 in reproducible JSON.
- Versioned scoring API: POST /v1/score, GET /health, /v1/model, /metrics and automatic /docs.
- Pydantic schema, negative-case error handling (including NaN), deterministic prediction and risk-review band, model version in result.
- Structured logs with request-id, route, status, latency and model version; no raw applicant data or IDs in logs.
- **20 pytest unit/integration tests passed locally**; CI runs the same suite.
- Example Dockerfile, Compose, Nginx, Jenkins, OpenShift configurations and technical integration documents (**examples only, not deployed/executed**).

## Checked model versions and rollback (locally tested)

- **1.0.0** is the default exported logistic model; **1.1.0** is a separately trained variant with regularization C=0.5. Both use exactly the same feature schema and deterministic synthetic train/validation/test split.
- **Selection at process start:** set `PD_MODEL_VERSION=1.1.0` to load the newer model. For rollback, restore `PD_MODEL_VERSION=1.0.0` and restart the service. No user-facing runtime switch is exposed.
- **Validation:** unit/API tests exercise v1, v2, request compatibility and a simulated rollback. Actual deployment to a bank or OpenShift cluster is not claimed.
- **Metric terminology:** `test_roc_auc` is computed on an independent synthetic test subset formed by row index, **not an out-of-time/OOT split**. No dates occur in the generated sample.

```bash
PD_MODEL_VERSION=1.1.0 uvicorn serving.api:app --port 8000
PD_MODEL_VERSION=1.0.0 uvicorn serving.api:app --port 8000  # rollback: separate restart
```

## Quick start
```bash
cd extensions/credit-risk-model-serving
python -m pip install -r requirements.txt
python -m pytest -q
python -m training.train_export
uvicorn serving.api:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/v1/score -H 'Content-Type: application/json' -d '{"application_id":"DEMO_0042","debt_to_income":0.35,"missed_payments_12m":1,"credit_history_months":48,"utilization":0.55}'
```

[Integration design](docs/integration_design.md) · [Requirements and acceptance tests](docs/test_cases_and_requirements.md).

**Limitations:** synthetic data and illustrative threshold, no financial decision, production monitoring, real bank integration or real deployment.
