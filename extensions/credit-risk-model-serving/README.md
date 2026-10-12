# Credit Risk Model Serving API — Python + FastAPI

**Educational Python model-integration prototype** for risk-strategy systems. All records and thresholds synthetic: no real bank data, approved decisions or production infrastructure.

## Implemented and locally tested
- scikit-learn LogisticRegression trained on 3,600 synthetic applications, train/validation/test 2,400/600/600; model 1.0.0 in reproducible JSON.
- Versioned scoring API: POST /v1/score, GET /health, /v1/model, /metrics and automatic /docs.
- Pydantic schema, negative-case error handling (including NaN), deterministic prediction and risk-review band, model version in result.
- Structured logs with request-id, route, status, latency and model version; no raw applicant data or IDs in logs.
- **15 pytest unit/integration tests passed locally**; CI runs the same suite.
- Example Dockerfile, Compose, Nginx, Jenkins, OpenShift configurations and technical integration documents (**examples only, not deployed/executed**).

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
