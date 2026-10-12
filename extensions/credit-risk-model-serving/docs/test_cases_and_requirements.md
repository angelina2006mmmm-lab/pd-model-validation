# Functional requirements and acceptance scenarios

| ID | Functional requirement | Acceptance check |
|---|---|---|
| FR-01 | Validate fields / types / NaN | HTTP 422; no raw sensitive values in error |
| FR-02 | Deterministic probability of default | Repeat request gives same probability for model version |
| FR-03 | Release traceability | Model version in both scoring and metadata routes |
| FR-04 | Reproducible training | Exported artifact matches committed JSON |
| FR-05 | Log diagnosis without PII | JSON request_id, path, status, version, latency; no applicant fields |
| FR-06 | Service readiness | GET /health; OpenAPI and Docker image example |
| FR-07 | Tests and CI | pytest unit and integration tests run on push |

## Change request template
Problem and source system, acceptance owner, functional/business requirements, field data dictionary, constraints, API schema, model version, edge cases, test matrix, monitoring and rollback.

## Collaboration workflow (example)
Jira: requirements and defects; Confluence: integration design/test cases/runbook; Bitbucket: branches/reviews/CI. These are workflow examples, not proof of integration with a corporate tenant.
