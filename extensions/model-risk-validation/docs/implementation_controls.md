# Serving-code and AI-agent risk controls

## Implementation parity / golden predictions

Compare the reference estimator with a hypothetical production scoring adapter for the same applications. Intentionally swap the values of `dti` and `utilization`; check whether predictions differ beyond tolerance. This simulates discovering an integration error **before** release. A real rollout would need a frozen golden dataset, code review, model checksum/version, performance and edge-case gates, shadow testing, rollback plan, incident owner and release approval.

## Agent evaluation

Eight synthetic trace records describe status, citations, policy versions and tool calls. The validator detects missing citations for grounded claims, stale policy versions, unauthorized tools and invalid statuses. These are **static traces** and deliberately planted failures; the findings do not estimate actual LLM prompt-injection resistance or hallucination rates.

## Suggested production questions (not implemented)

Data lineage, ownership, access rights, audit trail, monitoring periodicity, approved thresholds, incident escalation, independence of model-validation function and change governance.
