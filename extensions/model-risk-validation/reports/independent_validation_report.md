# Independent Model Risk Validation — Verified Synthetic Results

**Educational evidence, not certified independent bank validation.**

- Sample: 2428 train / 770 validation / 1602 chronological out-of-time applications.
- Champion selected on validation: **LogisticRegression**.
- Selected model OOT: **ROC-AUC 0.72165**, Gini 0.4433, KS 0.35882, Brier 0.0968, ECE 0.01715.
- Challenger OOT ROC-AUC: 0.69333.
- Score PSI from training to OOT: 0.03759.
- Intentional serving-feature mapping defect flagged on 120/120 application samples.
- Agent governance: 5/8 mocked synthetic traces have planted issues.
- Alerts: implementation_mapping_incident, synthetic_agent_trace_findings.

## Business interpretation
With an illustrative macro feature increase of 0.25, mean predicted PD increased 0.10347 → 0.11406.
Expected-loss proxy = PD × loan amount × assumed LGD=0.45. It is NOT bank revenue/loss prediction or Basel capital.

## Caveats
- Synthetic and randomly generated labels. Neither challenger performance nor alerts generalize to actual financial data.
- OOT period is chronological by synthetic application_date; this is not independent certification.
- Discovered implementation and agent failures are planted synthetic fixtures, not real-world detections.
- The governance/agent audit uses synthetic traces, not a live LLM accuracy benchmark.
