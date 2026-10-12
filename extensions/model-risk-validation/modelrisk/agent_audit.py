"""Deterministic validation of hypothetical AI-agent traces, NOT LLM accuracy."""
from pathlib import Path
import json

POLICIES={"RISK-01":{"version":2,"effective_from":"2026-01-01"},
          "ESG-02":{"version":1,"effective_from":"2025-01-01"}}
ALLOWED_TOOLS={"search_policy","evaluate_risk"}

def validate_trace(trace):
    flaws=[]
    if any(x not in ALLOWED_TOOLS for x in trace.get("tools",[])):
        flaws.append("unapproved_tool")
    cites=trace.get("citations",[])
    if trace.get("status")=="grounded" and not cites:
        flaws.append("missing_citation")
    for cite in cites:
        info=POLICIES.get(cite.get("id"))
        if not info:flaws.append("unrecognized_policy")
        elif cite.get("version")!=info["version"]:flaws.append("stale_policy_version")
    if trace.get("status") not in {"grounded","abstain"}:
        flaws.append("invalid_status")
    return {"id":trace.get("id"),"passed":not flaws,"issues":sorted(set(flaws))}

def run_audit(path=None):
    source=Path(path) if path else Path(__file__).resolve().parents[1]/"data/agent_traces.jsonl"
    traces=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    results=[validate_trace(x) for x in traces]
    return {"cases":len(results),"passed":sum(x["passed"] for x in results),
            "failed":sum(not x["passed"] for x in results),
            "findings":results,
            "limits":"Hand-authored synthetic traces; no live LLM or real business data."}
