"""Independent challenger, temporal validation and model-implementation incident checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .data import make_portfolio,validate_portfolio,split_vintages,FEATURES,NUMERIC,CATEGORICAL
from .metrics import (
 classification_metrics,population_stability_index,select_threshold,
 decision_proxy,expected_loss_proxy)
from .agent_audit import run_audit

def build_models(seed=1729):
    def preprocess():
        return ColumnTransformer([
          ("num",StandardScaler(),NUMERIC),
          ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CATEGORICAL)])
    return {
      "LogisticRegression":Pipeline([("features",preprocess()),
                    ("model",LogisticRegression(max_iter=700,random_state=seed))]),
      "HistGradientBoosting":Pipeline([("features",preprocess()),
                    ("model",HistGradientBoostingClassifier(
                      max_iter=110,max_leaf_nodes=10,l2_regularization=2.0,
                      learning_rate=.06,random_state=seed))])
    }

def monthly_monitor(data,prob,min_monthly=30):
    df=data[["application_date","default_12m"]].copy()
    df["pd"]=prob
    df["month"]=pd.to_datetime(df.application_date).dt.strftime("%Y-%m")
    rows=[]
    for month,part in df.groupby("month",sort=True):
        if len(part)<min_monthly or part.default_12m.nunique()!=2:continue
        metric=classification_metrics(part.default_12m,part.pd)
        rows.append({"month":month,"n":len(part),"roc_auc":metric["roc_auc"],
                    "brier":metric["brier"],"event_rate":metric["event_rate"]})
    return rows

def implementation_incident(model,frame,tolerance=1e-7):
    """Compare reference feature contract with an intentionally wrong serving adapter."""
    x=frame[FEATURES].copy()
    expected=model.predict_proba(x)[:,1]
    broken=x.copy()
    # Deliberate, documented bug: production maps utilization to DTI and vice versa.
    broken["dti"]=x["utilization"].to_numpy()
    broken["utilization"]=x["dti"].to_numpy()
    observed=model.predict_proba(broken)[:,1]
    gap=np.abs(expected-observed)
    return {
      "scenario":"deliberately swapped dti / utilization fields in serving adapter",
      "checked":int(len(gap)),
      "mismatches":int(np.count_nonzero(gap>tolerance)),
      "max_absolute_probability_delta":round(float(gap.max()),6),
      "severity":"incident" if np.any(gap>tolerance) else "ok"}

def stress_test(model,frame,macro_shock=.25,lgd=.45):
    initial=frame[FEATURES].copy()
    shifted=initial.copy()
    shifted["macro_stress"]=shifted["macro_stress"]+macro_shock
    base=model.predict_proba(initial)[:,1]
    shocked=model.predict_proba(shifted)[:,1]
    ead=frame["loan_amount"].to_numpy()
    return {"macro_shock":macro_shock,
       "mean_pd_baseline":round(float(base.mean()),5),
       "mean_pd_shocked":round(float(shocked.mean()),5),
       "expected_loss_proxy_baseline":expected_loss_proxy(base,ead,lgd),
       "expected_loss_proxy_shocked":expected_loss_proxy(shocked,ead,lgd),
       "assumption":"LGD=.45, exposure=loan amount; illustrative scenario only"}

def run_validation(n=4800,seed=1729,output_dir=None):
    data=make_portfolio(n,seed);validate_portfolio(data)
    train,valid,oot=split_vintages(data)
    models=build_models(seed)
    x_train,y_train=train[FEATURES],train.default_12m
    x_valid,y_valid=valid[FEATURES],valid.default_12m
    candidates={}
    for name,model in models.items():
        model.fit(x_train,y_train)
        p=model.predict_proba(x_valid)[:,1]
        candidates[name]=classification_metrics(y_valid,p)
    # Selection happens only on validation data. OOT is never used for tuning.
    winner=max(candidates,key=lambda k:candidates[k]["roc_auc"])
    model=models[winner]
    p_val=model.predict_proba(x_valid)[:,1]
    threshold=select_threshold(y_valid,p_val)
    train_prob=model.predict_proba(x_train)[:,1]
    oot_prob=model.predict_proba(oot[FEATURES])[:,1]
    oot_scores={name:classification_metrics(oot.default_12m,
                   m.predict_proba(oot[FEATURES])[:,1]) for name,m in models.items()}
    psi=population_stability_index(train_prob,oot_prob)
    alerts=[]
    if oot_scores[winner]["roc_auc"]<.65:alerts.append("low_oot_discrimination")
    if psi>.2:alerts.append("score_population_shift")
    if oot_scores[winner]["ece"]>.08:alerts.append("probability_calibration_gap")
    incident=implementation_incident(model,oot.iloc[:120])
    if incident["severity"]=="incident":alerts.append("implementation_mapping_incident")
    agents=run_audit()
    if agents["failed"]:alerts.append("synthetic_agent_trace_findings")
    result={
      "scope":"Independent student validation with synthetic applications and mocked agent traces",
      "seed":seed,"sample_counts":{"train":len(train),"validation":len(valid),"out_of_time":len(oot)},
      "split":{"train":"2023-01 through 2024-06",
              "validation":"2024-07 through 2024-12","out_of_time":"2025-01 through 2025-12"},
      "features":FEATURES,"selection":"Validation ROC-AUC, OOT untouched until final report",
      "validation_models":candidates,"selected_model":winner,
      "oot_models":oot_scores,"review_threshold_from_validation":threshold,
      "business_cost_proxy_on_oot":decision_proxy(oot.default_12m,oot_prob,threshold),
      "score_psi_train_to_oot":round(psi,5),
      "monthly_oot_monitoring":monthly_monitor(oot,oot_prob),
      "stress_scenario":stress_test(model,oot),
      "implementation_consistency":incident,
      "agent_trace_audit":agents,
      "alerts":sorted(alerts),
      "note":"Alert thresholds, costs and LGD are illustrative. No Basel compliance or production deployment."}
    if output_dir:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        (path/"validation_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        (path/"independent_validation_report.md").write_text(report_markdown(result),encoding="utf-8")
    return result

def report_markdown(r):
    score=r["oot_models"][r["selected_model"]]
    rows=[
      "# Independent Model Risk Validation Report",
      "",
      "**Synthetic educational evidence only. Not bank model approval.**",
      "",
      f"- Portfolio: {r['sample_counts']}, chronologically separated.",
      f"- Champion/challenger selected on 2024 validation: **{r['selected_model']}**.",
      f"- 2025 OOT: ROC-AUC {score['roc_auc']}, Gini {score['gini']}, KS {score['ks']}, Brier {score['brier']}, ECE {score['ece']}.",
      f"- Score PSI train versus OOT: {r['score_psi_train_to_oot']}.",
      f"- Intentional feature mapping bug found in {r['implementation_consistency']['mismatches']} of {r['implementation_consistency']['checked']} scored applications.",
      f"- Synthetic AI-agent trace findings: {r['agent_trace_audit']['failed']} of {r['agent_trace_audit']['cases']}.",
      "", "## Alerts",
      ", ".join(r["alerts"]) or "None under illustrative thresholds.",
      "", "## Governance caveats",
      "- Validation dataset is synthetic; no real bank portfolio or approved risk appetite.",
      "- Stress scenario uses a hypothetical macro input change, assumed LGD and simplified EAD.",
      "- Alternative models are independent challengers, not certified Basel PD/LGD/EAD models.",
      "- A real deployment needs data lineage, owners, release approvals, controls and monitoring SLAs.",
      "- Agent audit uses hand-authored traces and does not establish performance of a live LLM.",
      ""]
    return "\n".join(rows)
if __name__=="__main__":
    print(json.dumps(run_validation(output_dir=Path(__file__).resolve().parents[1]/"reports"),ensure_ascii=False,indent=2))
