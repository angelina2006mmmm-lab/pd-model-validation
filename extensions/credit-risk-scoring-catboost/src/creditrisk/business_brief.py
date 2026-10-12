"""Synthetic PD risk segmentation and a small, non-production business brief."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import FEATURES, NUM_FEATURES, CAT_FEATURES, create_applications, split_time
from .metrics import binary_report

def create_business_brief(n=4200,seed=42,output_dir=None):
    train,validation,test=split_time(create_applications(n=n,seed=seed))
    model=make_pipeline(ColumnTransformer([
        ("num",StandardScaler(),NUM_FEATURES),
        ("cat",OneHotEncoder(handle_unknown="ignore"),CAT_FEATURES)
    ]),LogisticRegression(max_iter=600,random_state=seed))
    model.fit(train[FEATURES],train.default_12m)
    p_val=model.predict_proba(validation[FEATURES])[:,1]
    p_oot=model.predict_proba(test[FEATURES])[:,1]
    limits=np.unique(np.quantile(p_val,np.linspace(0,1,6)))
    if len(limits)!=6:raise ValueError("Five risk bands require distinct validation quantiles")
    limits[0]=-np.inf;limits[-1]=np.inf
    groups=pd.cut(p_oot,bins=limits,labels=[1,2,3,4,5],include_lowest=True)
    summary=(pd.DataFrame({"group":groups.astype(int),"pd":p_oot,
                 "default":test.default_12m.to_numpy()})
          .groupby("group",observed=True)
          .agg(applications=("default","size"),average_pd=("pd","mean"),
               observed_default_rate=("default","mean")).reset_index())
    records=[{"group":int(r.group),"applications":int(r.applications),
              "average_pd":round(float(r.average_pd),4),
              "observed_default_rate":round(float(r.observed_default_rate),4)}
              for r in summary.itertuples(index=False)]
    assert sum(r["applications"] for r in records)==len(test)
    result={"data":"Synthetic credit applications, never live bank data",
            "n_out_of_time":len(test),"validation_bins":5,
            "oot_roc_auc":binary_report(test.default_12m,p_oot)["roc_auc"],
            "risk_groups":records,
            "controls":["Only validation used to fit risk-band boundaries",
               "Chronological data split and no target leakage",
               "Real use would require independent validation, fairness and governance"],
            "disclaimer":"Illustrative PD segments, not approved lending decisions."}
    if output_dir:
        folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
        (folder/"business_brief.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        lines=["# Business brief: synthetic PD segmentation","",
            "**Synthetic demo, not a bank scorecard or a lending recommendation.**","",
            f"OOT applications: {len(test)}, logistic model ROC-AUC: {result['oot_roc_auc']}.","",
            "| Risk group | Applications | Average estimated PD | Observed default rate |",
            "|---:|---:|---:|---:|"]
        lines.extend(f"| {g['group']} | {g['applications']} | {g['average_pd']:.4f} | {g['observed_default_rate']:.4f} |" for g in records)
        lines.extend(["","Risk group boundaries learned **on validation**, never on OOT.",
            "Risk groups are descriptive; thresholds require policy and asymmetric cost approval.",
            "Model risk: vintage shift, calibration, independent validation and governance are necessary before production."])
        (folder/"business_brief.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result

if __name__=="__main__":
    print(json.dumps(create_business_brief(output_dir=Path(__file__).resolve().parents[2]/"reports"),indent=2))
