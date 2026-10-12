"""Reproducible credit underwriting PoC: baseline vs CatBoost and untouched OOT test."""
from pathlib import Path
import json
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from .data import FEATURES,CAT_FEATURES,NUM_FEATURES,create_applications,split_time,assert_quality
from .metrics import binary_report,psi,choose_threshold,demo_cost
def run_experiment(output_dir=None,n=4200,seed=42,iterations=170):
    df=create_applications(n=n,seed=seed)
    assert_quality(df)
    train,val,test=split_time(df)
    X_train=train[FEATURES];X_val=val[FEATURES];X_test=test[FEATURES]
    y_train=train.default_12m;y_val=val.default_12m;y_test=test.default_12m
    baseline=make_pipeline(ColumnTransformer([
       ("num",StandardScaler(),NUM_FEATURES),
       ("cat",OneHotEncoder(handle_unknown="ignore"),CAT_FEATURES)]),
       LogisticRegression(max_iter=600,random_state=seed))
    baseline.fit(X_train,y_train)
    boosted=CatBoostClassifier(iterations=iterations,depth=5,learning_rate=.075,
        loss_function="Logloss",eval_metric="AUC",random_seed=seed,
        verbose=False,thread_count=2,allow_writing_files=False)
    boosted.fit(X_train,y_train,cat_features=CAT_FEATURES,
                eval_set=(X_val,y_val),early_stopping_rounds=40)
    candidates={"LogisticRegression":baseline,"CatBoost":boosted}
    validation_scores={name:binary_report(y_val,m.predict_proba(X_val)[:,1])
                       for name,m in candidates.items()}
    chosen=max(validation_scores,key=lambda name:validation_scores[name]["roc_auc"])
    model=candidates[chosen]
    p_val=model.predict_proba(X_val)[:,1]
    threshold=choose_threshold(y_val,p_val)
    p_train=model.predict_proba(X_train)[:,1]
    p_test=model.predict_proba(X_test)[:,1]
    importances=dict(sorted(zip(FEATURES,boosted.feature_importances_.tolist()),key=lambda kv:-kv[1]))
    out={
      "experiment":"synthetic 12-month loan default modeling, not real bank data",
      "seed":seed,"records":{"train":len(train),"validation":len(val),"out_of_time_test":len(test)},
      "temporal_split":{"train":"2023-01 to 2024-06","validation":"2024-07 to 2024-12",
                       "out_of_time_test":"2025-01 to 2025-12"},
      "features":FEATURES,
      "model_selection_validation_auc":{k:v["roc_auc"] for k,v in validation_scores.items()},
      "selected_model":chosen,"validation_threshold":round(float(threshold),3),
      "test":binary_report(y_test,p_test,threshold),
      "other_model_test":{k:binary_report(y_test,m.predict_proba(X_test)[:,1],threshold)
                          for k,m in candidates.items() if k!=chosen},
      "stability_psi_train_vs_oot":round(psi(p_train,p_test),4),
      "hypothetical_cost":{"false_negative":5,"false_positive":1,
                           "oot_avg_cost":demo_cost(y_test,p_test,threshold),
                           "note":"Illustrative asymmetric errors, not real financial economics"},
      "catboost_feature_importance":{k:round(v,2) for k,v in importances.items()},
      "leakage_control":"Target, ID and date excluded; validation controls selection; test never used for tuning."
    }
    if output_dir:
        folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
        (folder/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
        (folder/"report.md").write_text(make_report(out),encoding="utf-8")
    return out
def make_report(out):
    m=out["test"]
    return (
      "# Synthetic Credit Risk Scoring - Model Validation\n\n"
      "**Educational synthetic data only.**\n\n"
      f"- Chronological train/validation/OOT rows: {out['records']['train']} / {out['records']['validation']} / {out['records']['out_of_time_test']}.\n"
      f"- Selected {out['selected_model']} by validation ROC-AUC without tuning on OOT test.\n"
      f"- OOT ROC-AUC {m['roc_auc']}; Gini {m['gini']}; KS {m['ks']}; Brier {m['brier']}; PR-AUC {m['pr_auc']}.\n"
      f"- Validation threshold {out['validation_threshold']}; OOT PSI {out['stability_psi_train_vs_oot']}.\n"
      "- Decision cost is hypothetical, not banking revenue or approved risk appetite.\n"
      "- Next steps: genuine portfolio data, governance, model monitoring, independent validation and fairness/shift review.\n"
    )
if __name__=="__main__":
    out=run_experiment(output_dir=Path(__file__).resolve().parents[2]/"reports")
    print(json.dumps({"selected_model":out["selected_model"],"test":out["test"],
                      "psi":out["stability_psi_train_vs_oot"]},indent=2))
