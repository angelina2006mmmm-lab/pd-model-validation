"""Reproducible synthetic PD benchmarking: validation selects; OOT evaluates."""
import json
from pathlib import Path
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .data import CAT_FEATURES,NUM_FEATURES,FEATURES,create_applications,split_time,assert_quality
from .metrics import binary_report,psi,choose_threshold,demo_cost

def build_models(seed=42,rounds=180):
 def prep():
  return ColumnTransformer([("num",StandardScaler(),NUM_FEATURES),
      ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CAT_FEATURES)],sparse_threshold=0)
 return {
  "LogisticRegression":Pipeline([("features",prep()),("model",LogisticRegression(max_iter=600,random_state=seed))]),
  "XGBoost":Pipeline([("features",prep()),("model",XGBClassifier(n_estimators=rounds,max_depth=3,
      learning_rate=.045,subsample=.9,colsample_bytree=.9,eval_metric="logloss",
      tree_method="hist",random_state=seed,n_jobs=2))]),
  "LightGBM":Pipeline([("features",prep()),("model",LGBMClassifier(n_estimators=rounds,
      max_depth=3,num_leaves=7,learning_rate=.045,min_child_samples=25,subsample=.9,
      colsample_bytree=.9,random_state=seed,n_jobs=2,verbosity=-1))])
 }

def run_benchmark(n=4200,seed=42,rounds=180,output_dir=None):
 frame=create_applications(n=n,seed=seed);assert_quality(frame)
 train,val,oot=split_time(frame);models=build_models(seed,rounds)
 result={"data":"Synthetic loan applications; not real banking data","seed":seed,
         "rows":{"train":len(train),"validation":len(val),"oot":len(oot)},
         "selection_rule":"highest validation ROC-AUC; alphabetical tie",
         "no_leakage":"Only FEATURES; target, date and ID excluded","validation":{},"oot":{}}
 for name,model in models.items():
  model.fit(train[FEATURES],train["default_12m"])
  result["validation"][name]=binary_report(val["default_12m"],model.predict_proba(val[FEATURES])[:,1])
 winner=sorted(models,key=lambda name:(-result["validation"][name]["roc_auc"],name))[0]
 result["selected_model"]=winner
 for name,model in models.items():
  p_val=model.predict_proba(val[FEATURES])[:,1]
  threshold=choose_threshold(val["default_12m"],p_val)
  p_train=model.predict_proba(train[FEATURES])[:,1]
  p_oot=model.predict_proba(oot[FEATURES])[:,1]
  result["oot"][name]={**binary_report(oot["default_12m"],p_oot,threshold),
      "psi":round(psi(p_train,p_oot),4),
      "illustrative_cost_5_1":demo_cost(oot["default_12m"],p_oot,threshold)}
 if output_dir is not None:
  folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
  (folder/"boosting_comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
  lines=["# Synthetic credit risk: Logistic Regression vs XGBoost vs LightGBM","",
   "Synthetic credit applications, chronological split; not real banking performance.",
   f"Selected on validation: {winner}. OOT not used for model selection.","",
   "| Model | Validation ROC-AUC | OOT ROC-AUC | OOT Brier | OOT PSI |",
   "|---|---:|---:|---:|---:|"]
  for name in models:
   a=result["validation"][name];b=result["oot"][name]
   lines.append(f"| {name} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {b['brier']:.4f} | {b['psi']:.4f} |")
  lines+=["","Error costs are hypothetical, not real business profits."]
  (folder/"boosting_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 return result
if __name__=="__main__":
 out=run_benchmark(output_dir=Path(__file__).resolve().parents[2]/"reports")
 print(json.dumps({"selected":out["selected_model"],"validation_auc":{k:v["roc_auc"] for k,v in out["validation"].items()},
                   "oot_auc":{k:v["roc_auc"] for k,v in out["oot"].items()}},indent=2))
