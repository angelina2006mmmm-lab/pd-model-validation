"""End-to-end model comparison and versionable serving artifacts."""
import json
from pathlib import Path
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from .data import FEATURES,NUMERIC,make_sources,prepare_pandas,chronological_split
def build_models():
 prep=ColumnTransformer([("num",StandardScaler(),NUMERIC),
  ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["product"])])
 return {
  "LogisticRegression":Pipeline([("prep",prep),("est",LogisticRegression(max_iter=500,random_state=42))]),
  "HistGradientBoosting":Pipeline([("prep",prep),("est",HistGradientBoostingClassifier(max_iter=95,max_leaf_nodes=13,learning_rate=.05,random_state=42))])
 }
def run(output_dir=None,n=4200,seed=42):
 df=prepare_pandas(*make_sources(n,seed))
 tr,va,te=chronological_split(df);models=build_models();metrics={}
 for name,m in models.items():
  m.fit(tr[FEATURES],tr.default_12m)
  p=m.predict_proba(va[FEATURES])[:,1]
  metrics[name]={"validation_roc_auc":round(float(roc_auc_score(va.default_12m,p)),5)}
 winner=max(metrics,key=lambda key:metrics[key]["validation_roc_auc"])
 # This is the first and only access to out-of-time labels for evaluation.
 p=models[winner].predict_proba(te[FEATURES])[:,1]
 report={"dataset":"synthetic applications joined with synthetic monthly macro conditions",
  "n_train":len(tr),"n_validation":len(va),"n_oot":len(te),
  "time_split":{"train":"2023-01..2024-06","validation":"2024-07..2024-12","oot":"2025-01..2025-12"},
  "features":FEATURES,"excluded_from_features":["application_id","application_month","default_12m"],
  "validation":metrics,"selected_model":winner,
  "oot_metrics":{"roc_auc":round(float(roc_auc_score(te.default_12m,p)),5),
   "pr_auc":round(float(average_precision_score(te.default_12m,p)),5),
   "brier":round(float(brier_score_loss(te.default_12m,p)),5),
   "log_loss":round(float(log_loss(te.default_12m,p)),5)},
  "inference_contract":{"input_fields":FEATURES,"output":"12-month default probability","threshold":"business policy not defined"},
  "limitations":"Synthetic experiment; no actual credit decisions, revenues or compliance claims"}
 if output_dir:
  out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
  (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
  dump(models[winner],out/"model.joblib")
  (out/"model_card.md").write_text(model_card(report),encoding="utf-8")
 return report,models[winner]
def model_card(r):
 m=r["oot_metrics"]
 return (f"# Retail Credit ML Model Card\n\n**Synthetic demonstration only.**\n\n"
         f"Training/validation/OOT: {r['n_train']}/{r['n_validation']}/{r['n_oot']}.\n\n"
         f"Selected on validation: {r['selected_model']}. OOT ROC-AUC {m['roc_auc']}, "
         f"PR-AUC {m['pr_auc']}, Brier {m['brier']}.\n\n"
         f"Input features: {', '.join(FEATURES)}. No application ID, date, or target as features.\n\n"
         "Model selection did not use OOT data. No regulatory validation, production deployment "
         "or measured bank business impact. Threshold must be defined with stakeholders.\n")
if __name__=="__main__":
 print(json.dumps(run(Path(__file__).resolve().parents[1]/"reports")[0],indent=2,ensure_ascii=False))
