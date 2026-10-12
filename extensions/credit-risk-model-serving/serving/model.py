"""Deterministic scoring from approved versioned JSON artifacts; no remote model loading."""
import json, math
from pathlib import Path
MODEL_DIR=Path(__file__).resolve().parents[1]/'models'
VERSIONS={'1.0.0':'pd_logreg_v1.json','1.1.0':'pd_logreg_v2.json'}
FEATURES=['debt_to_income','missed_payments_12m','credit_history_months','utilization']
class PDModel:
 def __init__(self,path=None,version='1.0.0'):
  if path is None:
   if version not in VERSIONS:raise ValueError('Unsupported model version')
   path=MODEL_DIR/VERSIONS[version]
  a=json.loads(Path(path).read_text(encoding='utf-8'))
  if a['kind']!='logistic_regression' or a['trained_on']!='synthetic_only':raise ValueError('Wrong model')
  if a['features']!=FEATURES:raise ValueError('Schema mismatch')
  if not all(len(a[k])==4 for k in ['means','scales','coefficients']):raise ValueError('Bad dimensions')
  if not all(isinstance(x,(float,int)) and math.isfinite(x) for k in ('means','scales','coefficients') for x in a[k]):raise ValueError('Invalid numeric parameter')
  if not all(x>0 for x in a['scales']):raise ValueError('Bad scales')
  if not (0<a['review_threshold']<1):raise ValueError('Bad threshold')
  if a['version'] not in VERSIONS:raise ValueError('Unknown model version in artifact')
  if path==MODEL_DIR/VERSIONS.get(version) and a['version']!=version:raise ValueError('Model artifact version mismatch')
  self.info=a
 def predict(self,data):
  x=[data[n] for n in self.info['features']]
  z=self.info['intercept']+sum((v-m)/s*c for v,m,s,c in zip(x,self.info['means'],self.info['scales'],self.info['coefficients']))
  pd=1/(1+math.exp(-min(40,max(-40,z))))
  return pd,'review' if pd>=self.info['review_threshold'] else 'standard'
