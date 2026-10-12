"""Deterministic inference from a versioned, human-readable model artifact."""
import json,math
from pathlib import Path
MODEL=Path(__file__).resolve().parents[1]/'models/pd_logreg_v1.json'
class PDModel:
 def __init__(self,path=MODEL):
  a=json.loads(Path(path).read_text(encoding='utf-8'))
  if a['kind']!='logistic_regression' or a['trained_on']!='synthetic_only':raise ValueError('Wrong model')
  if a['features']!=['debt_to_income','missed_payments_12m','credit_history_months','utilization']:raise ValueError('Schema mismatch')
  if not all(len(a[k])==4 for k in ['means','scales','coefficients']):raise ValueError('Bad dimensions')
  if not all(x>0 for x in a['scales']):raise ValueError('Bad scales')
  self.info=a
 def predict(self,data):
  x=[data[n] for n in self.info['features']]
  z=self.info['intercept']+sum((v-m)/s*c for v,m,s,c in zip(x,self.info['means'],self.info['scales'],self.info['coefficients']))
  pd=1/(1+math.exp(-min(40,max(-40,z))))
  return pd,'review' if pd>=self.info['review_threshold'] else 'standard'
