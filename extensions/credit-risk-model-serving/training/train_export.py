"""Training a synthetic PD estimator and exporting a versioned plain JSON model."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
FEATURES=['debt_to_income','missed_payments_12m','credit_history_months','utilization']

def generate(n=3600,seed=28):
 r=np.random.default_rng(seed)
 dti=r.beta(2.4,4.4,n)*.95
 missed=r.poisson(.65,n).clip(max=8)
 history=r.integers(2,240,n)
 utilization=r.beta(2.5,2.7,n)
 x=np.column_stack([dti,missed,history,utilization])
 p=1/(1+np.exp(-(-3.6+2.2*dti+.56*missed-.006*history+1.4*utilization)))
 return x,r.binomial(1,p)

def train_export(output=None, version='1.0.0'):
 if version not in {'1.0.0', '1.1.0'}:
  raise ValueError('Unsupported model version')
 x,y=generate()
 tx,vx,ex=x[:2400],x[2400:3000],x[3000:]
 ty,vy,ey=y[:2400],y[2400:3000],y[3000:]
 scaler=StandardScaler().fit(tx)
 m=LogisticRegression(max_iter=500,random_state=28,C=(1.0 if version=='1.0.0' else 0.5)).fit(scaler.transform(tx),ty)
 pv=m.predict_proba(scaler.transform(vx))[:,1]
 # Validation only: illustrative five-to-one missed-risk versus false-alert costs.
 cuts=np.arange(.05,.501,.01)
 cutoff=min(cuts,key=lambda t:float(np.mean(5*(vy==1)*(pv<t)+(vy==0)*(pv>=t))))
 result={'version':version,'kind':'logistic_regression','trained_on':'synthetic_only',
  'features':FEATURES.copy(),'means':scaler.mean_.tolist(),'scales':scaler.scale_.tolist(),
  'coefficients':m.coef_[0].tolist(),'intercept':float(m.intercept_[0]),
  'review_threshold':round(float(cutoff),4),
  'samples':{'train':2400,'validation':600,'test':600},
  'test_roc_auc':round(float(roc_auc_score(ey,m.predict_proba(scaler.transform(ex))[:,1])),4)}
 if output is None:output=Path(__file__).parents[1]/('models/pd_logreg_v1.json' if version=='1.0.0' else 'models/pd_logreg_v2.json')
 Path(output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 return result
if __name__=='__main__':
 print(json.dumps({'v1':train_export(version='1.0.0'),'v2':train_export(version='1.1.0')},indent=2))
