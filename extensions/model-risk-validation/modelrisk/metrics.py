"""Model risk statistics: discrimination, calibration, drift and cost proxies."""
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,brier_score_loss,log_loss,average_precision_score

def _checked(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if len(y)!=len(p) or len(y)==0 or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError("Invalid prediction arrays")
    if len(np.unique(y))<2:raise ValueError("Both target classes required")
    return y,p

def expected_calibration_error(y,p,bins=10):
    y,p=_checked(y,p);edges=np.linspace(0,1,bins+1)
    idx=np.minimum(np.searchsorted(edges,p,side="right")-1,bins-1)
    return float(sum(np.sum(idx==i)/len(p)*abs(y[idx==i].mean()-p[idx==i].mean())
      for i in range(bins) if np.any(idx==i)))

def population_stability_index(reference,current,bins=10,eps=1e-6):
    a=np.asarray(reference,dtype=float);b=np.asarray(current,dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid PSI input")
    edges=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
    if len(edges)<3:return 0.0
    edges[0]=-np.inf;edges[-1]=np.inf
    qa=np.histogram(a,edges)[0]/len(a);qb=np.histogram(b,edges)[0]/len(b)
    return float(np.sum((qb-qa)*np.log((qb+eps)/(qa+eps))))

def classification_metrics(y,p):
    y,p=_checked(y,p)
    fpr,tpr,_=roc_curve(y,p)
    auc=float(roc_auc_score(y,p))
    return {
     "n":len(y),"event_rate":round(float(y.mean()),5),
     "roc_auc":round(auc,5),"gini":round(2*auc-1,5),
     "ks":round(float(np.max(tpr-fpr)),5),
     "pr_auc":round(float(average_precision_score(y,p)),5),
     "brier":round(float(brier_score_loss(y,p)),5),
     "log_loss":round(float(log_loss(y,p,labels=[0,1])),5),
     "ece":round(expected_calibration_error(y,p),5)}

def select_threshold(y,p,cost_missed=5.,cost_review=1.):
    y,p=_checked(y,p)
    t=np.linspace(.02,.75,74)
    costs=[float(np.mean(cost_missed*((y==1)&(p<v))+
          cost_review*((y==0)&(p>=v)))) for v in t]
    return round(float(t[int(np.argmin(costs))]),4)

def decision_proxy(y,p,threshold,cost_missed=5.,cost_review=1.):
    y,p=_checked(y,p)
    return round(float(np.mean(cost_missed*((y==1)&(p<threshold))+
        cost_review*((y==0)&(p>=threshold)))),5)

def expected_loss_proxy(pd_scores,ead,lgd=.45):
    p=np.asarray(pd_scores,dtype=float);amt=np.asarray(ead,dtype=float)
    if len(p)!=len(amt) or np.any(amt<0) or not np.isfinite(amt).all():
        raise ValueError("Invalid EAD")
    if not 0<=lgd<=1:raise ValueError("Invalid LGD")
    return round(float(np.sum(p*amt*lgd)),2)
