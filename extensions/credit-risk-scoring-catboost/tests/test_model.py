import numpy as np
from creditrisk.data import create_applications,split_time,FEATURES,assert_quality
from creditrisk.metrics import psi,ks_score,binary_report,choose_threshold,demo_cost
from creditrisk.experiment import run_experiment
def test_data_reproducibility():
    assert create_applications(350).equals(create_applications(350))
def test_no_leakage():
    a,b,c=split_time(create_applications(600))
    assert "default_12m" not in FEATURES and "application_date" not in FEATURES and "application_id" not in FEATURES
    assert a.application_date.max()<b.application_date.min()<c.application_date.min()
    assert set(a.application_id).isdisjoint(set(c.application_id))
def test_synthetic_data_quality():
    assert_quality(create_applications(300))
def test_psi_same_sample():
    x=np.linspace(.01,.99,100);assert abs(psi(x,x))<1e-12
def test_psi_detects_shift():
    x=np.linspace(.01,.95,100);assert psi(x,x+.04)>0
def test_ks_perfect_separation():
    assert ks_score([0,0,1,1],[.1,.2,.8,.9])==1
def test_auc_perfect_separation():
    assert binary_report([0,0,1,1],[.1,.2,.8,.9])["roc_auc"]==1
def test_validation_threshold():
    assert .05<=choose_threshold([0,0,1,1],[.2,.6,.7,.9])<=.8
def test_cost_nonnegative():
    assert demo_cost([1,0],[.7,.1],.5)>=0
def test_train_models_on_oot(tmp_path):
    r=run_experiment(output_dir=tmp_path,n=1200,iterations=35)
    assert r["selected_model"] in {"LogisticRegression","CatBoost"}
    assert r["test"]["roc_auc"]>.5
    assert sum(r["records"].values())==1200
    assert (tmp_path/"metrics.json").exists() and (tmp_path/"report.md").exists()
    assert r["hypothetical_cost"]["oot_avg_cost"]>=0
