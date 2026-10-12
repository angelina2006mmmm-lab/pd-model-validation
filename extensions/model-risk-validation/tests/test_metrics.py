import numpy as np
import pytest
from modelrisk.metrics import (classification_metrics,expected_calibration_error,
 population_stability_index,select_threshold,decision_proxy,expected_loss_proxy)

def test_perfect_auc_ks_gini():
    r=classification_metrics([0,0,1,1],[.01,.1,.9,.99])
    assert r['roc_auc']==1 and r['ks']==1 and r['gini']==1
def test_psi_identical_zero():
    a=np.linspace(.01,.99,100)
    assert population_stability_index(a,a)==0
def test_psi_shift_detected():
    a=np.linspace(.01,.99,100)
    assert population_stability_index(a,a+.10)>0
def test_calibration_ideal_two_bins():
    assert expected_calibration_error([0,1],[0,1])==0
def test_bad_probabilities_rejected():
    with pytest.raises(ValueError):classification_metrics([0,1],[.2,float("nan")])
    with pytest.raises(ValueError):classification_metrics([0,0],[.2,.3])
def test_decision_threshold_and_cost():
    y=[0,0,1,1];p=[.1,.2,.6,.8]
    t=select_threshold(y,p)
    assert .02<=t<=.75 and decision_proxy(y,p,t)>=0
def test_expected_loss_proxy():
    assert expected_loss_proxy([.1,.2],[100,200],.5)==25
    with pytest.raises(ValueError):expected_loss_proxy([.1],[100],1.4)
