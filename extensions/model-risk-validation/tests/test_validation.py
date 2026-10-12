import json
import numpy as np
from modelrisk.validation import (run_validation,build_models,implementation_incident,monthly_monitor)
from modelrisk.data import make_portfolio,split_vintages,FEATURES

def test_challenger_oot_and_reports(tmp_path):
    r=run_validation(n=1800,output_dir=tmp_path)
    assert sum(r['sample_counts'].values())==1800
    assert r['selected_model'] in {'LogisticRegression','HistGradientBoosting'}
    assert set(r['oot_models'])==set(r['validation_models'])
    assert r['oot_models'][r['selected_model']]['roc_auc']>.5
    assert r['agent_trace_audit']['failed']==5
    assert r['implementation_consistency']['mismatches']>0
    assert len(r['monthly_oot_monitoring'])>=4
    assert (tmp_path/'validation_results.json').exists()
    assert (tmp_path/'independent_validation_report.md').exists()
    assert json.loads((tmp_path/'validation_results.json').read_text())['selected_model']==r['selected_model']
def test_implementation_contract_detects_wrong_feature_mapping():
    a,b,c=split_vintages(make_portfolio(1500))
    model=build_models()['LogisticRegression'].fit(a[FEATURES],a.default_12m)
    detected=implementation_incident(model,c.iloc[:40])
    assert detected['mismatches']>0 and detected['severity']=='incident'
def test_stress_scenario_is_explicit_and_numerical():
    r=run_validation(n=1800)
    s=r['stress_scenario']
    assert s['macro_shock']==.25
    assert s['expected_loss_proxy_baseline']>=0
    assert s['expected_loss_proxy_shocked']>=0
