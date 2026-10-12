import json,logging
from fastapi.testclient import TestClient
from serving.api import app,model
from serving.model import PDModel
from training.train_export import train_export,generate
client=TestClient(app)
VALID={'application_id':'SYN-009','debt_to_income':.36,'missed_payments_12m':1,
       'credit_history_months':38,'utilization':.59}
def test_health_and_openapi():
 assert client.get('/health').json()['status']=='ok'
 assert '/v1/score' in client.get('/openapi.json').json()['paths']
def test_score_and_version():
 r=client.post('/v1/score',json=VALID)
 assert r.status_code==200 and r.headers['X-Request-ID']
 a=r.json();assert a['model_version']=='1.0.0' and 0<a['default_probability']<1
 assert a['risk_band'] in ['review','standard']
def test_determinism():
 assert client.post('/v1/score',json=VALID).json()==client.post('/v1/score',json=VALID).json()
def test_validation_rejects_bad_payloads():
 for bad in [{**VALID,'debt_to_income':-1},{**VALID,'utilization':2},{**VALID,'missed_payments_12m':-1},
     {**VALID,'credit_history_months':1000},{**VALID,'extra':'not allowed'},{**VALID,'application_id':'!'}]:
  assert client.post('/v1/score',json=bad).status_code==422
def test_missing_required_value():
 d=dict(VALID);d.pop('utilization')
 assert client.post('/v1/score',json=d).status_code==422
def test_nan_rejected():
 txt=json.dumps(VALID).replace('0.36','NaN')
 assert client.post('/v1/score',content=txt,headers={'content-type':'application/json'}).status_code==422
def test_model_metadata():
 assert client.get('/v1/model').json()['training_data']=='synthetic_only'
def test_monotonic_overdue():
 assert model.predict({**VALID,'missed_payments_12m':12})[0]>model.predict(VALID)[0]
def test_metrics_no_personal_input():
 client.post('/v1/score',json={**VALID,'application_id':'SECRET-ID-777'})
 t=client.get('/metrics').text
 assert 'SECRET-ID-777' not in t and 'debt_to_income' not in t
def test_json_logs_no_raw_payload(caplog):
 with caplog.at_level(logging.INFO,logger='risk_model_serving'):
  client.post('/v1/score',json={**VALID,'application_id':'SECRET-ID-888'})
 record=caplog.records[-1].message
 assert 'SECRET-ID-888' not in record
 assert json.loads(record)['model_version']=='1.0.0'
def test_bad_request_id_sanitized():
 r=client.post('/v1/score',json=VALID,headers={'X-Request-ID':'illegal space'})
 assert ' ' not in r.headers['X-Request-ID']
def test_export_reproducible(tmp_path):
 a=train_export(tmp_path/'a.json');b=train_export(tmp_path/'b.json')
 assert a==b and a['test_roc_auc']>.5
def test_checked_in_export_matches(tmp_path):
 assert_artifact_equivalent(train_export(tmp_path/'a.json'),PDModel().info)
def test_data_reproducibility():
 a,y=generate(20,28);b,z=generate(20,28)
 assert (a==b).all() and (y==z).all()
def test_bad_model_contract(tmp_path):
 import pytest
 a=train_export(tmp_path/'a.json');a['features'].append('bad')
 x=tmp_path/'bad.json';x.write_text(json.dumps(a))
 with pytest.raises(ValueError):PDModel(x)


def test_v2_artifact_reproducible(tmp_path):
 a=train_export(tmp_path/'v2.json',version='1.1.0')
 assert_artifact_equivalent(a,PDModel(version='1.1.0').info
 assert a['version']=='1.1.0' and a['test_roc_auc']>.5
 assert a['test_roc_auc']==PDModel(version='1.1.0').info['test_roc_auc']

def test_v1_v2_prediction_and_rollback():
 original=PDModel(version='1.0.0')
 candidate=PDModel(version='1.1.0')
 assert original.info['version']!=candidate.info['version']
 pd1,_=original.predict(VALID)
 pd2,_=candidate.predict(VALID)
 assert abs(pd1-pd2)>1e-7
 rolled_back=PDModel(version='1.0.0')
 assert rolled_back.predict(VALID)==original.predict(VALID)

def test_api_deployment_switch_then_rollback(monkeypatch):
 import serving.api as service
 with monkeypatch.context() as m:
  m.setattr(service,'model',PDModel(version='1.1.0'))
  active=client.post('/v1/score',json=VALID).json()
  assert active['model_version']=='1.1.0'
  assert client.get('/v1/model').json()['model_version']=='1.1.0'
 with monkeypatch.context() as m:
  m.setattr(service,'model',PDModel(version='1.0.0'))
  restored=client.post('/v1/score',json=VALID).json()
  assert restored['model_version']=='1.0.0'
  assert restored['default_probability']!=active['default_probability']

def test_unknown_version_rejected():
 import pytest
 with pytest.raises(ValueError,match='Unsupported model version'):
  PDModel(version='../../unexpected')

def test_model_endpoint_says_test_not_oot():
 meta=client.get('/v1/model').json()
 assert 'test_roc_auc' in meta and 'oot_roc_auc' not in meta

def assert_artifact_equivalent(actual,expected):
    """Sklearn minor releases can change coefficients by ~1e-16."""
    import math
    for k in ('features','samples','version','kind','trained_on'):
        assert actual[k]==expected[k]
    for k in ('coefficients','means','scales'):
        assert len(actual[k])==len(expected[k])
        for a,b in zip(actual[k],expected[k]):
            assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    for k in ('intercept','review_threshold','test_roc_auc'):
        assert math.isclose(actual[k],expected[k],rel_tol=1e-10,abs_tol=1e-8)
