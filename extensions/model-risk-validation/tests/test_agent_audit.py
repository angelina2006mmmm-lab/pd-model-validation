from modelrisk.agent_audit import run_audit,validate_trace
def test_mock_agent_audit():
    r=run_audit()
    assert r['cases']==8 and r['passed']==3 and r['failed']==5
def test_missing_citation_detection():
    r=validate_trace({'id':'x','status':'grounded','tools':['search_policy'],'citations':[]})
    assert 'missing_citation' in r['issues']
def test_disallowed_tool_detection():
    r=validate_trace({'id':'x','status':'abstain','tools':['send_email'],'citations':[]})
    assert 'unapproved_tool' in r['issues']
def test_stale_policy_detection():
    r=validate_trace({'id':'x','status':'grounded','tools':[],
                      'citations':[{'id':'RISK-01','version':1}]})
    assert 'stale_policy_version' in r['issues']
