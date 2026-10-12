import numpy as np
import pandas as pd
import pytest
from modelrisk.data import make_portfolio,validate_portfolio,split_vintages,FEATURES

def test_portfolio_reproducible():
    assert make_portfolio(900).equals(make_portfolio(900))
def test_quality_and_features():
    d=make_portfolio(900)
    assert validate_portfolio(d) and 'default_12m' not in FEATURES
    assert 'application_id' not in FEATURES and 'application_date' not in FEATURES
def test_duplicate_and_missing_rejected():
    d=make_portfolio(900)
    bad=d.copy();bad.loc[1,'application_id']=bad.loc[0,'application_id']
    with pytest.raises(ValueError,match="Duplicate"):validate_portfolio(bad)
    bad=d.copy();bad.loc[1,'dti']=np.nan
    with pytest.raises(ValueError):validate_portfolio(bad)
def test_time_split_and_disjoint():
    a,b,c=split_vintages(make_portfolio(1800))
    assert a.application_date.max()<b.application_date.min()<c.application_date.min()
    assert not(set(a.application_id)&set(b.application_id))
    assert not(set(c.application_id)&set(b.application_id))
def test_bad_small_time_segments():
    with pytest.raises(ValueError):split_vintages(make_portfolio(70))
