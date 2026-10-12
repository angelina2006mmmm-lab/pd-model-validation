import numpy as np
import pytest
from joblib import load
from pipeline.data import FEATURES,make_sources,prepare_pandas,chronological_split
from pipeline.experiment import run
def test_sources_reproducible():
 a,b=make_sources(350,seed=42);c,d=make_sources(350,seed=42)
 assert a.equals(c) and b.equals(d)
def test_join_and_quality():
 x=prepare_pandas(*make_sources(400,seed=8))
 assert len(x)==400 and not x.macro_stress.isna().any()
 assert np.allclose(x.loan_to_income,x.loan_amount/x.monthly_income,atol=1e-5)
def test_duplicate_rejection():
 import pandas as pd
 a,b=make_sources(300)
 with pytest.raises(ValueError,match="Duplicate applications"):
  prepare_pandas(pd.concat([a,a.iloc[[0]]]),b)
def test_missing_reference_month_rejection():
 a,b=make_sources(300)
 with pytest.raises(ValueError,match="Missing reference"):
  prepare_pandas(a,b.iloc[:1])
def test_chronological_oot():
 tr,va,te=chronological_split(prepare_pandas(*make_sources(900)))
 assert tr.application_month.max()<va.application_month.min()<te.application_month.min()
 assert "default_12m" not in FEATURES and "application_id" not in FEATURES
 assert not set(tr.application_id)&set(te.application_id)
def test_training_exports(tmp_path):
 r,m=run(output_dir=tmp_path,n=1300)
 assert r["oot_metrics"]["roc_auc"]>.5
 assert r["selected_model"] in {"LogisticRegression","HistGradientBoosting"}
 assert sum(r[k] for k in ("n_train","n_validation","n_oot"))==1300
 assert (tmp_path/"model_card.md").exists() and (tmp_path/"metrics.json").exists()
 assert np.allclose(m.predict_proba(prepare_pandas(*make_sources(1300))[FEATURES]),
  load(tmp_path/"model.joblib").predict_proba(prepare_pandas(*make_sources(1300))[FEATURES]))
def test_selection_does_not_use_oot():
 r,_=run(n=1200)
 assert len(r["validation"])==2
 assert r["excluded_from_features"]==["application_id","application_month","default_12m"]
def test_reproducible_report():
 r1,_=run(n=1300);r2,_=run(n=1300)
 assert r1==r2
