"""Reproducible synthetic loan applications without protected attributes or target leakage."""
import numpy as np
import pandas as pd
CAT_FEATURES=["product_type","employment_type"]
NUM_FEATURES=["loan_amount","term_months","monthly_income","dti","utilization","credit_history_months","late_payments_12m","macro_stress"]
FEATURES=NUM_FEATURES+CAT_FEATURES
def create_applications(n=4200,seed=42):
    rng=np.random.default_rng(seed)
    dates=pd.to_datetime("2023-01-01")+pd.to_timedelta(rng.integers(0,1095,n),unit="D")
    income=np.exp(rng.normal(np.log(105_000),.52,n)).round(0).clip(22_000,800_000)
    loan=np.exp(rng.normal(np.log(420_000),.83,n)).round(0).clip(20_000,4_000_000)
    term=rng.choice([12,24,36,48,60],n,p=[.13,.24,.30,.21,.12])
    dti=rng.beta(2.2,4.5,n)*.95
    utilization=rng.beta(2.2,2.9,n)
    history=rng.integers(2,210,n)
    late=np.minimum(rng.poisson(.55,n),6)
    product=rng.choice(["cash","consumer","auto"],n,p=[.46,.36,.18])
    employment=rng.choice(["salary","self_employed","other"],n,p=[.71,.20,.09])
    elapsed=(dates-pd.Timestamp("2023-01-01")).days.to_numpy()/1095
    macro=.3+.65*elapsed+rng.normal(0,.08,n)
    logits=(-3.95+2.5*dti+1.18*utilization+.61*late-.007*history+
        .48*(term>=48)+.43*(employment=="self_employed")+
        .40*(product=="cash")+.50*macro)
    p=1/(1+np.exp(-logits))
    y=rng.binomial(1,p)
    df=pd.DataFrame({
       "application_id":[f"SYN-{i:06d}" for i in range(n)],
       "application_date":dates,
       "loan_amount":loan,"term_months":term,"monthly_income":income,
       "dti":dti.round(5),"utilization":utilization.round(5),
       "credit_history_months":history,"late_payments_12m":late,
       "macro_stress":macro.round(5),"product_type":product,
       "employment_type":employment,"default_12m":y})
    return df.sort_values(["application_date","application_id"]).reset_index(drop=True)
def split_time(df):
    dates=pd.to_datetime(df["application_date"])
    train=df.loc[dates<"2024-07-01"].copy()
    val=df.loc[(dates>="2024-07-01")&(dates<"2025-01-01")].copy()
    test=df.loc[dates>="2025-01-01"].copy()
    for name,frame in [("train",train),("validation",val),("test",test)]:
        if len(frame)<30:raise ValueError(f"Not enough {name} rows")
    return train,val,test
def assert_quality(df):
    assert len(df)>0 and df["application_id"].is_unique
    assert df[FEATURES+["default_12m"]].notna().all().all()
    assert set(df["default_12m"].unique()).issubset({0,1})
    assert (df["loan_amount"]>0).all() and (df["monthly_income"]>0).all()
    assert df["dti"].between(0,1).all() and df["utilization"].between(0,1).all()
