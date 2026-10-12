"""Synthetic credit portfolio with chronological vintages; never actual borrower data."""
import numpy as np
import pandas as pd

NUMERIC=["loan_amount","monthly_income","term_months","dti","utilization",
         "credit_history_months","late_payments_12m","macro_stress"]
CATEGORICAL=["product_type"]
FEATURES=NUMERIC+CATEGORICAL

def make_portfolio(n=4800,seed=1729):
    rng=np.random.default_rng(seed)
    dates=pd.Timestamp("2023-01-01")+pd.to_timedelta(
        rng.integers(0,1096,n),unit="D")
    income=np.exp(rng.normal(np.log(105000),.48,n)).clip(22000,750000)
    amount=np.exp(rng.normal(np.log(400000),.72,n)).clip(20000,3800000)
    term=rng.choice([12,24,36,48,60],n)
    dti=rng.beta(2.5,4,n)*.92
    util=rng.beta(2.1,2.9,n)
    history=rng.integers(3,210,n)
    past_due=rng.poisson(.65,n).clip(0,7)
    product=rng.choice(["cash","auto","consumer"],n,p=[.45,.20,.35])
    elapsed=(dates-pd.Timestamp("2023-01-01")).days.to_numpy()/1096
    macro=.20+.65*elapsed+rng.normal(0,.06,n)
    logits=(-4.1+2.7*dti+1.15*util+.52*past_due-.0065*history
           +.40*(term>=48)+.35*(product=="cash")+.58*macro)
    pd_true=1/(1+np.exp(-logits))
    y=rng.binomial(1,pd_true)
    frame=pd.DataFrame({
      "application_id":[f"SYN-{i:06d}" for i in range(n)],
      "application_date":dates,"loan_amount":amount.round(2),
      "monthly_income":income.round(2),"term_months":term,
      "dti":dti.round(5),"utilization":util.round(5),
      "credit_history_months":history,
      "late_payments_12m":past_due,"macro_stress":macro.round(5),
      "product_type":product,"default_12m":y})
    return frame.sort_values(["application_date","application_id"]).reset_index(drop=True)

def validate_portfolio(df):
    required=FEATURES+["application_date","application_id","default_12m"]
    if df.empty or df[required].isna().any().any():
        raise ValueError("Missing values or empty sample")
    if not df.application_id.is_unique:
        raise ValueError("Duplicate applications")
    if not set(df.default_12m.unique()).issubset({0,1}):
        raise ValueError("Target must be binary")
    if not df.dti.between(0,1).all() or not df.utilization.between(0,1).all():
        raise ValueError("Invalid utilization or DTI")
    if not (df.loan_amount>0).all() or not (df.monthly_income>0).all():
        raise ValueError("Invalid amount")
    return True

def split_vintages(df):
    dates=pd.to_datetime(df.application_date)
    train=df.loc[dates<"2024-07-01"].copy()
    valid=df.loc[(dates>="2024-07-01")&(dates<"2025-01-01")].copy()
    oot=df.loc[dates>="2025-01-01"].copy()
    if min(len(train),len(valid),len(oot))<80:
        raise ValueError("Too few samples per chronological segment")
    if not (train.application_date.max()<valid.application_date.min()
            and valid.application_date.max()<oot.application_date.min()):
        raise ValueError("Chronological leakage")
    return train,valid,oot
