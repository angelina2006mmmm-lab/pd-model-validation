"""Independent synthetic application and macroeconomic input tables."""
import numpy as np
import pandas as pd
FEATURES=["loan_to_income","debt_to_income","utilization","credit_history_months","late_payments_12m","macro_stress","product"]
NUMERIC=FEATURES[:-1]
def make_sources(n=4200,seed=42):
 rng=np.random.default_rng(seed)
 months=pd.period_range("2023-01","2025-12",freq="M")
 macro=pd.DataFrame({"application_month":[str(x) for x in months],"macro_stress":np.linspace(.22,.75,len(months))+rng.normal(0,.03,len(months))})
 idx=rng.integers(0,len(months),n)
 income=np.exp(rng.normal(np.log(108000),.49,n)).clip(25000,650000)
 loan=np.exp(rng.normal(np.log(400000),.72,n)).clip(20000,2500000)
 dti=rng.beta(2.2,4.6,n);util=rng.beta(2.3,2.8,n)
 history=rng.integers(3,210,n);late=np.minimum(rng.poisson(.64,n),7)
 product=rng.choice(["cash","consumer","auto"],size=n,p=[.44,.38,.18])
 stress=macro.macro_stress.to_numpy()[idx]
 logits=-4.15+2.2*dti+1.18*util+.55*late-.004*history+.35*(product=="cash")+.48*stress+.16*(loan/income>7)
 prob=1/(1+np.exp(-logits))
 loans=pd.DataFrame({"application_id":[f"DEMO-{i:06d}" for i in range(n)],
  "application_month":[str(months[i]) for i in idx],"loan_amount":loan.round(2),
  "monthly_income":income.round(2),"debt_to_income":dti.round(5),"utilization":util.round(5),
  "credit_history_months":history,"late_payments_12m":late,"product":product,
  "default_12m":rng.binomial(1,prob)})
 return loans,macro
def prepare_pandas(loans,macro):
 if loans.application_id.duplicated().any():raise ValueError("Duplicate applications")
 if macro.application_month.duplicated().any():raise ValueError("Duplicate monthly macro source")
 if loans[["loan_amount","monthly_income","application_month","default_12m"]].isna().any().any():raise ValueError("Missing source fields")
 if (loans.loan_amount<=0).any() or (loans.monthly_income<=0).any():raise ValueError("Invalid loan or income")
 if not set(loans.default_12m.unique()).issubset({0,1}):raise ValueError("Invalid target")
 out=loans.merge(macro,on="application_month",how="left",validate="many_to_one")
 if out.macro_stress.isna().any():raise ValueError("Missing reference month")
 out["loan_to_income"]=(out.loan_amount/out.monthly_income).round(5)
 out=out.sort_values(["application_month","application_id"]).reset_index(drop=True)
 if out[FEATURES].isna().any().any():raise ValueError("Missing features")
 return out
def chronological_split(df):
 tr=df[df.application_month<"2024-07"].copy()
 va=df[(df.application_month>="2024-07")&(df.application_month<"2025-01")].copy()
 te=df[df.application_month>="2025-01"].copy()
 if min(len(tr),len(va),len(te))<30:raise ValueError("Insufficient data")
 if set(tr.application_id)&set(te.application_id):raise ValueError("ID leakage")
 return tr,va,te
