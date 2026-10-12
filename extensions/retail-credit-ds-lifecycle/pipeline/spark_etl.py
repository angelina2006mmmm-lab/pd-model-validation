"""Actual PySpark ETL; verifies parity with the Pandas feature pipeline.

Install optional requirements-spark.txt and Java 17/21.
"""
from .data import make_sources,prepare_pandas
def prepare_spark(loans,macro,spark):
 from pyspark.sql import functions as F
 left=spark.createDataFrame(loans)
 right=spark.createDataFrame(macro)
 if left.count()!=left.select("application_id").distinct().count():raise ValueError("Duplicate applications")
 if right.count()!=right.select("application_month").distinct().count():raise ValueError("Duplicate macro months")
 if left.filter((F.col("monthly_income")<=0)|(F.col("loan_amount")<=0)).limit(1).count():raise ValueError("Invalid inputs")
 if left.filter(F.col("application_month").isNull()).limit(1).count():raise ValueError("Null month")
 joined=left.join(F.broadcast(right),on="application_month",how="left").withColumn(
  "loan_to_income",F.round(F.col("loan_amount")/F.col("monthly_income"),5))
 if joined.filter(F.col("macro_stress").isNull()).limit(1).count():raise ValueError("Missing macro month")
 return joined
def run_parity(n=240,seed=42):
 from pyspark.sql import SparkSession
 spark=(SparkSession.builder.master("local[2]").appName("retail-credit-etl")
        .config("spark.ui.enabled","false").getOrCreate())
 try:
  loans,macro=make_sources(n,seed)
  original=prepare_pandas(loans,macro)
  observed={r["application_id"]:r["loan_to_income"] for r in prepare_spark(loans,macro,spark).select("application_id","loan_to_income").collect()}
  assert len(original)==len(observed)
  assert all(abs(observed[r.application_id]-r.loan_to_income)<1e-5 for r in original.itertuples())
  return {"status":"passed","compared_rows":len(observed),"sources":2}
 finally:spark.stop()
if __name__=="__main__":print(run_parity())
