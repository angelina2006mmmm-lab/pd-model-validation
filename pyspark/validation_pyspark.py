from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("credit-risk-validation").getOrCreate()

df = spark.read.option("header", True).option("inferSchema", True).csv(
    "data/credit_risk_synthetic.csv"
)

summary = df.select(
    F.count("*").alias("rows"),
    F.avg("default_12m").alias("default_rate"),
    *[
        F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls")
        for c in df.columns
        if c != "default_12m"
    ],
)

summary.show(truncate=False)

df.groupBy("sector_risk").agg(
    F.count("*").alias("n_obs"),
    F.avg("default_12m").alias("observed_default_rate"),
).orderBy("sector_risk").show()

spark.stop()
