import re
import os
import glob
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, NumericType, DecimalType

from utils.data_processing_bronze_table import (
    BRONZE_LMS_DIR,
    BRONZE_FIN_DIR,
    BRONZE_ATTR_DIR,
    BRONZE_CLICK_DIR,
    generate_first_of_month_dates,
)

SILVER_LOAN_DAILY_DIR = "datalake/silver/lms_loan_daily/"
SILVER_CUSTOMER_FEATURES_DIR = "datalake/silver/customer_features/"
SILVER_ATTRIBUTES_DIR = "datalake/silver/attributes/"
SILVER_CLICKSTREAM_DIR = "datalake/silver/clickstream/"

# to remove duplicate values in Type_of_Loan
def dedup_loan_types(loan_str):
    if loan_str is None:
        return None
    parts = [re.sub(r'^and\s+', '', p.strip()) for p in loan_str.split(",")]
    seen = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    return ", ".join(seen)


def cast_columns(df, type_map, strip_underscore_cols=()):
    strip = set(strip_underscore_cols)
    projected = []
    for c in df.columns:
        e = col(c)
        if c in strip:
            e = F.regexp_replace(e.cast(StringType()), "_", "")
        if c in type_map:
            e = e.cast(type_map[c])
        projected.append(e.alias(c))
    return df.select(*projected)


def run(spark):
    os.makedirs(SILVER_LOAN_DAILY_DIR, exist_ok=True)
    os.makedirs(SILVER_CUSTOMER_FEATURES_DIR, exist_ok=True)
    os.makedirs(SILVER_ATTRIBUTES_DIR, exist_ok=True)
    os.makedirs(SILVER_CLICKSTREAM_DIR, exist_ok=True)

    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    # ---------------- LMS + Clickstream: per-month passthrough ----------------
    lms_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for date_str in dates_str_lst:
        date_tag = date_str.replace('-', '_')

        # LMS
        lms_bronze_path = os.path.join(BRONZE_LMS_DIR, f"bronze_lms_loan_daily_{date_tag}.csv")
        df = spark.read.csv(lms_bronze_path, header=True, inferSchema=True)
        print(f"loaded {lms_bronze_path}, total row count: {df.count()}")
        df = cast_columns(df, lms_type_map)
        df.write.mode("overwrite").parquet(
            os.path.join(SILVER_LOAN_DAILY_DIR, f"silver_lms_loan_daily_{date_tag}.parquet"))

        # Clickstream passthrough (no transformations)
        cs_bronze_path = os.path.join(BRONZE_CLICK_DIR, f"bronze_feature_clickstream_{date_tag}.csv")
        df = spark.read.csv(cs_bronze_path, header=True, inferSchema=True)
        print(f"loaded {cs_bronze_path}, total row count: {df.count()}")
        df.write.mode("overwrite").parquet(
            os.path.join(SILVER_CLICKSTREAM_DIR, f"silver_feature_clickstream_{date_tag}.parquet"))

    # ---------------- Financial + Attributes: all-history single pass ----------------
    fin_paths = [os.path.join(BRONZE_FIN_DIR, f"bronze_features_financials_{d.replace('-','_')}.csv")
                 for d in dates_str_lst]
    attr_paths = [os.path.join(BRONZE_ATTR_DIR, f"bronze_features_attributes_{d.replace('-','_')}.csv")
                  for d in dates_str_lst]

    fin_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": DecimalType(),
        "Monthly_Inhand_Salary": DecimalType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DecimalType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": DecimalType(),
        "Amount_invested_monthly": DecimalType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DecimalType(),
        "snapshot_date": DateType(),
    }

    fin_df = spark.read.csv(fin_paths, header=True, inferSchema=True)
    print(f"loaded all bronze financial, total row count: {fin_df.count()}")

    fin_numeric_cols = {c for c, t in fin_type_map.items() if isinstance(t, (FloatType, IntegerType))}
    fin_df = cast_columns(fin_df, fin_type_map, strip_underscore_cols=fin_numeric_cols)

    # Replace bare "_" / "" with null across all columns
    fin_df = fin_df.select(*[
        F.when((F.col(c).cast("string") == "_") | (F.col(c).cast("string") == ""), None)
         .otherwise(F.col(c)).alias(c)
        for c in fin_df.columns
    ])

    # Keep only non-negative numeric values
    num_cols = [f.name for f in fin_df.schema.fields if isinstance(f.dataType, NumericType)]
    non_negative = None
    for col_name in num_cols:
        cond = col(col_name) >= 0
        non_negative = cond if non_negative is None else (non_negative & cond)
    if non_negative is not None:
        fin_df = fin_df.filter(non_negative)

    # Remove NULLs and sentinel values
    fin_df = fin_df.dropna()
    fin_df = fin_df.filter(col("Payment_of_Min_Amount") != "NM")
    fin_df = fin_df.filter(col("Payment_Behaviour") != "!@9#%8")

    # Dedup Type_of_Loan
    dedup_udf = udf(dedup_loan_types, StringType())
    fin_df = fin_df.withColumn("Type_of_Loan", dedup_udf(col("Type_of_Loan")))

    # Attributes
    attr_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    attr_df = spark.read.csv(attr_paths, header=True, inferSchema=True)
    print(f"loaded all bronze attributes, total row count: {attr_df.count()}")

    attr_numeric_cols = {c for c, t in attr_type_map.items() if isinstance(t, (FloatType, IntegerType))}
    attr_df = cast_columns(attr_df, attr_type_map, strip_underscore_cols=attr_numeric_cols)

    attr_df = attr_df.filter(
        (col("Age") >= 0) & (col("Age") <= 100)
        & (col("SSN") != "#F%$D@*&8")
        & (col("Occupation") != "_______")
    )
    attr_df = attr_df.withColumn("Name", F.regexp_extract(col("Name"), r'^"*([^"]+)', 1))
    attr_df = attr_df.cache()

    # Write attributes silver per month (preserve existing partition file convention)
    for date_str in dates_str_lst:
        date_tag = date_str.replace('-', '_')
        snap = datetime.strptime(date_str, "%Y-%m-%d").date()
        attr_df.filter(col("snapshot_date") == F.lit(snap)).write.mode("overwrite").parquet(
            os.path.join(SILVER_ATTRIBUTES_DIR, f"silver_features_attributes_{date_tag}.parquet"))

    # Join fin + attr into customer_features (join BEFORE IQR)
    customer_features_df = fin_df.join(
        attr_df.drop("snapshot_date"), on="Customer_ID", how="inner").cache()
    print(f"joined customer_features pre-IQR row count: {customer_features_df.count()}")

    # IQR clip on the joined frame, using all-history bounds
    cols_to_clip = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    for col_name in cols_to_clip:
        Q1, Q3 = customer_features_df.approxQuantile(col_name, [0.25, 0.75], 0.0)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        customer_features_df = customer_features_df.filter(
            (col(col_name) >= lower) & (col(col_name) <= upper))

    # Parse Credit_History_Age "X Years and Y Months" -> float years (moved from gold)
    yrs = F.coalesce(
        F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Years", 1).cast(IntegerType()),
        F.lit(0))
    mos = F.coalesce(
        F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Months", 1).cast(IntegerType()),
        F.lit(0))
    customer_features_df = customer_features_df.withColumn(
        "Credit_History_Age", (yrs + mos / F.lit(12.0)).cast(FloatType()))

    customer_features_df = customer_features_df.cache()
    total_cf_rows = customer_features_df.count()

    # Re-partition by snapshot_date into per-month parquet files
    for date_str in dates_str_lst:
        date_tag = date_str.replace('-', '_')
        snap = datetime.strptime(date_str, "%Y-%m-%d").date()
        customer_features_df.filter(col("snapshot_date") == F.lit(snap)).write.mode("overwrite").parquet(
            os.path.join(SILVER_CUSTOMER_FEATURES_DIR, f"silver_customer_features_{date_tag}.parquet"))

    print(f"customer_features total rows across all months: {total_cf_rows}")
