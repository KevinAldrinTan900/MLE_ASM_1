import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_processing_bronze_table import generate_first_of_month_dates
from utils.data_processing_silver_table import (
    SILVER_LOAN_DAILY_DIR,
    SILVER_CUSTOMER_FEATURES_DIR,
    SILVER_CLICKSTREAM_DIR,
)

GOLD_LABEL_STORE_DIR = "datalake/gold/label_store/"
GOLD_FEATURE_STORE_DIR = "datalake/gold/feature_store/"
GOLD_CLICKSTREAM_DIR = "datalake/gold/clickstream/"

# label definition: default = 30+ days past due, observed at 6 months on book
DPD = 30
MOB = 6

# fixed category lists so every monthly partition produces an identical schema
LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Not Specified",
    "Payday Loan", "Personal Loan", "Student Loan",
]
PAYMENT_BEHAVIOURS = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]
OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    return df


def process_features_gold_table(snapshot_date_str, silver_customer_features_directory, gold_feature_store_directory, spark):

    # connect to silver customer features table (cleaned financial + attributes join)
    partition_name = "silver_customer_features_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_customer_features_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # drop PII / non-predictive identifiers (keep Customer_ID + snapshot_date as join keys)
    df = df.drop("Name", "SSN")

    # --- feature encoding ---

    # Credit_Mix: ordinal label encoding (Bad < Standard < Good)
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "Bad", 0)
         .when(col("Credit_Mix") == "Standard", 1)
         .when(col("Credit_Mix") == "Good", 2)
         .cast(IntegerType()))

    # Payment_of_Min_Amount: binary 1/0
    df = df.withColumn("Payment_of_Min_Amount",
        F.when(col("Payment_of_Min_Amount") == "Yes", 1)
         .when(col("Payment_of_Min_Amount") == "No", 0)
         .cast(IntegerType()))

    # Type_of_Loan: multi-valued -> multi-hot indicator per loan type
    for loan_type in LOAN_TYPES:
        out_col = "Loan_" + loan_type.replace(" ", "_").replace("-", "_")
        df = df.withColumn(out_col,
            F.coalesce(col("Type_of_Loan").contains(loan_type).cast(IntegerType()), F.lit(0)))
    df = df.drop("Type_of_Loan")

    # Payment_Behaviour: one-hot
    for behaviour in PAYMENT_BEHAVIOURS:
        df = df.withColumn("PayBehav_" + behaviour,
            F.coalesce((col("Payment_Behaviour") == behaviour).cast(IntegerType()), F.lit(0)))
    df = df.drop("Payment_Behaviour")

    # Occupation: one-hot
    for occ in OCCUPATIONS:
        df = df.withColumn("Occ_" + occ,
            F.coalesce((col("Occupation") == occ).cast(IntegerType()), F.lit(0)))
    df = df.drop("Occupation")

    # save gold feature store - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_clickstream_gold_table(snapshot_date_str, silver_clickstream_directory, gold_clickstream_directory, spark):
    # passthrough copy from silver to gold, no transformations
    date_tag = snapshot_date_str.replace('-', '_')
    src = os.path.join(silver_clickstream_directory, f"silver_feature_clickstream_{date_tag}.parquet")
    df = spark.read.parquet(src)
    print('loaded from:', src, 'row count:', df.count())

    dst = os.path.join(gold_clickstream_directory, f"gold_feature_clickstream_{date_tag}.parquet")
    df.write.mode("overwrite").parquet(dst)
    print('saved to:', dst)
    return df


def run(spark):
    os.makedirs(GOLD_LABEL_STORE_DIR, exist_ok=True)
    os.makedirs(GOLD_FEATURE_STORE_DIR, exist_ok=True)
    os.makedirs(GOLD_CLICKSTREAM_DIR, exist_ok=True)

    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    total_label_rows = 0
    total_feature_rows = 0

    for date_str in dates_str_lst:
        label_df = process_labels_gold_table(
            date_str, SILVER_LOAN_DAILY_DIR, GOLD_LABEL_STORE_DIR, spark, DPD, MOB)
        feature_df = process_features_gold_table(
            date_str, SILVER_CUSTOMER_FEATURES_DIR, GOLD_FEATURE_STORE_DIR, spark)
        process_clickstream_gold_table(
            date_str, SILVER_CLICKSTREAM_DIR, GOLD_CLICKSTREAM_DIR, spark)
        total_label_rows += label_df.count()
        total_feature_rows += feature_df.count()

    print(f"gold label_store total rows across all months: {total_label_rows}")
    print(f"gold feature_store total rows across all months: {total_feature_rows}")