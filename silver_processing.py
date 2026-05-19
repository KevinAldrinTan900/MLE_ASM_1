import re
import os
import glob

import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, NumericType, DecimalType

import utils.data_processing_silver_table
from bronze_processing import BRONZE_LMS_DIR, BRONZE_FIN_DIR, BRONZE_ATTR_DIR, BRONZE_CLICK_DIR, generate_first_of_month_dates

SILVER_LOAN_DAILY_DIR = "datalake/silver/lms_loan_daily/"
SILVER_CUSTOMER_FEATURES_DIR = "datalake/silver/customer_features/"
SILVER_ATTRIBUTES_DIR = "datalake/silver/attributes/"

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

    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    total_cf_rows = 0

    for date_str in dates_str_lst:
        date_tag = date_str.replace('-', '_')

        # Silver Table Process for LMS
        lms_bronze_path = os.path.join(BRONZE_LMS_DIR, f"bronze_lms_loan_daily_{date_tag}.csv")
        df = spark.read.csv(lms_bronze_path, header=True, inferSchema=True)
        print(f"loaded {lms_bronze_path}, total row count: {df.count()}")

        column_type_map = {
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
        df = cast_columns(df, column_type_map)

        # NOTE: mob / installments_missed / first_missed_date / dpd are column-creating
        # augmentations that define the label -> moved to the gold label store.

        lms_silver_path = os.path.join(SILVER_LOAN_DAILY_DIR, f"silver_lms_loan_daily_{date_tag}.parquet")
        df.write.mode("overwrite").parquet(lms_silver_path)

        # Silver Table Process for Financial
        fin_bronze_path = os.path.join(BRONZE_FIN_DIR, f"bronze_features_financials_{date_tag}.csv")
        df = spark.read.csv(fin_bronze_path, header=True, inferSchema=True)
        print(f"loaded {fin_bronze_path}, total row count: {df.count()}")

        column_type_map = {
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
            "snapshot_date": DateType()
        }

        numeric_col_names = {c for c, t in column_type_map.items() if isinstance(t, (FloatType, IntegerType))}
        df = cast_columns(df, column_type_map, strip_underscore_cols=numeric_col_names)

        # Replace bare "_" / "" with null across all columns, in one projection
        fin_df = df.select(*[
            F.when((F.col(c).cast("string") == "_") | (F.col(c).cast("string") == ""), None)
             .otherwise(F.col(c)).alias(c)
            for c in df.columns
        ])

        # Keep only non-negative numeric values (single combined filter)
        num_cols = [f.name for f in fin_df.schema.fields if isinstance(f.dataType, NumericType)]
        non_negative = None
        for col_name in num_cols:
            cond = col(col_name) >= 0
            non_negative = cond if non_negative is None else (non_negative & cond)
        filtered_fin_df = fin_df.filter(non_negative) if non_negative is not None else fin_df

        # Remove Outliers
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
            Q1, Q3 = filtered_fin_df.approxQuantile(col_name, [0.25, 0.75], 0.0)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            filtered_fin_df = filtered_fin_df.filter((col(col_name) >= lower) & (col(col_name) <= upper))

        # Remove NULLs and other N/A values
        filtered_fin_df = filtered_fin_df.dropna()
        filtered_fin_df = filtered_fin_df.filter(col("Payment_of_Min_Amount") != "NM")
        filtered_fin_df = filtered_fin_df.filter(col("Payment_Behaviour") != "!@9#%8")

        # Remove duplicate values in Type_of_Loan
        dedup_udf = udf(dedup_loan_types, StringType())
        filtered_fin_df = filtered_fin_df.withColumn("Type_of_Loan", dedup_udf(col("Type_of_Loan")))

        # Silver Table Process for Attributes
        # Load all bronze attributes CSVs at once and process
        attr_bronze_path = os.path.join(BRONZE_ATTR_DIR, f"bronze_features_attributes_{date_tag}.csv")
        df = spark.read.csv(attr_bronze_path, header=True, inferSchema=True)
        print(f"loaded {attr_bronze_path}, total row count: {df.count()}")

        column_type_map = {
            "Customer_ID": StringType(),
            "Name": StringType(),
            "Age": IntegerType(),
            "SSN": StringType(),
            "Occupation": StringType(),
            "snapshot_date": DateType()
        }

        numeric_col_names = {c for c, t in column_type_map.items() if isinstance(t, (FloatType, IntegerType))}
        df = cast_columns(df, column_type_map, strip_underscore_cols=numeric_col_names)

        # Remove Invalid rows for Age, SSN, and Occupation (single combined filter)
        filtered_attr_df = df.filter(
            (col("Age") >= 0) & (col("Age") <= 100)
            & (col("SSN") != "#F%$D@*&8")
            & (col("Occupation") != "_______")
        )

        # Clean name column
        filtered_attr_df = filtered_attr_df.withColumn("Name", F.regexp_extract(col("Name"), r'^"*([^"]+)', 1))

        # Standalone silver table for cleaned attributes
        attr_silver_path = os.path.join(SILVER_ATTRIBUTES_DIR, f"silver_features_attributes_{date_tag}.parquet")
        filtered_attr_df.write.mode("overwrite").parquet(attr_silver_path)

        print(f"fin rows: {filtered_fin_df.count()}")
        print(f"attr rows: {filtered_attr_df.count()}")

        # check if any Customer_IDs actually overlap
        overlap = filtered_fin_df.select("Customer_ID").intersect(filtered_attr_df.select("Customer_ID"))
        print(f"overlapping Customer_IDs: {overlap.count()}")

        # Combine Financial and Attributes into Customer Features
        customer_features_df = filtered_fin_df.join(filtered_attr_df.drop("snapshot_date"), on="Customer_ID", how="inner")
        total_cf_rows += customer_features_df.count()
        customer_features_path = os.path.join(SILVER_CUSTOMER_FEATURES_DIR, f"silver_customer_features_{date_tag}.parquet")
        customer_features_df.write.mode("overwrite").parquet(customer_features_path)

    print(f"customer_features total rows across all months: {total_cf_rows}")
