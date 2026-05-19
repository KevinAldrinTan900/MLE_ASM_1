import os

import utils.data_processing_gold_table
from bronze_processing import generate_first_of_month_dates
from silver_processing import SILVER_LOAN_DAILY_DIR, SILVER_CUSTOMER_FEATURES_DIR

GOLD_LABEL_STORE_DIR = "datalake/gold/label_store/"
GOLD_FEATURE_STORE_DIR = "datalake/gold/feature_store/"

# label definition: default = 30+ days past due, observed at 6 months on book
DPD = 30
MOB = 6


def run(spark):
    os.makedirs(GOLD_LABEL_STORE_DIR, exist_ok=True)
    os.makedirs(GOLD_FEATURE_STORE_DIR, exist_ok=True)

    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    total_label_rows = 0
    total_feature_rows = 0

    for date_str in dates_str_lst:
        label_df = utils.data_processing_gold_table.process_labels_gold_table(
            date_str, SILVER_LOAN_DAILY_DIR, GOLD_LABEL_STORE_DIR, spark, DPD, MOB)
        feature_df = utils.data_processing_gold_table.process_features_gold_table(
            date_str, SILVER_CUSTOMER_FEATURES_DIR, GOLD_FEATURE_STORE_DIR, spark)
        total_label_rows += label_df.count()
        total_feature_rows += feature_df.count()

    print(f"gold label_store total rows across all months: {total_label_rows}")
    print(f"gold feature_store total rows across all months: {total_feature_rows}")
