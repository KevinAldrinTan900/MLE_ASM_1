import os
from datetime import datetime

import utils.data_processing_bronze_table

# Directories
BRONZE_LMS_DIR = "datalake/bronze/lms/"
BRONZE_FIN_DIR = "datalake/bronze/financial/"
BRONZE_ATTR_DIR = "datalake/bronze/attributes/"
BRONZE_CLICK_DIR = "datalake/bronze/clickstream/"


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return first_of_month_dates


def run(spark):
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    for directory in [BRONZE_LMS_DIR, BRONZE_FIN_DIR, BRONZE_ATTR_DIR, BRONZE_CLICK_DIR]:
        os.makedirs(directory, exist_ok=True)

    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(date_str, BRONZE_LMS_DIR, spark, "lms_loan_daily")
        utils.data_processing_bronze_table.process_bronze_table(date_str, BRONZE_FIN_DIR, spark, "features_financials")
        utils.data_processing_bronze_table.process_bronze_table(date_str, BRONZE_ATTR_DIR, spark, "features_attributes")
        utils.data_processing_bronze_table.process_bronze_table(date_str, BRONZE_CLICK_DIR, spark, "feature_clickstream")