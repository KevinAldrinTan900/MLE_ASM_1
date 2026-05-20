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


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark, file_name):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to source back end - IRL connect to back end source system
    csv_file_path = f"data/data/{file_name}.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = f"bronze_{file_name}_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def run(spark):
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

    for directory in [BRONZE_LMS_DIR, BRONZE_FIN_DIR, BRONZE_ATTR_DIR, BRONZE_CLICK_DIR]:
        os.makedirs(directory, exist_ok=True)

    for date_str in dates_str_lst:
        process_bronze_table(date_str, BRONZE_LMS_DIR, spark, "lms_loan_daily")
        process_bronze_table(date_str, BRONZE_FIN_DIR, spark, "features_financials")
        process_bronze_table(date_str, BRONZE_ATTR_DIR, spark, "features_attributes")
        process_bronze_table(date_str, BRONZE_CLICK_DIR, spark, "feature_clickstream")
