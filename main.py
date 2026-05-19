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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import bronze_processing
import silver_processing
import gold_processing

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# Disable whole-stage codegen to avoid 64KB method size limit on wide tables
spark.conf.set("spark.sql.codegen.wholeStage", "false")

# Disable constraint propagation: the deep withColumn chains in silver_processing
# make Catalyst's getAllValidConstraints pass explode the driver heap on the join
spark.conf.set("spark.sql.constraintPropagation.enabled", "false")

bronze_processing.run(spark)
silver_processing.run(spark)
gold_processing.run(spark)
