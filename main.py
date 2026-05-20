import pyspark

from utils import data_processing_bronze_table
from utils import data_processing_silver_table
from utils import data_processing_gold_table

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# Disable whole-stage codegen to avoid 64KB method size limit on wide tables
spark.conf.set("spark.sql.codegen.wholeStage", "false")

# Disable constraint propagation: the deep withColumn chains in silver_processing
# make Catalyst's getAllValidConstraints pass explode the driver heap on the join
spark.conf.set("spark.sql.constraintPropagation.enabled", "false")

data_processing_bronze_table.run(spark)
data_processing_silver_table.run(spark)
data_processing_gold_table.run(spark)
