from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

spark = SparkSession.builder.config(conf=SparkConf().setMaster('local[12]')).getOrCreate()

df = spark.read.csv('./GBA_step_200509.csv',header=True)

df.show()

df.schema


