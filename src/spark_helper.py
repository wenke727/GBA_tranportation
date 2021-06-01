from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from tqdm import tqdm

# Ref: https://blog.csdn.net/a5685263/article/details/102265838
conf = SparkConf().set("spark.executor.memory", "64g").set("spark.executor.cores", "16")
conf.set("spark.driver.memory", "16g")  # 这里是增加jvm的内存
conf.set("spark.driver.maxResultSize", "16g") # 这里是最大显示结果，这里是提示我改的。
conf.setMaster('local[*]')

sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()