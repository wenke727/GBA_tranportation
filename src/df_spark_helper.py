from utils.spark_helper import spark
from pyspark.sql.types import StringType, IntegerType
import pyspark.sql.functions as F


def df_pipline(df, funcs):
    for f in funcs:
        df = f(df)

    return df


def split_period(df, 
                 period_bins = ['2019-03-02 12:00:00', '2019-04-02 11:50:00', '2020-05-05 13:50:00', '2020-05-15 08:50:00', '2021-12-30 09:00:00'], 
                 verbose=False):
    """
    split period accoding to the timestamp
    """
    view_name = 'split_period_df'
    df.createOrReplaceTempView(view_name)
    
    sql = f"""
        SELECT *, 
            (case when '{period_bins[0]}' < timestamp and timestamp <= '{period_bins[1]}' then 1
                  when '{period_bins[1]}' < timestamp and timestamp <= '{period_bins[2]}' then 2
                  when '{period_bins[2]}' < timestamp and timestamp <= '{period_bins[3]}' then 3
                  when '{period_bins[3]}' < timestamp and timestamp <= '{period_bins[4]}' then 4
                  else -1 end) as period
        FROM {view_name}
    """
    if verbose: print(sql)
    
    return spark.sql(sql)


def convert_2_dateFormat(df, dateFormat="yyyyMMdd"):
    return df.withColumn( 'date', F.to_date( F.col('date').cast(StringType()), dateFormat) )


def parser_tripID_info(df, verbose=True, dateFormat = "yyyyMMdd"):
    
    df = df.withColumnRenamed('tripID', 'OD').withColumnRenamed('ID', 'tripID')
    df = df.withColumn('tripID', df.tripID.cast(StringType()))

    # `steps_all.csv`， ID = date(1/2) + time(4) + OD(1/2/3) + stepID(1/2)
    df = df.withColumn('day_num', F.length((df.date%100).cast(StringType())))
    df = df.withColumn('OD_num',  F.length(df.OD))
    df = convert_2_dateFormat(df, dateFormat)

    df = df.withColumn('t', df.tripID.substr(df.day_num+1, F.lit(4)))
    df = df.withColumn('timestamp', F.to_timestamp( F.concat( F.col('date').cast(StringType()), F.lit(' '), df.t) , 'yyyy-MM-dd HHmm' ))
    
    df = df.withColumn('t', df.t.substr(1, 2).cast(IntegerType()) + df.t.substr(3, 2).cast(IntegerType())/60)
    df = df.withColumn('stepID', df.tripID.substr(df.day_num + df.OD_num + 5, F.length(df.tripID)).cast(IntegerType()))
    
    df = df.drop('OD_num', 'day_num')

    # df.select('date').distinct().show()
    if verbose: df.show()

    return df
