#%%
import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from spark_helper import spark
from pyspark.sql.types import StringType, IntegerType
import pyspark.sql.functions as F

from utils.classes import PathSet
from df_helper import save_to_geojson, df_pipline
from df_spark_helper import parser_tripID_info, split_period, convert_2_dateFormat

# data initial
path_set = PathSet(cache_folder='../cache', file_name='wkt_step')
gdf_shps = path_set.get_shapes_with_bridges_info()
fids_bridge = gdf_shps[~gdf_shps.bridge.isnull()].fid.unique()


fig_title_font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 12,
}

#%%


""" preprocess csv data """
def add_date_to_file(fn):
    """add date to file

    Args:
        fn (function): [description]
    """
    date_ = '20'+fn.split("_")[-1].split('.')[0]
    df = pd.read_csv(fn)
    df.loc[:, "date"] = int(date_)
    
    print(f"{fn} -> {fn.replace('db/step', 'steps_csv')}")
    df.to_csv( fn.replace('db/step', 'steps_csv') , index=False)
    
    return


def add_date_batch(folder='/home/pcl/Data/GBA/db/step', n_jobs=56):
    from multiprocessing import Pool
    s = datetime.datetime.now()

    fns = [os.path.join(folder, f) for f in os.listdir(folder)]
    fns.sort()

    pools = Pool(n_jobs)
    data  = pools.map_async(add_date_to_file, fns).get()
    pools.close()
    pools.join()

    return


# if __name__ == '__main__':
    #data = read_data_batch()
    # add_date_batch()
    # s = datetime.datetime.now()
    # print(datetime.datetime.now() -s)


#%%
""" pyspark funcs """

def check_dataframe():
    """Check df
    """
    print('stepID Top 10')
    spark.sql(" select * from step order by stepID desc limit 10 ").show()
    
    print('time distribution')
    spark.sql("""select t, count(road) from step group by t order by t""").show(100)
    
    # 查看最大的step数值， 是小于100的
    print('stepID distribution')
    spark.sql("""select stepID, count(road) from step group by stepID order by stepID""").show(100)
    
    print('date distribution')
    spark.sql("""select date, count(road) from step group by date order by date """).show(30)

    return 


def check_num_dist():
    """对比分析shp and csv folder 可得，两者的记录是相符合的, 除了 2020-05-23
    """
    a = spark.sql('select date, count(tripID) as count from step group by date order by date').toPandas()
    b = spark.sql('select date, count(tripID) as count from csv group by date order by date').toPandas()
    c = b.merge(a, on='date', how='left')
    c.loc[:, 'same'] = (c.count_y - c.count_x) == 0
    
    return c


def check_tripID_dist(df_2):
    """check the distribution of the length of tripID

    Args:
        df_2 ([dataframe]): [description]
    """
    # check the number of tripID in p1 and p2
    steps = df_2.filter(F.col('date') < '2020-05-01')\
                .withColumn('tripID_num', F.length(df_2.tripID.cast(StringType())) )
    steps.groupBy(steps.date, steps.tripID_num).count().sort(steps.date).show()

    # check the number of tripID in p3 and p4
    steps = df_2.filter(F.col('date') >= '2020-05-01')\
                .withColumn('tripID_num', F.length(df_2.tripID.cast(StringType())) )
    steps.groupBy(steps.date, steps.tripID_num).count().sort(steps.date).show()


def read_and_tranform_data():
    """read origin csv data and save them in the parquet format

    Returns:
        [Boolean]: If success, return True
    """
    # 读取初始数据，然后写出到parquet格式, 
    df_1 = spark.read.format('csv').load("/home/pcl/Data/GBA/steps_all.csv", header=True, inferSchema=True,dateFormat="yyyyMMdd")
    df_1 = parser_tripID_info(df_1)
    df_1 = split_period(df_1)
    df_1 = df_1.repartition('date')

    # save to parquet format
    s = datetime.datetime.now()
    df_1.write.format('parquet').mode('overwrite').save('/home/pcl/Data/GBA/db/spark/steps_shp.parquet')
    print(datetime.datetime.now() -s)

    df_2 = spark.read.format('csv').load("/home/pcl/Data/GBA/steps_csv", header=True, inferSchema=True, dateFormat="yyyyMMdd")
    df_2 = convert_2_dateFormat(df_2)
    # check_tripID_dist(df_2)

    s = datetime.datetime.now()
    df_2.repartition('date').write.format('parquet').mode('overwrite').save('/home/pcl/Data/GBA/db/spark/steps_csv.parquet')
    print(datetime.datetime.now() -s)
    
    return True


def cal_step_freq(df, verbose=False):
    """Count the step frequency of each OD in each period
    
    Return [pd.DataFrame]
    """
    
    view_table = 'step_'
    df.createOrReplaceTempView("step_")
    sql = f"""SELECT a.OD, a.period, a.fid, a.nums, a.nums/b.totoal_nums AS freq
            FROM 
            (
                SELECT OD, fid, period, count(tripID) AS nums
                FROM {view_table}
                GROUP BY OD, period, fid
            ) AS a, 
            (
                SELECT OD, period, count(DISTINCT(timestamp)) AS totoal_nums 
                FROM {view_table}
                GROUP BY OD,  period
            ) as b
            WHERE a.OD == b.OD and a.period == b.period
            ORDER BY a.OD, a.period, freq
        """

    if verbose: print(sql)
    res = spark.sql(sql)

    return res.toPandas()


def selct_data_by_date(date_ = '2020-05-18'):
    df = spark.sql(f"select * from step where date == '{date_}'").cache()
    df_ = spark.sql(f"select * from csv where date == '{date_}'").cache()
    df.show()
    df_.show()
    
    return df, df_


#%%
# TODO 空值的处理， road里边的null为空值

df = spark.read.format('parquet').load('/home/pcl/Data/GBA/db/spark/steps_p3_p4.parquet')
# df = spark.read.format('parquet').load('/home/pcl/Data/GBA/db/spark/steps_shp.parquet')
df.createOrReplaceTempView('step')

df_csv = spark.read.format('parquet').load('/home/pcl/Data/GBA/db/spark/steps_csv.parquet')
df_csv.createOrReplaceTempView('csv')


#%%

def find_the_freq_steps(df_csv):
    # TODO
    df_csv.show()
    total_num = df_csv.count()

    cols_drop = ['tripID']
    cols_name = [ x for x in df_csv.columns if x not in cols_drop]
    unique_step = df_csv.select(cols_name).distinct().cache()

    unique_step.count()

    table_name= 'unique_step'

    unique_step.createOrReplaceTempView(table_name)

    cols_name = ",".join([ x for x in cols_name if x not in ['date']])

    #%%

    sql = f"""SELECT {cols_name}, COLLECT_SET(date)  AS date_lst 
            FROM  {table_name}
            GROUP BY {cols_name}
    """
    print(sql)
    tmp = spark.sql(sql)
    #%%
    tmp = tmp.withColumn('date_count', F.size(tmp.date_lst))
    tmp.show()
    tmp = tmp.where('date_count>48').sort('date_count').toPandas()


    return tmp



# %%
df.show()

# Count the step frequency of each OD in each period
trips_freq = cal_step_freq(df)

# %%
#! 识别 period 1、2 的桥上记录

bridge_map = {1:'Humen bridge', 2:'Nanshan bridge', 3:'HZMB'}

# bridge congestion
def extract_bridge_from_instruction(df):
    view_name = 'df_bridge'
    df.createOrReplaceTempView(view_name)
    sql = f"""SELECT *, (
                CASE 
                    WHEN instruction LIKE "%虎门大桥%" then 1
                    WHEN instruction LIKE "%南沙大桥%" then 2
                    WHEN instruction LIKE "%港珠澳大桥%" then 3
                ELSE -1 END) AS bridge
            FROM {view_name}
        """
    df = spark.sql(sql).where('bridge > 0 ')
    
    return df


def agg_speed_for_bridge(df):
    view_name = 'df_bridge'
    df.createOrReplaceTempView(view_name)
    sql = f"""SELECT period, timestamp, bridge, AVG(speed) as avg_speed
            FROM {view_name}
            GROUP BY period, timestamp, bridge
            ORDER BY period, timestamp, bridge
    """
    
    return spark.sql(sql)


def add_congestion_index(df):
    df.loc[:, 'bridge'] = df.bridge.apply( lambda x: bridge_map[x])

    df = df.merge(
        df.groupby('bridge')[['avg_speed']].max().rename(columns={"avg_speed":'max_speed'}).reset_index(),
        on='bridge'    
    )
    df.loc[:, 'ci'] = df.max_speed / df.avg_speed
    
    return df


# initialize
def df_initialize_for_p1(df_csv):
    df = df_csv.where("date < '2020-01-01' ")\
                .withColumn('t', F.substring(F.col('tripID'), -7, 4))\
                .withColumn('timestamp', F.to_timestamp( F.concat( F.col('date').cast(StringType()), F.lit(' '), F.col('t')) , 'yyyy-MM-dd HHmm'))\
                .withColumn('OD', F.substring(F.col('tripID'), -3, 3).cast(IntegerType()))\
                .withColumn('t', F.hour(F.col('timestamp')) + F.minute(F.col('timestamp'))/F.lit(60) )\
                .withColumn('instruction', F.trim(F.col('instruction')))\
                .withColumn('speed', F.col('distance')/F.col('duration')*F.lit(3.6))
    
    return df


def special_time_filter(data_bridge):
    data_bridge.loc[:, 't'] = data_bridge.timestamp.apply(lambda x: x.hour + x.minute/60)
    time_filter = []
    cur=.25
    while cur < 7:
        time_filter.append(cur)
        cur += .5

    data_bridge.query( 't not in @time_filter ', inplace=True )

    return data_bridge


p1 = df_initialize_for_p1(df_csv).cache()
p3 = df.withColumn('speed', F.col('distance') / F.col('duration') * F.lit(3.6)).drop('period')

p1_bridge = df_pipline(p1, [split_period, extract_bridge_from_instruction, agg_speed_for_bridge]).cache()
p3_bridge = df_pipline(p3, [split_period, extract_bridge_from_instruction, agg_speed_for_bridge]).cache()

data_bridge = pd.concat([p1_bridge.toPandas(), p3_bridge.toPandas()])
data_bridge = add_congestion_index(data_bridge)
data_bridge = special_time_filter(data_bridge)


#%%
# ! 绘制拥堵指数分布曲线


def add_ticks_day(x0, x1, ax,step_time=np.timedelta64(1,'D'), label_step=2 ):

    ticks = []
    tmp = x0
    while tmp <= x1:
        ticks.append(tmp)
        tmp += step_time

    try:
        ax.set_xticks(ticks)
        ax.set_xticklabels([ x._date_repr if i % label_step == 0 else '' for i, x in enumerate(ticks) ])
        return True
    except:
        return False


def get_weekends(x0, x1):
    fisrt_weekend_offset = 5 - x0.dayofweek if x0.dayofweek < 5 else 0
    cur = x0 + np.timedelta64(fisrt_weekend_offset,'D')

    ans = []
    while cur < x1:
        end = cur + np.timedelta64(2,'D')
        if end > x1:
            ans.append([cur, x1])
            break
        ans.append([cur, end])
        cur += np.timedelta64(7,'D')   

    return ans


def draw_congestion_index_by_date_p1(data_bridge, plot_params, y_max=4.2, ax=None, save_fn=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 6))
    plot_params['ax'] = ax

    data_bridge_sub = data_bridge.query( "period <= 2" )
    x0 = data_bridge_sub.timestamp.min()
    x1 = data_bridge_sub.timestamp.max()+np.timedelta64(15,'m')

    # add 
    # data_0406 = data_bridge.query(" '2019-04-07' <= timestamp < '2019-04-08'")
    # data_0406.loc[:, 'timestamp'] = data_0406.timestamp - np.timedelta64(1,'D')
    # sns.lineplot( data=data_0406, linestyle='-.', zorder=9, **plot_params, legend=False)

    sns.lineplot( data=data_bridge_sub.query("timestamp  < '2019-04-06 00:00:00' "), **plot_params, legend=False)
    sns.lineplot( data=data_bridge_sub.query("timestamp >= '2019-04-07 00:00:00' "), **plot_params)

    holiday = [np.datetime64('2019-04-05 00:00:00'), np.datetime64('2019-04-08 00:00:00')]
    ax.fill_between(holiday, 0, y_max, color='red', alpha=0.1, zorder=2, label='Qingming Festival')

    for index, weekend in enumerate(get_weekends(x0, x1)):
        ax.fill_between(weekend, 0, y_max, color='gray', alpha=0.2, zorder=1, **{'label':'Weekends'} if index==0 else {})
    
    ax.vlines(np.datetime64('2019-04-02 12:00:00'), 0 , y_max, colors='red', linestyle='--', lw=1, label='Nansha Bridge opened')

    ax.set_xlim(x0, x1)
    ax.set_ylim(0.90, y_max)
    add_ticks_day(x0, x1, ax)
    ax.legend()
    
    ax.set_xlabel('a) Average daily congestion index by date in period 1 and 2', fig_title_font)
    ax.set_ylabel("Congestion Index", fig_title_font)

    if save_fn:
        plt.savefig(save_fn, dpi=300)

    return ax


def draw_congestion_index_by_date_p3(data_bridge, plot_params, y_max=4.2, ax=None, save_fn=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 6))
    plot_params['ax'] = ax

    data_bridge_sub = data_bridge.query( "period > 2" )
    # x0 = data_bridge_sub.timestamp.min()+np.timedelta64(45+5*60,'m')
    # x1 = data_bridge_sub.timestamp.max()+np.timedelta64(15,'m')

    x0 = pd.to_datetime('2020-05-07 00:00:00')
    x1 = x0 + np.timedelta64(22, 'D')


    sns.lineplot( data=data_bridge_sub, **plot_params)

    for index, weekend in enumerate(get_weekends(x0, x1)):
        ax.fill_between(weekend, 0, y_max, color='gray', alpha=0.2, zorder=1, **{'label':'Weekends'} if index==0 else {})

    ax.vlines(np.datetime64('2020-05-15 09:00:00'), 0 , y_max, colors='blue', linestyle='--', lw=1, label='Humen Bridge reopened')

    ax.set_ylim(0.90, y_max)
    ax.set_xlim(x0, x1)
    add_ticks_day(x0, x1, ax)
    ax.legend()
    # plt.savefig('ci_dist_p3.jpg', dpi=300)

    ax.set_xlabel('b) Average daily congestion index by date in period 3 and 4', fig_title_font)
    ax.set_ylabel("Congestion Index", fig_title_font)

    
    return ax


def draw_congesion_index_dist(df, ax=None):
    """绘制周期内一天中的分布情况的分布情况

    Args:
        data_bridge ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.
    """
    data_bridge = df.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # sns.lineplot(data=data, x='t', y='ci', hue='period', style='bridge', ax=ax)
    idnx = data_bridge.query( " '2019-04-04 12:00:00' <= timestamp < '2019-04-08 09:00:00'  ").index
    data_bridge.loc[idnx, 'period'] = 5 

    sns.lineplot(data=data_bridge.query('period < 5'), x='t', y='ci', hue='bridge', style='period', size='period', hue_order=['Humen bridge', 'Nanshan bridge', "HZMB"],  sizes=[1.5, 1.5, 2, 0.5], ax=ax)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 28, 4))
    ax.set_xlabel('c) Daily congestion index distribution in the 4 periods', fig_title_font)
    ax.set_ylabel('Congestion Index', fig_title_font)
    ax.legend(loc='upper left')

    return data_bridge


def congesion_index_fig(data_bridge, out_fn='../result/fig/fig_1_congestion_distribution.jpg'):
    plot_params = {"x":'timestamp', 
                    "y":'ci', 
                    "hue":'bridge',
                    "hue_order": ['Humen bridge', 'Nanshan bridge', "HZMB"],
                    "zorder": 9,
                    "alpha": .9
                }

    fig, ax = plt.subplots(3, figsize=(15, 4*3))
    draw_congestion_index_by_date_p1(data_bridge, plot_params, ax=ax[0])
    draw_congestion_index_by_date_p3(data_bridge, plot_params, ax=ax[1])
    draw_congesion_index_dist(data_bridge, ax=ax[2])

    plt.tight_layout(h_pad=1.5)
    
    if out_fn is not None:
        fig.savefig(out_fn, dpi=300, pad_inches=0, bbox_inches='tight')

    return fig

congesion_index_fig(data_bridge)

# %%
