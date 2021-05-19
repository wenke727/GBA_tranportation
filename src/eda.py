#%%
import os
import math
from threading import Condition
from numpy.lib.function_base import copy
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.tools import reduce_mem_usage
from utils.classes import PathSet
import warnings

warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", 100)
    
#%%
"""
TripID:
    * path: 
        [:-3],      date
        [-7:-3],    time
        [-3:],      od
"""

"""global parameters"""
df = pd.read_hdf('/home/pcl/Data/GBA/gba_db.h5')
path_set = PathSet(load=True, cache_folder='../cache', file_name='path_features')

gba_area = gpd.read_file("../db/gba_boundary.geojson")
bridges = gpd.read_file("../db/bridges.geojson")
gdf_paths = path_set.convert_to_gdf()
gdf_paths = gpd.sjoin(left_df=gdf_paths, right_df=bridges[['bridge', 'geometry']], op='intersects', how='left') 

cities_lst = ["Hong Kong","Macao","Guangzhou","Shenzhen","Zhuhai","Foshan","Huizhou","Dongguan","Zhongshan","Jiangmen","Zhaoqing"]

#%%
"""tool functions"""
def period_split(df):
    # 区间左闭右开
    period_bins = ['20190302 1200', '20190402 1150', '20200505 1350', '20200515 0850', '20211230 0900']
    period_bins = [ pd.to_datetime(x) for x in period_bins]
    period_labels = [ i+1 for i in range(len(period_bins)-1)]
    df.loc[:, 'period'] = pd.cut(df.time, period_bins, labels=period_labels)
    
    return df

def outlier_filter(df, group_cols = ['OD', 'date'], col='speed', mul_factor=3):
    df_groups = df.groupby(group_cols)

    df_outlier_index = pd.concat([df_groups[col].mean(), df_groups[col].std()], axis=1)
    df_outlier_index.columns = ['_mean', '_std']
    df_outlier_index.loc[:, 'lower'] = df_outlier_index._mean - mul_factor * df_outlier_index._std
    df_outlier_index.loc[:, 'upper'] = df_outlier_index._mean + mul_factor * df_outlier_index._std
    df_outlier_index.reset_index(inplace=True)
    
    df_ = df.merge(df_outlier_index, on=group_cols).query("lower <= speed <= upper")[df.columns]
    df_.sort_values(by=['time', 'OD'], inplace=True)

    df_outliers = pd.concat([df, df_]).drop_duplicates(keep=False).sort_values(by=['time', 'OD'])
    print(f"outliers size: {df.shape[0]-df_.shape[0]}, percentage: {(1-df_.shape[0]/df.shape[0])*100:.2f}%")
    
    return df_, df_outliers

def add_df_info(df):
    pois = pd.read_excel("../db/cities_poi.xlsx",'poi').rename(columns={'id': 'OD'})
    reduce_mem_usage(pois)
    df = df.merge( pois[['OD', 'o', 'd']], on="OD")
    df.speed = df.speed.astype(np.float32)

    cities_order = ["Hong Kong","Macao","Guangzhou","Shenzhen","Zhuhai","Foshan","Huizhou","Dongguan","Zhongshan","Jiangmen","Zhaoqing"]
    for i in ["o", 'd']:
        df[i] = df[i].astype("category")
        df[i].cat.set_categories(cities_order, inplace=True)
        
    df.loc[:, 'dayofweek'] = df.time.dt.dayofweek
    holidays = ['20190404', '20190405'] # 特殊节假日
    df.loc[:, 'holiday'] = False
    df.loc[df.query(f"date in {holidays} ").index, 'holiday'] = True
    # df[df.period==2].groupby(['period','weekday','date'])['OD'].count()
    
    return df

def plot_trip(o='Shenzhen', d = 'Zhuhai', gdf_paths=gdf_paths):
    pids = paths.query(f" o == '{o}' and d == '{d}' ").fid_lst.values[0]
    pids = np.delete(pids, list(pids).index(-1))

    ax = gba_area.plot(figsize=(12,8))
    gdf_paths.loc[pids].plot(ax=ax, color='r')
    ax.set_title( f"{o} -> {d}" )
    
    return 


#%%
df = add_df_info(df)
df, df_outliers = outlier_filter(df, group_cols = ['OD', 'date'])
df_equal_weight = df.groupby(['o', 'd', 'time']).mean().dropna().reset_index()
print( f"df_equal_weight size shrink {df.shape[0]}->{df_equal_weight.shape[0]}, cut down {(1-df_equal_weight.shape[0]/df.shape[0])*100:.2f}%" )

df_equal_weight = period_split(df_equal_weight)
df = period_split(df)

df = df.merge( gdf_paths[['fid', 'shape', 'bridge']], on='fid', how='left')

# 增加拥堵系数
df = df.merge(df.groupby(['OD'])[['speed']].max().rename(columns={'speed': 'speed_max'}), on = "OD")
df.loc[:, 'ci'] = df.speed_max / df.speed

df.loc[:, 'holiday/weekend'] = (df.holiday) | (df.dayofweek >= 5)

#%%
# 仅仅是速度分布曲线，看不出什么来
for i in range(1,5,1):
    sns.kdeplot(df.query(f"period == {i}").speed, label = i)

#%%
# 
"""
The curves of congestion index of the whole net works in the four period:
    1: festival travel demand is too big for road capacity
    2: 虎门大桥的作用
TODO:
    1.受影响的出行
"""
def draw_curves_ci(df, holiday=True, group_cols=['OD'], sample=.1):
    df_ = df.copy()
    if holiday is not None:
        df_.query(f"dayofweek >=5 or holiday" if holiday else "dayofweek < 5 and not holiday", inplace=True)
    
    # df_speed = df_.groupby(group_cols)['speed'].mean().reset_index().dropna()
    df_speed = df_.merge(df_.groupby(['OD'])[['speed']].max().rename(columns={'speed': 'speed_max'}), on = "OD")
    
    df_speed.loc[:, 'ci'] = df_speed.speed_max / df_speed.speed
    # TODO 测试周五下午的情况
    df_speed.loc[:, 'holiday/weekend'] = (df_speed.holiday) | (df_speed.dayofweek >= 5)
    
    fig = plt.figure(figsize=(8, 16))
    ax = plt.subplot(3,1,1)    
    # http://seaborn.pydata.org/generated/seaborn.lineplot.html, confidence intervals
    sns.lineplot(x='t', y='ci', hue='period', style='holiday/weekend', data=df_speed, ax=ax  )
    ax.set_xlim(0, 24)
    ax.set_ylim(1, 1.6)
    ax.set_xticks(range(0, 28, 4))
    ax.set_title(f"all, group_cols: {group_cols}")
    
    ax = plt.subplot(3,1,2)    
    sns.lineplot(x='t', y='ci', hue='period', data=df_speed[~df_speed['holiday/weekend']], ax=ax  )
    ax.set_xlim(0, 24)
    ax.set_ylim(1, 1.6)
    ax.set_xticks(range(0, 28, 4))
    ax.set_title(f"weekdays, group_cols: {group_cols}")
    
    ax = plt.subplot(3,1,3)    
    sns.lineplot(x='t', y='ci', hue='period', dashes=False, data=df_speed[df_speed['holiday/weekend']], ax=ax  )
    ax.set_xlim(0, 24)
    ax.set_ylim(1, 1.6)
    ax.set_xticks(range(0, 28, 4))
    ax.set_title(f"holiday/weekend, group_cols: {group_cols}")
   
    plt.tight_layout(h_pad=.5)
    
    return fig

# _ = draw_curves_ci(df, holiday=None,  group_cols=['OD'])
_ = draw_curves_ci(df_equal_weight, holiday=None,  group_cols=['OD'])


#%%

def draw_heapmap_4_period(df, key='travelDis', time_period=None, aggfunc=np.average):
    # key = 'travelDis' # speed
    matrixs = {}
    df_ = df.query( f"{time_period[0]} <= t < {time_period[1]}" ) if time_period is not None else df.copy()
    for i in range(1, 5, 1):
        matrixs[i] = pd.pivot_table(df_.query( f"period == {i}" ), values=key, aggfunc=aggfunc, fill_value=np.nan, index=['o'], columns='d')

    base = matrixs[1].copy()
    for i, mat in matrixs.items():
        matrixs[i] = mat / base

    fig = plt.figure(figsize=(16,16))
    fig.suptitle(f"heapmap of {key} between {time_period}", fontsize=32)
    for i in range(1, 5, 1):
        plt.subplot(2,2,i)
        sns.heatmap(matrixs[i], vmax=1.2, vmin=0.8, center=1, annot=True,linewidths=.5, cmap=sns.cm.rocket_r, fmt='.2f')
        # sns.heatmap(matrixs[i], vmax=1.2, vmin=0.8, center=1, annot=True,linewidths=.5, cmap='rainbow', fmt='.2f')
    plt.tight_layout()    
    
    return fig

# for i in range(6, 20):
#     draw_heapmap_4_period(df, 'travelDis', None, np.std)

fig = draw_heapmap_4_period(df_equal_weight, 'speed', [16, 18], np.median)
fig = draw_heapmap_4_period(df_equal_weight.query("holiday==True"), 'speed', [16, 18], np.median)
fig = draw_heapmap_4_period(df_equal_weight.query("holiday==False"), 'speed', [16, 18], np.median)


# %%
key = 'speed' # speed
matrixs = {}
for i in range(1, 5, 1):
    matrixs[i] = pd.pivot_table(df.query( f"period == {i} and 15 <= t < 17 " ), values=key, aggfunc=np.average, fill_value=np.nan, index=['o'], columns='d')

base = matrixs[1].copy()
for key, mat in matrixs.items():
    matrixs[key] = mat / base

plt.figure(figsize=(16,16))
for i in range(1, 5, 1):
    plt.subplot(2,2,i)
    # sns.heatmap(matrixs[i], vmax=1.2, vmin=0.8, center=1, annot=True,linewidths=.5, cmap=sns.cm.rocket_r, fmt='.2f')
    sns.heatmap(matrixs[i], vmax=1.2, vmin=0.8, center=1, annot=True,linewidths=.5, cmap='rainbow', fmt='.2f')

# %%


paths = df.groupby(['o', 'd']).apply(lambda x: np.delete(x.fid.unique(), -1) ).reset_index().rename(columns={0: 'fid_lst'})
paths.loc[:, 'path_num'] = paths.fid_lst.apply(lambda x: len(x))

plot_trip('Shenzhen', 'Zhuhai')

# %%
def get_trip_related_to_bridges(df):
    tmp = df.groupby(['o', 'd', 'period','bridge'])[['OD']].count().reset_index()
    tmp_sum = tmp.groupby(['o', 'd', 'period']).sum().reset_index().rename(columns={'OD': '_sum'})
    tmp = tmp.merge(tmp_sum,  on=['o', 'd', 'period'] )   
    tmp.loc[:, 'percentage'] = tmp.OD / tmp._sum

    tmp = pd.pivot_table(tmp, values='percentage', aggfunc=np.mean, fill_value=np.nan, index=['o', 'd'], columns=['bridge','period']) 
    
    return tmp

df_od_related_to_bridges = get_trip_related_to_bridges(df)

# %%

od_related_to_bridge = pd.DataFrame(df_od_related_to_bridges.index, columns=['OD'])
od_related_to_bridge.loc[:, 'o'] = od_related_to_bridge.OD.apply(lambda x: x[0])
od_related_to_bridge.loc[:, 'd'] = od_related_to_bridge.OD.apply(lambda x: x[1])
df_related_to_bridges = df.merge(od_related_to_bridge[['o', 'd']], on=['o', 'd'])

# %%
df_equal_weight_2 = df_related_to_bridges.groupby(['o', 'd', 'time']).mean().dropna().reset_index()
print( f"df_equal_weight size shrink {df.shape[0]}->{df_equal_weight.shape[0]}, cut down {(1-df_equal_weight.shape[0]/df.shape[0])*100:.2f}%" )
df_equal_weight_2 = period_split(df_equal_weight_2)

_ = draw_curves_ci(df_equal_weight_2, holiday=None,  group_cols=['OD'])

# %%

# pd.pivot_table(df, index=['date'], columns=['o'], values='speed', aggfunc=np.average)

tmp = df.groupby(["date", "o"])[['speed']].mean().reset_index()
tmp.date = tmp.date.astype(str)
fig = plt.figure(figsize=(20,12))
sns.lineplot(x='date', y='speed',  data=tmp)

#%%


# %%

# 以星期为单元格，绘制情况

data = df.query( "period==4" )
print(data.dayofweek.unique())
data.dayofweek = 'day '+data.dayofweek.astype(str)
sns.lineplot( x='t', y='ci', hue='dayofweek', data=data )



# %%
def draw_speed_distribution_of_11_cities(df, key='ci', test=True):
    dates = df.date.unique()
    dates.sort()
    if test:
        dates = dates[:2]
    
    dist_cols, dist_rows = len(cities_lst), len(dates)
    fig = plt.figure(figsize=(6*dist_cols, 4*dist_rows))

    i = 0
    for date_index, date in tqdm(enumerate(dates)):
        for o in cities_lst:
            data = df.query( f" date =={date} and o =='{o}' " )
            ax = plt.subplot(dist_rows, dist_cols, i+1)
            tag = data.holiday.unique()[0] or (data.dayofweek >=5).unique()[0]
            if tag:
                sns.lineplot( x='t', y=key, color='orange', data=data)
            else:
                sns.lineplot( x='t', y=key, data=data)
                
            _mean = data[key].mean()
            ax.hlines(_mean, 0, 24, colors='gray', linestyles ='dotted', label=f"{_mean:.2f}, {data.speed.std():.2f}", )
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 6))
            ax.set_xlabel(o)
            ax.set_ylabel(str(date))
            ax.legend()
            i += 1

    plt.tight_layout()
    plt.show()
    
    return fig

fig = draw_speed_distribution_of_11_cities(df, 'ci')

# %%
# 查看不同城市间的速度分布情况
cities = ['Shenzhen', 'Hong Kong']
data = df.query( f" date =={date} and o in {cities} " )
sns.lineplot( x='t', y='speed', hue='o',data=data)



#%%


# %%
import copy

def dict_to_sql_sentence(con):
    condition = copy.deepcopy(con)
    for key, val in condition.items():
        if isinstance(val, str):
            condition[key] = f"'{val}'"
    
    sql = " and ".join( [ f"{key} == { val if isinstance(val, str) else val }"  for key, val in condition.items()])
    
    return sql    

def draw_lines(df, df_all, key = 'speed', sql=None, *args, **kwargs):
    data = df.query(sql) if sql is not None else df.copy()
    flag = False
    
    if data.shape[0] == 0:
        data = df_all.query(sql) if sql is not None else df.copy()
        kwargs['alpha'] = .08
        flag = True
        
    ax = sns.lineplot(x='t', y=key, data=data, *args, **kwargs)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    if flag:
        ax.set_alpha(.5)
        
    return ax

def draw_gba_intercities_index(df, df_all, holiday_or_weekend, key='ci', test=False, verbose=False, *args, **kwargs):
    dist_cols = dist_rows = len(cities_lst)

    fig = plt.figure(figsize=(6*dist_cols, 4*dist_rows))
    for i, o in tqdm(enumerate(cities_lst[3:4] if test else cities_lst)):
        for j, d in enumerate(cities_lst):
            if o == d:
                continue
            
            ax = plt.subplot(dist_rows, dist_cols, i*dist_cols+j+1, )
            con = {
                'o': o,
                'd': d,
                '`holiday/weekend`': holiday_or_weekend
            }
            sql = dict_to_sql_sentence(con)
            if verbose: print(sql)
            ax = draw_lines(df, df_all, key, sql, hue='period')

            if ax is None:
                continue
            ax.set_xlabel(f"{o}->{d}")

    plt.tight_layout()
    plt.show()
    
    return fig

fig = draw_gba_intercities_index(df_related_to_bridges, df, holiday_or_weekend=True, test=True, verbose=True)



# %%

o='Shenzhen'; d = 'Zhongshan'
con = {
    'o': o,
    'd': d,
    '`holiday/weekend`': False
}

sql = dict_to_sql_sentence(con)
fig = plt.figure()
ax = draw_lines(df_related_to_bridges, df, 'ci', sql, hue='period', alpha=.4)
ax.set_xticks(range(0,25,6),alpha=.4)
# ax.set_ylabel(alpha=.4)


# %%
