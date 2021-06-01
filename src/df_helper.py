import pandas as pd
import geopandas as gpd
import numpy as np

"""
TripID:
    * path: 
        [:-3],      date
        [-7:-3],    time
        [-3:],      od
"""

cities_order = ["Hong Kong","Macao","Guangzhou","Shenzhen","Zhuhai","Foshan","Huizhou","Dongguan","Zhongshan","Jiangmen","Zhaoqing"]

"""df helper"""
def df_query(df, sql):
    return df.query(sql)


def df_query_by_od(df, o, d):
    o = cities_order[o] if isinstance(o, int) else o
    d = cities_order[d] if isinstance(d, int) else d
    sql = f" o == '{o}' and d == '{d}' "
    
    return df_query(df, sql)


def df_pipline(df, funcs):
    for f in funcs:
        df = f(df)

    return df

def get_pois(fn="../db/cities_poi.xlsx"):
    """
    get city points
    """
    from haversine import haversine

    pois = pd.read_excel(fn, 'poi').rename(columns={'id': 'OD'})
    pois.loc[:,'great_circle_dis'] = pois.apply(lambda x: haversine((x.lat_0, x.lon_0), (x.lat_1, x.lon_1)), axis=1)
    return pois


"""shp parser releted func"""
def add_od(df, fn="../db/cities_poi.xlsx"):
    df.loc[:, "OD"] = df.tripID.astype(np.int) % 1000
    pois = pd.read_excel(fn, 'poi').rename(columns={'id': 'OD'})
    df = df.merge( pois[['OD', 'o', 'd']], on="OD")
    df.speed = df.speed.astype(np.float32)
    
    for i in ["o", 'd']:
        df[i] = df[i].astype("category")
        df[i].cat.set_categories(cities_order, inplace=True)
    
    return df


def add_holiday(df):
    df.loc[:, 'dayofweek'] = df.time.dt.dayofweek
    holidays = ['20190404', '20190405'] # 特殊节假日
    df.loc[:, 'holiday'] = False
    df.loc[df.query(f"date in {holidays} ").index, 'holiday'] = True
    
    return df


def period_split(df, period_bins = ['20190302 1200', '20190402 1150', '20200505 1350', '20200515 0850', '20211230 0900']):
    # 区间左闭右开
    period_bins = [ pd.to_datetime(x) for x in period_bins]
    period_labels = [ i+1 for i in range(len(period_bins)-1)]
    df.loc[:, 'period'] = pd.cut(df.time, period_bins, labels=period_labels)
    
    return df


def add_congestion_index(df):
    # 增加拥堵系数
    df = df.merge(df.groupby(['OD'])[['speed']].max().rename(columns={'speed': 'speed_max'}), on = "OD")
    df.loc[:, 'ci'] = df.speed_max / df.speed
    
    return df


def get_trip_related_to_bridges(df):
    """obtaines trips related to bridges

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    tmp = df.groupby(['o', 'd', 'period','bridge'])[['OD']].count().reset_index()
    tmp_sum = tmp.groupby(['o', 'd', 'period']).sum().reset_index().rename(columns={'OD': '_sum'})
    tmp = tmp.merge(tmp_sum,  on=['o', 'd', 'period'] )   
    tmp.loc[:, 'percentage'] = tmp.OD / tmp._sum

    tmp = pd.pivot_table(tmp, values='percentage', aggfunc=np.mean, fill_value=np.nan, index=['o', 'd'], columns=['bridge','period']) 
    
    return tmp