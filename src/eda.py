#%%
import os
import math
import pickle
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from spark_helper import sc, spark

from shapely import wkt
from shapely.geometry import LineString, box
from utils.tools import reduce_mem_usage
from utils.classes import PathSet
import warnings
warnings.filterwarnings('ignore')

from df_helper import df_query, df_query_by_od, add_od, period_split, add_congestion_index, get_trip_related_to_bridges, df_pipline, get_pois
from plot_helper import draw_subplot, gba_plot, plt_2_Image, draw_subplot_travel_path


pd.set_option("display.max_rows", 100)

RESULT_FOLDER = "../result"

#%%
"""tool functions"""
def outlier_filter(df, group_cols=['OD', 'date'], col='speed', mul_factor=3):
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


""" visualization"""
def plot_circuity_factor_under_4_period(df, test = False, fn ='../result/4个时期的距离线型图.pdf'):
    # 绕行系数
    pois = get_pois()
    
    dist_cols = dist_rows = len(cities_lst)
    fig = plt.figure(figsize=(2.5*dist_cols, 2*dist_rows))
    for i, o in tqdm(enumerate(cities_lst[3:4] if test else cities_lst)):
        for j, d in enumerate(cities_lst):
            if o == d:
                continue
            
            ax = plt.subplot(dist_rows, dist_cols, i*dist_cols+j+1, )
            data = df_query_by_od(df, o, d).merge(pois[['o', 'd', 'great_circle_dis']], on=['o', 'd'])
            
            data.loc[:, 'circuity_factor'] = data.travelDis/1000 / data.great_circle_dis
            sns.boxplot(x='period', y='circuity_factor', data=data, ax=ax)
            ax.set_xlabel(f"{o}->{d}")

    plt.tight_layout()

    if fn is not None:
        fig.savefig(fn, dpi=300)
    plt.close()
    
    return fig


def plot_dis_distribution_under_4_period(df, test = False, fn ='../result/4个时期的距离线型图.pdf'):
    dist_cols = dist_rows = len(cities_lst)
    fig = plt.figure(figsize=(2.5*dist_cols, 2*dist_rows))
    
    for i, o in tqdm(enumerate(cities_lst[3:4] if test else cities_lst)):
        for j, d in enumerate(cities_lst):
            if o == d:
                continue
            
            ax = plt.subplot(dist_rows, dist_cols, i*dist_cols+j+1, )
            data = df_query_by_od(df, o, d)
            
            data.travelDis = data.travelDis/1000
            sns.boxplot(x='period', y='travelDis', data=data, ax=ax)
            ax.set_xlabel(f"{o}->{d}")

    plt.tight_layout()

    if fn is not None:
        fig.savefig(fn, dpi=300)
    plt.close()
    
    return fig


#%%
"""global parameters"""

cities_lst = ["Hong Kong","Macao","Guangzhou","Shenzhen","Zhuhai","Foshan","Huizhou","Dongguan","Zhongshan","Jiangmen","Zhaoqing"]

df        = pd.read_hdf('/home/pcl/Data/GBA/gba_db_0522.h5')
pois      = get_pois()

gba_area  = gpd.read_file("../db/gba_boundary.geojson")
bridges   = gpd.read_file("../db/bridges.geojson")

path_set  = PathSet(load=True, cache_folder='../cache', file_name='wkt_path')
gdf_paths = path_set.get_shapes_with_bridges_info()

df = gpd.GeoDataFrame( df.merge(gdf_paths[['fid', 'geometry', 'bridge']], on='fid', how='left') )
df = df_pipline(df, [add_od, period_split, add_congestion_index])

steps_set = PathSet(load=True, cache_folder='../cache', file_name='wkt_step')
gdf_steps = steps_set.get_shapes_with_bridges_info()
gdf_steps[ gdf_steps.bridge.notnull() ].plot()

# df.loc[:, 'holiday_or_weekend'] = (df.holiday) | (df.dayofweek >= 5)
# df, df_outliers = outlier_filter(df, group_cols = ['OD', 'date'])
# df_equal_weight = df.groupby(['o', 'd', 'time']).mean().dropna().reset_index()
# print( f"df_equal_weight size shrink {df.shape[0]}->{df_equal_weight.shape[0]}, cut down {(1-df_equal_weight.shape[0]/df.shape[0])*100:.2f}%" )
# df_equal_weight = period_split(df_equal_weight)

#%%
"""绘制四个时期的出行路径情况"""

# test single trip
# i, j = 7, 9
# params = {'i':i, 'j':j, 'verbose':True, 'o': cities_lst[i],  'd': cities_lst[j], 'plot_config': { 'color':'r'},'bak_config': { 'edgecolor':'white', 'alpha':0.5} }
# res = draw_subplot_travel_path(params, df.query('period==4'))
# res['pic']

# draw all figs
def raw_subplot_travel_path_4_period(fn="../cache/path_figs_4_periods.pkl"):
    res = {}
    for period in range(1,5,1):
        print(period)
        axes, figs, fig = gba_plot(df=df.query(f"period == {period}"), 
                            func=draw_subplot_travel_path, 
                            plot_config={'color':'r'}, 
                            bak_config= {'edgecolor':'white', 'alpha':0.5}, 
                            n_jobs=32, 
                            verbose=True, 
                            fig_name=f"travel_path_period_{period}", 
                            savefolder="./tmp", 
                            focus_a=.4
                            )
        res[period] = figs
    
    pickle.dump(res, open(fn, 'wb'))
    
    return res

# if __name__ == '__main__':
    # res = raw_subplot_travel_path_4_period()
# load data

#%%
"""以城市为单位，重新组织位置，便于分析"""
def trip_path_distribution(i, focus=True, focus_a=.4):
    dist_cols, dist_rows = 5, 11
    fig = plt.figure(figsize=(4*dist_cols, 3*dist_rows))
    for j in tqdm(range(len(cities_lst)), cities_lst[i]):
        if i ==j: 
            continue
        for p in range(1, 6, 1):
            alpha = 1 if (focus and 'bridge' in df.columns and df_query_by_od(df, i, j).query(f"period =={p}").bridge.dropna().nunique() > 0) else focus_a

            ax= plt.subplot(dist_rows, dist_cols, j*dist_cols+p)
            if p < 5:
                ax.imshow(paths_figs[p][(i, j)], alpha=alpha)
                ax.set_axis_off()
            else:
                data = df_query_by_od(df, i, j).merge(pois[['o', 'd', 'great_circle_dis']], on=['o', 'd'])
                data.loc[:, 'circuity_factor'] = data.travelDis / 1000 / data.great_circle_dis
                sns.boxplot(x='period', y='circuity_factor', data=data, ax=ax)
                ax.set_title(f"{cities_lst[i]}->{cities_lst[j]}")           

    plt.tight_layout()
    plt.close()

    fig.savefig(os.path.join(RESULT_FOLDER, f"trip_path_distribution_{cities_lst[i]}.jpg"), dpi=300)

    return fig
    
def trip_path_distribution_batch(focus = True, focus_a = .4):
    for i in range(11):
        trip_path_distribution(i)
    
    return

paths_figs =  pickle.load( open("../cache/path_figs_4_periods.pkl", 'rb') )
# fig = trip_path_distribution(3)
trip_path_distribution_batch()

#%%
"""绕行系数 & 行驶距离 线型图"""
# _ = plot_circuity_factor_under_4_period(df, fn ='../result/boxplot_circuity_factor_under_4_periods.jpg')
# _ = plot_dis_distribution_under_4_period(df, fn ='../result/boxplot_distance_under_4_period.jpg')


#%%
# df_od_related_to_bridges = get_trip_related_to_bridges(df)
# df_od_related_to_bridges.to_excel("../result/bridge_usage_ratio.xlsx")
# df_od_related_to_bridges



#%%
""" heat map related """

from collections import deque
from utils.graph_helper import Digraph, plot_heat_map, combine_almost_equal_edges, add_points

def plot_heatmap(gdf, ax=None, column='count', cmap='Reds', legend=True, *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    gdf.sort_values(by=column, ascending=False).plot(column=column, cmap=cmap, legend=legend, ax=ax)
    
    return ax


def heatmap_count(df_od, *args, **kwargs):
    # 数量热力图

    def split_linestring(line):
        res = []
        coords = line.coords[:]
        for i in range(len(coords)-1):
            res.append( (LineString([coords[i], coords[i+1]]).wkt, 1) )
        
        return res

    rdd = sc.parallelize(df_od.geometry.dropna().values, 32)

    res = rdd.flatMap(lambda line: split_linestring(line))\
            .map(lambda x: (x, 1))\
            .reduceByKey(lambda x,y: x+y)\
            .map(lambda x: (x[0][0], x[1]))\
            .collect()


    heat_trips = gpd.GeoDataFrame(res, columns=['geometry', 'count'])
    heat_trips.geometry = heat_trips.geometry.apply(lambda x: wkt.loads(x))

    plot_heatmap(heat_trips)

    return heat_trips


def find_almost_equals_sindex(lines, item, decimal=4):
    inds = np.setdiff1d( 
                        lines.sindex.query(box(*item.geometry.bounds)),
                        [item.name]
    )   
    candidates = lines.loc[inds]
    idx = candidates[candidates.geometry.apply(lambda x: x.almost_equals(item.geometry, decimal))].index

    return list(idx) if len(idx) > 0 else None


def find_almost_equals_wkt_version(lines, _wkt, decimal=4):
    geom = wkt.loads(_wkt)
    candidates = lines.loc[lines.sindex.query(box(*geom.bounds))]
    idx = candidates[candidates.geometry.apply(lambda x: x.almost_equals(geom, decimal))].index

    return list(np.sort(idx)) if len(idx) > 1 else None


p = 4; i = 3; j=4
df_od = df_query_by_od(df, i, j).query(f"period =={p}")
heat_trips = heatmap_count(df_od)

# %%
def combine_almost_equal_edges(data, verbose=True):
    from copy import deepcopy
    df = deepcopy(data)
    if 'check' not in df.columns:
        df.loc[:, 'check'] = False
    
    df = add_points(df)
    
    df.loc[:, 'almost_equal_num'] = df.almost_equal.apply(lambda x: 0 if x is None else len(x))
    df.sort_values('freq', ascending=False, inplace=True)

    graph = Digraph(df[['start', 'end']].values)
    graph_bak = Digraph()

    queue = deque([df[~df.check].iloc[0].name])
    visited, line_map = set(), {}
    prev_size = df.shape[0]

    while True:
        while queue:
            # print(f"{queue}, visited: {len(visited)}")
            node = queue.popleft()
            if node in visited:
                continue
            
            lst = df.loc[node].almost_equal
            try:
                _sum = df.loc[lst].freq.sum()
            except:
                print( "node: ", node, " lst: ", lst)
                continue
            
            if _sum > 1:
                df.loc[lst, "check"] = True
                continue

            df.loc[node, "freq"] = _sum
            df.loc[node, "check"] = True
            another_id = [x for x in lst if x != node][0]

            line_map[df.loc[another_id]._wkt] = df.loc[node]._wkt

            remove_edge = (df.loc[another_id].start, df.loc[another_id].end)
            graph.remove_edge(*remove_edge)
            graph_bak.add_edge(*remove_edge)

            for i in lst:
                visited.add(i)

            df.drop(index=another_id, inplace=True)

            nxt_start, nxt_end = df.loc[node, ['end', 'start']]
            nxts = df.query(f"(start == @nxt_start or end == @nxt_end) and almost_equal_num == 2 ").index
            for nxt in nxts:
                if nxt in visited:
                    continue
                if nxt == 6871:
                    print(df.query(f"(start == @nxt_start or end == @nxt_end) and almost_equal_num == 2 "))
                queue.append(nxt)

        if verbose: print(f'check total number: {df[df.check].shape[0]}, remain {df[~df.check].shape[0]}')
        if prev_size == df[~df.check].shape[0]:
            break
        
        prev_size = df[~df.check].shape[0]
        queue.append(df[~df.check].iloc[0].name)

    return df


def heatmap_ratio(df_od, merge_almost_equal=True, verbose=False,*args, **kwargs):
    # 比例热力图
    def split_linestring(item):
        key, line = item
        res = []
        coords = line.coords[:]
        for i in range(len(coords)-1):
            res.append( (key, LineString([coords[i], coords[i+1]]).wkt) )
        
        return res

    rdd = sc.parallelize(df_od[['OD', 'geometry' ]].dropna().values)
    rdd1 = rdd.flatMap(split_linestring).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).cache()

    trip_od_nums = rdd1.map(lambda x: (x[0][0], x[1])).reduceByKey(lambda x, y: max(x, y)).collect()
    trip_od_nums = { key: val for key, val in trip_od_nums }

    rdd2 = rdd1.map(lambda x: (x[0][1], x[1]/trip_od_nums[x[0][0]]))\
            .groupByKey().mapValues(list)\
            .map(lambda x: (x[0], np.mean(x[1])))\
            .sortBy(lambda x: x[1])\
            .cache()

    heat_trips_res = rdd2.collect()

    heat_trips = gpd.GeoDataFrame(heat_trips_res, columns=['_wkt', 'freq'])
    heat_trips.loc[:, 'geometry'] = heat_trips._wkt.apply(wkt.loads)
    heat_trips.reset_index(inplace=True)
    
    if merge_almost_equal:
        lines = heat_trips.copy()
        res = rdd2.map(lambda x: (x[0], find_almost_equals_wkt_version(lines, x[0]), x[1])).collect()
        df2 = pd.DataFrame(res, columns=['_wkt', 'almost_equal', 'freq'])
        df2.loc[:, 'geometry'] = df2._wkt.apply(wkt.loads)
        df2 = gpd.GeoDataFrame(df2)
        heat_trips = df2

        heat_trips = combine_almost_equal_edges(heat_trips, verbose)


    print(f"heatmap_ratio { 'with' if merge_almost_equal else 'without' } merge almost equal edge")
    heat_trips.sort_values('freq', inplace=True)
    plot_heatmap(heat_trips, column='freq')
    
    # heat_trips.to_file("./tmp/heat_trips_shenzhen_zhuhai_test_4.geojson", driver="GeoJSON")

    return

# df3 = combine_almost_equal_edges(df2)
heat_trips_ratio = heatmap_ratio(df_od)

# import datetime
# s = datetime.datetime.now()
# df3.almost_equal = df3.almost_equal.astype(str)
# df3.to_file("./tmp/heat_trips_shenzhen_zhuhai_combine_p4.geojson", driver="GeoJSON")

#%%


# lines = heat_trips.copy()
# item = lines.loc[0]
# a = find_almost_equals_wkt_version(lines, 'LINESTRING (113.501821 22.4997, 113.502779 22.497385)')
# b = find_almost_equals_wkt_version(lines, 'LINESTRING (113.501816 22.499699, 113.502781 22.497387)')
# print(a, b)


# %%