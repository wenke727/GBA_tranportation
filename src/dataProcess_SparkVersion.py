#%%
import imp
import os, io
import copy
import datetime
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ogr
from shapely import wkt
from shapely.geometry import LineString

from utils.classes import PathSet

from multiprocessing import Pool, Lock
from PIL import Image

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

"""df helper"""
from df_helper import df_query, df_query_by_od, add_od
from plot_helper import plt_2_Image, draw_subplot, gba_plot

# spark = SparkSession.builder.config(conf=SparkConf().setMaster('local[12]')).getOrCreate()
# sc = SparkContext(conf=SparkConf().setMaster('local[6]').set("spark.executor.memory", "48g"))
# rdd = sc.parallelize(df[['id', '_wkt']])

cities_lst = ["Hong Kong","Macao","Guangzhou","Shenzhen","Zhuhai","Foshan","Huizhou","Dongguan","Zhongshan","Jiangmen","Zhaoqing"]

#%%
def add_geom_atts(df, att='coords'):
    df.loc[:, 'geometry'] = df[att].apply( lambda x: LineString(np.round(x, 6)))
    df.loc[:, '_wkt'] = df.geometry.apply(lambda x: x.wkt)

    return df


def apply_parallel(df, func, n_jobs = 56):
    from joblib import Parallel, delayed

    df.loc[:,'group'] = df.index % n_jobs
    df = df.groupby('group')
    results = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df)
    print("Done!")
    df = pd.concat(results)
    df = df.drop(columns=['coords', 'group']).sort_index()

    return df


def shp_parser_multiprocess():
    shpfile = '/home/pcl/Data/GBA/db/GBA_路径规划_190403.shp'
    shp = ogr.GetDriverByName("ESRI Shapefile").Open(shpfile)
    lyr = shp.GetLayerByIndex(0)
    lyd = lyr.GetLayerDefn()

    fields   = [lyd.GetFieldDefn(i).GetName() for i in range(lyd.GetFieldCount())]
    features = [ (i, lyr.GetFeature(i)) for i in tqdm(range(lyr.GetFeatureCount())) ]

    gdf = pd.DataFrame(features, columns=['id', 'obj'])
    for idx, key in enumerate(fields):
        gdf.loc[:, key] = gdf.obj.apply( lambda x: x.GetField(idx) )
    gdf.loc[:, 'coords'] = gdf.obj.apply( lambda x: x.GetGeometryRef().GetPoints())

    df = gdf.copy()
    n_jobs = 32
    df.drop(columns='obj', inplace=True)
    df.loc[:,'group'] = df.index % n_jobs
    dfs = [ i for _, i in df.groupby('group')]
    print(len(dfs))
    
    df_lst = []
    def collate_fn(df):
        print(df.shape[0])
        df_lst.append(df)
        
        return True
    
    pools = Pool(n_jobs)
    pools.map_async(add_geom_atts, dfs, callback=collate_fn)
    pools.close()
    pools.join()        

    print(len(df_lst))


def shp_parser(shpfile, n_jobs=16 ):
    # shpfile = '/home/pcl/Data/GBA/db/GBA_路径规划_190403.shp'
    shp = ogr.GetDriverByName("ESRI Shapefile").Open(shpfile)
    lyr = shp.GetLayerByIndex(0)
    lyd = lyr.GetLayerDefn()

    fields   = [lyd.GetFieldDefn(i).GetName() for i in range(lyd.GetFieldCount())]
    features = [ (i, lyr.GetFeature(i)) for i in range(lyr.GetFeatureCount()) ]

    gdf = pd.DataFrame(features, columns=['id', 'obj'])
    for idx, key in enumerate(fields):
        gdf.loc[:, key] = gdf.obj.apply( lambda x: x.GetField(idx) )
    gdf.loc[:, 'coords'] = gdf.obj.apply( lambda x: x.GetGeometryRef().GetPoints())
    
    if n_jobs is not None:
        df = gdf
        df.drop(columns=['obj'], inplace=True)
        s = datetime.datetime.now()
        df = apply_parallel(df, add_geom_atts, n_jobs)
        print(datetime.datetime.now() -s)

        return df

    s = datetime.datetime.now()
    gdf.loc[:, 'geometry'] = gdf.coords.apply( lambda x: LineString(np.round(x, 6)) )
    gdf.loc[:, '_wkt'] = gdf.geometry.apply(lambda x: x.wkt)
    print(f"shp_parser finished at: ", datetime.datetime.now() -s)

    return gpd.GeoDataFrame(gdf)


#%%
if __name__ == '__main__':
    # for params in params_lst[:4]:
    #     draw_subplot(params)

    # shpfile = '/home/pcl/Data/GBA/db/GBA_路径规划_190326.shp'
    shpfile = '/home/pcl/Data/GBA/db/GBA_路径规划_190403.shp'
    df = shp_parser(shpfile, n_jobs=32)
    df = add_od(df)
    gba_area  = gpd.read_file("../db/gba_boundary.geojson")
    df = gpd.GeoDataFrame(df)

    path_set = PathSet(cache_folder="../cache", file_name='wkt_path')
    path_set.addSet_df(gpd.GeoDataFrame(df).geometry.unique(), True)
    path_set.save()

    gdf_paths = path_set.convert_to_gdf()
    df = df.merge(gdf_paths[['_wkt', 'fid']], on='_wkt')
    df[['tripID', 'fid']]
    df = gpd.GeoDataFrame(df)

    axes,fig = gba_plot(df, draw_subplot, verbose=False, fig_name='travel_path_distribution')
    print("Done")

# %%
import time
def process_path_batch(path_set_path, folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps'):
    # path_set = path_set_path
    
    fn_lst, error_lst = [], []
    for fn in tqdm(os.listdir(folder)):
        if '路径规划_' not in fn or 'shp' not in fn:
            continue
        fn_lst.append(fn)
    fn_lst.sort()
    fn = fn_lst[0]
    
    for fn in fn_lst:
        try:
            print(f"{time.asctime( time.localtime(time.time()) )}, {fn}")
            df = process_path(fn, path_set_path, folder, save_folder)
        except:
            error_lst.append(fn)

    return error_lst
