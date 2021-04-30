#!/usr/bin/python
# -*- coding: <utf-8> -*-
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm 
from utils.classes import PathSet
import threading
from joblib import Parallel,delayed
from shapely.geometry import LineString

import warnings
warnings.filterwarnings('ignore')

folder = '/home/pcl/Data/GBA/period3_4'

path_set = PathSet(load=True)


def process_path(fn, path_set, folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps'):
    gdf = gpd.read_file( os.path.join(folder, fn), encoding='utf-8')
    df = pd.read_csv('/home/pcl/Data/GBA/period3_4/GBA_trajectory_info_200506.csv')

    shapes_lst = list(gdf.geometry.unique())
    shapes_dict = { idx: {'coords': str(i.coords[:])} for idx, i in enumerate(shapes_lst)}
    for key, val in tqdm(shapes_dict.items(), "add to path_set"):
        # print(key, val.keys())
        idx = path_set.add(val['coords'])
        val['index'] = idx


    def apply_parallel(df, func):
        n_jobs = 52
        df.loc[:,'group'] = df.index % n_jobs
        df = df.groupby('group')
        results = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df)
        print("Done!")
        return pd.concat(results)


    def helper(gdf):
        gdf.loc[:,'ids'] = gdf.geometry.apply( lambda x:  shapes_dict[shapes_lst.index(x)]['index'] )

        return gdf

    print('apply_parallel')
    gdf = apply_parallel(gdf, helper)
    gdf.drop(columns=['geometry', 'group'], inplace=True)

    save_fn = os.path.join(save_folder, fn.replace('.shp', '.csv'))
    gdf.to_csv( save_fn, encoding='utf-8' )

    return gdf


def process_path_batch(folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps'):
    
    error_lst = []
    for fn in tqdm(os.listdir(folder)):
        if '路径规划_' not in fn or 'shp' not in fn:
            continue
        print(fn)
        
        try:
            df = process_path(fn, path_set, folder, save_folder)
        except:
            error_lst.append(fn)

    return error_lst

if __name__ == '__main__':
    error_lst = process_path_batch()
    path_set.save()
    print(error_lst)
    print(f"path_set size: {path_set.size}")
    
            
    
