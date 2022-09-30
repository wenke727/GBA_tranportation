
#%%
import os, io
import sys
import copy
import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from tqdm import tqdm
from shapely import wkt
from shapely.geometry import LineString
from multiprocessing import Pool, Lock
from PIL import Image
from df_helper import df_query, df_query_by_od, add_od

from utils.classes import PathSet

gba_area   = gpd.read_file("../db/gba_boundary.geojson")
bridges    = gpd.read_file("../db/bridges.geojson")
cities_lst = ["Hong Kong","Macao","Guangzhou","Shenzhen","Zhuhai","Foshan","Huizhou","Dongguan","Zhongshan","Jiangmen","Zhaoqing"]

#%%
"""plot related func"""
def plt_2_Image(fig):
    # method 1: PIL -> Image
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg',pad_inches=0, bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_new = copy.deepcopy(Image.open(buf))
    buf.close()

    # method 2: PIL -> Image
    # img_new = fig2data(fig)
    return img_new  


def draw_subplot(params:dict, df):
    # tmp = df_query_by_od(params['df'], params['o'], params['d'])
    tmp = df_query_by_od(df, params['i'], params['j'])

    if 'verbose' in params and params['verbose']:
        print(f"{params['o']} -> {params['d']}: {tmp.shape[0]}")

    try:
        fig, ax = plt.subplots()
        ax.set_title(f"{params['o']} -> {params['d']}")
        gba_area.plot(ax=ax, **params['bak_config'] if 'bak_config' in params else {})
        tmp.plot(ax=ax, **params['plot_config'] if 'plot_config' in params else {}) 
        ax.set_axis_off()
        
        params['fig'], params['pic'] = fig, plt_2_Image(fig)
        plt.close()
    except:
        print('draw_subplot error')
        raise ValueError

    return params


def draw_subplot_travel_path(params:dict, df):
    tmp = df_query_by_od(df, params['o'], params['d'])

    if 'verbose' in params and params['verbose']:
        print(f"{params['o']} -> {params['d']}: {tmp.shape[0]}")

    try:
        fig, ax = plt.subplots()
        ax.set_title(f"{params['o']} -> {params['d']}")
        gba_area.plot(ax=ax, **params['bak_config'] if 'bak_config' in params else {})
        tmp.plot(ax=ax, **params['plot_config'] if 'plot_config' in params else {}) 
        
        bridges_tmp = bridges.copy()
        bridges_tmp.geometry = bridges_tmp.buffer(0.01)
        bridges_tmp.plot(ax=ax, zorder=9, color='gray', alpha =.9)
        
        ax.set_axis_off()
        
        # params['fig'] = fig
        params['pic'] = plt_2_Image(fig)
        plt.close()
    except:
        print('draw_subplot error')
        raise ValueError

    return params


def gba_plot(df, func, n_jobs=56, verbose=False, plot_config={}, bak_config={}, focus=True, focus_a=.5, fig_name='path_dis_new', savefolder='./tmp' ):
    def _call_back_figs(params):
        axes.append(params)
        return True
    
    t_s = datetime.datetime.now()
    dist_rows = dist_cols = len(cities_lst)
    params_lst, axes = [], []
    for i in range(dist_rows):
        for j in range(dist_cols):
            if i ==j: 
                continue
            
            params = {'i':i, 
                      'j':j, 
                      'verbose':verbose, 
                      'o': cities_lst[i], 
                      'd': cities_lst[j], 
                      'plot_config': plot_config,
                      'bak_config': bak_config
                    #   'df': df
                    #   'df': df_query_by_od(df, cities_lst[i], cities_lst[j])
                      }
            params_lst.append(params)

    pools = Pool(n_jobs)
    pools.map_async(partial(func, df=df), params_lst, callback=_call_back_figs)
    pools.close()
    pools.join()        
   
    figs = {(i['i'], i['j']): i['pic'] for i in axes[0] }
    fig = plt.figure(figsize=(4*dist_cols, 3*dist_rows))
    for i in tqdm(range(dist_rows)):
        for j in range(dist_cols):
            if i ==j: 
                continue
            alpha = 1 if (focus and 'bridge' in df.columns and df_query_by_od(df, i, j).bridge.dropna().nunique() > 0) else focus_a
            ax = plt.subplot(dist_rows, dist_cols, i*dist_cols+j+1)
            ax.imshow(figs[(i,j)], alpha=alpha)
            ax.set_axis_off()

    plt.tight_layout()
    if fig_name is not None:
        fig.savefig(f"{savefolder}/{fig_name}.jpg", dpi=300)
    plt.close() 

    print(f"gba_plot done, cost: {datetime.datetime.now() - t_s} s")

    return axes, figs, fig


def prepare_data(fn='/home/pcl/Data/GBA/path/gba_20190330.csv'):
    # create a functions
    gdf = pd.read_csv(fn)
    path_set = PathSet(load=True, cache_folder='../cache/', file_name='wkt_path')
    df_paths = path_set.convert_to_gdf()
    gdf = add_od(gdf)
    gdf = gpd.GeoDataFrame( gdf.merge(df_paths, on='fid') )
    
    return gdf


def check_single():
    i = 3
    j = 4
    params = {'i':i, 
                'j':j, 
                'verbose':True, 
                'o': cities_lst[i], 
                'd': cities_lst[j], 
                'df': gdf,
                'plot_config': { 'column':'fid', 'legend':True, 'categorical':True}      
                }

    func = partial(draw_subplot, df=gdf)

    return func(params)['pic']


#%%
if __name__ == '__main__':
    fn = '/home/pcl/Data/GBA/path/gba_20200527.csv'
    # fn = '/home/pcl/Data/GBA/path/gba_20190403.csv'
    name = fn.split("/")[-1].split('.')[0]
    gdf = prepare_data(fn)
    # axes, fig = gba_plot(df=gdf, func=draw_subplot, n_jobs=56, verbose=False, fig_name=name, savefolder="./tmp")


    check_single()

# %%
