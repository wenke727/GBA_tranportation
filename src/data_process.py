#%%
import os
import numpy as np
import geopandas as gpd
from numpy.lib.arraysetops import unique
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm 
from utils.classes import PathSet
import threading
import time
from joblib import Parallel,delayed
from shapely.geometry import LineString
from utils.tools import reduce_mem_usage

import warnings
warnings.filterwarnings('ignore')

folder = '/home/pcl/Data/GBA/period3_4'

pd.set_option("display.max_rows", 100)

# path_set = PathSet(load=True, cache_folder='../cache')
# path_set_tmcs = PathSet(load=True, cache_folder='../cache', file_name='path_set_tmcs_tmcs')
# path_set_step = PathSet(load=True, cache_folder='../cache', file_name='path_set_step')

#%%
def process_path(fn, path_set, folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps'):
    gdf = gpd.read_file( os.path.join(folder, fn), encoding='utf-8')

    gdf.loc[:, 'path'] = gdf.geometry.apply( lambda x: np.round(x.coords[:], 6).tolist())
    gdf.loc[:, 'geometry'] = gdf.path.apply( lambda x: LineString(x))
    gdf.path = gdf.path.astype(str)

    unique_geoms = gdf.geometry.unique()
    path_set.addSet_df(unique_geoms)
    path_set.save()

    gdf = gdf.merge(path_set.convert_to_gdf(), on='path')
    gdf.drop(columns=['geometry', 'path', 'shape'], inplace=True)
    
    # shapes_lst = list(unique_geoms)
    # shapes_dict = { idx: {'coords': str(i.coords[:])} for idx, i in enumerate(shapes_lst)}
    # for key, val in tqdm(shapes_dict.items(), "add to path_set"):
    #     # print(key, val.keys())
    #     # idx = path_set.add(val['coords'])
    #     val['index'] = path_set.get_key(val['coords'])
    
    # def apply_parallel(df, func, n_jobs = 16):
    #     df.loc[:,'group'] = df.index % n_jobs
    #     df = df.groupby('group')
    #     results = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df)
    #     print("Done!")
    #     return pd.concat(results)

    # def helper(gdf):
    #     # gdf.loc[:,'ids'] = gdf.geometry.apply( lambda x:  shapes_dict[shapes_lst.index(x)]['index'] )
    #     gdf.loc[:,'ids'] = gdf.geometry.apply( lambda x:  path_set.get_key(str(x.coords[:])) )

    #     return gdf

    # print('apply_parallel')
    # gdf = apply_parallel(gdf, helper)
    
    # gdf.loc[:,'ids'] = gdf.geometry.apply( lambda x:  path_set.get_key(str(x.coords[:])) )
    # gdf.drop(columns=['geometry', 'group'], inplace=True)
    gdf.sort_index(inplace=True)
    
    save_fn = os.path.join(save_folder, fn.replace('.shp', '.csv'))
    gdf.to_csv( save_fn, encoding='utf-8' )

    return gdf


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


def process_steps_old(step_file, path_set, folder=folder):
    df_step = gpd.read_file(os.path.join(folder, step_file), encoding='utf-8')
    df_step.loc[:, 'day']     = df_step.ID.apply( lambda x: x [:2])
    df_step.loc[:, 'time']    = df_step.ID.apply( lambda x: x [2:6])
    df_step.loc[:, 'step_id'] = df_step.ID.apply( lambda x: x [6:]).astype(int) - df_step.tripID.astype(int)*100
    df_step.loc[:, 'path']    = df_step.geometry.apply(lambda x: str(x.coords[:]))
    # df_step.step_id.value_counts().sort_index()

    path_set.addSet(df_step['path'].unique())
    df_step.loc[:, 'path_id'] = df_step['path'].apply( lambda x: path_set.get_key(x) )

    df_step.drop(columns=['geometry', 'path'], inplace=True)
    df_step.loc[:, 'speed'] = df_step['distance'] / df_step['duration'] * 3.6
    # sns.kdeplot(df_step.sample(10**3).loc[:, 'speed'])
    
    return df_step


def process_steps(fn, save_folder='/home/pcl/Data/GBA/steps', verbose=True):
    gdf = gpd.read_file(fn, encoding='utf-8')
    df  = pd.read_csv(fn.replace('shp', 'csv'), encoding='utf-8')

    gdf.rename(columns={'tripID': 'OD'}, inplace=True)
    atts = ['OD', 'distance', 'duration', 'tolls']
    gdf[atts] = gdf[atts].astype(int)
    reduce_mem_usage(gdf, True)
    reduce_mem_usage(df, True)

    # check the data is the same or not
    atts = ['distance', 'duration']
    assert df.shape[0] == gdf.shape[0], "check"
    assert sum([(df[i] - gdf[i]).sum() for i in atts]) == 0, "check"

    gdf = gdf[['ID', 'OD', 'geometry']].merge(df, left_index=True, right_index=True)
    gdf.loc[:, 'step_id'] = gdf.tripID.apply(lambda x: x.split("_")[-1]).astype(int)

    path_set_step.addSet_df(gdf['geometry'].unique())

    gdf.loc[:, 'fid'] = gdf['geometry'].apply( lambda x: path_set_step.get_key(str(x.coords[:])) )

    date_ = '20'+fn.split("_")[-1].split('.')[0]
    gdf.loc[:, 'date'] = int(date_)
    
    def cal_time(x):
        offset = 2 if str(x.date)[-2:] == x.ID[:2] else 1
        t = x.ID[offset: offset+4]
        t = int(t)//100 + int(str(t)[-2:])/60
        return t 
    
    gdf.loc[:, 't'] = gdf.apply(cal_time, axis=1)

    drop_atts = ['tripID', 'geometry']
    gdf.drop(columns=drop_atts, inplace=True)
    gdf.rename(columns={'ID': 'tripID'}, inplace=True)

    atts_order = [ 'OD', 'step_id', 'tripID', 'instruction', 'orientation', 'road', 'distance', 'tolls', 'toll_distance', 'toll_road', 'duration', 'action',  'assistant_action', 'fid', 'date', 't']
    gdf = gdf[atts_order]

    if save_folder is not None:
        to_fn = os.path.join(save_folder, fn.split('/')[-1].replace('.shp', '.h5') )
        pd.DataFrame(gdf).to_hdf(to_fn, 'steps')

    return gdf


def process_steps_ver_for_period_1_2(fn, save_folder='/home/pcl/Data/GBA/steps', verbose=True):
    df = pd.read_csv(fn)
    date_ = '20'+fn.split("_")[-1].split('.')[0]

    df.loc[:, "OD"] = df.tripID.apply( lambda x: x % 1000 )
    df.loc[:, "date"] = int(date_)

    def cal_time(x):
        offset = 2 if str(x.date)[-2:] == str(x.tripID)[:2] else 1
        t = str(x.tripID)[offset: offset+4]
        t = int(t)//100 + int(str(t)[-2:])/60
        return t 

    df.loc[:, 't'] = df.apply(cal_time, axis=1)

    df_start_step = df.reset_index().groupby(['OD', 't'])[['index']].min().reset_index().rename(columns={'index': 'start_step_index'})

    df = df.merge( df_start_step, on = ['OD', 't'])
    df.loc[:,'step_id'] = df.index - df.start_step_index
    df.drop(columns='start_step_index', inplace=True)

    df.loc[:, 'fid'] = None
    atts_order = [ 'OD', 'step_id', 'tripID', 'instruction', 'orientation', 'road', 'distance', 'tolls', 'toll_distance', 'toll_road', 'duration', 'action',  'assistant_action', 'fid', 'date', 't']
    df = df[atts_order]
    
    if save_folder is not None:
        to_fn = os.path.join(save_folder, fn.split('/')[-1].replace('.csv', '.h5') )
        df.to_hdf(to_fn, 'steps')

    return df


def process_steps_start():
    folder = '/home/pcl/Data/GBA/period3_4'
    fn_lst = []
    for f in os.listdir(folder):
        if 'shp' in f and 'step' in f:
            fn_lst.append(os.path.join(folder, f))
    fn_lst.sort()

    for i in tqdm(fn_lst):
        process_steps(i)

    folder = "/home/pcl/Data/GBA/1904/"
    fn_lst = []
    for f in os.listdir(folder):
        if 'csv' in f and 'step' in f:
            fn_lst.append(os.path.join(folder, f))
    fn_lst.sort()

    for i in tqdm(fn_lst):
        df = process_steps_ver_for_period_1_2(i)


def process_steps_batch(folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps'):
    error_lst = []
    for fn in tqdm(os.listdir(folder)):
        if 'step_' not in fn or 'shp' not in fn:
            continue 

        save_fn = os.path.join(save_folder, fn.replace('.shp', '.csv'))

        if os.path.exists(save_fn):
            continue

        print("path_set.num: ", path_set.num)
        
        try:
            df = process_steps(fn, path_set, folder)
            pd.DataFrame(df).to_csv(save_fn, encoding='utf-8')
        except:
            error_lst.append(fn)
    
    return error_lst


def process_tmcs(fn, path_set_tmcs, folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps', test=False):
    if test:
        gdf = gpd.read_file( os.path.join(folder, fn), encoding='utf-8', rows=2000)
    else:
        gdf = gpd.read_file( os.path.join(folder, fn), encoding='utf-8')

    for att in ['lineID', 'tripID', 'distance', 'status']:
        gdf[att] = gdf[att].astype(int)
    gdf = reduce_mem_usage(gdf)

    # shapes_lst = list(gdf.geometry.unique())
    # shapes_dict = { idx: {'coords': str(i.coords[:])} for idx, i in enumerate(shapes_lst)}
    # for key, val in tqdm(shapes_dict.items(), "add to path_set_tmcs"):
    #     idx = path_set_tmcs.add(val['coords'])
    #     val['index'] = idx
    path_set_tmcs.addSet_df(gdf.geometry.unique(), verbose=False)

    ids = gdf.geometry.apply( lambda x: path_set_tmcs.get_key(str(x.coords[:])) )

    gdf.drop(columns=['geometry'], inplace=True)
    gdf.loc[:, 'path_id'] = ids
    reduce_mem_usage(gdf)
    
    # save_fn = os.path.join(save_folder, fn.replace('.shp', '.csv'))
    # gdf.to_csv( save_fn, encoding='utf-8' )
    h5 = pd.HDFStore(os.path.join(folder, fn.replace('.csv', '.h5')),'w', complevel=4, complib='blosc')
    h5[fn.split(".")[0]] = df
    h5.close()

    return gdf


def process_tmcs_batch(folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps', test=False ):
    fn_lst, error_lst = [], []
    for i in os.listdir(folder):
        if 'tmcs' in i and 'shp' in i:
            fn_lst.append(i)
    fn_lst.sort()

    for fn in fn_lst:
        try:
            localtime = time.asctime( time.localtime(time.time()) )
            print(f"{fn}, path_set_tmcs size: {path_set_tmcs.size}, start at {localtime}")
            process_tmcs(fn, path_set_tmcs, folder, save_folder, test=test)
        except:
            error_lst.append(fn)

    path_set_tmcs.save()
    print("error list: ", error_lst)
    
    return 


#%%
if __name__ == '__main__':
    # for path_analysis
    path_set_path = PathSet(load=False, cache_folder='../cache', file_name='path_set_path')
    process_path_batch(folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps', path_set_path=path_set_path)
    process_path_batch(folder = '/home/pcl/Data/GBA/period1_2', save_folder='/home/pcl/Data/GBA/steps', path_set_path=path_set_path)
    
