#%%
import os
import math
import numpy as np
import geopandas as gpd
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm 
from utils.classes import PathSet
import threading
import time
from joblib import Parallel, delayed
from shapely.geometry import LineString
from utils.tools import reduce_mem_usage

import warnings
warnings.filterwarnings('ignore')


#%%
def tranfer_time(x):
    m, h = math.modf(x.t)
    return f"{x.date} {int(h):02d}:{int(m*60):02d}"

def path_process(fn, folder, path_set, save_folder=None, check = True):
    date_ = '20'+fn.split("_")[-1].split('.')[0]
    if os.path.exists(f"{save_folder}/gba_{date_}.csv"):
        return

    gdf = gpd.read_file( os.path.join(folder, fn), encoding='utf-8')
    if check:
        gdf_origin = gdf.copy()

    gdf.loc[:, 'path'] = gdf.geometry.apply( lambda x: np.round(x.coords[:], 6).tolist())
    gdf.loc[:, 'geometry'] = gdf.path.apply( lambda x: LineString(x))
    gdf.path = gdf.path.astype(str)
    gdf.rename(columns={'tripID':'OD', 
                        'traffic_lig':'traffic_lights', 
                        'toll_distan':'toll_distance', 
                        'verycongest':'verycongested'}, 
               inplace=True)
    atts = ['OD', 'duration','travelDis','toll_distance','traffic_lights']
    gdf[atts] = gdf[atts].astype(np.int)

    unique_geoms = gdf.geometry.unique()
    path_set.addSet_df(unique_geoms)
    path_set.save()

    csv_fn = fn.replace('路径规划', 'trajectory_info').replace('.shp', '.csv')
    df_csv = pd.read_csv(os.path.join(folder, csv_fn), encoding='utf-8')
    tmp = gdf.merge(df_csv, left_index=True, right_index=True)
    assert ((tmp['duration_x'] - tmp['duration_y'])!=0).sum() == 0 and \
        ((tmp.travelDis_x - tmp.travelDis_y)!=0).sum() == 0, \
            f"check the order of the driving direction path in {fn}"

    gdf = gdf.merge(df_csv[['tripID']], left_index=True, right_index=True)

    gdf.loc[:, "date"] = date_
    gdf.loc[:, "t"] = gdf.tripID.apply(lambda x: str(x)[-7:-3])
    gdf.loc[:, "t"] = gdf.t.apply(lambda x:  int(x)//100 + int(str(x)[-2:])/60 ) 

    gdf = gdf.merge(path_set.convert_to_gdf(), on='path')
    gdf.sort_values(['t','OD'], inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    gdf.drop(columns=['geometry', 'path', 'shape'], inplace=True)
    
    if check:
        status = ((gdf.duration - gdf_origin.duration != 0).sum() == 0) 
        assert status, f"check the order of the driving direction path in {fn}"

    if save_folder is not None:
        gdf.to_csv( f"{save_folder}/gba_{date_}.csv", encoding='utf-8' )

    return 

def start_path_process(folder = '/home/pcl/Data/GBA/db', save_folder = '/home/pcl/Data/GBA/path'):
    path_set = PathSet(load=True, cache_folder='../cache', file_name='path_features')

    filter_str = '路径'
    driving_path_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'shp' in i]
    driving_path_lst.sort()
    print(driving_path_lst)
    
    for fn in tqdm(driving_path_lst):
        print(f"{fn}, {path_set.size}")
        path_process(fn, folder, path_set, save_folder, True )

def post_process(folder_ori = '/home/pcl/Data/GBA/db', folder = '/home/pcl/Data/GBA/path', store_fn = '/home/pcl/Data/GBA/gba_db.h5' ):
    visited_date = [ fn.replace('gba_20', 'GBA_trajectory_info_') for fn in os.listdir(folder) ]
    visited_date.sort()

    un_visited_fns = [ fn for fn in os.listdir(folder_ori) if  ("GBA_trajectory_info_" in fn) and ('csv' in fn) and (fn not in visited_date)]
    un_visited_fns.sort()

    dfs = []
    for fn in os.listdir(folder):
        dfs.append(pd.read_csv(os.path.join(folder, fn), encoding='utf-8'))
    df_visited = pd.concat(dfs)

    dfs = []
    for fn in un_visited_fns:
        try:
            date_ = '20'+fn.split("_")[-1].split('.')[0]
            df = pd.read_csv(os.path.join(folder_ori, fn), encoding='utf-8')
            df.loc[:, 'date'] = date_
            df.loc[:, 'OD'] = df.tripID
            df.loc[:, "t"] = df.tripID.apply(lambda x: str(x)[-7:-3])
            df.loc[:, "t"] = df.t.apply(lambda x:  int(x)//100 + int(str(x)[-2:])/60 ) 
            dfs.append(df)
        except:
            print(f"load {fn} error")
    df_unvisited = pd.concat(dfs)
    del dfs

    df = pd.concat([df_visited, df_unvisited] )
    df.loc[:, "OD"] = df.loc[:, "OD"] % 1000

    t_filter = [0.25,  0.75,  1.25,  1.75,  2.25,  2.75, 3.25,  3.75,  4.25,  4.75,  5.25,  5.75]
    df.query(f't not in {t_filter}', inplace=True)


    df.loc[:, "time"] = df.apply(tranfer_time, axis=1)
    df.loc[:, "time"] = pd.to_datetime( df.loc[:, "time"] )

    df.sort_values( ['time', 'OD'], inplace=True ) 
    df.reset_index(drop=True, inplace=True)
    df.fid = df.fid.fillna(-1).astype(np.int)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    int_atts = ['OD','duration', 'travelDis','toll_distance','tolls','traffic_lights','tripID','date']
    df[int_atts] = df[int_atts].astype(np.int)

    period_bins = ['20190302 1200', '20190402 1200', '20200505 1400', '20200515 0900', '20211230 0900']
    period_bins = [ pd.to_datetime(x) for x in period_bins]
    period_labels = [ i+1 for i in range(len(period_bins)-1)]

    df.loc[:, 'period'] = pd.cut(df.time, period_bins, labels=period_labels)
    reduce_mem_usage(df, True)

    t_filter = [0.25,  0.75,  1.25,  1.75,  2.25,  2.75, 3.25,  3.75,  4.25,  4.75,  5.25,  5.75]
    df.query(f't not in {t_filter}', inplace=True)

    df.drop(columns='strategy', inplace=True) # FIXME `strategy`: '速度最快'

    if store_fn is not None:
        df.to_hdf(store_fn, key='trajectory_info', format="table")

    return df

# 原来eda整理数据
def read_path_file(fn, verbose=False):
    if verbose: print(f"read_path_file: {fn}")
    date_ = '20'+fn.split("_")[-1].split('.')[0]
    df = pd.read_csv(fn)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'index', }).set_index('index').sort_index()
    df.rename(columns={'tripID':'OD'}, inplace=True)
    
    csv_fn = fn.replace('路径规划', 'trajectory_info')
    if "路径规划" in fn and os.path.exists(csv_fn):
        df_origin = pd.read_csv(csv_fn)
        tmp = df.merge(df_origin, left_index=True, right_index=True)
        assert (tmp['duration_x'] - tmp['duration_y']).sum() == 0  and \
            (tmp.travelDis_x - tmp.travelDis_y).sum()==0, \
                f"check the order of the driving direaction path in {fn}"

        df = df.merge(df_origin[['tripID']], left_index=True, right_index=True)
        df.rename(columns={'traffic_lig':'traffic_lights', 'toll_distan':'toll_distance', 'verycongest':'verycongested'}, inplace=True)
    else:
        df.loc[:, 'tripID'] = df.OD
        df.loc[:, 'OD'] = df.OD % 1000
    
    df.loc[:, "date"] = date_
    df.loc[:, "t"] = df.tripID.apply(lambda x: str(x)[-7:-3])
    df.loc[:, "t"] = df.t.apply(lambda x:  int(x)//100 + int(str(x)[-2:])/60 ) 

    return df

def tranfer_time(x):
    m, h = math.modf(x.t)
    return f"{x.date} {int(h):02d}:{int(m*60):02d}"

def read_path_files(store_fn = '/home/pcl/Data/GBA/gba_db.h5', folder = '/home/pcl/Data/GBA/path'):
    filter_str = '路径'
    driving_path_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'csv' in i]
    driving_path_lst += [os.path.join(folder, f"GBA_trajectory_info_1903{i}.csv") for i in range(22, 26, 1)] 

    # folder = '/home/pcl/Data/GBA/1904'
    # filter_str = 'trajectory_info'
    # driving_path_lst += [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'csv' in i]
    # driving_path_lst.sort()

    driving_path_lst.sort()
    df_lst, error_lst = [], []
    for fn in driving_path_lst:
        # try:
        df = read_path_file(fn, True)
        df_lst.append(df)
        # except:
        #     error_lst.append(fn)
        #     print(f"{fn} error")
    
    df = pd.concat(df_lst)

    atts = ['duration', 'travelDis', 'toll_distance', 'tolls', 'traffic_lights']
    df[atts] = df[atts].astype(int)

    df.loc[:, "time"] = df.apply(tranfer_time, axis=1)
    df.loc[:, "time"] = pd.to_datetime( df.loc[:, "time"] )

    df.sort_values( ['time', 'OD'], inplace=True ) 
    df.reset_index(drop=True, inplace=True)
    df.fid = df.fid.fillna(-1).astype(np.int)

    period_bins = ['20190302 1200', '20190402 1200', '20200505 1400', '20200515 0900', '20211230 0900']
    period_bins = [ pd.to_datetime(x) for x in period_bins]
    # period_labels = [f'Period {i+1}' for i in range(len(period_bins)-1)]
    period_labels = [ i+1 for i in range(len(period_bins)-1)]

    df.loc[:, 'period'] = pd.cut(df.time, period_bins, labels=period_labels)
    reduce_mem_usage(df, True)

    t_filter = [0.25,  0.75,  1.25,  1.75,  2.25,  2.75, 3.25,  3.75,  4.25,  4.75,  5.25,  5.75]
    df.query(f't not in {t_filter}', inplace=True)
 
    df.drop(columns='strategy', inplace=True) # FIXME `strategy`: '速度最快'
    if store_fn is not None:
        df.to_hdf(store_fn, key='trajectory_info', format="table")

    return df



if __name__ == '__main__':
    start_path_process()
    df = post_process()

    # df = read_path_files('/home/pcl/Data/GBA/gba_db.h5')
    # check
    # gdf = gpd.read_file("/home/pcl/Data/GBA/db/GBA_路径规划_200602.shp", encoding='utf-8')
    # df_0602 = pd.read_csv("/home/pcl/Data/GBA/path/gba_20200602.csv", encoding='utf-8')
    # df_0602


#%%
# # %%
# from multiprocessing import Process, Lock, Pool

# def path_process(fn, folder, path_set, save_folder=None, check=True, lock=None):
#     # def path_process(params_dict):
#     # print(params_dict)
#     # fn = params_dict['fn']
#     # folder = params_dict['folder']
#     # path_set = params_dict['path_set']
#     # save_folder = params_dict['save_folder']
#     # check = params_dict['check']
#     # lock = params_dict['lock']
    
#     print("Worker process id for {0}".format(os.getpid())) 
    
#     gdf = gpd.read_file( os.path.join(folder, fn), encoding='utf-8')
#     if check:
#         gdf_origin = gdf.copy()

#     date_ = '20'+fn.split("_")[-1].split('.')[0]

#     gdf.loc[:, 'path'] = gdf.geometry.apply( lambda x: np.round(x.coords[:], 6).tolist())
#     gdf.loc[:, 'geometry'] = gdf.path.apply( lambda x: LineString(x))
#     gdf.path = gdf.path.astype(str)
#     gdf.rename(columns={'tripID':'OD', 
#                         'traffic_lig':'traffic_lights', 
#                         'toll_distan':'toll_distance', 
#                         'verycongest':'verycongested'}, 
#                inplace=True)
#     atts = ['OD', 'duration','travelDis','toll_distance','traffic_lights']
#     gdf[atts] = gdf[atts].astype(np.int)

#     unique_geoms = gdf.geometry.unique()
    
#     if lock is not None: 
#         lock.acquire()
#     path_set.addSet_df(unique_geoms)
#     path_set.save()
#     print(f"{fn}, {path_set.size}")
#     if lock is not None: 
#         lock.release()

#     csv_fn = fn.replace('路径规划', 'trajectory_info').replace('.shp', '.csv')
#     df_csv = pd.read_csv(os.path.join(folder, csv_fn), encoding='utf-8')
#     tmp = gdf.merge(df_csv, left_index=True, right_index=True)
#     assert ((tmp['duration_x'] - tmp['duration_y'])!=0).sum() == 0 and \
#         ((tmp.travelDis_x - tmp.travelDis_y)!=0).sum() == 0, \
#             f"check the order of the driving direction path in {fn}"

#     gdf = gdf.merge(df_csv[['tripID']], left_index=True, right_index=True)

#     gdf.loc[:, "date"] = date_
#     gdf.loc[:, "t"] = gdf.tripID.apply(lambda x: str(x)[-7:-3])
#     gdf.loc[:, "t"] = gdf.t.apply(lambda x:  int(x)//100 + int(str(x)[-2:])/60 ) 

#     gdf = gdf.merge(path_set.convert_to_gdf(), on='path')
#     gdf.sort_values(['t','OD'], inplace=True)
#     gdf.reset_index(drop=True, inplace=True)
#     gdf.drop(columns=['geometry', 'path', 'shape'], inplace=True)
    
#     if check:
#         status = ((gdf.duration - gdf_origin.duration != 0).sum() == 0) 
#         assert status, f"check the order of the driving direction path in {fn}"

#     if save_folder is not None:
#         gdf.to_csv( f"{save_folder}/gba_{date_}.csv", encoding='utf-8' )

#     return 

# def start_path_process(folder = '/home/pcl/Data/GBA/db', save_folder = '/home/pcl/Data/GBA/path/parrellel'):
#     path_set = PathSet(load=True, cache_folder='../cache', file_name='path_features_parrellel')

#     filter_str = '路径'
#     driving_path_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'shp' in i]
#     driving_path_lst.sort()
#     print(driving_path_lst)

#     lock = Lock()
#     f = partial(path_process, folder=folder, path_set=path_set, save_folder=save_folder, check=True, lock=lock)
    
    
#     # for i in range(0, len(driving_path_lst) + 5, 6):
        
#     f_lst = []
#     for fn in driving_path_lst[:6]:
#         f_lst.append(Process(target=path_process, args=(fn, folder, path_set, save_folder, True, lock)))
#         # f_lst.append(Process(target=f, args=(fn,)))
    
#     for i in f_lst:
#         i.start()

#     for i in f_lst:
#         i.join()


    
# if __name__ == '__main__':
#     start_path_process()
    
# # %%
# def f(a, b, c):
#     print("{} {} {}".format(a, b, c))

# def main():
#     iterable = [1, 2, 3, 4, 5]
#     pool = Pool()
#     a_ = "hi"
#     b_ = "there"
#     func = partial(f, a=a_, b=b_)
#     pool.map(func, iterable)
#     pool.close()
#     pool.join()

# if __name__ == "__main__":
#     main()
# %%
