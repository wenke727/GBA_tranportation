#%%
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm 
from utils.classes import PathSet
import threading

from shapely.geometry import LineString

import warnings
warnings.filterwarnings('ignore')

folder = '/home/pcl/Data/GBA/period3_4'

path_set = PathSet(load=True)


#%%
def process_steps(step_file, path_set, folder=folder):
    df_step = gpd.read_file(os.path.join(folder, step_file), encoding='utf-8')
    df_step.loc[:, 'day'] = df_step.ID.apply( lambda x: x [:2])
    df_step.loc[:, 'time'] = df_step.ID.apply( lambda x: x [2:6])
    df_step.loc[:, 'step_id'] = df_step.ID.apply( lambda x: x [6:]).astype(int) - df_step.tripID.astype(int)*100
    df_step.loc[:, 'path'] = df_step.geometry.apply(lambda x: str(x.coords[:]))
    # df_step.step_id.value_counts().sort_index()

    path_set.addSet(df_step['path'].unique())
    df_step.loc[:, 'path_id'] = df_step['path'].apply( lambda x: path_set.get_key(x) )

    df_step.drop(columns=['geometry', 'path'], inplace=True)
    df_step.loc[:, 'speed'] = df_step['distance'] / df_step['duration'] * 3.6
    # sns.kdeplot(df_step.sample(10**3).loc[:, 'speed'])
    
    return df_step

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


if __name__ == '__main__':
    process_steps_batch()
    

#%%
from joblib import Parallel,delayed



fn = 'GBA_路径规划_200506.shp'

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

process_path('GBA_路径规划_200508.shp', path_set)

#%%
def process_path_batch(folder = '/home/pcl/Data/GBA/period3_4', save_folder='/home/pcl/Data/GBA/steps'):
    
    error_lst = []
    for fn in tqdm(os.listdir(folder)):
        if '路径规划_' not in fn or 'shp' not in fn:
            continue
        print(fn)
        
        try:
            df = process_path(fn, path_set, folder)
        except:
            error_lst.append(fn)
                    
process_path_batch()



#%%


def collect_path(fn):
    try:
        df_step = gpd.read_file(fn, encoding='utf-8')
    except:
        return 
    print(f"starting: {fn}\n")
    
    global res 
    df_step.loc[:, 'day'] = df_step.ID.apply( lambda x: x [:2])
    df_step.loc[:, 'time'] = df_step.ID.apply( lambda x: x [2:6])
    df_step.loc[:, 'step_id'] = df_step.ID.apply( lambda x: x [6:]).astype(int) - df_step.tripID.astype(int)*100
    df_step.loc[:, 'path'] = df_step.geometry.apply(lambda x: str(x.coords[:]))
    df = df_step['path'].unique()
    res[fn] = df
    
    return df

lst = []
for fn in os.listdir(folder):
    if 'GBA_step_' not in fn or 'shp' not in fn:
        continue 
    if os.path.exists(fn.replace('.shp', '.csv')):
        continue
    
    lst.append( os.path.join(folder, fn))

res = {}
tasks = []
for fn in lst:
    tasks.append( threading.Thread(target=collect_path, args=(fn,)))

n_jonbs = 5
for i in range(0, len(tasks), n_jonbs):
    new_tasks = tasks[i: i+n_jonbs]
    print(i, new_tasks)

    for task in new_tasks:
        task.start()
    for task in new_tasks:
        task.join()
    
[(i, len(item)) for i, item in res.items()]




#%%

# fn = '/home/pcl/Data/GBA/output/GBA_路径规划_200509.shp'
# df = gpd.read_file(fn, encoding='utf-8')
# df.info()
# df.iloc[0].geometry


# fn = '/home/pcl/Data/GBA/output/GBA_路径规划_200509.shp'
# df = gpd.read_file(fn, encoding='utf-8')

# df.iloc[0].geometry


# #%%

# df_whole = pd.read_csv(os.path.join(folder, whole_csv))

# # df_whole.loc[:, 'day'] 
# df_whole.loc[:, 'tripID'] = df_whole.tripID.astype(str)
# df_whole.tripID.apply(lambda x: [x[:-7], x[-7:-3], x[-3:]])

# # %%


# # %%


# # %%
# df_step.to_csv('./test.csv', encoding='utf-8')


# #%%


# #%%



# # %%

# df_step.set_crs(epsg=4326, inplace=True)

# # %%
