#%%
import imp
import os
import ogr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import geometry
from tqdm import tqdm
from shapely.geometry import LineString
from shapely import wkt
from utils.decorators import timer
from utils.classes import PathSet

#%%

from spark_helper import sc

path_set = PathSet(cache_folder='../cache', file_name='wkt_step')

#%%
def fid_matching(df):
    values = list(df[['id','coords']].values)
    rdd = sc.parallelize( values, 56 )

    rdd1 = rdd.map(lambda x: (x[0], LineString(np.round(x[1], 6)).wkt))

    unique_lines = rdd1.map(lambda x: x[1]).distinct().collect()
    lines = gpd.GeoDataFrame(unique_lines, columns=['geometry'])
    lines.geometry = lines.geometry.apply(lambda x: wkt.loads(x))

    line_dict = { line: i for i, line in enumerate(unique_lines) }
    lines_dict = path_set.addSet_df(lines.geometry.values)

    relations = rdd1.map(lambda x: (x[0], line_dict[x[1]])).collect()
    relations = pd.DataFrame(relations, columns=['id', 'fid'])

    df = df.merge(relations, on='id')

    df.drop(columns='coords', inplace=True)
    
    return df


def shp_parser(fn, n_jobs=16, test=True, addFeilds=True, path_set=None, save_folder='/home/pcl/Data/GBA/steps' ):
    date_ = '20'+fn.split("_")[-1].split('.')[0]
    shp = ogr.GetDriverByName("ESRI Shapefile").Open(fn)
    lyr = shp.GetLayerByIndex(0)
    lyd = lyr.GetLayerDefn()

    fields   = [lyd.GetFieldDefn(i).GetName() for i in range(lyd.GetFieldCount())]
    features = [(i, lyr.GetFeature(i)) for i in tqdm(range(lyr.GetFeatureCount() if not test else 100) , date_) ]

    gdf = pd.DataFrame(features, columns=['id', 'obj'])
    gdf.loc[:, 'date'] = date_

    if addFeilds:
        for idx, key in tqdm(enumerate(fields)):
            gdf.loc[:, key] = gdf.obj.apply( lambda x: x.GetField(idx) )
        
        gdf.loc[:, 'coords'] = gdf.obj.apply( lambda x: np.round(x.GetGeometryRef().GetPoints(), 6) )
        gdf.drop(columns='obj', inplace=True)

    # print(gdf['date'].unique(), gdf.shape[0])
    # gdf.loc[:, '_wkt'] = gdf.obj.apply( lambda x: x.GetGeometryRef().ExportToWkt())
    # gdf.geometry = gdf.coords.apply(lambda x: LineString(x))
    
    if path_set is not None:
        gdf = fid_matching(gdf)
        path_set.save()
    if save_folder is not None:
        gdf.to_hdf(os.path.join(save_folder, f"{date_}.h5"), 'step')

    return gdf, fields


def read_shp_batch(folder, path_set, filter_str = 'step', n_jobs=8, test=True):
    from multiprocessing import Pool, Lock
    step_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'shp' in i]
    step_lst.sort()
    print(step_lst, len(step_lst))

    df_lst, error_lst = pd.DataFrame(), []
    # def collate_fn(df):
    #     df_lst.append(df)

    # pools = Pool(n_jobs)
    # pools.map_async(shp_parser, step_lst[:2], callback=collate_fn)
    # # pools.apply_async(shp_parser, step_lst[:5], callback=collate_fn)
    # pools.close()
    # pools.join()        

    for fn in tqdm(step_lst, "read_shp_batch"):
        try:
            df, fields = shp_parser(fn, test=test, addFeilds=True, path_set=path_set)
            # df_lst = df_lst.append(df, ignore_index=True)
        except:
            error_lst.append(fn)
    print("error_lst: ", error_lst)
            
    return True


if __name__ == '__main__':
    df, _ = read_shp_batch(folder='/home/pcl/Data/GBA/db', path_set=path_set, test=False)
    # df.to_hdf("./tmp/steps_all.h5", 'step')
 
#%%

fn = '/home/pcl/Data/GBA/steps/20200506.h5'
df = pd.read_hdf(fn)

# %%
# 'id', 'date', 'ID', 'tripID', 'road', 'distance', 'duration', 'tolls', 'fid'
df[[ 'road', 'distance', 'duration', 'tolls','fid']].drop_duplicates()
# %%
df[[ 'road', 'distance', 'duration', 'tolls']].drop_duplicates()

# %%
