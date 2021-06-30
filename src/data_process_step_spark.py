#%%
import os
import ogr
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import LineString
from shapely import wkt
from utils.classes import PathSet
from spark_helper import sc

# path_set = PathSet(cache_folder='../cache', file_name='wkt_step')
path_set = PathSet(cache_folder='../cache', file_name='wkt_step_0617')

#%%
def fid_matching(df):
    """matching feature with PathSet object

    Args:
        df (pd.DataFrame): [description]

    Returns:
        [pd.DataFrame]: [description]
    """
    values = list(df[['id','coords']].values)
    rdd = sc.parallelize( values, 56 )

    rdd1 = rdd.map(lambda x: (x[0], LineString(np.round(x[1], 6)).wkt))

    unique_lines = rdd1.map(lambda x: x[1]).distinct().collect()
    lines = gpd.GeoDataFrame(unique_lines, columns=['geometry'])
    lines.geometry = lines.geometry.apply(lambda x: wkt.loads(x))

    lines_dict = path_set.addSet_df(lines.geometry.values)

    relations = rdd1.map(lambda x: (x[0], lines_dict[x[1]])).collect()
    relations = pd.DataFrame(relations, columns=['id', 'fid'])

    df = df.merge(relations, on='id')

    df.drop(columns='coords', inplace=True)
    
    return df


def shp_parser(fn, test=False, addFeilds=True, path_set=None, save_folder='/home/pcl/Data/GBA/steps' ):
    date_ = '20'+fn.split("_")[-1].split('.')[0]
    try:
        shp = ogr.GetDriverByName("ESRI Shapefile").Open(fn)
        lyr = shp.GetLayerByIndex(0)
        lyd = lyr.GetLayerDefn()
    except:
        print(f"shp_parser {fn} error")
        return None

    fields   = [lyd.GetFieldDefn(i).GetName() for i in range(lyd.GetFieldCount())]
    features = [(i, lyr.GetFeature(i)) for i in tqdm(range(lyr.GetFeatureCount() if not test else 100), date_) ]

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

    return gdf


def read_shp_batch(folder, path_set, filter_str = 'step', n_jobs=4, test=True):
    from multiprocessing import Pool, Lock
    step_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'shp' in i]
    step_lst.sort()
    print(step_lst, len(step_lst))

    df_lst, error_lst = pd.DataFrame(), []
    # For the sake of order, the parallel scheme is not used
    # pools = Pool(n_jobs)
    # res = pools.map_async(shp_parser, step_lst).get()
    # pools.close()
    # pools.join() 

    for fn in tqdm(step_lst, "read_shp_batch"):
        try:
            shp_parser(fn, test=test, addFeilds=True, path_set=path_set)
        except:
            error_lst.append(fn)
    print("error_lst: ", error_lst)
            
    return True

#%%
if __name__ == '__main__':
    read_shp_batch(folder='/home/pcl/Data/GBA/db', path_set=path_set, test=False)
    # df.to_hdf("./tmp/steps_all.h5", 'step')

#%%

gdf = shp_parser(fn='/home/pcl/Data/GBA/db/GBA_step_200523.shp', save_folder=None)

#%%
