#%%
import os
import sys
import ogr
import pandas as pd
import numpy as np
import datetime
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import LineString
from shapely import wkt
from utils.classes import PathSet
from utils.spark_helper import sc, spark
from config import STEP_CSV_FOLDER, STEP_SHP_FOLDER

TMP_FOLDER = '/home/pcl/Data/GBA/steps/test_0803'
path_set = PathSet(cache_folder='../cache', file_name='wkt_step')

# TODO load error Segmentation fault (core dumped)

#%%
def matching_fid(df, lock=None):
    """matching feature with PathSet object

    Args:
        df (pd.DataFrame): [description]

    Returns:
        [pd.DataFrame]: [description]
    """
    values = list(df[['id','coords']].values)
    rdd = sc.parallelize( values, 56 )

    rdd1 = rdd.map(lambda x: (x[0], LineString(np.round(x[1], 6)).wkt)).cache()
    unique_lines = rdd1.map(lambda x: x[1]).distinct().collect()
    
    lines = gpd.GeoDataFrame(unique_lines, columns=['geometry'])
    lines.geometry = lines.geometry.apply(lambda x: wkt.loads(x))
    
    if lock is not None:
        lock.acquire()
    
    lines_dict = path_set.addSet_df(lines.geometry.values, auto_save=True)
    relations = rdd1.map(lambda x: (x[0], lines_dict[x[1]])).collect()

    if lock is not None:
        lock.release()

    relations = pd.DataFrame(relations, columns=['id', 'fid'])
    df = df.merge(relations, on='id')
    df.drop(columns='coords', inplace=True)
    
    return df


def batch_read_shp(folder, path_set, filter_str='step', test=False):
    fn_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'shp' in i]
    fn_lst.sort()
    print(fn_lst, len(fn_lst))

    df_lst, error_lst = pd.DataFrame(), []

    for fn in tqdm(fn_lst, "batch_read_shp"):
        try:
            shp_parser(fn, test=test, add_shp_fields=True, path_set=path_set)
        except:
            error_lst.append(fn)
    print("error_lst: ", error_lst)
            
    return True


def shp_parser(fn, 
               test=False, 
               add_shp_fields=True, 
               path_set=None, 
               save_folder=TMP_FOLDER,
               drop_atts=['road', 'distance', 'duration', 'tolls'],
               format='csv',
               verbose=True
               ):
    try:
        shp = ogr.GetDriverByName("ESRI Shapefile").Open(fn)
        lyr = shp.GetLayerByIndex(0)
        lyd = lyr.GetLayerDefn()
    except:
        print(f"shp_parser {fn} error")
        return None

    date_ = '20'+fn.split("_")[-1].split('.')[0]
    fields   = [lyd.GetFieldDefn(i).GetName() for i in range(lyd.GetFieldCount()) if lyd.GetFieldDefn(i).GetName() not in drop_atts]
    features = [(i, lyr.GetFeature(i)) for i in tqdm(range(lyr.GetFeatureCount() if not test else 100), date_) ]

    gdf = pd.DataFrame(features, columns=['id', 'obj'])
    gdf.loc[:, 'date'] = date_
    
    if add_shp_fields:
        for idx, key in tqdm(enumerate(fields), f'\t add fields in shp {date_} file'):
            gdf.loc[:, key] = gdf.obj.apply( lambda x: x.GetField(idx) )
        
        gdf.loc[:, 'coords'] = gdf.obj.apply( lambda x: np.round(x.GetGeometryRef().GetPoints(), 6) )
        gdf.drop(columns='obj', inplace=True)

    # gdf.loc[:, '_wkt'] = gdf.obj.apply( lambda x: x.GetGeometryRef().ExportToWkt())
    if path_set is not None:
        gdf = matching_fid(gdf)

    if save_folder is not None:
        if format =='csv':
            gdf.to_csv(os.path.join(save_folder, f"{date_}.csv"), index=False)
        if format =='h5':
            gdf.to_hdf(os.path.join(save_folder, f"{date_}.h5"), 'step')
        
    if verbose:
        print(f"{sys._getframe(0).f_code.co_name} {date_} done!")

    return gdf


def parallel_read_shp():
    from multiprocessing import Pool, Lock
    from functools import partial
    
    # TODO: The program takes up too much memory
    folder = STEP_SHP_FOLDER
    filter_str = 'step'

    step_lst  = [os.path.join(folder, i) for i in os.listdir(folder) if filter_str in i and 'shp' in i]
    step_lst.sort()

    f = partial(shp_parser)

    pools = Pool(16)
    pools.map_async(f, step_lst)
    pools.close()
    pools.join()
    
    return True


def merge_with_step_csv(fn, csv_folder=STEP_CSV_FOLDER, verbose=True):
    if verbose: 
        print(f'merge_with_step_csv {fn}, {os.getpid()} .')
  
    data = pd.read_csv(fn).rename(columns={'id':'index'})
    
    df = pd.read_csv( os.path.join( csv_folder, f"GBA_step_{fn.split('/')[-1][2:8]}.csv" ) )
    data = data.merge(df[ [ x for x in df.columns if x not in data.columns] ], left_on='index', right_index=True)
    
    data.to_csv(fn, index=False)
    if verbose: 
        print(f'\tmerge_with_step_csv {fn} done.')
    
    return data


def merge_with_step_csv_batch(folder=TMP_FOLDER, csv_folder=STEP_CSV_FOLDER, n_jobs=32, filter=None):
    from multiprocessing import Pool
    s = datetime.datetime.now()

    fns = [os.path.join(folder, f) for f in os.listdir(folder)]
    fns.sort()

    pools = Pool(n_jobs)
    pools.map_async(merge_with_step_csv, fns)
    pools.close()
    pools.join()

    print(f"load data done, {datetime.datetime.now() -s}")
    
    return True


def convert_2_spark_df(input=TMP_FOLDER, out_fn='/home/pcl/Data/GBA/db/spark/steps_p3_p4.parquet'):
    from utils.df_spark_helper import df_pipline, parser_tripID_info, split_period

    df_1 = spark.read.format('csv').load(input, header=True, inferSchema=True, dateFormat="yyyyMMdd")
    df_1 = df_pipline(df_1, [parser_tripID_info, split_period])
    df_1 = df_1.repartition('date')

    s = datetime.datetime.now()
    print('begin to write')
    df_1.write.format('parquet').mode('overwrite').save(out_fn)
    print(datetime.datetime.now() -s)
    
    return True

#%%
if __name__ == '__main__':
    """single file"""
    # fn = f'{STEP_SHP_FOLDER}/GBA_step_200523.shp'
    # gdf = shp_parser(fn, path_set=path_set, save_folder=None, test=False)
    
    # step 1: read files in sequence 
    batch_read_shp(folder=STEP_SHP_FOLDER, path_set=path_set)

    # step 2: merge data
    merge_with_step_csv_batch()

    # step 3: convert to spark dataframe
    convert_2_spark_df(TMP_FOLDER, '/home/pcl/Data/GBA/db/spark/steps_p3_p4_210803.parquet')

# %%