#%%
import os
import pickle
import pyproj
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import transform
from shapely import wkt
import geopandas as gpd

class PathSet():
    def __init__(self, in_epsg=4326, out_epsg=2435, cache_folder="../../cache", file_name="path_set",load=True):
        self.path_to_key = {}
        self.key_to_elem = {}
        self.num = 0
        self.cache_folder = cache_folder
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        self.fn = os.path.join(cache_folder, file_name+".pkl")
        self.in_epsg = in_epsg
        self.out_epsg = out_epsg
        
        wgs84 = pyproj.CRS(f'EPSG:{in_epsg}')
        utm = pyproj.CRS(f'EPSG:{out_epsg}')
        self.project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        
        if load:
            self.load()
    
    def add(self, item):
        if item in self.path_to_key:
            return self.path_to_key[item]
        
        self.path_to_key[item] = self.num
        shape = wkt.loads(item)
        self.key_to_elem[self.num] = {'_wkt': item, 
                                      'geometry': shape,
                                      'length': self.cal_distance(shape)
                                      }
        self.num += 1

        return self.path_to_key[item]
    
    def addSet(self, lst):
        for path in list(lst):
            self.add(path)
        print('addSet success')
        
        return

    def addSet_df(self, lst, auto_save=True, verbose=False):
        import geopandas as gpd
        paths = gpd.GeoDataFrame(geometry = lst)
        
        paths.drop_duplicates(inplace=True)
        paths.set_crs(epsg = self.in_epsg, inplace=True)
        paths.loc[:, '_wkt']   = paths.geometry.apply( lambda x: x.wkt )
        paths.loc[:, 'length'] = paths.to_crs(epsg=self.out_epsg).length
        paths.rename(columns={'geometry': 'geometry'}, inplace=True)
        paths_dict = paths.to_dict(orient='records')

        res = {}
        prev_sum = self.num
        for item in paths_dict:
            if item['_wkt'] in self.path_to_key:
                res[item['_wkt']] = self.path_to_key[item['_wkt']]
                continue
            
            self.path_to_key[item['_wkt']] = self.num
            self.key_to_elem[self.num] = item.copy()
            self.num += 1
            if verbose:
                print(f"add {item['_wkt']} to db")
            res[item['_wkt']] = self.path_to_key[item['_wkt']]
        
        if auto_save and self.num > prev_sum:
            self.save()
        
        return res
    
    def get_key(self, path:str):
        if path not in self.path_to_key:
            return None
        
        return self.path_to_key[path]

    def get_path(self, index):
        if index not in self.key_to_elem:
            return None

        return self.key_to_elem[index]
    
    def get_shape(self, index):
        if index not in self.key_to_elem:
            return None

        return self.key_to_elem[index]['geometry']

    def get_shapes(self, fids):
        shapes = []
        for i in fids:
            shapes.append(self.key_to_elem[i])

        return gpd.GeoDataFrame(shapes).set_geometry('geometry')

    def convert_to_gdf(self):
        gdf = gpd.GeoDataFrame(self.key_to_elem).T
        gdf.set_crs(epsg=self.in_epsg, inplace=True)
        gdf.loc[:, 'fid'] = gdf.index
        
        return gdf
        
    def remove(self, path):
        if path not in self.path_to_key:
            return False
        
        index = self.path_to_key[path]
        del self.path_to_key[path]
        del self.key_to_elem[index]
        
        return True

    def cal_distance(self, path):
        return transform(self.project, path).length

    def save(self):
        data = {
            'path_to_key': self.path_to_key,
            "key_to_elem": self.key_to_elem,
            "num": self.num
        }
        
        pickle.dump(data, open(self.fn, 'wb'))

        return True
    
    def load(self):
        if not os.path.exists(self.fn):
            return False
        
        data = pickle.load( open(self.fn, 'rb') )
        self.path_to_key = data['path_to_key']
        self.key_to_elem = data['key_to_elem']
        self.num = data['num']
        
        return True
    
    @property
    def size(self):
        return len(self.key_to_elem)

    def get_shapes_with_bridges_info(self, bridges_fn="../db/bridges.geojson"):
        gdf_paths = self.convert_to_gdf()
        bridges = gpd.read_file(bridges_fn)
        
        gdf_paths = gpd.sjoin(left_df=gdf_paths, right_df=bridges[['bridge', 'geometry']], op='intersects', how='left') 
        
        return gdf_paths


#%%
if __name__ == '__main__':
    paths = PathSet()
    paths.addSet(["[(1,2), [3,4]]", '[(1,2), [3,5]]'])
    paths.key_to_elem
    paths.path_to_key
    paths.get_key("[(1,2), [3,4]]")
    paths.get_path(0)
    paths.get_shape(0)
    
    paths.remove('[(1,2), [3,5]]')

    paths.cal_distance(paths.get_shape(0))

    paths.add('[(1,2), [3,5]]')
    paths.add('[(1,2), [3,4]]')
    
    
    paths.save()

    #%%
    path_set = PathSet(load=True, cache_folder='../../cache', file_name='path_set_tmcs')
    gdf = path_set.convert_to_gdf()
    gdf
    # %%


    gdf.plot()

# %%
