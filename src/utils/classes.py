#%%
import os
import pickle
import pyproj
import numpy as np
from shapely.geometry import Point
from shapely.ops import transform
from shapely.geometry import LineString
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
        shape = LineString( eval(item) )
        self.key_to_elem[self.num] = {'path': item, 
                                      'shape': shape,
                                      'length': self.cal_distance(shape)
                                      }
        self.num += 1

        return self.path_to_key[item]
    
    def addSet(self, lst):
        for path in list(lst):
            self.add(path)
        print('addSet success')
        
        return

    def addSet_df(self, lst, verbose=False):
        import geopandas as gpd
        paths = gpd.GeoDataFrame(geometry = lst)
        paths.drop_duplicates(inplace=True)
        paths.set_crs(epsg = self.in_epsg, inplace=True)
        paths.loc[:, 'path']   = paths.geometry.apply( lambda x: str(np.round(x.coords[:], 6).tolist()) )
        paths.loc[:, 'length'] = paths.to_crs(epsg=self.out_epsg).length
        paths.rename(columns={'geometry': 'shape'}, inplace=True)
        paths_dict = paths.to_dict(orient='records')

        res = {}
        for item in paths_dict:
            if item['path'] in self.path_to_key:
                res[item['path']] = self.path_to_key[item['path']]
                continue
            
            self.path_to_key[item['path']] = self.num
            self.key_to_elem[self.num] = item.copy()
            self.num += 1
            if verbose:
                print(f"add {item['path']} to db")
            res[item['path']] = self.path_to_key[item['path']]
        
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

        return self.key_to_elem[index]['shape']

    def get_shapes(self, fids):
        shapes = []
        for i in fids:
            shapes.append(self.key_to_elem[i])

        return gpd.GeoDataFrame(shapes).set_geometry('shape')

    def convert_to_gdf(self):
        gdf = gpd.GeoDataFrame(self.key_to_elem).T
        gdf.set_geometry("shape", inplace=True)
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
