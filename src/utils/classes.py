import os
import pickle
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from shapely.geometry import LineString

class PathSet():
    def __init__(self, in_epsg=4326, out_epsg=2435, cache_folder="../../cache", load=True):
        self.path_to_key = {}
        self.key_to_elem = {}
        self.num = 0
        self.cache_folder = cache_folder
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        
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
        
        pickle.dump(data, open(f'{self.cache_folder}/path_set.pkl', 'wb'))

        return True
    
    def load(self):
        data = pickle.load( open(f'{self.cache_folder}/path_set.pkl', 'rb') )

        self.path_to_key = data['path_to_key']
        self.key_to_elem = data['key_to_elem']
        self.num = data['num']
        
        return True
    
    @property
    def size(self):
        return len(self.key_to_elem)


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
