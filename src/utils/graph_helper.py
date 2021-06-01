#%%
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from collections import deque

from shapely.ops import transform

#%%

class Node:
    """
    Define the node in the road network 
    """

    def __init__(self, id):
        self.val = id
        self.x, self.y = [float(i) for i in id.split(',')]
        self.prev = set()
        self.nxt = set()
        self.indegree = 0
        self.outdegree = 0

    def add(self, point):
        self.nxt.add(point)
        self.outdegree += 1

        point.prev.add(self)
        point.indegree += 1

    def check_0_out_more_2_in(self):
        return self.outdegree == 0 and self.indegree >= 2

    def move_nxt_to_prev(self, node):
        if node not in self.nxt:
            return False

        self.nxt.remove(node)
        self.prev.add(node)
        self.indegree += 1
        self.outdegree -= 1
        return True

    def move_prev_to_nxt(self, node):
        if node not in self.prev:
            return False

        self.prev.remove(node)
        self.nxt.add(node)
        self.indegree -= 1
        self.outdegree += 1
        return True

class Digraph:
    def __init__(self, edges=None, cal_degree=True, *args, **kwargs):
        """[summary]

        Args:
            edges ([list], optional): [description]. Defaults to None.
        """
        self.graph = {}
        self.prev = {}
        if edges is not None:
            self.build_graph(edges)
        
        if cal_degree:
            self.calculate_degree()

    def __str__(self):
        return ""

    def add_edge(self, start, end):
        for p in [start, end]:
            for g in [self.graph, self.prev]:
                if p in g:
                    continue
                g[p] = set()

        self.graph[start].add(end)
        self.prev[end].add(start)
        pass

    def remove_edge(self, start, end):
        self.graph[start].remove(end)
        if len(self.graph[start]) == 0:
            del self.graph[start]
        
        self.prev[end].remove(start)
        if len(self.prev[end]) == 0:
            del self.prev[end]
        pass

    def build_graph(self, edges):
        for edge in edges:
            self.add_edge(*edge)
        return self.graph

    def clean_empty_set(self):
        """
        clean the empty node
        """
        for item in [self.prev, self.graph]:
            for i in list(item.keys()):
                if len(item[i]) == 0:
                    del item[i]
        pass
        
    def calculate_degree(self,):
        """caculate the indegree and outdegree of each node 

        Returns:
            [dataframe]: the degree dataframe of nodes
        """
        self.clean_empty_set()
        self.degree = pd.merge(
            pd.DataFrame([[key, len(self.prev[key])]
                          for key in self.prev], columns=['node_id', 'indegree']),
            pd.DataFrame([[key, len(self.graph[key])]
                          for key in self.graph], columns=['node_id', 'outdegree']),
            how='outer',
            on='node_id'
        ).fillna(0)
        self.degree[['outdegree', 'indegree']] = self.degree[['outdegree', 'indegree']].astype(np.int)
        
        return self.degree

    def get_origin_point(self,):
        self.calculate_degree()
        return self.degree.query( "indegree == 0 and outdegree != 0" ).node_id.values.tolist()

    def get_end_point(self, point):
        if point in self.graph:
            return self.graph[point]
        return None
    
    def _combine_edges_helper(self, origins, result=None, pre=None, roads=None, vis=False):
        """combine segment based on the node degree

        Args:
            origins ([type]): [description]
            result (list, optional): [Collection results]. Defaults to None.
            pre ([type], optional): The previous points, the case a node with more than 2 children. Defaults to None.
            roads (gpd.Geodataframe, optional): 道路数据框，含有属性 's' 和 'e'. Defaults to None.
            vis (bool, optional): [description]. Defaults to False.
        """
        for o in origins:
            pre_node = o
            path = []
            if pre is not None:
                path = [[pre,o]]
                self.remove_edge(pre,o)

            if o not in self.graph:
                return 
            
            # case: 0 indegree, > 2 outdegree
            if len(self.graph[o]) > 1:
                o_lst = list( self.graph[o] )
                self._combine_edges_helper( o_lst, result, o, roads, vis )
                return
            
            while o in self.graph and len(self.graph[o]) == 1:
                o = list(self.graph[o])[0]
                self.remove_edge( pre_node, o )
                path.append([pre_node, o])
                pre_node = o

            if roads is not None:
                assert hasattr(roads, 's') and hasattr(roads, 'e'), "attribute is missing"
                tmp = gpd.GeoDataFrame(path, columns=['s','e']).merge( roads, on=['s','e'] )
            
                ids = []
                for i in tmp.rid.values:
                    if len(ids) == 0 or ids[-1] != i:
                        ids.append(i)
                # ids = '_'.join(map(str, ids))

                if vis: map_visualize(tmp, 's')
                if result is not None: result.append([tmp, ids ])

            else:
                if result is not None: result.append([path, []])
            
        return

    def combine_edges(self, roads=None, vis=False):
        """roads 是一开始传入的roads的df文件

        Args:
            roads ([type], optional): [description]. Defaults to None.
            vis (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        import copy
        graph_bak = copy.deepcopy(self.graph)
        prev_back = copy.deepcopy(self.prev.copy())
        
        result = [] # path, road_id
        origins = self.get_origin_point()
        while len(origins) > 0:
            self._combine_edges_helper(origins, result, roads=roads)
            origins = self.get_origin_point()

        if roads is not None and vis:
            for i, _ in result:
                map_visualize(i, 's')
        
        self.graph = graph_bak
        self.prev = prev_back
        
        return result


def add_points(df, demical=6):
    df.loc[:, 'start'] = df.geometry.apply(lambda x: Point(np.round(x.coords[0], demical)).wkt)
    df.loc[:, 'end']   = df.geometry.apply(lambda x: Point(np.round(x.coords[-1], demical)).wkt)

    return df


def combine_almost_equal_edges(data, verbose=True):
    from copy import deepcopy
    df = deepcopy(data)
    if 'check' not in df.columns:
        df.loc[:, 'check'] = False
    
    df = add_points(df)
    
    df.loc[:, 'almost_equal_num'] = df.almost_equal.apply(lambda x: 0 if x is None else len(x))
    df.sort_values('freq', ascending=False, inplace=True)

    graph = Digraph(df[['start', 'end']].values)
    graph_bak = Digraph()

    queue = deque([df[~df.check].iloc[0].name])
    visited, line_map = set(), {}
    prev_size = df.shape[0]

    while True:
        while queue:
            # print(f"{queue}, visited: {len(visited)}")
            node = queue.popleft()
            if node in visited:
                continue
            
            lst = df.loc[node].almost_equal
            _sum = df.loc[lst].freq.sum()
            if _sum > 1:
                df.loc[lst, "check"] = True
                continue

            df.loc[node, "freq"] = _sum
            df.loc[node, "check"] = True
            another_id = [x for x in lst if x != node][0]

            line_map[df.loc[another_id]._wkt] = df.loc[node]._wkt

            remove_edge = (df.loc[another_id].start, df.loc[another_id].end)
            graph.remove_edge(*remove_edge)
            graph_bak.add_edge(*remove_edge)

            for i in lst:
                visited.add(i)

            df.drop(index=another_id, inplace=True)

            nxt_start, nxt_end = df.loc[node, ['end', 'start']]
            nxts = df.query(f"(start == @nxt_start or end == @nxt_end) and almost_equal_num == 2 ").index
            for nxt in nxts:
                if nxt in visited:
                    continue
                queue.append(nxt)

        if verbose: print(f'check total number: {df[df.check].shape[0]}, remain {df[~df.check].shape[0]}')
        if prev_size == df[~df.check].shape[0]:
            break
        
        prev_size = df[~df.check].shape[0]
        queue.append(df[~df.check].iloc[0].name)

    return df


def plot_heat_map(df):
    ax = df.plot(column='freq', legend=True)
    
    return ax

#%%

if __name__ == '__main__':
    df = gpd.read_file('./heat_maps.geojson').query("almost_equal != 'None' ")
    df.almost_equal = df.almost_equal.apply(eval)
    post = combine_almost_equal_edges(df)

    plot_heat_map(df)

    # df.almost_equal.apply(len).value_counts()


    # lst = [3911, 5114, 6008] # 有一条不是类似的曲线，其freq已超过1，可用hause_dorff_distance计算其记录
    # lst = [3081, 5114, 6008]
    # lst = [260, 4506, 4908] # freq sum == 1
    # lst = [699, 4530, 4885]
    # ax = df.loc[lst].plot(column='start', legend=True)
    # ax.set_title( df.loc[lst].freq.sum() )

