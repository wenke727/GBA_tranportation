
# %%
from shapely import wkt
from shapely.geometry import LineString, box
from utils.graph_helper import *


def find_almost_equals_wkt_version(lines, _wkt, decimal=4):
    geom = wkt.loads(_wkt)
    candidates = lines.loc[lines.sindex.query(box(*geom.bounds))]
    idx = candidates[candidates.geometry.apply(lambda x: x.almost_equals(geom, decimal))].index

    return list(np.sort(idx)) if len(idx) > 1 else None


trips = trips_freq.query('OD==372 and period==4')
heatmap_data = trips.sort_values('nums', ascending=True)
heatmap_data = gdf_shps[['fid', 'geometry','_wkt']].merge(heatmap_data, on ='fid', how='right')

# find_almost_equals_wkt_version(heatmap_data, gdf_shps.iloc[2206]._wkt, 4)
# heatmap_data.loc[:, 'freq'] = heatmap_data.nums 
heatmap_data.loc[:, 'almost_equal'] = heatmap_data._wkt.apply(lambda x: find_almost_equals_wkt_version(heatmap_data, x))
heatmap_data.loc[[17, 56]]

#%%

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
            print(f"{queue}, visited: {len(visited)}")
            node = queue.popleft()
            if node in visited:
                continue
            
            lst = df.loc[node].almost_equal
            try:
                _sum = df.loc[lst].freq.sum()
            except:
                print( "node: ", node, " lst: ", lst)
                continue
            
            # if _sum > 1:
            #     df.loc[lst, "check"] = True
            #     continue

            df.loc[node, "freq"] = _sum
            df.loc[node, "check"] = True
            another_id = [x for x in lst if x != node][0]

            line_map[df.loc[another_id]._wkt] = df.loc[node]._wkt

            remove_edge = (df.loc[another_id].start, df.loc[another_id].end)
            try:
                graph.remove_edge(*remove_edge)
                graph_bak.add_edge(*remove_edge)
            except:
                print(f"\tremove {remove_edge} error!")

            for i in lst:
                visited.add(i)

            size_ = df.shape[0]
            df.drop(index=another_id, inplace=True)
            print(f"\t{lst}, drop records: {size_} -> {df.shape[0]}")

            nxt_start, nxt_end = df.loc[node, ['end', 'start']]
            nxts = df.query(f"(start == @nxt_start or end == @nxt_end) and almost_equal_num == 2 ").index
            for nxt in nxts:
                if nxt in visited:
                    continue
                if nxt == 6871:
                    print(df.query(f"(start == @nxt_start or end == @nxt_end) and almost_equal_num == 2 "))
                queue.append(nxt)

        if verbose: print(f'check total number: {df[df.check].shape[0]}, remain {df[~df.check].shape[0]}')
        if prev_size == df[~df.check].shape[0]:
            break
        
        prev_size = df[~df.check].shape[0]
        queue.append(df[~df.check].iloc[0].name)

    return df

heatmap_data = combine_almost_equal_edges(heatmap_data)


# %%
# heatmap_data[heatmap_data.almost_equal.isna()].plot(column='freq', legend=True)
# heatmap_data[~heatmap_data.almost_equal.isna()].plot(column='freq', legend=True)
heatmap_data.sort_values('freq').plot(column='freq', legend=True, cmap='Reds')

# %%
save_to_geojson(heatmap_data, 'od_372_p4', ['almost_equal'])
# TODO 精细化的绘制：1）地图匹配, 肯能这才是正解；2）拆分为最小点对，然后almost_equal，再group


# %%

def split_line_to_coord_pairs(line):
    coords = line.coords[:]
    res = [ (coords[i], coords[i+1]) for i in range(len(coords)-1) ]
    
    return res

heatmap_data.loc[:,'coords'] = heatmap_data.geometry.apply(split_line_to_coord_pairs)
heatmap_data_fine = gpd.GeoDataFrame( pd.DataFrame(heatmap_data).explode('coords').reset_index(drop=True) )
heatmap_data_fine.loc[:, 'geometry'] = heatmap_data_fine.coords.apply(LineString)
heatmap_data_fine.loc[:, 'start'] = heatmap_data_fine.coords.apply(lambda x: Point(x[0]))
heatmap_data_fine.loc[:, 'end']   = heatmap_data_fine.coords.apply(lambda x: Point(x[1]))
heatmap_data_fine

heatmap_data_fine.loc[:, 'almost_equal'] = heatmap_data_fine.geometry.apply(lambda x: find_almost_equals_wkt_version(heatmap_data_fine, x.to_wkt()))

#%%

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
            print(f"{queue}, visited: {len(visited)}")
            node = queue.popleft()
            if node in visited:
                continue
            
            lst = df.loc[node].almost_equal
            try:
                _sum = df.loc[lst].freq.sum()
            except:
                print( "node: ", node, " lst: ", lst)
                continue
            
            # if _sum > 1:
            #     df.loc[lst, "check"] = True
            #     continue

            df.loc[node, "freq"] = _sum
            df.loc[node, "check"] = True
            another_id = [x for x in lst if x != node][0]

            line_map[df.loc[another_id]._wkt] = df.loc[node]._wkt

            remove_edge = (df.loc[another_id].start, df.loc[another_id].end)
            try:
                graph.remove_edge(*remove_edge)
                graph_bak.add_edge(*remove_edge)
            except:
                print(f"\tremove {remove_edge} error!")

            for i in lst:
                visited.add(i)

            size_ = df.shape[0]
            df.drop(index=another_id, inplace=True)
            print(f"\t{lst}, drop records: {size_} -> {df.shape[0]}")

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

    print(f"dataframe size: {data.shape[0]} -> {df.shape[0]}, sum: {data.freq.sum():.0f} -> {df.freq.sum():.0f}")
    
    return df.sort_values('freq')


tmp = combine_almost_equal_edges(heatmap_data_fine)
tmp.plot(column='freq', legend=True, cmap='Reds')
tmp
save_to_geojson(tmp, 'tmp', ['almost_equal', 'coords'])


# %%


# p3 p4 records
sql = f"""SELECT  * 
        FROM step 
        WHERE fid IN ({','.join(list(fids_bridge.astype(str)))})
        """
steps = spark.sql(sql).distinct().cache()
steps.show()

# %%
steps.count()

# %%
steps.select('road').distinct().show()
steps.select('instruction').distinct().collect()


