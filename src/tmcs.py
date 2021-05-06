import geopandas as gpd

df = gpd.read_file('../../GBA_tmcs_200528.shp', iterator=True)

df.info()

