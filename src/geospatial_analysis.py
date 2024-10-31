#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import features
import numpy as np
import os
from pathlib import Path
from shapely.geometry import Point
# %%
main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]

raw_dir = data_main_path+"/raw"
preprocessed_dir = data_main_path+"/preprocessed"
preprocessed_csv_dir=preprocessed_dir+"/csv/"
preprocessed_raster_dir=preprocessed_dir+"/rasters/"
os.makedirs(preprocessed_raster_dir, exist_ok=True)

parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
GEE_data_path=raw_dir+"/GEE/"
#%%
d = {'col1': ['name1','name2'], 'geometry': [Point(7.902716,51.064265),Point(7.902716,50.064265)]}
gdf = gpd.GeoDataFrame(d, crs="epsg:4326")

gdf = gdf.to_crs(crs="epsg:3035")
#%%

with rio.open(GEE_data_path+"oc.tif") as src:
    #for val in src.sample([tuple(np.array(gdf["geometry"].iloc[0]))]): 
    for val in src.sample(list(zip(np.array(gdf.geometry.x),np.array(gdf.geometry.y)))): 
        print(val)
    
# %%
tuple(np.array(gdf.geometry))

# %%
geometry=gpd.points_from_xy(x=LUCAS_preprocessed_imported.th_long,y=LUCAS_preprocessed_imported.th_lat)
# %%
geometry_3035=gpd.GeoDataFrame({"geometry":geometry},crs="epsg:4326").to_crs("epsg:3035")
# %%
geometry_3035[:100]
# %%
with rio.open(GEE_data_path+"oc.tif") as src:
    #for val in src.sample([tuple(np.array(gdf["geometry"].iloc[0]))]): 
    for val in src.sample(list(zip(np.array(geometry_3035[:10000].geometry.x),np.array(geometry_3035[:10000].geometry.y)))): 
        print(val)
# %%
