#%%
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.windows import from_bounds
import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
import numpy as np
import zipfile
import xarray
import matplotlib.pyplot as plt
# %%
main_path=str(Path(Path(os.path.abspath(__file__)).parents[1]))
# %%
#here you can download the shapefiles of the NUTS regions:
# https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics

all_NUTS_years=[2003,2006,2010,2013,2016,2021]
#loop through all NUTS years
year=2021 #this would be assigned in a loop

NUTS_shapefile=gpd.read_file(main_path+"/data/raw/NUTS_RG_01M_2021_3035/NUTS_RG_01M_"+str(year)+"_3035.shp")
# %%

output_shape=rio.open(main_path+"/data/preprocessed/clipped_nuts.tif")
# %%
transform=output_shape.transform
# %%
#get the NUTS level from NUTS codes. The NUTS level is equal to the len of the NUTS code - 2 (countries have level 0)
NUTS_shapefile["LEVL_CODE"]=np.char.str_len(np.array(NUTS_shapefile.FID).astype(str))-2
#%%
NUTS0=gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"]==0])
NUTS1=gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"]==1])
NUTS2=gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"]==2])
NUTS3=gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"]==3])
# %%

NUTS0_regs=np.unique(NUTS0["FID"])
NUTS0_index=np.arange(len(NUTS0_regs))+1
NUTS0.sort_values(by="FID",inplace=True)
NUTS0["index"]=NUTS0_index
NUTS1_regs=np.unique(NUTS1["FID"])
NUTS1_index=np.arange(len(NUTS1_regs))+1
NUTS1.sort_values(by="FID",inplace=True)
NUTS1["index"]=NUTS1_index
NUTS2_regs=np.unique(NUTS2["FID"])
NUTS2_index=np.arange(len(NUTS2_regs))+1
NUTS2.sort_values(by="FID",inplace=True)
NUTS2["index"]=NUTS2_index
NUTS3_regs=np.unique(NUTS3["FID"])
NUTS3_index=np.arange(len(NUTS3_regs))+1
NUTS3.sort_values(by="FID",inplace=True)
NUTS3["index"]=NUTS3_index


#%%
geom_value = ((geom,value) for geom, value in zip(NUTS0.geometry, NUTS0.index))
NUTS0_raster=features.rasterize(
    geom_value,
    out_shape=output_shape.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS1.geometry, NUTS1.index))
NUTS1_raster=features.rasterize(
    geom_value,
    out_shape=output_shape.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS2.geometry, NUTS2.index))
NUTS2_raster=features.rasterize(
    geom_value,
    out_shape=output_shape.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS3.geometry, NUTS3.index))
NUTS3_raster=features.rasterize(
    geom_value,
    out_shape=output_shape.shape,
    transform=transform,
)
# %%
show(NUTS2_raster)

# %%
#TODO: Export raster with NUTS index and save in separate file which NUTS ID the index refers to
# %%
