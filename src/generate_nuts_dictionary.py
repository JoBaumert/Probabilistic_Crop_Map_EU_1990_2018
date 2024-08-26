#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio import features
from rasterio.transform import from_bounds
import numpy as np
import os
from pathlib import Path

main_path=str(Path(Path(os.path.abspath(__file__)).parents[1]))
result_dir = main_path+"/data/results/"
os.makedirs(result_dir, exist_ok=True)

# List of years to process
years = [2003, 2006, 2010, 2013, 2016, 2021]

#%%
# Dictionary

combined_df = pd.DataFrame()

for year in years:

    
    file_path = main_path+"/data/raw/nuts_shapedata/NUTS_RG_01M_" + str(year) + "_3035.shp/NUTS_RG_01M_" + str(year) + "_3035.shp"
    
    # Load the shapefile
    NUTS_shapefile = gpd.read_file(file_path, encoding='latin1')
    
    #Get the NUTS level from NUTS codes
    NUTS_shapefile["LEVL_CODE"] = np.char.str_len(np.array(NUTS_shapefile.NUTS_ID).astype(str)) - 2
    
    # Create NUTS levels
    NUTS0 = gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"] == 0])
    NUTS1 = gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"] == 1])
    NUTS2 = gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"] == 2])
    NUTS3 = gpd.GeoDataFrame(NUTS_shapefile[NUTS_shapefile["LEVL_CODE"] == 3])
    
    #Assign index and sort
    NUTS0.sort_values(by="NUTS_ID", inplace=True)
    NUTS0["index"] = np.arange(len(NUTS0)) + 1
    
    NUTS1.sort_values(by="NUTS_ID", inplace=True)
    NUTS1["index"] = np.arange(len(NUTS1)) + 1
    
    NUTS2.sort_values(by="NUTS_ID", inplace=True)
    NUTS2["index"] = np.arange(len(NUTS2)) + 1
    
    NUTS3.sort_values(by="NUTS_ID", inplace=True)
    NUTS3["index"] = np.arange(len(NUTS3)) + 1
    
    temp_df = pd.concat([NUTS0, NUTS1, NUTS2, NUTS3], ignore_index=True)
    temp_df["year"] = year
    
    temp_df = temp_df[["NUTS_ID", "LEVL_CODE", "index", "year"]]

    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# Generate Dictionary 
combined_df.to_csv(os.path.join(result_dir, "nuts_regions_dictionary.csv"), index=False)


# %%
